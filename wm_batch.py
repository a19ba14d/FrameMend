"""Run independent watermark jobs with local routing and restartable batch records."""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import cache, partial
from hashlib import sha256
from inspect import getsource
from itertools import combinations
import json
from math import ceil, floor
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from subprocess import CalledProcessError
import sys
from threading import BoundedSemaphore, Lock
from time import monotonic
from zipfile import BadZipFile

import cv2
import numpy as np

import wm_grok
from benchmark_yolo_watermark import detector, predict_video
from wm_director import (ask_with_timeout_recovery, box, classify_layer, confirm_discovery_target, distinct_native_boxes, expand,
                         grok_request_method, job_trace, load_settings, native_discovery, native_tile_boxes, observed_boxes,
                         recipe_coverage_findings, recheck_job,
                         review_method, review_watermark_hints, review_watermark_visibility, run_job, sparse_discovery_samples,
                         uncovered_overlay_regions, uncovered_track_regions, verify_discovery_presence,
                         write_discovery_recipe, xywh)
from wm_discovery import (LAYER_FIELDS, MAX_TEMPLATES, MAX_TRACKS, agrees_in_halves, discovery_tracks, enroll, evidence_threshold,
                          learned_motif, luma_extent_groups, observe, pack_tracks, recover_chroma, remember,
                          remembered_proposals)
from wm_discovery_render import consensus_layers_for_area
from wm_grounding import (CANDIDATE_DECISIONS, CANDIDATE_PROMPT, FIT_INSTRUCTION, LOCAL_ADMISSION_SECONDS,
                          ask_local, filled_mask_evidence,
                          frame_evidence, LocalEvidenceCache, evidence_identity, local_service, locate_named,
                          locate_reference_overlay, locate_source_overlays, preview_residue)
from wm_media import Video, clipped_plane, digest, get_frame, read_frames, yuv_to_rgb
from wm_reference_matte import make_recipe, selected_anchors
from wm_refinements import FILENAME as REFINEMENT_FILE, merge_proposals, read_refinements, save_refinements
from wm_release import exact_coverage_accepted
from wm_track import highpass, load_track
from wm_runtime import WorkflowLimit, check_budget, current_runtime, processing_phase, source_runtime
from workflow import (build_plan, execute_plan, fingerprints, input_files, lock_job, read_object,
                      resume_state, write_object)


FIRST_TURN_GROK_SECONDS = 300
FIRST_TURN_GROK_SLOT_WAIT = 75
COMPATIBLE_ABSENCE_RULES = {
    # Only semaphore waiting acquired a deadline; prompts, images and verdict checks are unchanged.
    'a9df5ea96a93b67a660cc33f3d24ebf28b3837fa44e8efe4f3e788a7e6102d2c':
        'e2a8fcdb612837c21112ef8c1ef9ea33fa7182f6fc1227df034378271e9aa4fa',
    # The native OCR fallback adds evidence before the same independent Grok check.
    'a650e36c512b36cbb5a1cfd39c389fea6c5b9f4455cb8f4d5d5fdea4d598ca28':
        '006933ff7de0f96305c2d28790396e16b20cf034e533ca4182e7a90c98c53fe7',
}


class LocalDetector:
    def __init__(self, weights, *, device=0):
        self.device = device
        self.model = detector(weights, .25, device=device)
        self.lock = Lock()
        self.screen_gate = BoundedSemaphore(3)

    def sample(self, video, cuda, limit=None, *, detail=False):
        step = max(1, (video.count+limit-1)//limit if limit is not None else video.count//16)
        with self.screen_gate:
            sampled = [(i, video.planes(raw)) for i, raw in read_frames(video, cuda=cuda, step=step)]
            images = [yuv_to_rgb(parts, video.matrix)[..., ::-1].copy() for _, parts in sampled]
            with self.lock:
                predictions = self.model.predict(images, imgsz=1280 if detail else 640, batch=8 if detail else 16,
                                                 device=self.device, half=self.device != 'cpu', conf=.15 if detail else .25,
                                                 max_det=80, verbose=False)
        candidates = {}
        for (i, _), result in zip(sampled, predictions, strict=True):
            candidates[i] = [{'rect': box[:4].tolist(), 'confidence': float(box[4])}
                             for box in result.boxes.data.cpu().numpy()]
        return sampled, candidates

    def recipe(self, video, directory, cuda, trace):
        sampled, candidates = self.sample(video, cuda)
        candidates = [candidates.get(i, []) for i in range(video.count)]
        anchors = selected_anchors(video, candidates)
        trace('route.local_screened', sampled=len(sampled), anchors=[i for i, _ in anchors])
        if len(anchors) < 4:
            raise ValueError('Too few samples contain a local watermark candidate.')
        indices = {i for i, _ in anchors}
        signals = {i: highpass(parts[0], 2.2) for i, parts in sampled if i in indices}
        _, _, _, checks = evidence_threshold(learned_motif(anchors, signals), anchors, signals)
        agreement = sum(row['corroborates_visual_box'] for row in checks)
        trace('route.local_agreement', agreement=agreement, total=len(checks), observations=checks)
        if agreement < .75*len(checks):
            raise ValueError('The candidate positions and pixel evidence disagree.')
        report = predict_video(self.model, video.path, video.count, .25, prediction_lock=self.lock, device=self.device)
        write_object(directory/'detections.json', report)
        trace('route.local_scanned', frames=video.count, seconds=report['seconds'])
        return make_recipe(video.path, report['rows'], directory/'reference-plan', None, cuda)


def review_layer_visibility(video, layer, frames, config, cuda, trace, *, require_local=False, hide=None):
    from wm_layers import visible_box

    references = []
    for anchor, (x, y, score) in layer.matches.items():
        left, top, width, height = visible_box(layer.motif, x, y)
        rect = xywh(expand((left, top, left+width, top+height), video, 0, 0))
        references.append((rect[2]*rect[3], score, anchor.frame, rect))
    _, _, frame, rect = max(references)
    return review_watermark_visibility(video, layer.label, [frame, *rect], frames, config, cuda, trace,
                                        require_local=require_local, hide=hide)


def layer_match_evidence(video, layer, frame, position, cuda, *, with_mask=True):
    from wm_layers import crop_for, visible_box

    x, y = position
    footprint = visible_box(layer.motif, x, y)
    region = crop_for(video, layer.motif, position, footprint)
    preview = {'outline': footprint, 'mask_at': (layer.motif.mask, x, y)} if with_mask else {}
    image = frame_evidence(video, frame, cuda, max_side=800, region=region, **preview)
    return frame, region, footprint, image


def confirm_layer_match(video, layer, frame, point, config, cuda, trace):
    from wm_layers import candidate_identifiers

    _, _, footprint, image = layer_match_evidence(video, layer, frame, point[:2], cuda)
    label = json.dumps(layer.label, ensure_ascii=False)
    prompt = (f'Check whether the outlined location contains this distribution watermark: {label}. '
              'Read the actual lettering and emblem in the Original panel. Copy the visible wording, '
              'especially the exact website or account number, into "label". '
              'Use the cyan tint in Mask preview only to check coverage. '
              'Use "matches" only for the same graphic with all its wording and emblem covered. '
              'Use "variant" for a visible distribution overlay with different wording or incomplete coverage. '
              'Use "different" for clear scene content, or "uncertain" if the image is insufficient. '
              'Return JSON with decision, label, fit and a short reason.'+FIT_INSTRUCTION)
    try:
        answer = ask_local(config['discovery_local'], prompt, image, 22, ('matches', 'variant', 'different', 'uncertain'),
                           json_mode=True)
    except (OSError, RuntimeError, ValueError) as error:
        answer = {'decision': 'uncertain', 'reason': str(error)}
    expected = candidate_identifiers(layer.label)
    observed = candidate_identifiers(answer.get('label'))
    conflict = bool(expected and observed and expected.isdisjoint(observed))
    trace('layers.weak_match_checked', frame=frame, footprint=list(footprint), score=point[2], answer=answer,
          identifier_conflict=conflict)
    if answer['decision'] == 'variant' or (answer['decision'] == 'matches' and (conflict or answer.get('fit') == 'partial')):
        return 'replan'
    if answer['decision'] != 'matches' or answer.get('fit') != 'complete':
        return 'unverified'
    required = config.get('required_overlay_regions')
    observation = [frame, *footprint]
    if required is not None and observation not in required:
        required.append(observation)
    return 'matches'


def confirm_fading_frames(video, layer, probes, anchor, config, cuda, trace, *, clipped=False):
    reference_frame, position = anchor
    identity_choices = ('same', 'different', 'uncertain')
    target = json.dumps(layer.label, ensure_ascii=False)
    local_prompt = (f'In the Original panel find the already tracked distribution overlay described as {target}. '
                    'Use the icon and readable strokes to identify the graphic. Do not guess faint characters. '
                    'Use same only when this looks like the described graphic and different for clearly different '
                    'content. Use uncertain if the overlay is unclear. In the cyan Mask preview judge whether '
                    'the visible graphic and lettering are completely covered. Return JSON with decision (same, '
                    'different or uncertain), fit (complete, partial or uncertain) and a short reason.')
    compare = ('The first crop shows the independently verified overlay in a nearby frame. '
               'Compare the actual artwork in each Original panel with that reference. Mark identity same only '
               'for matching visible strokes, icon and layout. Mark different for visibly changed artwork or '
               'lettering, or uncertain if too little is visible. Do not guess or transcribe faint characters. '
               'Use the cyan Mask preview only for fit: complete if all visible target pixels are covered, '
               'partial if a visible part lies outside the mask, and uncertain if you cannot judge. ')
    original_pair = ('Compare two original crops. The first image is a verified reference for a distribution graphic. '
                     'In the second image check whether the visible strokes, lettering and emblem match that graphic. '
                     'Ignore scene content, brightness and opacity. Return JSON with decision same if the visible '
                     'artwork matches, different if the artwork or readable wording visibly changes, or uncertain if '
                     'too little is visible. Do not guess faint characters. Give a short reason.')
    if clipped:
        boundary = 'The graphic can cross an image boundary. Judge only pixels that are actually visible. '
        local_prompt += ' '+boundary
        compare += boundary
        original_pair += ' '+boundary
    candidates = {}
    reference = None
    for frame, point in probes:
        evidence = layer_match_evidence(video, layer, frame, point[:2], cuda)
        try:
            answer = ask_local(config['discovery_local'], local_prompt, evidence[3], 22, identity_choices,
                               max_tokens=170, json_mode=True)
        except (OSError, RuntimeError, ValueError) as error:
            answer = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.fading_match_local', frame=frame, reference=reference_frame, score=point[2], answer=answer)
        if answer.get('fit') != 'complete':
            if not clipped:
                continue
            try:
                filled = filled_mask_evidence(video, frame, cuda, 800, [(layer.motif.mask, *point[:2])],
                                              region=evidence[1])
                preview = preview_residue(config['discovery_local'], filled, 22)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                preview = {'decision': 'uncertain', 'reason': str(error)}
            trace('layers.clipped_match_preview', frame=frame, answer=preview)
            if preview.get('decision') != 'none' or preview.get('label') != '':
                continue
        identity = answer['decision']
        if identity != 'same':
            reference = reference or layer_match_evidence(video, layer, reference_frame, position[:2], cuda,
                                                          with_mask=False)
            raw = layer_match_evidence(video, layer, frame, point[:2], cuda, with_mask=False)
            try:
                comparison = ask_local(config['discovery_local'], original_pair, raw[3], 22, identity_choices,
                                       max_tokens=160, json_mode=True, reference=reference[3])
            except (OSError, RuntimeError, ValueError) as error:
                comparison = {'decision': 'uncertain', 'reason': str(error)}
            trace('layers.fading_match_original_pair', frame=frame, reference=reference_frame, answer=comparison)
            identity = comparison['decision']
        if identity == 'same':
            candidates[frame] = evidence
    if not candidates:
        return set()
    reference = reference or layer_match_evidence(video, layer, reference_frame, position[:2], cuda, with_mask=False)
    images = [(f'Nearby confirmed reference, frame {reference_frame}', reference[3][1])]
    images.extend((f'Candidate Original and Mask preview, frame {i}', row[3][1]) for i, row in candidates.items())
    prompt = (compare+f'Check only the following candidate frames: {list(candidates)}. Return JSON with decision '
              'matches if every identity is same and every fit is complete, review otherwise, a short reason, and '
              'checks with one object per candidate containing integer frame, identity, and fit. No tools.')
    try:
        timeout = min(75 if clipped else 45, config['timeout_seconds'])
        answer = ask_with_timeout_recovery({**config, 'timeout_seconds': timeout},
                                           prompt, images, ('matches', 'review'), trace)
    except (OSError, RuntimeError, ValueError) as error:
        answer = {'decision': 'review', 'reason': str(error)}
    trace('layers.fading_match_review', frames=list(candidates), reference=reference_frame, answer=answer)
    checks = answer.get('checks')
    if (not isinstance(checks, list) or len(checks) != len(candidates)
            or not all(isinstance(row, dict) and type(row.get('frame')) is int
                       and row.get('identity') in identity_choices
                       and row.get('fit') in ('complete', 'partial', 'uncertain')
                       for row in checks)
            or {row['frame'] for row in checks} != set(candidates)):
        return set()
    accepted = {row['frame'] for row in checks if row['identity'] == 'same' and row['fit'] == 'complete'}
    if (answer['decision'] == 'matches') != (accepted == set(candidates)):
        return set()
    required = config.get('required_overlay_regions')
    if required is not None:
        for i in sorted(accepted):
            observation = [i, *candidates[i][2]]
            if observation not in required:
                required.append(observation)
    return accepted


def confirm_short_frames(video, layer, probes, anchor, config, cuda, trace):
    unavailable = False

    def observe_review(event, **fields):
        nonlocal unavailable
        if event == 'layers.fading_match_review':
            answer = fields['answer']
            unavailable = not answer.get('checks') and transient_grok_error(answer.get('reason'))
        trace(event, **fields)

    accepted = confirm_fading_frames(video, layer, probes, anchor, config, cuda, observe_review)
    if unavailable and len(probes) > 1:
        trace('layers.short_review_split', frames=[i for i, _ in probes if i not in accepted])
        for probe in probes:
            if probe[0] not in accepted:
                accepted.update(confirm_fading_frames(video, layer, [probe], anchor, config, cuda, trace))
    return accepted


IDENTITY_CHOICES = ('same', 'variant', 'different', 'uncertain')
IDENTITY_READING = (
    'Read the exact readable website, @handle, or account ID printed in the distribution overlay in each original crop. '
    'Put the first text in reference_identifier and the second in candidate_identifier. '
    'Use an empty string for unreadable or missing text. Use "account ID 123456789" format for an account number. '
    'Different readable identifiers are distinct overlays even if the icon and brand look alike. '
    'Ignore text belonging to the scene. Do not guess missing characters.'
)
IDENTITY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'decision': {'enum': list(IDENTITY_CHOICES)}, 'reference_identifier': {'type': 'string'},
                   'candidate_identifier': {'type': 'string'}, 'reason': {'type': 'string'}},
    'required': ['decision', 'reference_identifier', 'candidate_identifier', 'reason'],
}


def conflicting_pair(answer, decision, expected, alternatives):
    from wm_layers import candidate_identifiers, identifier_kind

    reference = candidate_identifiers(answer.get('reference_identifier'))
    candidate = candidate_identifiers(answer.get('candidate_identifier'))
    if (decision in ('variant', 'different') and len(reference) == len(candidate) == 1 and reference == expected
            and candidate.isdisjoint(expected) and candidate <= alternatives):
        identifier = next(iter(candidate))
        if identifier_kind(identifier) == identifier_kind(next(iter(reference))):
            return identifier
    return None


def read_native_identifier(video, frame, footprint, choices, backend, cuda, trace, read_text):
    from wm_layers import candidate_identifiers, overlap

    x, y, w, h = footprint
    region = xywh(expand((x, y, x+w, y+h), video, 0, 0))
    if min(region[2:]) < 8:
        return None
    try:
        readings = {}
        for text, bounds in read_text(video, frame, region, cuda):
            box = xywh(bounds)
            if min(box[2:]) <= 0:
                continue
            for identifier in candidate_identifiers(text) & choices:
                readings.setdefault(identifier, []).append(box)
        repeated = {identifier for identifier, boxes in readings.items()
                    if any(overlap(a, b) >= .8*min(a[2]*a[3], b[2]*b[3]) for a, b in combinations(boxes, 2))}
        if len(repeated) != 1:
            return None
        image = frame_evidence(video, frame, cuda, 1000, region=region)
        names = locate_source_overlays(backend, image, 22, read_labels=True)
        visual = set().union(*(candidate_identifiers(name) for name, _ in names))
        trace('layers.identifier_native_read', frame=frame, region=region, ocr=sorted(repeated), visual=sorted(visual))
        return next(iter(repeated)) if visual == repeated else None
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        trace('layers.identifier_native_unavailable', frame=frame, reason=str(error))
        return None


def confirm_alternate_identifiers(video, layer, candidates, config, cuda, trace, *, hide, read_text=None):
    from wm_layers import candidate_identifiers

    expected = candidate_identifiers(layer.label)
    anchor = max(layer.matches, key=lambda item: layer.matches[item][2])
    _, _, reference_box, reference = layer_match_evidence(video, layer, anchor.frame, layer.matches[anchor][:2],
                                                           cuda, with_mask=False)
    local_prompt = (IDENTITY_READING+' Return JSON with decision same if the printed design and readable identifier '
                    'match, variant if the brand is visually similar but the readable identifier differs, different '
                    'for unrelated content, or uncertain when the evidence is insufficient. Give a brief reason.')
    backend = config['discovery_local']
    reference_read = None
    validated = []
    for i, (point, alternatives) in candidates.items():
        _, _, footprint, raw = layer_match_evidence(video, layer, i, point[:2], cuda, with_mask=False)
        try:
            answer = ask_local(backend, local_prompt, raw, 22, IDENTITY_CHOICES, max_tokens=200,
                               schema=IDENTITY_SCHEMA, reference=reference)
        except (OSError, RuntimeError, ValueError) as error:
            answer = {'decision': 'uncertain', 'reason': str(error)}
        alternate = conflicting_pair(answer, answer['decision'], expected, alternatives)
        trace('layers.identifier_pair_local', label=layer.label, frame=i, answer=answer, alternate=alternate)
        if not alternate and read_text is not None and len(expected) == 1 and answer['decision'] != 'same':
            if reference_read is None:
                reference_read = read_native_identifier(video, anchor.frame, reference_box, expected, backend,
                                                        cuda, trace, read_text) or ''
            if reference_read:
                candidate_read = read_native_identifier(video, i, footprint, alternatives, backend, cuda,
                                                         trace, read_text)
                alternate = conflicting_pair({'reference_identifier': reference_read,
                                              'candidate_identifier': candidate_read},
                                             'variant', expected, alternatives)
        if alternate:
            validated.append((i, raw, alternate))
    accepted = set()
    for offset in range(0, len(validated), 2):
        group = validated[offset:offset+2]
        frames = {i for i, _, _ in group}
        images = [('Verified reference crop, first image', reference[1])]
        for i, raw, _ in group:
            images.append((f'Original candidate crop for frame {i}', raw[1]))
            images.append(filled_mask_evidence(video, i, cuda, 900, hide[i]))
        prompt = (IDENTITY_READING+f' The labeled candidate frames are {sorted(frames)}. For each frame, read the '
                  'candidate identifier from its Original crop. The full preview replaces only pixels covered by '
                  'independently located overlay masks. In that full view search anywhere for the exact reference '
                  'overlay. Mark remaining present if it is still recognizable, uncertain if any surviving fragment '
                  'could be the reference, or absent if only distinctly different overlays or scene content remain. '
                  'Return JSON with decision checked, reason, and checks with integer frame, reference_identifier, '
                  'candidate_identifier, identity (same, variant, different, uncertain), and remaining for each frame. No tools.')
        try:
            answer = ask_with_timeout_recovery({**config, 'timeout_seconds': min(60, config['timeout_seconds'])},
                                               prompt, images, ('checked',), trace)
        except (OSError, RuntimeError, ValueError) as error:
            answer = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.identifier_pair_review', label=layer.label, frames=sorted(frames), answer=answer)
        checks = answer.get('checks')
        if (not isinstance(checks, list) or len(checks) != len(frames)
                or not all(isinstance(row, dict) and type(row.get('frame')) is int for row in checks)
                or {row['frame'] for row in checks} != frames):
            continue
        by_frame = {i: alternate for i, _, alternate in group}
        accepted.update(row['frame'] for row in checks if row.get('remaining') == 'absent' and conflicting_pair(
            row, row.get('identity'), expected, {by_frame[row['frame']]}))
    return accepted


def joined_component_boxes(rows):
    joined = []
    for a, b in combinations(rows, 2):
        aw, ah, bw, bh = a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]
        dx, dy = max(a[0], b[0])-min(a[2], b[2]), max(a[1], b[1])-min(a[3], b[3])
        aligned = (0 <= dx <= .55*min(aw, bw) and -dy >= .45*min(ah, bh)
                   or 0 <= dy <= .55*min(ah, bh) and -dx >= .45*min(aw, bw))
        if aligned:
            rect = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            if (rect[2]-rect[0])*(rect[3]-rect[1]) <= 2.2*(aw*ah+bw*bh) and rect not in joined:
                joined.append(rect)
        if len(joined) == 2:
            break
    return joined


def preview_residual_boxes(video, frame, region, hint, motif, backend, cuda, trace):
    left, top, *_ = motif.bounds
    origin = hint[0]-left, hint[1]-top
    image = filled_mask_evidence(video, frame, cuda, 1000, [(motif.mask, *origin)], region=region)
    rows = locate_named(backend, None, image, 22)
    x, y, w, h = region
    result = []
    for a, b, c, d in rows:
        left, top = x+floor(a*w/1000), y+floor(b*h/1000)
        width, height = x+ceil(c*w/1000)-left, y+ceil(d*h/1000)-top
        if min(width, height) < 3 or width*height > .5*w*h:
            continue
        filled, _ = clipped_plane(motif.mask, (left-origin[0], top-origin[1], width, height))
        if int(filled.sum()) < .5*width*height:
            result.append([a, b, c, d])
    trace('layers.preview_residual_proposals', frame=frame, region=region, proposals=result, detected=len(rows))
    return result


def refine_layer_candidates(video, candidates, gap, backend, cuda, trace, *, sampled=None):
    from wm_layers import build_layer, diverse_frames, overlap, proposals_for, remainder_observations, visible_box

    extra = {}
    hints = dict(gap.hints)
    hidden = getattr(gap, 'hidden_masks', {})
    masked_frames = sorted(hidden)
    for index in np.linspace(0, len(masked_frames)-1, min(4, len(masked_frames)), dtype=int):
        i = masked_frames[index]
        motif = gap.layer.motif
        anchors = diverse_frames(gap.layer.matches)
        reference, position = min(anchors, key=lambda row: (abs(row[0].frame-i), -row[1][2]))
        x, y, w, h = visible_box(motif, *position[:2])
        scope = xywh(expand((x, y, x+w, y+h), video, 0, 0))
        try:
            original = frame_evidence(video, reference.frame, cuda, 640, scope)
            preview = filled_mask_evidence(video, i, cuda, 1000, hidden[i])
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('layers.remainder_grounding_unavailable', frame=i, reason=str(error))
            continue
        rows = []
        for locate in (partial(locate_named, backend, None, preview, 22),
                       partial(locate_reference_overlay, backend, preview, original, 22)):
            try:
                rows.extend(row for row in locate() if row not in rows)
            except (OSError, RuntimeError, ValueError) as error:
                trace('layers.remainder_grounding_unavailable', frame=i, reason=str(error))
        for x, y, w, h in (xywh(box(row, video)) for row in rows):
            if min(w, h) < 8 or w*h > .18*video.width*video.height:
                continue
            filled = np.zeros((h, w), dtype=bool)
            for mask, u, v in hidden[i]:
                view, section = clipped_plane(mask, (x-u, y-v, w, h))
                filled[section] |= view > 0
            if filled.mean() >= .50:
                continue
            if i not in extra:
                hints[i] = (x, y, w, h)
            rect = [x, y, x+w, y+h]
            extra.setdefault(i, []).append({'rect': rect, 'provider': 'qwen-visible-remainder'})
            trace('layers.remainder_proposal', frame=i, rect=rect, reference=reference.frame)
            if len(extra[i]) == 3:
                break
    remainders = set(extra)
    reviewed = sampled if sampled is not None else set()
    focused = [i for i in gap.frames if i in hints and i not in reviewed]
    missing = [i for i in gap.frames if i not in hints and i not in reviewed]
    focused_count = min(len(focused), 5 if missing else 6)
    indices = [focused[i] for i in np.linspace(0, len(focused)-1, focused_count, dtype=int)]
    indices += [missing[i] for i in np.linspace(0, len(missing)-1, min(6-focused_count, len(missing)), dtype=int)]
    visible = gap.visible_frames
    sampled_visible = [visible[i] for i in np.linspace(0, len(visible)-1, min(3, len(visible)), dtype=int)]
    reported = list(dict.fromkeys([*extra, *sampled_visible]))[:3]
    offset = max(2, round(float(video.fps)/5))
    tiled_probes = set()
    cluster, cluster_span = [], None
    if reported:
        spans = np.split(np.asarray(gap.frames), np.flatnonzero(np.diff(gap.frames) > 1)+1)
        for span in sorted((span for span in spans if len(span) >= 3), key=len, reverse=True):
            low, high = int(span[0]), int(span[-1])
            probes = [i for i in visible if low <= i <= high]
            if not probes:
                continue
            center = probes[len(probes)//2]
            neighbors = [max(low, min(high, center+step*offset)) for step in (-1, 1, -2, 2)]
            cluster = list(dict.fromkeys([center, *neighbors, (low+high)//2]))[:3]
            cluster_span = low, high
            break
        nearby = [i+step for step in (-offset, offset) for i in reported if 0 <= i+step < video.count]
        indices = list(dict.fromkeys([*cluster, *reported, *nearby, *indices]))[:6]
        tiled_probes.update(indices[:3])
        trace('layers.visible_probes_replanned', frames=reported, sampled=indices, tiled=indices[:3], cluster=cluster)
    contextual = set(indices[::2]) if gap.needs_classification else set()
    if (gap.partial_mask or gap.needs_classification) and len(indices) < 6:
        motif = gap.layer.motif
        left, top, _, _ = motif.bounds
        for frame in indices[:]:
            for i in (frame+offset, frame-offset):
                if len(indices) == 6:
                    break
                if not 0 <= i < video.count or i in candidates or i in hints or i in reviewed:
                    continue
                x, y, w, h = hints[frame]
                radius = min(48, max(12, w//4, h//2))
                signal = highpass(video.planes(get_frame(video, i, cuda))[0], 2.2)
                a, b, score = motif.local(signal, (x-left, y-top), radius, radius, reversible=True)
                if abs(score) >= max(.50, gap.layer.floor-.12) and agrees_in_halves(
                        motif, signal, a, b, -1 if score < 0 else 1):
                    hints[i] = visible_box(motif, a, b)
                    x, y, w, h = hints[i]
                    rect = [x, y, x+w, y+h]
                    extra[i] = [{'rect': rect, 'provider': 'temporal-pixels'}]
                    indices.append(i)
                    trace('layers.temporal_proposal', frame=i, anchor=frame, rect=rect, score=score,
                          needs_classification=gap.needs_classification)
    for i in indices:
        reviewed.add(i)
        hint = hints.get(i)
        region = None
        residual, separate = [], []
        if hint is not None and i not in contextual:
            x, y, w, h = hint
            if gap.partial_mask or i in reported:
                px, py = max(16, w//2), max(16, h)
            else:
                px, py = max(12, w//5), max(12, h//2)
            left, top = max(0, x-px), max(0, y-py)
            right, bottom = min(video.width, x+w+px), min(video.height, y+h+py)
            if right-left >= 8 and bottom-top >= 8:
                region = left, top, right-left, bottom-top
        try:
            image = frame_evidence(video, i, cuda, max_side=1440, region=region)
            label = '' if i in reported or gap.needs_classification else gap.layer.label
            rows = locate_named(backend, label, image, 22, tight=True) if label else []
            if not rows:
                rows = locate_named(backend, None, image, 22)
            if region is not None and gap.partial_mask:
                try:
                    residual = preview_residual_boxes(video, i, region, hint, gap.layer.motif, backend, cuda, trace)
                except (OSError, RuntimeError, ValueError, cv2.error) as error:
                    trace('layers.preview_residual_unavailable', frame=i, reason=str(error))
                else:
                    rows += [row for row in residual if row not in rows]
            if region is None:
                if i in tiled_probes:
                    rows += native_tile_boxes(video, i, cuda, backend, trace)
                rows = distinct_native_boxes(video, rows)
                joined = joined_component_boxes(rows) if i in tiled_probes else []
                boxes = [box(row, video) for row in [*joined, *rows]]
            else:
                x, y, w, h = region
                boxes = []
                joined = joined_component_boxes(rows) if gap.partial_mask else []
                for a, b, c, d in [*joined, *rows]:
                    if (c-a)*(d-b) > 900000:
                        continue
                    left, top = x+floor(a*w/1000), y+floor(b*h/1000)
                    right, bottom = x+ceil(c*w/1000), y+ceil(d*h/1000)
                    footprint = left, top, right-left, bottom-top
                    if overlap(footprint, hint) >= .50*min(footprint[2]*footprint[3], hint[2]*hint[3]):
                        boxes.append((left, top, right, bottom))
                    elif [a, b, c, d] in residual and min(footprint[2:]) >= 8:
                        separate.append([left, top, right, bottom])
                        u, v, width, height = hint
                        joined = (max(0, min(u, left)), max(0, min(v, top)),
                                  min(video.width, max(u+width, right)), min(video.height, max(v+height, bottom)))
                        if joined not in boxes:
                            boxes.append(joined)
        except (OSError, RuntimeError, ValueError) as error:
            trace('layers.refinement_rejected', frame=i, reason=str(error))
            continue
        if separate:
            extra[i] = ([{'rect': rect, 'provider': 'qwen-visible-remainder'} for rect in separate]
                        + extra.get(i, []))
            remainders.add(i)
            trace('layers.disconnected_remainders', frame=i, hint=hint, proposals=separate)
        if boxes:
            proposals = [{'rect': list(rect), 'provider': 'qwen-crop' if region else 'qwen-native'} for rect in boxes]
            extra[i] = extra[i]+proposals if i in remainders else proposals+extra.get(i, [])
        trace('layers.refined_candidates', frame=i, label=gap.layer.label, boxes=boxes, hint=hint, crop=region)
    if cluster_span is not None and not gap.partial_mask:
        center = cluster[0]
        near = {i: rows for i, rows in extra.items() if abs(i-center) <= 2*offset}
        proposals = proposals_for(video, near)
        if remainders.intersection(near):
            proposals = remainder_observations(video, near, proposals)
        sampled = sorted({item.frame for item in proposals})
        repeated = len(sampled) >= 3 and build_layer(observe(video, [(i, None) for i in sampled], cuda), proposals)
        if not repeated:
            step = max(1, round(float(video.fps)/15))
            low, high = max(0, cluster_span[0]-step), min(video.count-1, cluster_span[1]+step)
            dense = [center+delta*step for delta in (-1, 1, -2, 2, -3, 3, -4, 4)]
            dense = [i for i in dense if low <= i <= high and i not in indices][:7]
            if dense:
                try:
                    decision = native_discovery(video, dense, cuda, backend, '', trace, tiled=True)
                    extra = merge_visual_proposals(video, extra, {'rounds': [{'frames': dense, 'decision': decision}]},
                                                   provider='qwen-dense-replan')
                    trace('layers.dense_replan_proposals', anchor=center, frames=dense,
                          observations=len(decision.get('observations', [])))
                except (OSError, RuntimeError, ValueError, cv2.error) as error:
                    trace('layers.dense_replan_unavailable', anchor=center, reason=str(error))
    return extra


@cache
def motif_review_rules():
    from wm_grounding import CANDIDATE_SCHEMA, classify_candidate, crop_detail, jpeg, local_request_body, parse_local_reply
    from wm_layers import (ACCOUNT_ID, IDENTIFIER, INDEPENDENT_CONTEXT, candidate_identifiers, confirm_layer,
                           crop_for, layer_review, overlay_fit, review_anchors, visible_box)

    rules = (classify_layer, classify_candidate, frame_evidence, crop_detail, jpeg, ask_local,
             local_request_body, parse_local_reply,
             review_anchors, layer_review, crop_for, visible_box, candidate_identifiers, overlay_fit, confirm_layer)
    text = '\n'.join([CANDIDATE_PROMPT, FIT_INSTRUCTION, json.dumps(CANDIDATE_SCHEMA, sort_keys=True),
                      IDENTIFIER.pattern, ACCOUNT_ID.pattern, str(INDEPENDENT_CONTEXT),
                      *(getsource(rule) for rule in rules)])
    return sha256(text.encode('utf-8')).hexdigest()


def motif_review_method(config):
    return {'rules_sha256': motif_review_rules(), **grok_request_method(config),
            'local': config['discovery_local']['model']}


@cache
def absence_review_rules():
    from wm_director import (VISUAL_ONLY_RETRY, ask, ask_cached_visibility, compact_images, local_visibility_crops,
                             reference_read_for_absence, review_watermark_visibility, visibility_reply)
    from wm_grounding import (SOURCE_LABEL_PROMPT, crop_detail, filled_mask_evidence, frame_tiles, jpeg,
                              local_request_body, parse_local_reply)
    from wm_layers import (ACCOUNT_ID, IDENTIFIER, MASKED_REVIEW_MINIMUM, absence_identity, absence_intervals,
                           candidate_identifiers, identifier_kind,
                           masked_proof_keys, overlapping_identifiers, review_anchored_matches, review_fading_matches,
                           review_overlapping_matches, uncovered_absence)

    rules = (review_layer_visibility, review_watermark_visibility, local_visibility_crops, reference_read_for_absence,
             confirm_layer_match, confirm_fading_frames,
             confirm_alternate_identifiers, read_native_identifier, locate_source_overlays,
             conflicting_pair, overlapping_identifiers, identifier_kind,
             layer_match_evidence, frame_evidence, filled_mask_evidence, frame_tiles, preview_residue, crop_detail, jpeg, ask_local,
             local_request_body, parse_local_reply,
             candidate_identifiers,
             ask_with_timeout_recovery, ask, ask_cached_visibility, visibility_reply, compact_images, absence_identity, absence_intervals,
             masked_proof_keys, review_anchored_matches, review_fading_matches, review_overlapping_matches,
             uncovered_absence)
    text = '\n'.join([FIT_INSTRUCTION, VISUAL_ONLY_RETRY, IDENTIFIER.pattern, ACCOUNT_ID.pattern, IDENTITY_READING,
                      SOURCE_LABEL_PROMPT,
                      json.dumps(IDENTITY_SCHEMA, sort_keys=True), str(MASKED_REVIEW_MINIMUM),
                      *(getsource(rule) for rule in rules)])
    return sha256(text.encode('utf-8')).hexdigest()


def absence_review_method(config):
    return {'rules_sha256': absence_review_rules(), **grok_request_method(config),
            'local': config['discovery_local']['model']}


def same_absence_scope(saved, scope):
    if saved.get('candidate_sha256') != scope.get('candidate_sha256'):
        return False
    if any(saved.get(key) != value for key, value in scope.items() if key != 'review_method'):
        return False
    method = scope['review_method']
    earlier = saved.get('review_method')
    legacy = COMPATIBLE_ABSENCE_RULES.get(method.get('rules_sha256'))
    return earlier == method or (legacy is not None and earlier == {**method, 'rules_sha256': legacy})


@cache
def review_checkpoint_rules():
    return sha256(b'\n'.join(Path(__file__).with_name(name).read_bytes() for name in
                            ('wm_layers.py', 'wm_overlap.py', 'wm_discovery.py', 'wm_gpu_search.py'))).hexdigest()


def prior_review_checkpoint(directories, identity, method):
    candidates = []
    for directory in directories:
        for recipe in directory.glob('pass-*/coverage-checkpoint-*/discovery/recipe.json'):
            try:
                profile = read_object(recipe.parent/read_object(recipe)['profile'])['discovery']
                if profile.get('coverage_review_only') is not True or profile.get('review_checkpoint_method') != method:
                    continue
                asset = Path(profile['track'])
                if digest(asset) != profile['sha256']:
                    continue
                with np.load(asset, allow_pickle=False) as saved:
                    if str(saved['source_sha256']) != identity:
                        continue
                candidates.append(recipe)
            except (OSError, ValueError, KeyError, BadZipFile, EOFError):
                continue
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def reconcile_layer_reviews(answers, frames):
    from wm_layers import candidate_identifiers, overlay_fit

    if not answers:
        return None, None
    roles = {answer['decision'] for answer in answers}
    identifiers = {frozenset(names) for answer in answers if answer['decision'] == 'watermark'
                   and (names := candidate_identifiers(answer.get('label')))}
    if len(roles) > 1 or len(identifiers) > 1:
        return {'decision': 'uncertain', 'reason': 'Recorded reviews of the same pixels disagree about the overlay.'}, 'identity'
    partials = [answer for answer in answers if answer['decision'] == 'watermark' and overlay_fit(answer, frames) == 'partial']
    return (partials or answers)[0], 'fit' if 0 < len(partials) < len(answers) else None


def cached_layer_reviewer(video, config, cuda, trace, cache_path, *, prior_paths=()):
    from wm_layers import overlay_fit

    scope = {'source_sha256': config['absence_source_sha256'], 'review_method': motif_review_method(config)}
    ambiguous = {}
    peers = dict.fromkeys(path for path in prior_paths if path != cache_path)

    def load_entries(path):
        try:
            saved = read_object(path)
            if (all(saved.get(key) == value for key, value in scope.items())
                    and isinstance(saved.get('entries'), dict)):
                return saved['entries']
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            trace('layers.review_cache_rejected', path=str(path), reason=str(error))
        return {}

    entries = load_entries(cache_path)

    def prior_entries(path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            peers[path] = None
            return {}
        except OSError as error:
            trace('layers.review_cache_rejected', path=str(path), reason=str(error))
            return {}
        stamp = stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        previous = peers[path]
        if previous is not None and previous[0] == stamp:
            return previous[1]
        fresh = load_entries(path)
        peers[path] = stamp, fresh
        if fresh:
            trace('layers.prior_reviews_imported', path=str(path), verdicts=len(fresh))
        return fresh

    def reusable(answer, frames):
        if not isinstance(answer, dict) or not isinstance(answer.get('reason'), str) or not answer['reason'].strip():
            return False
        return (answer.get('decision') == 'protected'
                or (answer.get('decision') == 'watermark' and overlay_fit(answer, frames) in ('complete', 'partial')))

    def cached_answer(key, frames):
        answers = [answer for records in (entries, *(prior_entries(path) for path in peers))
                   if reusable(answer := records.get(key), frames)]
        answer, conflict = reconcile_layer_reviews(answers, frames)
        if conflict == 'identity':
            trace('layers.review_identity_conflict', frames=frames, request_sha256=key, verdicts=len(answers))
        elif conflict == 'fit':
            trace('layers.review_fit_conflict', frames=frames, request_sha256=key, retained='partial')
        return answer

    def review(evidence):
        frames = [frame for frame, *_ in evidence]
        parts = [(frame, rect, footprint, sha256(image[1]).hexdigest())
                 for frame, rect, footprint, image in evidence]
        key = sha256(json.dumps(parts, separators=(',', ':')).encode('utf-8')).hexdigest()
        if (answer := cached_answer(key, frames)) is not None:
            trace('layers.review_reused', frames=frames)
            return answer
        if key in ambiguous:
            trace('layers.ambiguous_review_reused', frames=frames, request_sha256=key)
            return ambiguous[key]

        observed = None

        def review_trace(event, **fields):
            nonlocal observed
            if event == 'grok.decision':
                observed = fields['response']
            trace(event, **fields)

        answer = classify_layer(video, evidence, config=config, cuda=cuda, trace=review_trace)
        if reusable(answer, frames):
            entries[key] = answer
            try:
                write_object(cache_path, {**scope, 'entries': entries})
            except OSError as error:
                trace('layers.review_cache_write_failed', reason=str(error))
            return cached_answer(key, frames)
        elif observed is not None and answer is observed and answer.get('decision') == 'uncertain':
            # Replanning must gather different evidence. Service errors still get a retry.
            ambiguous[key] = answer
        return answer

    return review


def local_tracks(video, candidates, config, cuda, trace, *, fallback=None, persist_layers=False, read_text=None,
                 positive_only=False):
    from wm_layers import (LayerNeedsEvidence, candidate_identifiers, confirm_layer, discover_layers, expand_reviewed_layer,
                           load_layer_evidence, proposals_for, retain_confirmed_layer, save_layer_evidence, scan_layers)

    trace('layers.started', sampled_frames=len(candidates), candidates=sum(map(len, candidates.values())))
    fallback = fallback or partial(classify_layer, video, config=config, cuda=cuda, trace=trace)
    visibility = partial(review_layer_visibility, video, config=config, cuda=cuda, trace=trace)

    def borderline(layer, frames, **options):
        backend = {**config['discovery_local'], 'admission_budget': {'remaining': LOCAL_ADMISSION_SECONDS}}
        return visibility(layer, frames, config={**config, 'discovery_local': backend}, require_local=True, **options)

    corroborate = partial(confirm_layer_match, video, config=config, cuda=cuda, trace=trace)
    screen_short = partial(corroborate, config={**config, 'required_overlay_regions': None})
    corroborate_fade = partial(confirm_fading_frames, video, config=config, cuda=cuda, trace=trace)
    corroborate_short = partial(confirm_short_frames, video, config=config, cuda=cuda, trace=trace)
    corroborate_edge = partial(corroborate_fade, clipped=True)
    corroborate_identifiers = partial(confirm_alternate_identifiers, video, config=config, cuda=cuda, trace=trace,
                                      read_text=read_text)
    backend = config['discovery_local']
    proof_file = config.get('absence_proof_file')
    ledger = None
    prior_checkpoints = [Path(directory) for directory in config.get('prior_layer_checkpoints', ())]
    if proof_file:
        proof_file = Path(proof_file)
        launcher = backend.get('launcher', {})
        provenance = {'source_sha256': config['absence_source_sha256'], 'review_method': absence_review_method(config),
                      'local_endpoint': backend['endpoint'],
                      'local_launcher': {key: launcher.get(key) for key in ('executable', 'weights', 'projector', 'device')}}
        if candidate_identity := config.get('absence_candidate_sha256'):
            provenance['candidate_sha256'] = candidate_identity
        ledger = {**provenance, 'proofs': {}}
        for path in [*(directory/proof_file.name for directory in prior_checkpoints), proof_file]:
            if not path.is_file():
                continue
            try:
                saved = read_object(path)
                if (same_absence_scope(saved, provenance)
                        and isinstance(saved.get('proofs'), dict)
                        and all(isinstance(value, dict) for value in saved['proofs'].values())):
                    for layer, proofs in saved['proofs'].items():
                        ledger['proofs'].setdefault(layer, {}).update(proofs)
                    if path != proof_file:
                        trace('layers.prior_absence_proofs_imported', path=str(path),
                              intervals=sum(map(len, saved['proofs'].values())))
            except (OSError, ValueError) as error:
                trace('layers.absence_cache_rejected', path=str(path), reason=str(error))

    def save_proofs():
        write_object(proof_file, ledger)

    cache = proof_file.with_name('confirmed-motifs.npz') if proof_file and persist_layers else None
    motif_provenance = {**provenance, 'review_method': motif_review_method(config)} if cache else None
    refinement_path = cache.with_name(REFINEMENT_FILE) if cache else None
    saved_refinements = {}
    if refinement_path:
        source_size = video.count, video.width, video.height
        source_identity = provenance['source_sha256']
        peer_paths = map(Path, config.get('prior_refinement_checkpoints', ()))
        for path in dict.fromkeys([*peer_paths, refinement_path]):
            recovered = read_refinements(path, source_identity, size=source_size)
            if path == refinement_path:
                saved_refinements = recovered
            if recovered:
                priority = {i: [row for row in candidates.get(i, ()) if row.get('provider') == 'confirmed-source']
                            + rows for i, rows in recovered.items()}
                candidates = merge_proposals(candidates, priority)
                trace('layers.refinements_resumed', path=str(path), frames=len(recovered),
                      proposals=sum(map(len, recovered.values())))

    def save_confirmed(layers):
        try:
            save_layer_evidence(cache, layers, motif_provenance)
        except (OSError, ValueError) as error:
            trace('layers.motif_cache_write_failed', reason=str(error))

    retained = []
    saved_signals = {}
    required = config.get('required_overlay_regions')
    checkpoints = ([cache, *(directory/cache.name for directory in reversed(prior_checkpoints))]
                   if cache else [])
    for checkpoint in dict.fromkeys(checkpoints):
        if not checkpoint.is_file():
            continue
        try:
            rejected = partial(trace, 'layers.motif_checkpoint_layer_rejected', path=str(checkpoint))
            saved, earlier = load_layer_evidence(checkpoint, video, provenance['source_sha256'], on_rejected=rejected)
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as error:
            trace('layers.motif_cache_rejected', path=str(checkpoint), reason=str(error))
        else:
            same_review = earlier == motif_provenance
            if checkpoint != cache and not same_review:
                trace('layers.prior_motif_method_changed', path=str(checkpoint),
                      changed=sorted(key for key in earlier.keys() | motif_provenance.keys()
                                     if earlier.get(key) != motif_provenance.get(key)))
                continue
            if not same_review:
                same_review = earlier == {**provenance, 'review_method': review_method(config)}
            approved = saved
            if not same_review:
                approved = []
                frames = sorted({item.frame for layer in saved for item in layer.matches})
                signals = observe(video, [(i, None) for i in frames], cuda)
                for layer in saved:
                    status = confirm_layer(video, layer, signals, backend, cuda, trace, fallback, prior_watermark=True)
                    if status == 'partial' and layer.identifier_support is None:
                        layer, status = expand_reviewed_layer(video, layer, signals, backend, cuda, trace, fallback)
                    if status == 'confirmed':
                        approved.append(layer)
            deferred = 0
            primary_checkpoint = not retained
            known_identifiers = {name for layer in retained for name in candidate_identifiers(layer.label)}
            for layer in approved:
                if required is not None:
                    required.extend(region for region in map(list, layer.reviewed_regions) if region not in required)
                if retain_confirmed_layer(retained, layer, saved_signals, video, cuda, trace):
                    continue
                names = candidate_identifiers(layer.label)
                adds_identity = len(names) == 1 and names.isdisjoint(known_identifiers)
                if len(retained) < MAX_TEMPLATES and (primary_checkpoint or adds_identity):
                    retained.append(layer)
                    known_identifiers.update(names)
                else:
                    deferred += 1
            trace('layers.confirmed_motifs_resumed', loaded=len(saved), retained=len(retained), deferred=deferred,
                  reviewed_again=not same_review, path=str(checkpoint), prior_run=checkpoint != cache)
    del saved_signals

    def proof_count():
        return sum(map(len, ledger['proofs'].values())) if ledger else 0

    runtime = current_runtime()
    location_cache = (runtime.locations if runtime is not None else {}) if video.count <= 12000 else None
    partial_samples = {}
    mask_replans = 0
    coverage_replans = 0
    while True:
        check_budget()
        layers = None
        before = proof_count()
        try:
            layers = discover_layers(video, candidates, backend, cuda, trace, fallback=fallback,
                                     required=required, retained=retained,
                                     save_confirmed=save_confirmed if cache else None,
                                     read_text=read_text)
            tracks = scan_layers(video, layers, cuda, trace, confirm_absence=visibility, confirm_match=corroborate,
                                 confirm_borderline=borderline, confirm_masked=borderline, confirm_fade=corroborate_fade,
                                 confirm_identifiers=corroborate_identifiers, confirm_edge=corroborate_edge,
                                 screen_short=screen_short, confirm_short=corroborate_short,
                                 absence_proofs=ledger['proofs'] if ledger else None,
                                 save_proofs=save_proofs if ledger else None,
                                 replan_on_visible=coverage_replans == 0, location_cache=location_cache,
                                 gpu_search=config.get('gpu_pixel_search', False),
                                 pixel_workers=config.get('pixel_verify_workers', 4), positive_only=positive_only)
            break
        except LayerNeedsEvidence as gap:
            if runtime is not None and not gap.partial_mask:
                key = (gap.layer.label, sha256(gap.layer.motif.mask.tobytes()).hexdigest())
                if not runtime.gap_progress(key, gap.frames):
                    trace('layers.replan_stalled', reason=str(gap), frames=list(gap.frames))
                    raise
            sample_options = {}
            if gap.partial_mask:
                if mask_replans >= config.get('max_mask_replans', 3):
                    raise
                mask = gap.layer.motif.mask
                positions = tuple(sorted((i, tuple(rect)) for i, rect in gap.hints.items()))
                signature = mask.shape, sha256(mask.tobytes()).digest(), positions
                reviewed = partial_samples.get(signature)
                if reviewed is not None and (not reviewed or not set(gap.hints).difference(reviewed)):
                    trace('layers.replan_repeated_mask', reason=str(gap))
                    raise
                if reviewed:
                    trace('layers.replan_fresh_mask_samples', remaining=sorted(set(gap.hints).difference(reviewed)),
                          previously_sampled=sorted(reviewed))
                sample_options['sampled'] = partial_samples.setdefault(signature, set())
            elif coverage_replans >= config.get('max_coverage_replans', 2):
                raise
            advanced = bool(layers) and proof_count() > before
            refined = {} if advanced and not (gap.visible_frames or gap.hints or gap.partial_mask) else (
                refine_layer_candidates(video, candidates, gap, backend, cuda, trace, **sample_options))
            if refined and refinement_path:
                try:
                    previous = saved_refinements
                    saved_refinements = save_refinements(refinement_path, source_identity, source_size,
                                                        previous, refined)
                    if saved_refinements != previous:
                        trace('layers.refinements_saved', path=str(refinement_path), frames=len(saved_refinements),
                              proposals=sum(map(len, saved_refinements.values())))
                except OSError as error:
                    trace('layers.refinements_save_failed', path=str(refinement_path), reason=str(error))
            if not refined and not advanced:
                raise
            if ((mask_replans or coverage_replans) and not advanced
                    and set(proposals_for(video, refined)).issubset(proposals_for(video, candidates))):
                trace('layers.replan_no_new_mask_proposals', reason=str(gap))
                raise
            if layers and gap.visible_frames and len(refined) >= 3:
                candidates = refined
            else:
                candidates = {**candidates, **{i: rows+list(candidates.get(i, ())) for i, rows in refined.items()}}
            if layers or gap.confirmed:
                retained = layers or gap.confirmed
            if gap.partial_mask:
                mask_replans += 1
            else:
                coverage_replans += 1
            trace('layers.replanning', reason=str(gap), sampled_frames=len(candidates), added_proposals=len(refined),
                  new_absence_evidence=proof_count()-before, round=mask_replans+coverage_replans,
                  partial_mask=gap.partial_mask)
    labels = [layer.label for layer in layers]
    trace('layers.verified', labels=labels, tracks=len(tracks), frames=video.count)
    return tracks, labels


def local_layer_recipe(video, directory, identity, candidates, config, cuda, trace, *, fallback=None, read_text=None):
    tracks, labels = local_tracks(video, candidates, config, cuda, trace, fallback=fallback, persist_layers=True,
                                  read_text=read_text)
    track = pack_tracks(tracks)
    tightened = consensus_layers_for_area(video, track, cuda, trace)
    if tightened:
        missing = uncovered_track_regions(video, tightened, config.get('required_overlay_regions'))
        if missing:
            trace('layers.mask_area_consensus_coverage_rejected', regions=missing)
            raise ValueError('Tightening the proposed masks would omit previously confirmed overlays.')
    settings = {'consensus_mask': True} if tightened else {}
    if (runtime := current_runtime()) is not None and runtime.overlap_frames:
        from wm_overlap import review_samples
        settings['overlap_review_frames'] = review_samples(runtime.overlap_frames)
    directory.mkdir()
    return write_discovery_recipe(video, directory, identity, track, None, ' / '.join(labels), {}, trace,
                                  projection=True, retained_settings=settings)


def feedback_proposals(video, hint, review_frames, config, cuda, trace, *, retained=()):
    frame = hint['frame']
    if type(frame) is not int or not 0 <= frame < video.count:
        raise ValueError('The reported watermark frame is outside the source.')

    def clipped(bounds):
        if len(bounds) != 4 or any(type(value) is not int for value in bounds):
            raise ValueError('The reported watermark needs a source rectangle.')
        x, y, w, h = bounds
        rect = xywh(expand((x, y, x+w, y+h), video, 0, 0))
        if min(rect[2:]) < 8:
            raise ValueError('The reported watermark rectangle is too small or outside the source.')
        return rect

    x, y, w, h = clipped(hint['rect'])
    backend = config['discovery_local']
    search = xywh(expand((x, y, x+w, y+h), video, max(16, w//3), max(12, h//2)))
    verified = {}
    for row in hint.get('observations', [])[:6]:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            continue
        i, *bounds = row
        if type(i) is int and 0 <= i < video.count:
            try:
                verified[i] = clipped(bounds)
            except ValueError:
                continue

    def evidence(i, region):
        hidden = [(layer['mask'], *map(int, layer['positions'][i])) for layer in retained
                  if 'present' not in layer or layer['present'][i]]
        if hidden:
            return filled_mask_evidence(video, i, cuda, 1200, hidden, region=region)
        return frame_evidence(video, i, cuda, max_side=1200, region=region)

    def identify(i, region, previous):
        try:
            answer = ask_local(backend, CANDIDATE_PROMPT+' This crop has no outline. Read the visible '
                               'distribution overlay only from actual lettering and emblems.',
                               evidence(i, region), 22, CANDIDATE_DECISIONS, json_mode=True)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('feedback.identity_unavailable', frame=i, reason=str(error))
            return previous
        trace('feedback.identity', frame=i, region=region, answer=answer)
        label = answer.get('label')
        return label.strip()[:100] if answer['decision'] == 'watermark' and isinstance(label, str) and label.strip() else previous

    def locate(i, region, label, *, generic=False):
        try:
            image = evidence(i, region)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('feedback.grounding_unavailable', frame=i, reason=str(error))
            return []
        x0, y0, width, height = region or (0, 0, video.width, video.height)
        rows = []
        for name in ([label, None] if label and (region is not None or generic) else [label]):
            try:
                detected = locate_named(backend, name, image, 22, tight=True)
            except (OSError, RuntimeError, ValueError) as error:
                trace('feedback.grounding_unavailable', frame=i, named=bool(name), reason=str(error))
                continue
            for a, b, c, d in detected:
                if (c-a)*(d-b) >= 850000:
                    continue
                rect = [x0+round(a*width/1000), y0+round(b*height/1000),
                        x0+round(c*width/1000), y0+round(d*height/1000)]
                if min(rect[2]-rect[0], rect[3]-rect[1]) >= 8:
                    rows.append({'rect': rect, 'provider': 'review-qwen'})
            if rows:
                break
        trace('feedback.grounded', frame=i, region=region, rects=rows)
        return rows

    label = identify(frame, search, hint['label'])
    _, nearby = sparse_discovery_samples(video, {frame, *review_frames, *verified}, {frame, *verified})
    anchor_frame = frame
    anchor_region = search
    anchors = locate(frame, search, label)
    if not anchors and frame in verified:
        a, b, width, height = verified[frame]
        anchors = [{'rect': [a, b, a+width, b+height], 'provider': 'review-pixels'}]
    if not anchors:
        extra = list(dict.fromkeys([frame, *nearby, *review_frames]))[:5]
        for i in extra:
            scopes = (None,) if i == frame else (search, None)
            for scope in scopes:
                if anchors := locate(i, scope, label, generic=True):
                    anchor_frame = i
                    anchor_region = scope or (0, 0, video.width, video.height)
                    trace('feedback.grounding_recovered', reported_frame=frame, anchor_frame=i, region=scope)
                    break
            if anchors:
                break
    if not anchors:
        raise ValueError('Qwen did not locate the reported overlay in the original frame or the follow-up frames.')
    near = min(anchors, key=lambda row: abs(row['rect'][0]+row['rect'][2]-2*x-w)
               +abs(row['rect'][1]+row['rect'][3]-2*y-h))
    left, top, right, bottom = near['rect']
    width, height = right-left, bottom-top
    sx, sy, sw, sh = anchor_region
    extent = (left-2*width*(left <= sx+4), top-2*height*(top <= sy+4),
              right+2*width*(right >= sx+sw-4), bottom+2*height*(bottom >= sy+sh-4))
    recentered = xywh(expand(extent, video, max(24, width//2), max(14, height//2)))
    label = identify(anchor_frame, recentered, label)
    dense = []
    if anchor_frame != frame or anchor_region != search:
        step = max(1, round(float(video.fps)/8))
        dense = [i for offset in (-1, 1, -2, 2, -3, 3)
                 if 0 <= (i := anchor_frame+offset*step) < video.count]
        trace('feedback.nearby_sampling', anchor_frame=anchor_frame, frames=dense)
    frames = list(dict.fromkeys([frame, anchor_frame, *verified, *dense, *nearby, *review_frames,
                                *np.linspace(0, video.count-1, 7, dtype=int).tolist()]))[:11]
    regions = {anchor_frame: recentered}
    for i, (a, b, width, height) in verified.items():
        regions[i] = xywh(expand((a, b, a+width, b+height), video, max(24, width//2), max(14, height//2)))
    proposals = {}
    for i in frames:
        nearby_region = regions[min(regions, key=lambda anchor: abs(i-anchor))]
        proposals[i] = locate(i, nearby_region, label)
        if i in verified:
            a, b, width, height = verified[i]
            proposals[i].insert(0, {'rect': [a, b, a+width, b+height], 'provider': 'review-pixels'})
    anchored_rects = {tuple(row['rect']) for row in anchors}
    proposals[anchor_frame] = anchors+[row for row in proposals[anchor_frame] if tuple(row['rect']) not in anchored_rects]
    if sum(bool(rows) for rows in proposals.values()) < 3:
        for i in [i for i in frames if not proposals[i]][:4]:
            proposals[i] = locate(i, None, label, generic=True)
    return proposals


class FeedbackTrackOverlap(ValueError):
    pass


def join_feedback_tracks(retained, found, count, *, source_verified=False):
    prior = discovery_tracks(retained)
    if len(prior)+len(found) > MAX_TRACKS:
        raise ValueError('The additional watermark would exceed the layer limit.')
    for a in prior:
        for b in found:
            bx, by, bw, bh = cv2.boundingRect(b['mask'])
            b_start = b['positions']+[bx, by]
            a_end = a['positions']+a['mask'].shape[::-1]+2
            b_end = b_start+[bw, bh]+2
            overlap = (a.get('present', np.ones(count, bool)) & b['present']
                       & np.all(a['positions'] < b_end, axis=1) & np.all(b_start < a_end, axis=1))
            if overlap.any() and (not source_verified or not a['verified'][overlap].all()
                                  or not b['verified'][overlap].all()):
                raise FeedbackTrackOverlap('The reported watermark overlaps a retained layer and needs a joint plan.')
    layers = [{key: layer[key] for key in LAYER_FIELDS if key in layer} for layer in [*prior, *found]]
    for layer in layers:
        layer.setdefault('present', np.ones(count, bool))
    layers[0]['cuts'] = np.union1d(retained['cuts'], found[0]['cuts']).astype(np.int32)
    combined = pack_tracks(layers)
    for key in ('extent_signal', 'extent_frames', 'extent_sigma'):
        if key in retained:
            combined[key] = retained[key]
    for fields in luma_extent_groups(retained):
        combined.update({key: retained[key] for key in fields})
    return combined, len(prior)


def feedback_recipe(video, directory, identity, record, config, cuda, trace, *, fallback=None, read_text=None):
    from wm_layers import overlap

    hint = record['decision']['watermark_hint']
    attempt = Path(record['directory'])
    recipe = Path(record['recipe'])
    plan = build_plan(video.path, attempt, recipe, not cuda)
    state, finished = resume_state(attempt, plan, fingerprints(input_files(video.path, recipe, attempt, plan)))
    if state['status'] != 'complete' or finished != len(plan):
        raise ValueError('The preceding candidate must pass all local checks before retaining its track.')
    verification = read_object(attempt/'verification.json')
    if not all(verification.get(key) is True for key in ('same_video_timing_and_size',
                                                        'same_audio_payload_and_timing', 'full_decode_ok')):
        raise ValueError('The previous candidate must preserve source timing, audio and decodability.')
    previous = read_object(recipe)
    profile = (recipe.parent/previous['profile']).resolve()
    settings = read_object(profile)['discovery']
    retained = load_track(video, profile, attempt/'track.npz')
    if 'matte_positions' in retained:
        raise ValueError('The preceding mask requires a joint restoration plan.')
    if record.get('whole_frame_review', {}).get('hints'):
        from wm_dynamic import dynamic_tracks
        try:
            found, dynamic = dynamic_tracks(video, Video.open(attempt/'result.mp4'),
                                           record['whole_frame_review']['hints'], retained, cuda, trace,
                                           clean_frames=record['whole_frame_review'].get('clean_frames', ()))
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('feedback.dynamic_rejected', reason=str(error))
        else:
            combined, first = join_feedback_tracks(retained, found, video.count, source_verified=True)
            directory.mkdir()
            return write_discovery_recipe(video, directory, identity, combined, None, settings['label'], {}, trace,
                                          projection=True, feedback={'frame': hint['frame'], 'first_layer': first},
                                          retained_settings={**settings, 'coverage_review_only': True,
                                                             'dynamic_feedback': dynamic},
                                          final_render=previous.get('final_render'))
    proposals = feedback_proposals(video, hint, record['frames'], config, cuda, trace,
                                   retained=discovery_tracks(retained))
    frame = hint['frame']
    anchors = [xywh(row['rect']) for row in proposals[frame]] or [tuple(hint['rect'])]

    def verified_join(found):
        visible = []
        for layer in found:
            if layer['present'][frame]:
                x, y = layer['positions'][frame]
                left, top, width, height = cv2.boundingRect(layer['mask'])
                visible.append((x+left, y+top, width, height))
        if not any(overlap(a, b) >= .5*min(a[2]*a[3], b[2]*b[3]) for a in anchors for b in visible):
            raise ValueError('The independently verified track does not match the local observation at the reported frame.')
        return join_feedback_tracks(retained, found, video.count, source_verified=True)

    positive_only = bool(record.get('whole_frame_review'))
    found, labels = local_tracks(video, proposals, config, cuda, trace, fallback=fallback, read_text=read_text,
                                positive_only=positive_only)
    if positive_only:
        settings = {**settings, 'coverage_review_only': True}
    try:
        combined, first = verified_join(found)
    except FeedbackTrackOverlap:
        candidate = Video.open(attempt/'result.mp4')
        candidate_digest = digest(candidate.path)
        tracking_config = config
        if proof_file := config.get('absence_proof_file'):
            proof_file = Path(proof_file).with_name(f'feedback-absence-{candidate_digest[:16]}.json')
            tracking_config = {**config, 'absence_proof_file': str(proof_file),
                               'absence_source_sha256': identity, 'absence_candidate_sha256': candidate_digest}
        trace('feedback.overlap_recheck', candidate_sha256=candidate_digest, original_sha256=identity)
        found, labels = local_tracks(candidate, proposals, tracking_config, cuda, trace, read_text=read_text)
        combined, first = verified_join(found)
    label = ' / '.join([settings['label'], *labels])
    directory.mkdir()
    trace('feedback.tracks_joined', frame=frame, retained=first, added=len(found), labels=labels)
    return write_discovery_recipe(video, directory, identity, combined, None, label, {}, trace,
                                  projection=True, feedback={'frame': frame, 'first_layer': first},
                                  retained_settings=settings,
                                  final_render=previous.get('final_render'))


def merge_visual_proposals(video, candidates, plan, *, provider='visual-planner', include_samples=False):
    updated = {i: list(rows) for i, rows in candidates.items()}

    def add(i, x, y, w, h):
        rect = [x, y, x+w, y+h]
        rows = updated.setdefault(i, [])
        if not any(list(proposal['rect']) == rect for proposal in rows):
            rows.insert(0, {'rect': rect, 'provider': provider})

    for reviewed in plan['rounds']:
        decision = reviewed['decision']
        frames = reviewed['frames']
        if include_samples:
            for i in frames:
                if type(i) is int and 0 <= i < video.count:
                    updated.setdefault(i, [])
        observations = decision.get('observations')
        if decision['decision'] != 'track' or not isinstance(observations, list):
            continue
        for row in observations[:60*len(frames)]:
            try:
                _, found = observed_boxes(video, {**decision, 'observations': [row]}, frames)
            except ValueError:
                continue
            for i, rect in found.items():
                add(i, *rect)
    if include_samples:
        for row in plan.get('local_seed_observations', ()):
            add(*row)
    return updated


def reuse_visual_proposals(video, candidates, history, trace):
    rounds = []
    for directory, original in history:
        for path in sorted(directory.glob('pass-*/*/director.json')):
            report = read_object(path)
            if report.get('source') == original:
                rounds.extend(report.get('planning', {}).get('rounds', []))
    if not rounds:
        return candidates
    updated = merge_visual_proposals(video, candidates, {'rounds': rounds}, provider='prior-visual')
    added = sum(map(len, updated.values()))-sum(map(len, candidates.values()))
    if added:
        trace('layers.prior_visual_proposals_reused', rounds=len(rounds), candidates=added)
    return updated


def reuse_motif_proposals(video, candidates, history, identity, trace):
    from wm_layers import diverse_frames, load_layer_evidence, overlap, visible_box

    recovered = {}
    count = 0
    paths = dict.fromkeys(directory/'confirmed-motifs.npz' for directory, _ in reversed(history))
    for path in paths:
        if not path.is_file():
            continue
        try:
            layers, _ = load_layer_evidence(path, video, identity)
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as error:
            trace('layers.prior_motif_proposals_rejected', path=str(path), reason=str(error))
            continue
        for layer in layers:
            fresh = []
            for item, (x, y, score) in diverse_frames(layer.matches):
                if score < layer.floor:
                    continue
                left, top, w, h = visible_box(layer.motif, x, y)
                bounds = expand((left, top, left+w, top+h), video, 0, 0)
                rect = xywh(bounds)
                if min(rect[2:]) < 8 or rect[2]*rect[3] > .18*video.width*video.height:
                    continue
                prior = [*recovered.get(item.frame, ()), *candidates.get(item.frame, ())]
                if len(prior) >= 60 or any(overlap(rect, other) >= .80*max(rect[2]*rect[3], other[2]*other[3])
                                          for other in (xywh(row['rect']) for row in prior)):
                    continue
                fresh.append((item.frame, bounds))
            take = min(40, len(fresh), 360-count)
            if not take:
                continue
            for at in np.linspace(0, len(fresh)-1, take, dtype=int):
                frame, bounds = fresh[at]
                recovered.setdefault(frame, []).append({'rect': list(bounds), 'provider': 'prior-motif'})
            count += take
        if count == 360:
            break
    if not recovered:
        return candidates
    trace('layers.prior_motif_proposals_reused', frames=len(recovered), proposals=count)
    return {**candidates, **{i: [*candidates.get(i, ()), *rows] for i, rows in recovered.items()}}


def source_layer_candidates(video, local, text_proposals, config, cuda, trace, identity, history, *, proposal_window=None):
    sampled, candidates = [], {}
    if local is not None:
        try:
            sampled, candidates = local.sample(video, cuda, limit=24, detail=True)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('layers.detector_unavailable', reason=str(error))
    if text_proposals is not None and sampled:
        try:
            extra = text_proposals.sample(video, sampled[::2], trace)
        except (OSError, RuntimeError, ValueError) as error:
            trace('layers.ocr_unavailable', reason=str(error))
        else:
            for i, rows in extra.items():
                candidates[i] = rows+candidates.get(i, [])
    bank_samples = sampled[::2] or ((i, video.planes(raw)) for i, raw in
                                    read_frames(video, cuda=cuda, step=max(1, ceil(video.count/12))))
    try:
        known = remembered_proposals(video, bank_samples, Path(config['learned_marks']), identity, trace)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        trace('layers.library_unavailable', reason=str(error))
    else:
        for i, rows in known.items():
            candidates[i] = rows+candidates.get(i, [])
    candidates = reuse_visual_proposals(video, candidates, history, trace)
    candidates = native_layer_candidates(video, candidates, config, cuda, trace, proposal_window=proposal_window)
    return reuse_motif_proposals(video, candidates, history, identity, trace)


def native_layer_candidates(video, candidates, config, cuda, trace, *, proposal_window=None):
    count = min(video.count, config.get('local_proposal_frames', 0))
    if not count:
        return candidates
    backend = config['discovery_local']
    if proposal_window is not None:
        proposal_window['remaining'] = 5.
        backend = {**backend, 'proposal_window': proposal_window}

    def propose(frames, stage):
        try:
            options = {'direct': True} if config.get('local_direct_proposals') else {}
            decision = native_discovery(video, frames, cuda, backend, '', trace, **options)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            trace('layers.local_proposals_unavailable', stage=stage, reason=str(error))
            return {'decision': 'review'}
        trace('layers.local_proposals', stage=stage, frames=frames, observations=len(decision.get('observations', [])),
              decision=decision['decision'])
        return decision

    from wm_quality import opening_samples
    frames = sorted(set([*opening_samples(video), *np.linspace(0, video.count-1, count, dtype=int).tolist()]))
    decision = propose(frames, 'initial')
    rounds = [{'frames': frames, 'decision': decision}]
    anchors = {row['frame'] for row in decision.get('observations', [])}
    if 0 < len(anchors) < 3:
        followup, nearby = sparse_discovery_samples(video, set(frames), anchors)
        if nearby:
            trace('layers.local_proposals_followup', anchors=sorted(anchors), nearby=nearby)
            rounds.append({'frames': followup, 'decision': propose(followup, 'followup')})
    elif len(anchors) >= 3:
        step = max(1, round(float(video.fps)*.75))
        near_ends = (step, 2*step, video.count-1-2*step, video.count-1-step)
        boundary = list(dict.fromkeys(i for i in near_ends if 0 <= i < video.count and i not in frames))
        if boundary:
            rounds.append({'frames': boundary, 'decision': propose(boundary, 'boundary')})
    candidates = merge_visual_proposals(video, candidates, {'rounds': rounds}, provider='local-vision')
    for reviewed in rounds:
        for i in reviewed['frames']:
            candidates.setdefault(i, [])
    if proposal_window is not None and proposal_window.get('deferred'):
        trace('layers.optional_proposals_deferred', retained=sum(map(len, candidates.values())))
    return candidates


def reason_for(report):
    planning = report.get('planning', {})
    last = report['attempts'][-1] if report.get('attempts') else {}
    return (report.get('error') or planning.get('rejection') or planning.get('reason')
            or ' / '.join(last.get('local_findings', []))
            or (last.get('decision') or {}).get('reason') or (planning.get('decision') or {}).get('reason'))


def transient_grok_error(message):
    error = (message or '').lower()
    return 'grok' in error and any(word in error for word in ('timed out', 'tool call', 'malformed',
                                                              'closed its connection', 'invalid decision',
                                                              'invalid api response', 'api temporarily unavailable',
                                                              'review deferred'))


def transient_review_error(report):
    return transient_grok_error(report.get('error'))


def awaiting_visual_review(report):
    if (not report or report['status'] not in ('failed', 'needs_review')
            or not transient_review_error(report) or not report.get('attempts')):
        return False
    last = report['attempts'][-1]
    if last.get('local_findings') != []:
        verification = Path(last['directory'])/'verification.json'
        if not verification.is_file() or not all(read_object(verification).get(key) is True for key in (
                'same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok')):
            return False
    state = Path(last['directory'])/'run.json'
    return state.is_file() and read_object(state).get('status') == 'complete'


def needs_another_strategy(report):
    check_budget()
    if report and report.get('attempts') and report['attempts'][-1].get('whole_frame_review'):
        # The output-driven repair loop has already consumed its bounded attempts.
        # Do not restart discovery from scratch and discard the completed candidate.
        return False
    return report is None or (report['status'] != 'complete' and not awaiting_visual_review(report))


def route_summary(name, report, path):
    attempts = report.get('attempts', [])
    last = attempts[-1] if attempts else {}
    return {'route': name, 'status': report['status'], 'report': str(path), 'reason': reason_for(report),
            'attempts_recorded': len(attempts),
            'local_checks_passed': last['local_findings'] == [] if 'local_findings' in last else None,
            'visual_decision': (last.get('decision') or {}).get('decision')}


def run_route(source, directory, name, recipe, config, cuda, render_gate, trace, routes, *, local_refinement=None,
              reuse=None):
    trace('route.started', route=name, recipe=str(recipe) if recipe else None)
    target = directory/name
    result = run_job(source, target, recipe, config, cuda, reuse=reuse, render_gate=render_gate,
                     local_refinement=local_refinement)
    if awaiting_visual_review(result):
        trace('route.review_retry', route=name, reason=result['error'])
        result = recheck_job(source, target, config, cuda, render_gate=render_gate)
    row = route_summary(name, result, target/'director.json')
    routes.append(row)
    trace('route.finished', **row)
    return result


def next_attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while (attempt := directory/f'pass-{index:02d}').exists():
        index += 1
    attempt.mkdir()
    return attempt


def color_retry_recipe(source, directory):
    if any(directory.glob('pass-*/color-retry/director.json')):
        return None
    for path in sorted(directory.glob('pass-*/*/director.json'), reverse=True):
        report = read_object(path)
        if report.get('source') != str(source) or report.get('status') == 'complete':
            continue
        for attempt in report.get('attempts', []):
            repair = (attempt.get('decision') or {}).get('repair') or {}
            if repair.get('kind') == 'discovery-detail' and attempt.get('local_findings') == []:
                return Path(attempt['recipe'])
    return None


def feedback_hints(record):
    primary = (record.get('decision') or {}).get('watermark_hint') or record.get('provisional_watermark_hint')
    context = record.get('context_watermark_hints', ())
    hints, seen, same_mark = [], set(), set()
    for hint in [primary, *(context if isinstance(context, (list, tuple)) else ())]:
        if not isinstance(hint, dict):
            continue
        frame, rect, label = hint.get('frame'), hint.get('rect'), hint.get('label')
        if (type(frame) is not int or not isinstance(rect, (tuple, list)) or len(rect) != 4
                or not all(type(value) is int for value in rect) or not isinstance(label, str)):
            continue
        key = frame, tuple(rect)
        mark = label.strip().casefold(), tuple(rect)
        if key not in seen and mark not in same_mark:
            seen.add(key)
            same_mark.add(mark)
            hints.append(hint)
    return hints[:30]


def recorded_context_hints(source, path, report, record):
    decision = record.get('decision') or {}
    frames = record.get('frames')
    events = path.parent/'events.jsonl'
    if (report.get('status') != 'needs_review' or decision.get('decision') != 'review'
            or not isinstance(frames, list) or not frames or not events.is_file()):
        return []
    attempt = Path(record['directory']).resolve()
    candidate = attempt/'result.mp4'
    if attempt.parent != path.parent.resolve() or not candidate.is_file() or not (attempt/'run.json').is_file():
        return []
    state = read_object(attempt/'run.json')
    if state.get('status') != 'complete' or state.get('inputs', {}).get(str(source)) != digest(source):
        return []
    render = next((row for row in state['stages'] if row['name'] == 'render' and row.get('status') == 'complete'), {})
    if render.get('artifacts', {}).get(str(candidate)) != digest(candidate):
        return []
    for line in reversed(events.read_text(encoding='utf-8').splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        answer = row.get('response')
        if (row.get('event') == 'grok.decision' and row.get('phase') == 'review' and 'comparison' not in row
                and row.get('attempt') == len(report['attempts']) and isinstance(answer, dict)
                and answer.get('decision') == 'review' and answer.get('reason') == decision.get('reason')):
            return review_watermark_hints(answer, Video.open(source), frames)
    return []


def resume_candidate_review(source, directory, config, cuda, routes, trace, *, render_gate=None):
    for path in sorted(directory.glob('pass-*/*/director.json'), key=lambda report: report.stat().st_mtime_ns, reverse=True):
        report = read_object(path)
        attempts = report.get('attempts', [])
        if report.get('source') != str(source) or not attempts:
            continue
        last = attempts[-1]
        if (config.get('local_layers') and report.get('status') in ('failed', 'needs_review')
                and last.get('local_findings') == [] and last.get('source_comparison', {}).get('decision') == 'accept'):
            if not feedback_hints(last) and (recovered := recorded_context_hints(source, path, report, last)):
                last['context_watermark_hints'] = recovered
                write_object(path, report)
                trace('route.recovered_context_feedback', report=str(path), hints=len(feedback_hints(last)))
            if feedback_hints(last):
                trace('route.resume_feedback', report=str(path))
                return report
        interrupted = report.get('status') in ('running', 'rechecking', 'interrupted')
        changed = (report.get('status') == 'needs_review' and last.get('review_method') != review_method(config))
        retryable = awaiting_visual_review(report)
        if not interrupted and not changed and not retryable:
            continue
        attempt = Path(last['directory'])
        state_path = attempt/'run.json'
        if not state_path.is_file():
            continue
        state = read_object(state_path)
        if state.get('status') != 'complete':
            if not interrupted:
                continue
            rendered = any(row['name'] == 'render' and row.get('status') == 'complete' for row in state['stages'])
            if not rendered:
                continue
            recipe = Path(last['recipe'])
            plan = build_plan(source, attempt, recipe, not cuda)
            try:
                with lock_job(attempt):
                    state, first = resume_state(attempt, plan, fingerprints(input_files(source, recipe, attempt, plan)))
                    trace('route.resume_verification', report=str(path), stages_left=len(plan)-first)
                    execute_plan(attempt, state, first, trace)
            except (OSError, RuntimeError, ValueError, CalledProcessError) as error:
                trace('route.resume_verification_rejected', report=str(path), reason=str(error))
                continue
        if interrupted:
            report['status'] = 'interrupted'
            write_object(path, report)
        trace('route.resume_review', report=str(path), updated_method=changed, interrupted_review=retryable)
        try:
            with processing_phase(allow_planning_exhausted=True):
                result = recheck_job(source, path.parent, config, cuda, render_gate=render_gate)
        except (OSError, RuntimeError, ValueError) as error:
            trace('route.resume_review_rejected', report=str(path), reason=str(error))
            continue
        routes.append(route_summary('resumed-review', result, path))
        return result
    return None


def interrupted_recipe(source, directory, identity, history=()):
    for job, original in [(directory, str(source)), *reversed(history)]:
        for path in sorted(job.glob('pass-*/*/director.json'), reverse=True):
            report = read_object(path)
            if (report.get('source') != original or report.get('status') not in ('running', 'interrupted', 'failed')
                    or not report.get('attempts')):
                continue
            if job != directory and report.get('status') == 'running':
                try:
                    with lock_job(path.parent), lock_job(Path(report['attempts'][-1]['directory'])):
                        pass
                except ValueError:
                    continue
            last = report['attempts'][-1]
            recipe = Path(last['recipe'])
            if not recipe.is_file():
                continue
            attempt = Path(last['directory'])
            state_path = attempt/'run.json'
            if not state_path.is_file():
                if job == directory and report['status'] == 'running':
                    return recipe, None
                continue
            state = read_object(state_path)
            inputs = state.get('inputs', {})
            if inputs.get(original) != identity or inputs.get(str(recipe)) != digest(recipe):
                continue
            profile = (recipe.parent/read_object(recipe)['profile']).resolve()
            if not profile.is_file() or inputs.get(str(profile)) != digest(profile):
                continue
            pending = next((stage for stage in state['stages'] if stage.get('status') != 'complete'), None)
            if pending is not None and job == directory and report['status'] in ('running', 'interrupted'):
                return recipe, attempt
            if pending is None or pending['name'] != 'render' or pending.get('status') not in ('running', 'failed'):
                continue
            return recipe, attempt if original == str(source) else None
    return None


def overlay_regions(directory, identity):
    path = directory/'overlay-evidence.json'
    if not path.is_file():
        return []
    evidence = read_object(path)
    if evidence['source_sha256'] != identity:
        raise ValueError('The recorded overlay evidence belongs to another source.')
    return evidence['regions']


def import_overlay_regions(directory, identity, incoming):
    required = overlay_regions(directory, identity)
    combined = [list(rect) for rect in dict.fromkeys(map(tuple, [*required, *incoming]))]
    if combined == required:
        return required, 0
    directory.mkdir(parents=True, exist_ok=True)
    write_object(directory/'overlay-evidence.json', {'source_sha256': identity, 'regions': combined})
    return combined, len(combined)-len(set(map(tuple, required)))


def prior_regions(roots):
    regions = {}
    for root in roots:
        if not (root/'manifest.json').is_file():
            raise ValueError(f'The prior evidence directory is not a watermark batch: {root}')
        for path in root.glob('jobs/*/overlay-evidence.json'):
            evidence = read_object(path)
            regions.setdefault(evidence['source_sha256'], []).extend(evidence['regions'])
    return regions


def prior_visual_history(roots):
    history = {}
    for root in roots:
        for item in read_object(root/'manifest.json')['sources']:
            directory = root/'jobs'/f"{item['index']:03d}"
            history.setdefault(item['source_sha256'], []).append((directory, item['source']))
    return history


def completed_route(item, root, required=(), *, coverage_request=None):
    directory = root/'jobs'/f"{item['index']:03d}"
    for path in sorted(directory.glob('pass-*/*/director.json'), reverse=True):
        report = read_object(path)
        attempts = report.get('attempts', [])
        if report.get('source') != item['source'] or report['status'] != 'complete' or not attempts:
            continue
        last = attempts[-1]
        attempt = Path(last['directory'])
        state = attempt/'run.json'
        if not state.is_file() or not (attempt/'verification.json').is_file():
            continue
        record = read_object(state)
        if record.get('status') != 'complete' or record['inputs'].get(item['source']) != item['source_sha256']:
            continue
        decision = last.get('decision') or {}
        output = report.get('result')
        if not output or decision.get('decision') != 'accept' or last.get('local_findings') != [] or not Path(output).is_file():
            continue
        output_digest = digest(Path(output))
        render = next((row for row in record['stages'] if row['name'] == 'render'), {})
        if render.get('artifacts', {}).get(str(attempt/'result.mp4')) != output_digest:
            continue
        proof = None
        if required:
            recipe = Path(last['recipe'])
            if not recipe.is_file() or record['inputs'].get(str(recipe)) != digest(recipe):
                continue
            profile = (recipe.parent/read_object(recipe)['profile']).resolve()
            track = attempt/'track.npz'
            scan = next((row for row in record['stages'] if row['name'] == 'scan'), {})
            if (not profile.is_file() or record['inputs'].get(str(profile)) != digest(profile)
                    or not track.is_file() or scan.get('artifacts', {}).get(str(track)) != digest(track)):
                continue
            video = Video.open(Path(item['source']))
            if not read_object(profile).get('discovery'):
                continue
            if missing := uncovered_overlay_regions(video, track, required):
                proof = path.with_name('coverage-review.json')
                if not exact_coverage_accepted(video, path, report, missing, item['source_sha256'], output_digest,
                                               proof, **(coverage_request or {})):
                    continue
        route = route_summary('recorded-verdict', report, path)
        if proof is not None:
            route.update(route='exact-coverage-review', coverage_proof=str(proof))
        return {**item, 'status': 'complete', 'reason': reason_for(report), 'result': output,
                'result_sha256': output_digest, 'routes': [route]}
    return None


def released_motif_source(item, batch, accepted):
    directory = batch/'jobs'/f"{item['index']:03d}"
    required = overlay_regions(directory, item['source_sha256'])
    verified = completed_route(item, batch, required)
    if not verified or verified['result_sha256'] != accepted['result_sha256']:
        raise ValueError('The prior release no longer matches its accepted batch result.')
    report_path = Path(verified['routes'][0]['report']).resolve(strict=True)
    last = read_object(report_path)['attempts'][-1]
    attempt = Path(last['directory']).resolve(strict=True)
    if attempt.parent != report_path.parent or not attempt.is_relative_to(directory.resolve()):
        raise ValueError('The prior release attempt is outside its job directory.')
    recipe = Path(last['recipe']).resolve(strict=True)
    profile = read_object((recipe.parent/read_object(recipe)['profile']).resolve(strict=True))
    if not profile.get('discovery'):
        return None
    if last.get('source_comparison', {}).get('decision') != 'accept':
        raise ValueError('The prior watermark needs an accepted source comparison.')
    stages = read_object(attempt/'run.json')['stages']
    outputs = {str(attempt/name) for name in ('track.npz', 'verification.json', 'watermark-audit.json')}
    if not outputs.issubset(path for stage in stages for path in stage['outputs']):
        raise ValueError('The prior release lacks recorded watermark and technical checks.')
    inputs = fingerprints(input_files(Path(item['source']), recipe, attempt, stages))
    state, finished = resume_state(attempt, stages, inputs)
    if state['status'] != 'complete' or finished != len(stages):
        raise ValueError('The prior release still has unverified stages.')
    technical, audit = (read_object(attempt/name) for name in ('verification.json', 'watermark-audit.json'))
    if (not all(technical.get(key) is True for key in ('same_video_timing_and_size',
                                                       'same_audio_payload_and_timing', 'full_decode_ok'))
            or audit.get('discovery_passed') is not True
            or audit.get('discovery_verified_frames') != technical['frames']
            or any(layer.get('discovery_passed') is not True for layer in audit.get('discovery_layers', []))):
        raise ValueError('The prior release did not verify all watermark layers and source timing.')
    with np.load(attempt/'track.npz', allow_pickle=False) as track:
        if str(track['source_sha256']) != item['source_sha256']:
            raise ValueError('The prior watermark track belongs to another source.')
    return attempt


def learn_from_batches(roots, library, trace):
    for batch in roots:
        manifest = {row['index']: row for row in read_object(batch/'manifest.json')['sources']}
        for job in read_object(batch/'batch.json')['jobs']:
            if job['status'] != 'complete':
                continue
            index = job['index']
            try:
                item = manifest[index]
                if job['source_sha256'] != item['source_sha256']:
                    raise ValueError('The prior batch result belongs to another source.')
                if attempt := released_motif_source(item, batch, job):
                    for path, added in remember(library, attempt):
                        trace('batch.learned_prototype', batch=str(batch), index=item['index'],
                              source_sha256=item['source_sha256'], prototype=str(path), new=added)
            except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as error:
                trace('batch.learning_rejected', batch=str(batch), index=index, reason=str(error))


def prior_observations(source, directory):
    proposals = []
    for log in directory.glob('pass-*/adaptive/events.jsonl'):
        if not (log.parent/'director.json').is_file():
            continue
        report = read_object(log.parent/'director.json')
        if report.get('source') != str(source):
            continue
        with log.open(encoding='utf-8') as stream:
            current = None
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event['event'] == 'discovery.learned':
                    current = [(row['frame'], row['visual_box']) for row in event['observations']]
                if (event['event'] == 'discovery.coverage' and current and
                        .80*event['total'] <= event['verified'] < event['total']):
                    proposals.append((event['verified'], current, log))
    return max(proposals, key=lambda row: row[0]) if proposals else None


def replay_discovery(video, directory, item, config, cuda, trace, proposal):
    count, observations, log = proposal
    trace('route.replay_evidence', observations=observations, previous_verified=count, log=str(log))
    track, prior = enroll(video, observations, Path(config['learned_marks']), item['source_sha256'], cuda,
                          trace, tolerate_outlier=True)
    recover_chroma(video, track, cuda, trace)
    verify_discovery_presence(video, track, None, config, cuda, trace)
    verified = int(track['verified'].sum())
    trace('route.replay_coverage', verified=verified, total=video.count)
    if verified != video.count:
        return None
    label = None
    if config.get('local_layers'):
        classification, label, _ = confirm_discovery_target(video, track, config, cuda, trace)
        if classification != 'confirmed':
            raise ValueError('The replayed target could not be confirmed as a complete distribution watermark.')
    return write_discovery_recipe(video, directory, item['source_sha256'], track, prior, label, {}, trace)


def run_source(item, root, config, local, cuda, render_gate, text_proposals=None, *, history=()):
    with source_runtime(config) as runtime:
        row = _run_source(item, root, config, local, cuda, render_gate, text_proposals, history=history)
        row['workflow_stalled'] = runtime.stalled
        row['analysis_cache'] = {'frame_hits': runtime.frame_cache_hits, 'frame_bytes': runtime.frame_bytes,
                                 'pixel_passes': len(runtime.pixel_passes)}
        candidates = []
        for path in sorted((root/'jobs'/f"{item['index']:03d}").glob('pass-*/*/director.json')):
            report = read_object(path)
            if report.get('source') == item['source'] and (candidate := report.get('review_candidate')):
                if Path(candidate).is_file():
                    candidates.append(candidate)
        if candidates:
            row['review_candidate'] = candidates[-1]
        return row


def _run_source(item, root, config, local, cuda, render_gate, text_proposals=None, *, history=()):
    source = Path(item['source'])
    directory = next_attempt(root/'jobs'/f"{item['index']:03d}")
    write_trace = job_trace(directory/'routing.jsonl')
    transient_local = False
    transient_grok = False

    def trace(event, **details):
        nonlocal transient_local, transient_grok
        if event == 'vision.local_transport_unavailable':
            transient_local = True
        elif event == 'grok.api.failed' and transient_grok_error(details.get('error')):
            transient_grok = True
        write_trace(event, **details)
        if event == 'layers.overlap_recovered' and (runtime := current_runtime()) is not None:
            runtime.overlap_frames.update(details['frames'])

    routes = []
    result = None
    stage = 'source_validation'
    runtime = current_runtime()

    def render_review_checkpoint(reason):
        if runtime is None or runtime.review_recipe is None:
            return None
        trace('route.coverage_review_fallback', reason=reason, recipe=str(runtime.review_recipe))
        with processing_phase(allow_planning_exhausted=True):
            fallback_config = {**config, 'max_repairs': 0}
            candidate = run_route(source, directory, 'coverage-review', runtime.review_recipe, fallback_config,
                                  cuda, render_gate, trace, routes)
            return recover_review(candidate)

    try:
        if digest(source) != item['source_sha256']:
            raise ValueError('The source changed after the batch manifest was created.')
        video = Video.open(source)
        video.check_timestamps()
        stage = 'routing'
        evidence_path = directory.parent/'overlay-evidence.json'
        required = overlay_regions(directory.parent, item['source_sha256'])
        config = {**config, 'required_overlay_regions': required,
                  'absence_proof_file': str(directory.parent/'absence-evidence.json'),
                  'absence_source_sha256': item['source_sha256'],
                  'prior_refinement_checkpoints': [str(path/REFINEMENT_FILE) for path, _ in history]}
        if runtime is not None:
            checkpoint_method = {'motifs': motif_review_method(config) if config.get('discovery_local') else {},
                                 'pixel_rules': review_checkpoint_rules()}
            runtime.review_recipe = prior_review_checkpoint(
                [directory.parent, *(path for path, _ in history)], item['source_sha256'], checkpoint_method)
            if runtime.review_recipe:
                trace('layers.review_checkpoint_resumed', recipe=str(runtime.review_recipe))
            def save_review_checkpoint(tracks, labels):
                from wm_overlap import review_samples
                target = directory/f'coverage-checkpoint-{runtime.review_checkpoints+1:02d}'
                target.mkdir()
                settings = {'coverage_review_only': True,
                            'overlap_review_frames': review_samples(runtime.overlap_frames),
                            'review_checkpoint_method': checkpoint_method}
                runtime.review_recipe = write_discovery_recipe(
                    video, target, item['source_sha256'], pack_tracks(tracks), None, ' / '.join(labels), {}, trace,
                    projection=True, retained_settings=settings)
                runtime.review_checkpoints += 1
                trace('layers.review_checkpoint_saved', recipe=str(runtime.review_recipe),
                      tracks=len(tracks), confirmed_placements=sum(int(track['present'].sum()) for track in tracks))
            runtime.save_review_checkpoint = save_review_checkpoint
        if required and config.get('discovery_local'):
            checkpoints = tuple(dict.fromkeys(path/'confirmed-motifs.npz' for path, _ in
                                              [*history, (directory.parent, item['source'])]))
            request = {'config': config, 'cuda': cuda, 'trace': trace, 'checkpoints': checkpoints}
            if verdict := completed_route(item, root, required, coverage_request=request):
                trace('route.reused_verified_output', result=verdict['result'], routes=verdict['routes'])
                return {**verdict, 'routing_log': str(directory/'routing.jsonl')}
        read_text = None
        if text_proposals is not None:
            read_text = text_proposals if callable(text_proposals) else text_proposals.crop
        prior_reviews = [path/'layer-reviews.json' for path, _ in history]
        if signature := config.get('local_evidence_identity'):
            prior_caches = [path/'local-vision-cache.json' for path, _ in history]
            cache = LocalEvidenceCache(directory.parent/'local-vision-cache.json',
                                       {'source_sha256': item['source_sha256'], 'backend': signature}, trace,
                                       prior_paths=prior_caches)
            config['prior_layer_checkpoints'] = [str(path.parent) for path in cache.imported_paths]
            for name in ('discovery_local', 'grounding_local'):
                backend = config.get(name)
                if backend and all(backend[key] == signature[key] for key in ('endpoint', 'model')):
                    config[name] = {**backend, 'evidence_cache': cache}
        backend = config.get('discovery_local')
        read_pixels = getattr(read_text, 'read_pixels', None)
        if backend is not None and callable(read_pixels):
            config['discovery_local'] = {**backend, 'read_pixels': read_pixels}
        local_layers = config.get('local_layers', False)
        review_layer = (cached_layer_reviewer(video, config, cuda, trace, directory.parent/'layer-reviews.json',
                                             prior_paths=prior_reviews)
                        if local_layers else None)

        def save_evidence():
            if required:
                write_object(evidence_path, {'source_sha256': item['source_sha256'], 'regions': required})

        def plan_layers(name, candidates):
            confirmed = {}
            for i, x, y, w, h in required:
                confirmed.setdefault(i, []).append({'rect': [x, y, x+w, y+h], 'provider': 'confirmed-source'})
            if confirmed:
                candidates = {**candidates, **{i: rows+list(candidates.get(i, ())) for i, rows in confirmed.items()}}
                trace('layers.confirmed_proposals_reused', frames=len(confirmed), regions=len(required))
            try:
                return local_layer_recipe(video, directory/name, item['source_sha256'], candidates, config, cuda,
                                          trace, fallback=review_layer, read_text=read_text)
            finally:
                save_evidence()

        feedback_rounds = 0
        feedback_seen = set()

        def recover_review(result):
            nonlocal feedback_rounds
            if (feedback_rounds >= config.get('max_repairs', 2) or not config.get('local_layers') or not result
                    or result['status'] not in ('needs_review', 'failed') or not result['attempts']):
                return result
            record = result['attempts'][-1]
            decision = record.get('decision') or {}
            hints = feedback_hints(record)
            if (not hints or not record.get('whole_frame_review') and
                    (record.get('local_findings') != [] or record.get('source_comparison', {}).get('decision') != 'accept')):
                return result
            feedback_rounds += 1
            for number, hint in enumerate(hints[:3], 1):
                signature = (hint['frame'], tuple(hint['rect']), hint['label'])
                if signature in feedback_seen:
                    continue
                feedback_seen.add(signature)
                trace('route.review_feedback_started', hint=hint, number=number)
                for region in hint.get('observations', []):
                    if region not in required:
                        required.append(region)
                name = f'feedback-plan-{feedback_rounds:02d}-{number:02d}'
                try:
                    review = {**record, 'decision': {**decision, 'watermark_hint': hint}}
                    with processing_phase(allow_planning_exhausted=True):
                        recipe = feedback_recipe(video, directory/name, item['source_sha256'], review,
                                                 config, cuda, trace, fallback=review_layer, read_text=read_text)
                except (OSError, RuntimeError, ValueError, cv2.error) as error:
                    trace('route.review_feedback_rejected', reason=str(error), number=number)
                    continue
                finally:
                    save_evidence()
                with processing_phase(allow_planning_exhausted=True):
                    refined = run_route(source, directory, f'review-feedback-{feedback_rounds:02d}', recipe,
                                        config, cuda, render_gate, trace, routes)
                    return recover_review(refined)
            return result

        def attempt_route(name, recipe, route_config=config, *, local_refinement=None, reuse=None):
            try:
                return recover_review(run_route(source, directory, name, recipe, route_config,
                                                cuda, render_gate, trace, routes, local_refinement=local_refinement,
                                                reuse=reuse))
            finally:
                save_evidence()

        result = recover_review(resume_candidate_review(source, directory.parent, config, cuda, routes, trace,
                                                        render_gate=render_gate))
        if result is None:
            for prior_directory, prior_source in history:
                if Path(prior_source).resolve() != source.resolve():
                    continue
                result = recover_review(resume_candidate_review(source, prior_directory, config, cuda, routes, trace,
                                                                render_gate=render_gate))
                if result is not None:
                    break
        if result is None and runtime is not None and runtime.review_recipe is not None:
            result = render_review_checkpoint('Resuming confirmed placements from an unfinished coverage review.')
            return {**item, 'status': result['status'], 'reason': reason_for(result), 'coverage_review_only': True,
                    'routes': routes, 'routing_log': str(directory/'routing.jsonl')}

        if needs_another_strategy(result) and (pending := interrupted_recipe(
                source, directory.parent, item['source_sha256'], history)):
            recipe, prepared = pending
            result = attempt_route('resumed-render', recipe, reuse=prepared)
        if needs_another_strategy(result) and (recipe := color_retry_recipe(source, directory.parent)):
            result = attempt_route('color-retry', recipe)
        sampled_candidates = {}
        if local_layers and needs_another_strategy(result):
            proposal_window = ({} if required and any((Path(path)/'confirmed-motifs.npz').is_file()
                                                       for path in config.get('prior_layer_checkpoints', ())) else None)
            try:
                sampled_candidates = source_layer_candidates(video, local, text_proposals, config, cuda, trace,
                                                             item['source_sha256'],
                                                             [*history, (directory.parent, str(source))],
                                                             proposal_window=proposal_window)
                try:
                    recipe = plan_layers('layer-plan', sampled_candidates)
                    if proposal_window and proposal_window.get('deferred'):
                        if missing := recipe_coverage_findings(video, recipe, required):
                            raise ValueError(' / '.join(missing))
                except (OSError, RuntimeError, ValueError, cv2.error) as error:
                    if not proposal_window or not proposal_window.get('deferred'):
                        raise
                    trace('route.layers_visual_proposals_resumed', reason=str(error))
                    sampled_candidates = native_layer_candidates(video, sampled_candidates, config, cuda, trace)
                    recipe = plan_layers('layer-plan-with-visual-proposals', sampled_candidates)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('route.layers_rejected', reason=str(error))
            else:
                result = attempt_route('local-layers', recipe)
        if local is not None and needs_another_strategy(result):
            try:
                recipe = local.recipe(video, directory, cuda, trace)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('route.local_rejected', reason=str(error))
            else:
                result = attempt_route('reference', recipe)
        def refine_visual_plan(plan):
            try:
                proposals = merge_visual_proposals(video, sampled_candidates, plan, include_samples=True)
                return plan_layers('visual-layer-plan', proposals)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('route.visual_layers_rejected', reason=str(error))
                plan['reason'] = str(error)
                return None

        if needs_another_strategy(result):
            result = attempt_route('adaptive', None, local_refinement=refine_visual_plan if local_layers else None)
        plan = result.get('planning', {})
        if needs_another_strategy(result) and (plan.get('multiple_layers')
                                               or (local_layers and plan.get('local_layer_refinement'))):
            if local_layers:
                if (recipe := refine_visual_plan(plan)) is not None:
                    result = attempt_route('visual-layers', recipe)
            else:
                layers = {**config, 'recognition': 'known', 'detect_known_layout': False,
                          'scan_samples': 5, 'sample_width': min(config['sample_width'], 440),
                          'scan_frames': plan['layer_frames'][:3]}
                result = attempt_route('layered', None, layers)
        if needs_another_strategy(result) and (prior := prior_observations(source, directory.parent)):
            try:
                recipe = replay_discovery(video, directory, item, config, cuda, trace, prior)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('route.replay_rejected', reason=str(error))
            else:
                if recipe is not None:
                    result = attempt_route('replayed-discovery', recipe)
            finally:
                save_evidence()
        if result['status'] != 'complete' and not result.get('review_candidate'):
            try:
                if candidate := render_review_checkpoint(reason_for(result)):
                    result = candidate
            except (OSError, RuntimeError, ValueError, CalledProcessError, cv2.error) as error:
                trace('route.coverage_review_failed', reason=str(error))
        output = result.get('result')
        if awaiting_visual_review(result):
            trace('route.visual_review_pending', reason=result['error'])
        return {**item, 'status': result['status'], 'reason': reason_for(result), 'routes': routes,
                'result': output, 'result_sha256': digest(Path(output)) if output else None,
                'review_candidate': result.get('review_candidate'),
                'review_candidate_sha256': result.get('review_candidate_sha256'),
                'routing_log': str(directory/'routing.jsonl'), 'transient_local_vision': transient_local,
                'transient_grok_vision': transient_grok}
    except wm_grok.SourceReviewDeferred as error:
        trace('route.deferred', reason=str(error))
        return {**item, 'status': 'needs_review', 'reason': str(error), 'routes': routes,
                'routing_log': str(directory/'routing.jsonl'), 'visual_deferred': True}
    except (OSError, RuntimeError, ValueError, CalledProcessError, cv2.error) as error:
        reason = ('FFprobe could not read the source video or its frame timestamps.'
                  if stage == 'source_validation' and isinstance(error, CalledProcessError) else str(error))
        trace('route.error', stage=stage, reason=reason, error_type=type(error).__name__)
        if stage != 'source_validation':
            try:
                if candidate := render_review_checkpoint(reason):
                    return {**item, 'status': candidate['status'], 'reason': reason_for(candidate),
                            'planning_stop_reason': reason, 'coverage_review_only': True,
                            'routes': routes, 'routing_log': str(directory/'routing.jsonl'),
                            'budget_exhausted': isinstance(error, WorkflowLimit)}
            except (OSError, RuntimeError, ValueError, CalledProcessError, cv2.error) as fallback_error:
                trace('route.coverage_review_failed', reason=str(fallback_error))
        return {**item, 'status': 'needs_review' if isinstance(error, WorkflowLimit) else 'failed',
                'failure_stage': stage, 'reason': reason, 'routes': routes,
                'budget_exhausted': isinstance(error, WorkflowLimit),
                'routing_log': str(directory/'routing.jsonl'), 'transient_local_vision': transient_local,
                'transient_grok_vision': transient_grok}


def source_inventory(source):
    items = []
    for path in sorted(source.iterdir()):
        if path.suffix.lower() != '.mp4':
            continue
        item = {'index': len(items)+1, 'source': str(path)}
        try:
            mode = path.stat().st_mode
            if S_ISDIR(mode):
                continue
            if not S_ISREG(mode):
                raise OSError('The source is not a regular file.')
            item['source_sha256'] = digest(path)
        except OSError as error:
            item.update(source_sha256=None, intake_error=f'Could not fingerprint the source: {error}')
        items.append(item)
    if not items:
        raise ValueError('The source directory contains no MP4 files.')
    return items


def source_intake_error(item, *, resume):
    if item.get('intake_error'):
        return item['intake_error']
    if resume:
        try:
            actual = digest(item['source'])
        except OSError as error:
            return f'Could not read the recorded source: {error}'
        if actual != item['source_sha256']:
            return 'The source changed after the batch manifest was created.'
    return None


def publish_progress(root, source, jobs, scheduler):
    rows = [jobs[index] for index in sorted(jobs)]
    counts = Counter(row['status'] for row in rows)
    unresolved = Counter((row['status'], row.get('reason') or 'No reason was recorded.')
                         for row in rows if row['status'] in ('failed', 'needs_review'))
    finished = counts['complete']+unresolved.total()
    timings = [row['timing'] for row in rows if 'timing' in row]
    metrics = {'finished': finished, 'failed_rate': counts['failed']/finished if finished else None,
               'unresolved_rate': unresolved.total()/finished if finished else None,
               'timed_sources': len(timings), 'active_source_seconds': round(sum(row['active_seconds'] for row in timings), 3),
               'failure_reasons': [{'status': status, 'reason': reason, 'count': count}
                                   for (status, reason), count in unresolved.most_common()]}
    write_object(root/'batch.json', {'source_directory': str(source), 'total': len(rows),
                 'counts': dict(counts), 'metrics': metrics, 'jobs': rows, 'scheduler': scheduler})


def restore_scheduler(saved, log):
    scheduler = saved.get('scheduler')
    if (isinstance(scheduler, dict)
            and all(isinstance(scheduler.get(key), list) for key in ('running', 'waiting', 'transport_retried'))
            and isinstance(scheduler.get('evidence_retries'), dict)):
        if isinstance(scheduler.get('transport_retries'), dict):
            return scheduler
        return {**scheduler, 'transport_retries': {str(index): 1 for index in scheduler['transport_retried']}}

    rows = saved.get('jobs', [])
    running = [row['index'] for row in rows if row['status'] in ('running', 'interrupted')]
    waiting = [row for row in rows if row['status'] == 'queued']
    waiting.sort(key=lambda row: (bool(row.get('visual_deferred')), row['index']))
    appended, retries, transport = {}, {}, Counter()
    if rows and log.is_file():
        with log.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind, index = event.get('event'), event.get('index')
                if kind == 'batch.started':
                    appended.clear()
                    retries.clear()
                    transport.clear()
                    snapshot = event.get('scheduler', {})
                    retries.update(snapshot.get('evidence_retries', {}))
                    transport.update(snapshot.get('transport_retries', {}))
                elif kind in ('batch.source_requeued', 'batch.source_yielded') and type(index) is int:
                    appended.pop(index, None)
                    appended[index] = None
                    if kind == 'batch.source_requeued' and type(count := event.get('retry')) is int:
                        retries[str(index)] = count
                        if event.get('transient_vision_error') is True:
                            transport[str(index)] += 1
    waiting_indices = {row['index'] for row in waiting}
    order = [row['index'] for row in waiting if row['index'] not in appended]
    order.extend(index for index in appended if index in waiting_indices)
    return {'running': running, 'waiting': order, 'evidence_retries': retries,
            'transport_retried': sorted(map(int, transport)), 'transport_retries': dict(transport)}


def retry_evidence(directory, identity):
    proofs = 0
    for path in (directory/'absence-evidence.json', *directory.glob('feedback-absence-*.json')):
        try:
            saved = read_object(path)
        except (OSError, ValueError):
            saved = {}
        if saved.get('source_sha256') == identity and isinstance(saved.get('proofs'), dict):
            proofs += sum(len(items) for items in saved['proofs'].values() if isinstance(items, dict))
    refinements = read_refinements(directory/REFINEMENT_FILE, identity)
    return len(overlay_regions(directory, identity)), proofs, sum(map(len, refinements.values()))


def run_batch(source, root, config, local, cuda, workers, render_workers, resume, retry, *,
              text_proposals=None, prior_runs=(), learn_runs=(), evidence_retries=0):
    if config.get('discovery_local'):
        # Later edits in the workspace must not change the rule identity of this worker.
        motif_review_rules()
        absence_review_rules()
    imported = prior_regions(prior_runs)
    history = prior_visual_history(prior_runs)
    manifest = root/'manifest.json'
    if not resume:
        items = source_inventory(source)
        root.mkdir(parents=True, exist_ok=False)
    with lock_job(root):
        if resume:
            inventory = read_object(manifest)
            items = inventory['sources']
            if (not items or inventory.get('source_directory', str(source)) != str(source)
                    or any(Path(item['source']).parent != source for item in items)):
                raise ValueError('The batch belongs to a different source directory.')
        else:
            write_object(manifest, {'source_directory': str(source), 'sources': items})
        saved = read_object(root/'batch.json') if resume and (root/'batch.json').is_file() else {}
        previous = saved.get('jobs', [])
        schedule = restore_scheduler(saved, root/'batch-events.jsonl')
        unfinished = {row['index'] for row in previous if row['status'] in ('running', 'queued', 'interrupted')}
        jobs = {row['index']: row for row in previous}
        deferred = {row['index'] for row in previous if row.get('visual_deferred')}
        pending = []
        trace = job_trace(root/'batch-events.jsonl')
        if learn_runs:
            config = {**config, 'learned_marks': config.get('learned_marks', str(root/'learned-marks'))}
            learn_from_batches(learn_runs, Path(config['learned_marks']), trace)
        for item in items:
            index, identity = item['index'], item['source_sha256']
            row = jobs.get(index, {})
            timing = {'timing': row['timing']} if 'timing' in row else {}
            if reason := source_intake_error(item, resume=resume):
                jobs[index] = {**item, **timing, 'status': 'failed', 'failure_stage': 'intake', 'reason': reason}
                trace('batch.input_rejected', index=index, reason=reason)
                continue
            directory = root/'jobs'/f'{index:03d}'
            required, added = import_overlay_regions(directory, identity, imported.get(identity, ()))
            if added:
                trace('batch.evidence_imported', index=index, regions=added)
            if row.get('status') == 'complete':
                path = Path(row['result'])
                try:
                    unchanged = path.is_file() and digest(path) == row['result_sha256']
                except OSError:
                    unchanged = False
                if not unchanged:
                    reason = 'The accepted output is missing, unreadable or changed.'
                    row = jobs[index] = {**item, **timing, 'status': 'failed',
                                         'failure_stage': 'release_validation', 'reason': reason}
                    trace('batch.acceptance_withheld', index=index, reason=reason, previous_result=str(path))
                elif not required:
                    continue
            if resume and (verdict := completed_route(item, root, required)):
                if row.get('status') != 'complete' or verdict['result'] != row.get('result'):
                    jobs[index] = {**verdict, **timing}
                continue
            if row.get('status') == 'complete':
                reason = 'No verified published candidate covers the recorded overlay evidence.'
                row = jobs[index] = {**item, **timing, 'status': 'needs_review', 'reason': reason,
                                     'routes': row.get('routes', [])}
                trace('batch.acceptance_withheld', index=index, reason=reason, previous_result=str(path))
            if row.get('status') in ('needs_review', 'failed') and not retry:
                continue
            jobs[index] = {**item, **timing, 'status': 'queued', 'visual_deferred': index in deferred}
            pending.append(item)
        pending.sort(key=lambda item: item['index'] in deferred)
        by_index = {item['index']: item for item in pending}
        queue = deque(by_index.pop(index) for index in [*schedule['running'], *schedule['waiting']]
                      if type(index) is int and index in by_index)
        queue.extend(by_index.values())
        unfinished.intersection_update(item['index'] for item in pending)
        saved_retries = schedule['evidence_retries']
        requeued = Counter({index: count for index in unfinished
                            if type(count := saved_retries.get(str(index))) is int and count > 0})
        transport_requeued = Counter({index: count for index in unfinished
                                      if type(count := schedule['transport_retries'].get(str(index))) is int and count > 0})
        futures = {}

        def publish(starting=None):
            running = [item['index'] for item, _, _ in futures.values()]
            if starting is not None:
                running.append(starting)
            publish_progress(root, source, jobs, {'running': running, 'waiting': [item['index'] for item in queue],
                             'evidence_retries': dict(requeued), 'transport_retried': sorted(transport_requeued),
                             'transport_retries': dict(transport_requeued)})

        publish()
        if resume and previous:
            trace('batch.scheduler_restored', interrupted=len(schedule['running']), waiting=len(queue),
                  retries=sum(requeued.values()), transport_retried=len(transport_requeued))
        trace('batch.started', workers=workers, render_workers=render_workers, queued=len(pending), total=len(items), model=config['model'],
              evidence_retries=evidence_retries,
              scheduler={'evidence_retries': dict(requeued), 'transport_retries': dict(transport_requeued)},
              effort=config['reasoning_effort'], local_detector=local is not None, local_ocr=text_proposals is not None,
              local_detector_device=config.get('local_detector_device', 'gpu') if local is not None else None,
              local_vision=config.get('discovery_local', {}).get('model'),
              local_proposal_frames=config.get('local_proposal_frames', 0))
        render_gate = BoundedSemaphore(render_workers)
        config = {**config, 'render_workers': render_workers}
        backend = config.get('discovery_local') if pending else None
        with local_service(backend, root/'local-vision.log', trace), ThreadPoolExecutor(max_workers=workers) as pool:
            if signature := evidence_identity(backend):
                config = {**config, 'local_evidence_identity': signature}
                trace('batch.local_evidence_cache_enabled', model=signature['model'],
                      server_sha256=signature['server_sha256'])
            def evidence(item):
                return retry_evidence(root/'jobs'/f"{item['index']:03d}", item['source_sha256'])

            def refresh_regions(item):
                index, identity = item['index'], item['source_sha256']
                incoming = [rect for peer, _ in history.get(identity, ()) for rect in overlay_regions(peer, identity)]
                required, added = import_overlay_regions(root/'jobs'/f'{index:03d}', identity, incoming)
                if added:
                    trace('batch.evidence_imported', index=index, regions=added)
                return required, added

            def refresh_release(item, row):
                required, added = refresh_regions(item)
                if not added or row['status'] != 'complete':
                    return row
                verdict = completed_route(item, root, required)
                if verdict is not None and verdict['result_sha256'] == row.get('result_sha256'):
                    return row
                reason = 'The completed repair does not cover the latest confirmed overlay regions.'
                trace('batch.acceptance_withheld', index=item['index'], reason=reason,
                      previous_result=row.get('result'))
                return {**item, 'status': 'needs_review', 'reason': reason, 'routes': row.get('routes', [])}

            def process_source(item, prior, budget, bounded_local_wait):
                remaining = config.get('source_budget_seconds', 1800)-jobs[item['index']].get('timing', {}).get('active_seconds', 0)
                if remaining <= 0:
                    return {**item, 'status': 'needs_review', 'budget_exhausted': True,
                            'reason': 'The video exhausted its processing budget across batch retries.'}, 0.
                source_config = {**config, 'source_budget_seconds': remaining}
                if bounded_local_wait and signature:
                    admission = {'remaining': LOCAL_ADMISSION_SECONDS}
                    for name in ('discovery_local', 'grounding_local'):
                        backend = config.get(name)
                        if backend and all(backend[key] == signature[key] for key in ('endpoint', 'model')):
                            source_config[name] = {**backend, 'source_admission_budget': admission}
                    trace('batch.local_admission_budget', index=item['index'], seconds=LOCAL_ADMISSION_SECONDS)
                with wm_grok.source_review_budget(budget, max_slot_wait=FIRST_TURN_GROK_SLOT_WAIT):
                    started = monotonic()
                    row = run_source(item, root, source_config, local, cuda, render_gate, text_proposals, history=prior)
                    return row, round(monotonic()-started, 3)

            def submit_next():
                if not queue:
                    publish()
                    return
                item = queue.popleft()
                refresh_regions(item)
                before = evidence(item) if evidence_retries else None
                index = item['index']
                jobs[index].update(status='running', visual_deferred=index in deferred)
                prior = history.get(item['source_sha256'], ())
                budget = FIRST_TURN_GROK_SECONDS if queue and index not in deferred else None
                bounded_local_wait = budget is not None and evidence_retries > 0 and index not in transport_requeued
                publish(starting=index)
                started = monotonic()
                future = pool.submit(process_source, item, prior, budget, bounded_local_wait)
                futures[future] = item, before, started

            for _ in range(min(workers, len(queue))):
                submit_next()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in [task for task in futures if task in done]:
                    item, before, started = futures.pop(future)
                    seconds = round(monotonic()-started, 3)
                    try:
                        result, seconds = future.result()
                        row = refresh_release(item, result)
                    except Exception as error:
                        row = {**item, 'status': 'failed', 'reason': str(error)}
                        trace('batch.worker_error', index=item['index'], error_type=type(error).__name__, reason=str(error))
                    index = item['index']
                    previous_timing = jobs[index].get('timing', {})
                    row['timing'] = {'attempts': previous_timing.get('attempts', 0)+1,
                                     'active_seconds': round(previous_timing.get('active_seconds', 0.)+seconds, 3)}
                    if index in deferred:
                        row['visual_deferred'] = True
                    jobs[index] = row
                    trace('batch.source_finished', index=item['index'], status=row['status'], reason=row.get('reason'),
                          result=row.get('result'), seconds=seconds)
                    print(f'{item["index"]:03d} {row["status"]}', flush=True)
                    if row.get('visual_deferred') and index not in deferred:
                        deferred.add(index)
                        queue.append(item)
                        jobs[index] = {**row, 'status': 'queued'}
                        trace('batch.source_yielded', index=index, queued=len(queue), reason=row.get('reason'))
                    elif (row['status'] in ('needs_review', 'failed') and requeued[index] < evidence_retries
                          and not row.get('workflow_stalled') and not row.get('budget_exhausted')):
                        after = evidence(item)
                        gained = any(new > old for old, new in zip(before, after, strict=True))
                        transport = transport_requeued[index] < 10 and (
                            row.get('transient_local_vision') or row.get('transient_grok_vision')
                            or transient_grok_error(row.get('reason')))
                        if gained or transport:
                            if transport:
                                transport_requeued[index] += 1
                            requeued[index] += 1
                            queue.append(item)
                            jobs[index] = {**row, 'status': 'queued'}
                            trace('batch.source_requeued', index=index, retry=requeued[index],
                                  new_regions=after[0]-before[0], new_proofs=after[1]-before[1],
                                  new_refinements=after[2]-before[2],
                                  transient_vision_error=transport)
                    submit_next()
            if history:
                for item in items:
                    index = item['index']
                    if jobs[index]['status'] == 'complete':
                        row = jobs[index]
                        refreshed = refresh_release(item, row)
                        if 'timing' in row:
                            refreshed['timing'] = row['timing']
                        jobs[index] = refreshed
                publish()
        counts = dict(Counter(row['status'] for row in jobs.values()))
        trace('batch.finished', counts=counts)
        print(counts, flush=True)
        return 0 if counts.get('complete') == len(items) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('job', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'profiles'/'grok-director.json')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--detector-deps', type=Path)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 25))
    parser.add_argument('--render-workers', type=int, default=1, choices=(1, 2))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-unresolved', action='store_true')
    parser.add_argument('--evidence-retries', type=int, choices=range(31),
                        help='Retry an unresolved source after the queue if it gained verified evidence or had a temporary visual error.')
    parser.add_argument('--prior-run', type=Path, action='append', default=[],
                        help='Reuse confirmed overlay regions from this batch for sources with identical bytes.')
    parser.add_argument('--learn-from-run', type=Path, action='append', default=[],
                        help='Learn reusable watermark proposals from verified releases in this batch.')
    parser.add_argument('--cpu-decode', action='store_true')
    args = parser.parse_args()
    if args.retry_unresolved and not args.resume:
        parser.error('--retry-unresolved requires --resume.')
    source, root = args.source.resolve(strict=True), args.job.resolve()
    if not source.is_dir():
        parser.error('Choose a directory of original MP4 files.')
    if args.detector_deps:
        sys.path.insert(0, str(args.detector_deps.resolve(strict=True)))
    cv2.setNumThreads(2)
    config_path = args.config.resolve(strict=True)
    config = load_settings(config_path)
    config.update(batch_routing=True)
    config.setdefault('learned_marks', str(root/'learned-marks'))
    if config.get('local_layers') and not (args.weights or config.get('local_proposal_frames')):
        parser.error('Local layer discovery requires --weights or local_proposal_frames.')
    device = 'cpu' if config.get('local_detector_device') == 'cpu' else 0
    local = LocalDetector(args.weights.resolve(strict=True), device=device) if args.weights else None
    text_proposals = None
    if config.get('local_layers') and local is not None and (paths := config.get('ocr_python_dirs', [])):
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip() for path in paths):
            parser.error('ocr_python_dirs must list Python dependency directories.')
        for path in reversed(paths):
            sys.path.insert(0, str((config_path.parent/Path(path)).resolve(strict=True)))
        from wm_ocr import TextProposals
        text_proposals = TextProposals()
    return run_batch(source, root, config, local, not args.cpu_decode, args.workers, args.render_workers,
                     args.resume, args.retry_unresolved, text_proposals=text_proposals,
                     prior_runs=[path.resolve(strict=True) for path in args.prior_run],
                     learn_runs=[path.resolve(strict=True) for path in args.learn_from_run],
                     evidence_retries=config.get('evidence_retries', 0) if args.evidence_retries is None
                     else args.evidence_retries)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
