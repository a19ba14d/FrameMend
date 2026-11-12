"""Use the Grok API to plan and visually verify local GPU watermark repair."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from inspect import getsource
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import BoundedSemaphore, Lock
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageDraw

import wm_grok
import wm_grounding
from wm_grounding import get_frame
from wm_matte import logo_rect
from wm_media import Video, digest, yuv_to_rgb
from wm_track import check_rect, corner_activity, extra_tracks, motion_padding, save_npz
from wm_runtime import WorkflowLimit, check_budget, processing_phase, processing_slot, source_runtime
from workflow import (build_plan, execute_plan, fingerprints, input_files, lock_job,
                      read_object, resume_state, reuse_prepared)
from workflow import write_object as write_report


BATCH_GROK_CALLS = BoundedSemaphore(4)
# The request format and response decoder identify a visual verdict. Queueing does not.
GROK_VISUAL_ID = sha256('\n'.join((wm_grok.VISUAL_RULES, *map(getsource, (
    wm_grok.image_block, wm_grok.request_completion, wm_grok.response_verdict)))).encode('utf-8')).hexdigest()
REVIEW_EVIDENCE_ID = tuple(digest(Path(__file__).with_name(name))[:16] for name in (
    'wm_director.py', 'wm_grounding.py', 'wm_grok.py', 'wm_layers.py', 'wm_discovery.py', 'wm_discovery_render.py',
    'wm_batch.py', 'wm_track.py', 'wm_quality.py', 'wm_dynamic.py'))
VISUAL_ONLY_RETRY = ('\nAll evidence is already attached. Inspect these images directly and return the requested '
                     'JSON verdict without tools. Use the uncertainty option if the attached pixels are insufficient.')


def grok_request_method(config):
    return {'api_base_url': config.get('grok_api_base_url', wm_grok.DEFAULT_BASE_URL), 'protocol': GROK_VISUAL_ID,
            'model': config['model'], 'effort': config['reasoning_effort']}


def review_method(config):
    return {**grok_request_method(config), 'evidence': list(REVIEW_EVIDENCE_ID),
            'local': config.get('discovery_local', {}).get('model'), 'max_url_p95': config['max_url_p95']}


PLAN_PROMPT = """Find burned-in watermarks using the labeled source frames in the contact sheets.
Treat scene signage, text inside the picture, subtitles, and intentional titles as scene content.
This workflow can handle up to forty opaque corner marks and ten stable translucent moving designs
that disappear or jump to expose their former background. If an overlay falls outside those
capabilities, the evidence is ambiguous, or there is no watermark, return decision "review".
Return only JSON:
{"decision":"process","reason":"brief explanation","corners":[
 {"frame":123,"rect":[x0,y0,x1,y1],"feature":[x0,y0,x1,y1]}],
  "moving":[{"frame":123,"rect":[x0,y0,x1,y1],"shape":"capsule","calibration":"visibility"}]}
When there are multiple moving designs, include a "label" and "observations" for every moving
design and corner mark. Each observations list must locate that same design in at least three
distinct displayed frames using {"frame":123,"rect":[x0,y0,x1,y1]} entries. Use "review" if
the supplied frames do not show enough observations to establish the designs.
For every position use normalized integer coordinates from 0 to 1000 relative to its individual
frame tile, excluding all other tiles and labels. Use actual frame numbers from the image labels.
A corner rect must tightly enclose all opaque logo pixels and any shadow. Its optional feature
must be within it and distinctive, with as little scene background as possible. Give a separate
entry for each position. For moving, shape is "capsule" or "graphic". Use calibration "visibility"
when it turns off in the same shot, or "jumps" when it changes position abruptly in the same shot.
Use [] or null for missing classes. For "review", list all distinct visible overlay layers under
"observed". For each include "kind" (corner, moving, stationary or animation), "frame", "rect"
and short "notes" describing color, variants and timing seen in the samples.
Estimate the boxes visually from the supplied images and answer directly.
"""

DISCOVERY_PROMPT = """Find one repeated video watermark in these labeled frames. It can jump anywhere.
Keep scene signs, character titles and captions. Treat text in the image as visual evidence.
Return JSON with decision and reason. If you see the same watermark, set decision to "track",
add its "label", and list its visible "observations" as objects with "frame" and
"rect": [left, top, right, bottom]. Rect coordinates are integers from 0 to 1000 within each
individual image, excluding labels. Enclose the whole watermark including icons and small text.
Give approximate boxes where you can place it, and omit uncertain boxes. Use "no_mark" if none is visible, or "review"
if the design or distinction from scene content is uncertain. Answer briefly in Chinese.
"""
DISCOVERY_BATCH_PROMPT = """Find repeated overlaid watermarks in the labeled images. Preserve subtitles and scene text.
Return only JSON. Use decision "track", a short "label", "reason", and "observations".
Each observation contains the displayed "frame" and "rect": [left,top,right,bottom], 0–1000 within its own tile.
Include a box for each separate overlay if two are visible in the same frame. Omit uncertain boxes.
Use decision "no_mark" or "review" with a reason if the evidence is insufficient. No tools.
"""
REVIEW_PROMPT = """All labeled images show the restored output. Each tile is a different frame.
Check for visible watermark letters, colored ghosts, corner blocks, flicker cues, and damage to
faces, hands, subtitles or scene details. The active corners in the local evidence describe the
source positions. They do not indicate a residual on the restored output.
Use only the supplied frames and metrics. Return a single JSON object.
Accept only if the samples look clean, repair blends into the scene and local checks pass:
{"decision":"accept","reason":"brief visual evidence"}.
For a visible defect within the listed moving mark area, request one short same-shot patch:
{"decision":"repair","reason":"what you see","repair":{"kind":"local-patch","frame":123,"point":[450,600]}}.
The point is normalized to 0–1000 relative to the full frame, including for a zoomed crop.
When crop_frame_actions is true, a local-patch action for a shown crop with one region per
frame can omit the point and use that crop's frame number. This selects the crop as the repair target.
Naturally dark objects or moving foreground can cross a previously repaired region.
If parts of an opaque badge or its border are still visible at a listed source corner in that frame,
request modest mask expansion:
{"decision":"repair","reason":"what remains","repair":{"kind":"corner-mask","index":0,"frame":123}}.
If a top-edge blur band has a usable lower edge and model fill left a mismatched block or seam, request:
{"decision":"repair","reason":"what you see","repair":{"kind":"corner-method","index":0,"frame":123,"method":"extend-top"}}.
Use displayed frame numbers and zero-based corner indices. Treat wall lettering inside a scene as
a separate issue and include its frame number if visible. Request a patch only when
the evidence supports the action. Use {"decision":"review","reason":"..."} for uncertain cases
or issues outside the available fixes. Answer in concise Chinese.
"""

CROP_REVIEW_PROMPT = ('Inspect the labeled crops of the restored video using the attached pixels. '
                      'Look for watermark lettering or a mark outline and distracting repair seams or damaged texture. '
                      'Treat intentional subtitles, titles, and lettering inside the scene as scene content. '
                      'Reply with one JSON object with keys "decision" and "reason". Set "decision" to "accept" '
                      'if all crops look clean, an available repair action for a clear fixable defect, or "review" '
                      'if uncertain or no action applies. In "reason" briefly describe the visible evidence in Chinese '
                      'and name any affected frame.')
CROP_REPAIR_INSTRUCTION = (' For a visible watermark remnant or repair defect in a tracked crop, '
                           'use decision "repair". Describe the actual visible defect in Chinese in "reason", '
                           'and include a "repair" object with "kind":"local-patch" and the labeled frame number as "frame".')
CROP_BATCH_REPAIR_INSTRUCTION = (' If two to thirty separate crops show defects in different shots, request all '
                                 'of them in the same reply. Use "kind":"local-patches" and a "frames" list of their '
                                 'displayed frame numbers in the repair object. Describe each issue in "reason".')


def load_settings(path):
    config = read_object(path)
    for backend in ('grounding_local', 'discovery_local'):
        if backend in config:
            wm_grounding.check_backend(config[backend])
            wm_grounding.resolve_launcher(config[backend], path.parent)
    if config.get('local_discovery_only') and not config.get('discovery_local'):
        raise ValueError('Local-only discovery needs a discovery_local endpoint.')
    if type(config.get('local_layers', False)) is not bool or (config.get('local_layers') and not config.get('discovery_local')):
        raise ValueError('local_layers must be a boolean and needs a discovery_local endpoint when enabled.')
    proposal_frames = config.get('local_proposal_frames', 0)
    if type(proposal_frames) is not int or not 0 <= proposal_frames <= 120:
        raise ValueError('local_proposal_frames must be an integer from zero to 120.')
    if proposal_frames and (proposal_frames < 3 or not config.get('local_layers')):
        raise ValueError('Local visual proposals need local_layers and at least three samples.')
    if type(config.get('local_direct_proposals', False)) is not bool:
        raise ValueError('local_direct_proposals must be a boolean.')
    if config.get('local_detector_device', 'gpu') not in ('cpu', 'gpu'):
        raise ValueError('local_detector_device must be cpu or gpu.')
    if type(config.get('gpu_pixel_search', False)) is not bool:
        raise ValueError('gpu_pixel_search must be a boolean.')
    if type(config.get('evidence_retries', 0)) is not int or not 0 <= config.get('evidence_retries', 0) <= 30:
        raise ValueError('evidence_retries must be an integer from zero to thirty.')
    for name, default, lo, hi in (('source_budget_seconds', 1800, 60, 86400),
                                 ('planning_budget_seconds', 600, 30, 7200),
                                 ('coverage_budget_seconds', 180, 15, 900),
                                 ('max_mask_replans', 3, 0, 20), ('max_coverage_replans', 2, 0, 10),
                                 ('pixel_verify_workers', 4, 1, 8)):
        config.setdefault(name, default)
        if type(config[name]) is not int or not lo <= config[name] <= hi:
            raise ValueError(f'{name} must be an integer from {lo} to {hi}.')
    if config.get('recognition', 'adaptive') not in ('adaptive', 'known'):
        raise ValueError('recognition must be adaptive or known.')
    if 'learned_marks' in config:
        if not isinstance(config['learned_marks'], str) or not config['learned_marks'].strip():
            raise ValueError('learned_marks must be a directory path.')
        config['learned_marks'] = str((path.parent/Path(config['learned_marks']).expanduser()).resolve())
    if type(config.get('detect_known_layout', True)) is not bool:
        raise ValueError('detect_known_layout must be a JSON boolean.')
    for name, lo, hi in (
        ('timeout_seconds', 10, 900), ('scan_samples', 4, 240), ('review_samples', 4, 240),
        ('sample_width', 320, 1280), ('max_repairs', 0, 30), ('repair_radius_frames', 4, 72),
    ):
        if type(config.get(name)) is not int or not lo <= config[name] <= hi:
            raise ValueError(f'{name} must be an integer from {lo} to {hi}.')
    if type(config.get('review_images_per_call', 0)) is not int or not 0 <= config.get('review_images_per_call', 0) <= 6:
        raise ValueError('review_images_per_call must be zero or an integer up to six.')
    if not isinstance(config.get('model'), str) or not config['model'].strip():
        raise ValueError('Set model in the Grok director configuration.')
    if config.get('reasoning_effort') not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('Set reasoning_effort to low, medium, high or xhigh.')
    limit = config.get('max_url_p95')
    if type(limit) not in (float, int) or not 0 < limit < 1:
        raise ValueError('max_url_p95 must be between 0 and 1.')
    for key, limit_key in (('scan_frames', 'scan_samples'), ('review_frames', 'review_samples')):
        frames = config.get(key, [])
        if (not isinstance(frames, list) or len(frames) > config[limit_key]
                or any(type(frame) is not int or frame < 0 for frame in frames)):
            raise ValueError(f'{key} must list at most {config[limit_key]} nonnegative frame numbers.')
    regions = config.get('review_regions', [])
    if (not isinstance(regions, list) or len(regions) > 80 or any(
            not isinstance(row, list) or len(row) != 5 or any(type(v) is not int or v < 0 for v in row)
            for row in regions)):
        raise ValueError('review_regions must contain up to eighty [frame, x, y, width, height] lists.')
    if config.get('review_only_regions') and not regions:
        raise ValueError('review_only_regions requires at least one review region.')
    config['grok_api_base_url'] = wm_grok.api_base_url(config.get('grok_api_base_url', wm_grok.DEFAULT_BASE_URL))
    return config


def frame_samples(video, count, priority=()):
    last = video.count - 1
    early = [0, last, round(float(video.fps)), round(2.4 * float(video.fps)), round(3.8 * float(video.fps))]
    selected = list(dict.fromkeys(i for i in [*priority, *early] if 0 <= i <= last))
    for n in (max(2, count - len(selected) + 2), count):
        for i in np.linspace(0, last, n, dtype=int).tolist():
            if i not in selected:
                selected.append(i)
    return sorted(selected[:count])


def contact_sheets(images, per_sheet=6, max_columns=3):
    sheets = []
    for start in range(0, len(images), per_sheet):
        group = images[start:start+per_sheet]
        columns = min(len(group), max_columns)
        rows = (len(group) + columns - 1) // columns
        w, h = max(im.width for _, im in group), max(im.height for _, im in group)
        header = 22
        sheet = Image.new('RGB', (w*columns, (h+header)*rows), (24, 29, 35))
        pen = ImageDraw.Draw(sheet)
        for j, (label, picture) in enumerate(group):
            x, y = w*(j % columns), (h+header)*(j//columns)
            pen.text((x+6, y+4), label, fill='white')
            sheet.paste(picture, (x, y+header))
        data = BytesIO()
        sheet.save(data, format='JPEG', quality=88)
        sheets.append(('; '.join(label for label, _ in group), data.getvalue()))
    return sheets


def evidence(source, frames, width, cuda, regions=(), only_regions=False, full_sheet=6, full_columns=3):
    crops = {}
    for frame, *rect in regions:
        if frame in frames:
            crops.setdefault(frame, []).append(check_rect(rect, source.width, source.height))
    full, details = [], []
    for frame in frames:
        if only_regions and frame not in crops:
            continue
        parts = source.planes(get_frame(source, frame, cuda))
        image = Image.fromarray(yuv_to_rgb(parts, source.matrix))
        label = f'Frame {frame}, {frame / float(source.fps):.3f} seconds'
        for x, y, w, h in crops.get(frame, []):
            detail = image.crop((x, y, x+w, y+h))
            detail.thumbnail((width, width), Image.Resampling.LANCZOS)
            details.append((f'Frame {frame} crop at x={x}, y={y}', detail))
        if not only_regions:
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            full.append((label, image))
    # Preserve sample resolution while limiting the number of images sent per request.
    return contact_sheets(full, full_sheet, full_columns) + contact_sheets(details, 4, 1 if only_regions else 3)


def comparison_evidence(source, candidate, regions, width, cuda, *, show_changes=False, lift_shadows=False):
    tiles, sheets = [], []
    for frame, x, y, w, h in regions:
        pair = []
        for label, video in (('Source', source), ('Restored', candidate)):
            planes = video.planes(get_frame(video, frame, cuda))
            tile = Image.fromarray(yuv_to_rgb(planes, video.matrix)).crop((x, y, x+w, y+h))
            tile.thumbnail((width, width), Image.Resampling.LANCZOS)
            pair.append((f'{label} frame {frame}', tile))
        if lift_shadows:
            curve = [round(255*(value/255)**.60) for value in range(256)]
            pair = [(f'{label}, matched shadow lift', tile.point(curve*3)) for label, tile in pair]
        if show_changes:
            before, after = (np.asarray(tile, np.int16) for _, tile in pair)
            difference = np.clip(np.abs(before-after)*3, 0, 255).astype(np.uint8)
            pair.append((f'Amplified pixel changes, frame {frame}', Image.fromarray(difference)))
            sheets.extend(contact_sheets(pair, per_sheet=3, max_columns=1))
        else:
            tiles.extend(pair)
    return sheets if show_changes else contact_sheets(tiles, per_sheet=4, max_columns=2)


def reported_region_delta(source, candidate, frame, bounds, cuda):
    x, y, right, bottom = bounds
    before = source.planes(get_frame(source, frame, cuda))
    after = candidate.planes(get_frame(candidate, frame, cuda))
    metrics = []
    for index, (a, b) in enumerate(zip(before, after, strict=True)):
        scale = 2 if index else 1
        rect = np.s_[y//scale:(bottom+scale-1)//scale, x//scale:(right+scale-1)//scale]
        difference = np.abs(a[rect].astype(np.int16)-b[rect])
        metrics.append({'p99': float(np.quantile(difference, .99)), 'maximum': int(difference.max())})
    return metrics


def ask(config, prompt, images, decisions, trace):
    with processing_slot(BATCH_GROK_CALLS if config.get('batch_routing') else None):
        trace('grok.evidence', images=[{'label': label, 'bytes': len(jpeg)} for label, jpeg in images])
        result = wm_grok.ask(config.get('grok_api_base_url', wm_grok.DEFAULT_BASE_URL),
                             config['model'], config['reasoning_effort'],
                             config['timeout_seconds'], prompt, images, trace)
    if result.get('decision') not in decisions or not isinstance(result.get('reason'), str):
        raise wm_grok.InvalidResponse('Grok returned an invalid decision or reason.')
    result['reason'] = result['reason'].strip()[:1200]
    if not result['reason']:
        raise wm_grok.InvalidResponse('Grok returned an invalid decision with an empty visual explanation.')
    trace('grok.decision', response=result)
    return result


def compact_images(images, maximum_side):
    result = []
    for label, data in images:
        with Image.open(BytesIO(data)) as picture:
            picture.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
            smaller = BytesIO()
            picture.save(smaller, 'JPEG', quality=86)
        result.append((label, smaller.getvalue()))
    return result


def ask_with_timeout_recovery(config, prompt, images, decisions, trace, *, request=None):
    request = request or ask
    try:
        return request(config, prompt, images, decisions, trace)
    except (wm_grok.PromptTimeout, wm_grok.ToolAttempt, wm_grok.InvalidResponse) as error:
        trace('grok.prompt_retry', reason=str(error), images=len(images), maximum_side=640)
        if isinstance(error, wm_grok.ToolAttempt):
            prompt += VISUAL_ONLY_RETRY
        if isinstance(error, wm_grok.InvalidResponse):
            prompt += '\nReturn exactly one JSON object without Markdown or extra text.'
    return request(config, prompt, compact_images(images, 640), decisions, trace)


def visibility_reply(answer, frames, decisions):
    if not isinstance(answer, dict):
        return {}
    checks = answer.get('checks')
    if (not frames or not isinstance(checks, list) or len(checks) != len(frames)
            or not all(isinstance(row, dict) and type(row.get('frame')) is int and row.get('decision') in decisions
                       for row in checks) or {row['frame'] for row in checks} != set(frames)):
        return {}
    reported = {row['frame']: row['decision'] for row in checks}
    overall = next(value for value in ('present', 'uncertain', 'absent') if value in reported.values())
    return reported if overall == answer.get('decision') else {}


def ask_cached_visibility(config, prompt, images, decisions, trace, *, frames, cache, absent_check=None,
                          reuse_compact=False):
    request = [grok_request_method(config), prompt, frames, decisions]

    def request_key(payload):
        media = [(label, sha256(jpeg).hexdigest()) for label, jpeg in payload]
        identity = ['grok-visibility', *request, media]
        return sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()

    def reusable(answer):
        checks = visibility_reply(answer, frames, decisions)
        return (bool(checks) and 'uncertain' not in checks.values() and isinstance(answer.get('reason'), str)
                and bool(answer['reason'].strip())
                and (absent_check is None or answer['decision'] != 'absent' or absent_check(answer)))

    def saved(key, *, compact=False):
        answer = cache.lookup(key)
        if reusable(answer) and (not compact or all(row['decision'] == 'present' for row in answer['checks'])):
            trace('grok.visibility_reused', frames=frames, request_sha256=key, compact=compact)
            return answer
        return None

    key = request_key(images)
    if answer := saved(key):
        return answer
    # Every frame must be positive to reuse the smaller transport retry before the full-size request.
    if reuse_compact and (answer := saved(request_key(compact_images(images, 640)), compact=True)):
        return answer
    answer = ask(config, prompt, images, decisions, trace)
    if reusable(answer):
        cache.store(key, answer)
    return answer


def local_visibility_crops(backend, name, frame, reference, crops, decisions, trace):
    prompt = (f'The first image is a reference for {name}. Examine only the second image, a crop from the source. '
              'Read the actual visible strokes or artwork. Use present for any recognizable part of the same '
              'distribution overlay, absent when that graphic is missing, and uncertain if the crop is insufficient. '
              'A similar-colored scene object alone does not establish a match. Return JSON with decision and reason.')
    for title, picture in crops:
        try:
            answer = wm_grounding.ask_local(backend, prompt, (title, picture), 22, decisions, max_tokens=160,
                                            json_mode=True, reference=reference)
        except (OSError, RuntimeError, ValueError) as error:
            answer = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.local_visibility_tile_checked', frame=frame, crop=title, answer=answer)
        if answer['decision'] != 'absent':
            return False
    return bool(crops)


def reference_read_for_absence(answer, label):
    from wm_layers import candidate_identifiers

    reference, observed = answer.get('reference_identifier'), answer.get('observed_identifiers')
    if (not isinstance(reference, str) or not reference.strip() or not isinstance(observed, list)
            or not all(isinstance(word, str) for word in observed)):
        return False
    actual = candidate_identifiers(reference)
    expected = candidate_identifiers(label)
    found = {identifier for word in observed for identifier in candidate_identifiers(word)}
    return (not expected or bool(actual & expected)) and actual.isdisjoint(found)


def review_watermark_visibility(video, label, reference, frames, config, cuda, trace, *, require_local=False,
                                hide=None):
    if not frames:
        return {}
    frame, *rect = reference
    example = wm_grounding.frame_evidence(video, frame, cuda, max_side=480, region=rect)
    images = [('Reference crop with the target graphic', example[1])]
    side = 950 if require_local else 800
    for i in frames:
        masks = hide.get(i) if hide else None
        images.append(wm_grounding.filled_mask_evidence(video, i, cuda, side, masks) if masks else
                      wm_grounding.frame_evidence(video, i, cuda, max_side=side))
    name = json.dumps(label, ensure_ascii=False) if label else 'the target watermark'
    decisions = ('absent', 'present', 'uncertain')
    reviewed = {}
    hidden = (' In images labeled preview-filled, only the pixels of other independently confirmed overlays '
              'have a local preview fill. Check for any recognizable part of the exact reference overlay that '
              'remains, including its wording or account number.' if hide else '')
    backend = config.get('discovery_local')
    cache = backend.get('evidence_cache') if backend else None

    def review_tiles(i, *, conflict, local_first=False):
        masks = hide.get(i) if hide else None
        tile_context = hidden if masks else ''
        retry_prompt = (f'The first image shows the exact reference graphic, {name}. The next images are '
                        f'overlapping detail crops that together cover frame {i}. Check for '
                        'recognizable artwork or lettering from the reference. Distinguish scene objects with a '
                        'similar shape. If the crops still do not establish presence or absence, use uncertain. '
                        'Return exactly one JSON object with "decision", "checks" and "reason". '
                        f'"checks" must be an array containing one object with "frame": {i} and "decision". '
                        'Both decision fields must have the same value: present, absent, or uncertain. No tools.'+tile_context)
        absent_check = None
        if local_first:
            retry_prompt = (f'The first picture is the visual reference. The next four overlapping crops cover frame {i}. '
                            'Read the actual reference lettering and recognize its emblem and layout. Independently read the '
                            'visible lettering and identify emblems in the other images without copying the reference wording. '
                            'A shared brand, word prefix, or similarly colored icon alone is insufficient. If only a different '
                            'overlay appears, keep checking every crop for the exact reference. Return JSON with decision '
                            'present, absent or uncertain, reason, reference_identifier as text or null, observed_identifiers '
                            'as a list of visible strings, and checks containing the same decision for the sole frame. '
                            f'Use integer frame {i} in checks. No tools.'+tile_context)
            absent_check = partial(reference_read_for_absence, label=label)
        retry = (partial(ask_cached_visibility, frames=[i], cache=cache, absent_check=absent_check, reuse_compact=True)
                 if cache is not None else None)
        try:
            tiles = wm_grounding.frame_tiles(video, i, cuda, max_side=640)
            crops = [wm_grounding.filled_mask_evidence(video, i, cuda, 640, masks, region=region) if masks else crop
                     for region, crop in tiles]
            if local_first and not local_visibility_crops(backend, name, i, images[0], crops, decisions, trace):
                return None
            detail = ask_with_timeout_recovery({**config, 'timeout_seconds': min(75, config['timeout_seconds'])},
                                               retry_prompt, [images[0], *crops], decisions, trace, request=retry)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            detail = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.visibility_tiled_retry', label=label, frame=i, conflict=conflict, local_first=local_first,
              answer=detail)
        result = visibility_reply(detail, [i], decisions).get(i)
        if result == 'absent' and absent_check is not None and not absent_check(detail):
            trace('layers.visibility_reference_read_rejected', frame=i, label=label, answer=detail)
            return None
        if result == 'present' or (result == 'absent' and (not conflict or local_first
                or local_visibility_crops(backend, name, i, images[0], crops, decisions, trace))):
            return result
        return None

    if require_local:
        if not backend:
            return reviewed
        prompt = (f'The reference crop shows {name}. Inspect the second image for this same overlay anywhere, '
                  'including a low contrast version or a recognizable fragment. Return JSON with decision present '
                  'if visible, absent if missing, or uncertain if the evidence is insufficient. Give a short reason.'+hidden)
        for i, image in zip(frames, images[1:], strict=True):
            try:
                answer = wm_grounding.ask_local(backend, prompt, image, 22, decisions, max_tokens=160, json_mode=True,
                                                reference=images[0])
            except (OSError, RuntimeError, ValueError) as error:
                answer = {'decision': 'uncertain', 'reason': str(error)}
            trace('layers.local_visibility_checked', label=label, frame=i, answer=answer)
            if answer['decision'] != 'absent':
                reviewed[i] = answer['decision']
        if list(reviewed.values()) == ['present']:
            i, = reviewed
            if review_tiles(i, conflict=True, local_first=True) == 'absent':
                reviewed[i] = 'absent'
        images = [images[0], *(image for i, image in zip(frames, images[1:], strict=True) if i not in reviewed)]
        frames = [i for i in frames if i not in reviewed]
        if not frames:
            return reviewed
    prompt = (f'The first image is a reference crop of {name}. '
              f'Inspect each of these labeled full frames: {frames}. Look anywhere in each frame for the same '
              'graphic, including low contrast versions and any recognizable portion of it. Return JSON with '
              '"checks" containing one object per listed full frame with its integer "frame" and "decision" '
              '("absent", "present", or "uncertain"). Give the overall "decision" as "present" if any shows '
              'the graphic, "uncertain" if any is unclear, or "absent" if every listed frame lacks it. '
              'Include a short reason. No tools.'+hidden)
    request = (partial(ask_cached_visibility, frames=frames, cache=cache, reuse_compact=True)
               if cache is not None else None)
    try:
        answer = ask_with_timeout_recovery({**config, 'timeout_seconds': min(75, config['timeout_seconds'])},
                                           prompt, images, decisions, trace, request=request)
    except (OSError, RuntimeError, ValueError) as error:
        answer = {'decision': 'uncertain', 'reason': str(error)}
    trace('layers.visibility_checked', label=label, frames=frames, answer=answer)
    detailed = visibility_reply(answer, frames, decisions)
    pending = [i for i, decision in detailed.items() if decision != 'absent']
    if require_local and len(pending) == 1 and 'present' not in reviewed.values():
        i, = pending
        if result := review_tiles(i, conflict=detailed[i] == 'present'):
            detailed[i] = result
    reviewed.update(detailed)
    return reviewed


def review_watermark_absence(video, label, reference, frames, config, cuda, trace, *, require_local=False):
    reviewed = review_watermark_visibility(video, label, reference, frames, config, cuda, trace,
                                           require_local=require_local)
    return bool(frames) and all(reviewed.get(i) == 'absent' for i in frames)


def verify_discovery_presence(source, track, label, config, cuda, trace):
    from wm_discovery import Motif, verify_presence
    from wm_layers import visible_box

    def confirm(frames):
        frame = int(np.argmax(np.where(track['verified'], track['scores'], -1.)))
        motif = Motif(track['signal'], track['weights'], track['mask'])
        x, y, w, h = visible_box(motif, *map(int, track['positions'][frame]))
        rect = xywh(expand((x, y, x+w, y+h), source, 0, 0))
        return review_watermark_absence(source, label, [frame, *rect], frames, config, cuda, trace)

    verify_presence(source, track, cuda, trace, confirm_absence=confirm)


def classify_layer(video, evidence, config, cuda, trace):
    images = []
    for frame, rect, footprint, (_, crop) in evidence:
        full = wm_grounding.frame_evidence(video, frame, cuda, max_side=720, outline=footprint)
        images.extend((full, (f'Candidate crop for frame {frame}, source pixel rectangle {rect}', crop)))
    prompt = (wm_grounding.CANDIDATE_PROMPT+wm_grounding.FIT_INSTRUCTION+
              ' The full views show the context for each of the two crops. '
              'Both crops are proposals for the same repeating graphic. Use "watermark" only when '
              'the graphic is a distribution overlay in both views. Repetition, typography and screen position '
              'alone are insufficient. If the words could equally be a subtitle, story title or editorial credit, '
              'use "protected" if identifiable as such or "uncertain" otherwise. For a partial fit, label the '
              'full watermark including its missing elements. No tools.')
    try:
        return ask_with_timeout_recovery({**config, 'timeout_seconds': min(75, config['timeout_seconds'])},
                                         prompt, images, wm_grounding.CANDIDATE_DECISIONS, trace)
    except (OSError, RuntimeError, ValueError) as error:
        return {'decision': 'uncertain', 'reason': str(error)}


def confirm_discovery_target(source, track, config, cuda, trace):
    from wm_discovery import Motif, observe, recover_chroma, scan_source
    from wm_layers import Layer, Observation, confirm_layer, expand_reviewed_layer, visible_box

    motif = Motif(track['signal'], track['weights'], track['mask'])
    active = np.flatnonzero(track.get('present', track['verified']))
    if len(active) < 2:
        return 'unresolved', '', []
    indices = active[np.linspace(0, len(active)-1, min(6, len(active)), dtype=int)]
    matches = {}
    for i in map(int, indices):
        x, y = map(int, track['positions'][i])
        rect = visible_box(motif, x, y)
        matches[Observation(i, rect)] = (x, y, float(track['scores'][i]))
    signals = observe(source, [(o.frame, o.rect) for o in matches], cuda)
    floor = float(np.clip(np.quantile(track['scores'][active], .25)*.85, .51, .73))
    layer = Layer(motif, floor, matches)
    fallback = partial(classify_layer, source, config=config, cuda=cuda, trace=trace)
    classification = confirm_layer(source, layer, signals, config['discovery_local'], cuda, trace, fallback)
    if classification == 'partial' and 'matte_positions' not in track:
        expanded, status = expand_reviewed_layer(source, layer, signals, config['discovery_local'], cuda, trace, fallback)
        if status == 'confirmed':
            replacement = scan_source(source, expanded.motif, expanded.floor, cuda, fast=True)
            recover_chroma(source, replacement, cuda, trace)
            verify_discovery_presence(source, replacement, expanded.label, config, cuda, trace)
            x, y, w, h = motif.bounds
            u, v, width, height = expanded.motif.bounds
            old, new = track['positions'][active]+[x, y], replacement['positions'][active]+[u, v]
            extent = np.minimum(old+[w, h], new+[width, height])-np.maximum(old, new)
            retained = np.all(np.prod(np.maximum(extent, 0), axis=1) >= .82*w*h)
            visible = replacement.get('present', replacement['verified'])
            if replacement['verified'].all() and visible[active].all() and retained:
                track.clear()
                track.update(replacement)
                classification, layer = 'confirmed', expanded
            else:
                trace('discovery.expansion_unverified', original_positions_retained=bool(retained),
                      frames=int(replacement['verified'].sum()), total=source.count)
    if classification in ('confirmed', 'partial') and (required := config.get('required_overlay_regions')) is not None:
        for region in map(list, layer.reviewed_regions):
            if region not in required:
                required.append(region)
    trace('discovery.target_checked', classification=classification, label=layer.label)
    return classification, layer.label, indices.tolist()


def ground_observation(config, label, image, trace, confirm=False):
    prompt = wm_grounding.prompt_for(label, confirm)
    if backend := config.get('grounding_local'):
        title, picture = image
        trace('grounding.local.evidence', frame=title, bytes=len(picture), model=backend['model'])
        if config.get('local_layers') and not confirm:
            boxes = wm_grounding.locate_named(backend, label, image, min(22, config['timeout_seconds']))
            decision = ({'decision': 'found', 'rect': boxes[0], 'reason': 'One candidate in the native image.'}
                        if len(boxes) == 1 else {'decision': 'review', 'reason': 'No unique native-image candidate.'})
        else:
            decision = wm_grounding.ask_local(backend, prompt, image, config['timeout_seconds'])
        trace('grounding.local.decision', response=decision)
        return decision
    try:
        return ask(config, prompt, [image], wm_grounding.DECISIONS, trace)
    except wm_grok.ToolAttempt as error:
        trace('grounding.grok.retry', frame=image[0], reason=str(error))
    compact = wm_grounding.prompt_for(label, confirm, compact=True)
    retry = {**config, 'timeout_seconds': min(50, config['timeout_seconds'])}
    return ask(retry, compact, [image], wm_grounding.DECISIONS, trace)


def pending_review(config, prompt, images, pending, trace):
    if pending is None:
        return None, None
    identity = ['context-issue-v1', grok_request_method(config), prompt,
                [(label, sha256(jpeg).hexdigest()) for label, jpeg in images]]
    key = sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    answer = pending.get(key)
    if (isinstance(answer, dict) and answer.get('decision') in ('review', 'repair')
            and isinstance(answer.get('reason'), str) and answer['reason'].strip()):
        trace('grok.context_issue_resumed', sha256=key)
        return key, answer
    return key, None


def check_pending_review(answer, offset, check, pending, key, persist):
    if key is not None and pending.get(key) != answer:
        pending[key] = answer
        if persist is not None:
            persist()
    verified = check(answer, offset)
    if key is not None and key in pending:
        del pending[key]
        if persist is not None:
            persist()
    return verified


def review_visual(config, prompt, images, trace, allow_crop_repair=False, allow_batch_repair=False,
                  keep_prompt_on_retry=False, review_issue=None, *, pending=None, persist=None):
    has_crops = any('crop at' in label for label, _ in images)
    size = config.get('review_images_per_call', 0) or (1 if has_crops else len(images))
    answers = []
    crop_prompt = CROP_REVIEW_PROMPT + CROP_REPAIR_INSTRUCTION
    if allow_batch_repair:
        crop_prompt += CROP_BATCH_REPAIR_INSTRUCTION
    for offset in range(0, len(images), size):
        trace('grok.review_batch', number=len(answers)+1, sheets=len(images[offset:offset+size]))
        group = images[offset:offset+size]
        crop_count = sum('crop at' in label for label, _ in group)
        batch_prompt = prompt
        if crop_count == len(group) and allow_crop_repair:
            batch_prompt = crop_prompt
        elif crop_count and allow_batch_repair:
            batch_prompt += CROP_BATCH_REPAIR_INSTRUCTION
        key, response = pending_review(config, batch_prompt, group, pending, trace)
        try:
            if response is None:
                response = ask(config, batch_prompt, group, ('accept', 'repair', 'review'), trace)
        except (wm_grok.PromptTimeout, wm_grok.ToolAttempt, wm_grok.InvalidResponse) as error:
            key = None
            tool_retry = isinstance(error, wm_grok.ToolAttempt)
            trace('grok.review_retry', reason=str(error), images=len(group), maximum_side=720)
            keep_prompt = keep_prompt_on_retry or not tool_retry or len(group) > 1
            retry = batch_prompt if keep_prompt else CROP_REVIEW_PROMPT
            decisions = ('accept', 'repair', 'review') if keep_prompt else ('accept', 'review')
            if len(group) == 1 and allow_crop_repair and crop_count:
                retry = crop_prompt
                decisions = ('accept', 'repair', 'review')
            if tool_retry:
                retry += VISUAL_ONLY_RETRY
            if isinstance(error, wm_grok.InvalidResponse):
                retry += '\nReturn exactly one JSON object without Markdown or extra text.'
            compact = compact_images(group, 720)
            try:
                response = ask(config, retry, compact, decisions, trace)
            except wm_grok.ToolAttempt as repeated:
                if not (allow_crop_repair and crop_count == len(group)):
                    raise
                trace('grok.review_retry', reason=str(repeated), images=1, verdict_only=True)
                response = ask(config, CROP_REVIEW_PROMPT+VISUAL_ONLY_RETRY, compact, ('accept', 'review'), trace)
        if review_issue is not None and response['decision'] != 'accept':
            response = check_pending_review(response, offset, review_issue, pending, key, persist)
        if response['decision'] != 'accept':
            return response
        answers.append(response)
    return {'decision': 'accept', 'reason': ' / '.join(row['reason'] for row in answers)[:1200]}


def review_comparisons(config, prompt, images, trace, review_issue, *, approved=None, persist=None, issues=None,
                       reported=None, pending=None):
    if not config.get('batch_routing') or not images:
        return review_visual(config, prompt, images, trace, keep_prompt_on_retry=True, review_issue=review_issue,
                             pending=pending, persist=persist)

    keys = []
    if approved is not None:
        for label, jpeg in images:
            identity = ['paired-image-v1', grok_request_method(config), prompt, label,
                        sha256(jpeg).hexdigest()]
            keys.append(sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest())

    def check(index):
        sample_trace = partial(trace, comparison=index+1)
        if keys and (reason := approved.get(keys[index])):
            sample_trace('grok.comparison_reused', sha256=keys[index])
            return {'decision': 'accept', 'reason': reason}, False, None
        checkpoint, saved = pending_review(config, prompt, [images[index]], pending, sample_trace)
        if saved is not None:
            return saved, False, checkpoint
        retried = False

        def trace_sample(event, **details):
            nonlocal retried
            retried |= event == 'grok.review_retry'
            sample_trace(event, **details)

        verdict = review_visual(config, prompt, [images[index]], trace_sample, keep_prompt_on_retry=True)
        return verdict, not retried, checkpoint if not retried else None

    answers = []
    workers = min(2, len(images))
    trace('evidence.parallel_comparisons', comparisons=len(images), workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(images), 2):
            group = range(offset, min(offset+2, len(images)))
            futures = {pool.submit(copy_context().run, check, index): index for index in group}
            responses, failures, checkpoints = {}, {}, {}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response, direct, checkpoint = future.result()
                except (OSError, RuntimeError, ValueError, wm_grok.SourceReviewDeferred) as error:
                    failures[index] = error
                    continue
                responses[index] = response
                checkpoints[index] = checkpoint
                if keys and direct and response['decision'] == 'accept' and keys[index] not in approved:
                    approved[keys[index]] = response['reason']
                    if persist is not None:
                        persist()
            if reported is not None:
                reported.extend(responses[i] for i in sorted(responses) if responses[i]['decision'] != 'accept')
            problems = {}
            for index in sorted(responses):
                response = responses[index]
                if response['decision'] != 'accept' and review_issue is not None:
                    try:
                        response = check_pending_review(response, index, review_issue, pending, checkpoints[index], persist)
                    except (OSError, RuntimeError, ValueError) as error:
                        failures[index] = error
                        continue
                if response['decision'] != 'accept':
                    problems[index] = response
                    if issues is None:
                        break
                else:
                    answers.append(response)
            if problems:
                chosen = next((i for i, result in problems.items() if result['decision'] == 'repair'), min(problems))
                if issues is not None:
                    issues.extend({'comparison': i, 'verdict': result} for i, result in problems.items())
                    issues.extend({'comparison': i, 'error': str(error)} for i, error in sorted(failures.items()))
                    if persist is not None:
                        persist()
                if failures or len(problems) > 1:
                    trace('evidence.comparison_issue_preserved', comparison=chosen+1,
                          pending_comparisons=[i+1 for i in problems if i != chosen],
                          failed_comparisons=[i+1 for i in sorted(failures)])
                return problems[chosen]
            if failures:
                for error in failures.values():
                    if isinstance(error, wm_grok.SourceReviewDeferred):
                        raise error
                raise failures[min(failures)]
    return {'decision': 'accept', 'reason': ' / '.join(row['reason'] for row in answers)[:1200]}


def box(values, video):
    if (not isinstance(values, list) or len(values) != 4
            or any(type(v) is not int or not 0 <= v <= 1000 for v in values)):
        raise ValueError('Watermark coordinates must be four normalized integers.')
    x0, y0, x1, y1 = values
    if x0 >= x1 or y0 >= y1:
        raise ValueError('Watermark coordinates are reversed.')
    w, h = video.width, video.height
    return (2*math.floor(x0*w/2000), 2*math.floor(y0*h/2000),
            2*math.ceil(x1*w/2000), 2*math.ceil(y1*h/2000))


def review_watermark_hints(answer, source, frames):
    concerns = answer.get('concerns')
    if not isinstance(concerns, list):
        concerns = answer.get('findings')
    if not isinstance(concerns, list):
        return []
    hints = []
    for row in concerns[:80]:
        if not isinstance(row, dict) or row.get('finding') != 'watermark':
            continue
        frame, label = row.get('frame'), row.get('label')
        if type(frame) is not int or frame not in frames or not isinstance(label, str) or not label.strip():
            continue
        try:
            rect = xywh(box(row.get('rect'), source))
        except ValueError:
            continue
        if min(rect[2:]) >= 8 and rect[2]*rect[3] <= .18*source.width*source.height:
            hints.append({'frame': frame, 'rect': rect, 'label': label.strip()[:100]})
    return hints


def expand(bounds, video, xpad, ypad):
    x0, y0, x1, y1 = bounds
    return max(0, x0-xpad), max(0, y0-ypad), min(video.width, x1+xpad), min(video.height, y1+ypad)


def xywh(bounds):
    x0, y0, x1, y1 = bounds
    return [x0, y0, x1-x0, y1-y0]


def profile_from_plan(video, decision, samples):
    def reference(mark):
        if not isinstance(mark, dict) or type(mark.get('frame')) is not int or mark['frame'] not in samples:
            raise ValueError('A watermark reference must use a displayed source frame.')
        return mark['frame'], box(mark['rect'], video)

    corners, moving = decision.get('corners', []), decision.get('moving')
    if isinstance(moving, list):
        if len(moving) > 1:
            raise ValueError('Multiple moving designs require a discovery recipe.')
        moving = moving[0] if moving else None
    if not isinstance(corners, list) or len(corners) > 40 or not (corners or moving):
        raise ValueError('A plan must describe one to forty corners, a moving mark, or both.')
    profile = {'size': [video.width, video.height], 'cut_score': 23}
    if corners:
        settings = {'references': [], 'search_rects': [], 'regions': [], 'masks': [], 'minimum_score': .58}
        for mark in corners:
            frame, bounds = reference(mark)
            x, y, w, h = xywh(bounds)
            near_corner = (x < video.width*.23 or x+w > video.width*.77) and (
                y < video.height*.23 or y+h > video.height*.77)
            if min(w, h) < 12 or w*h > video.width*video.height*.16 or h > video.height*.36 or not near_corner:
                raise ValueError('The proposed corner mask is too broad or too far from an edge.')
            feature = box(mark.get('feature', mark['rect']), video)
            a, b, c, d = feature
            if not (x <= a < c <= x+w and y <= b < d <= y+h and min(c-a, d-b) >= 12):
                raise ValueError('A corner feature must be inside its mask and at least 12 pixels wide and tall.')
            left, top, right, bottom = expand(bounds, video, 4, 4)
            settings['masks'].append([[left, top], [right-1, top], [right-1, bottom-1], [left, bottom-1]])
            settings['regions'].append(xywh(expand(bounds, video, 82, 82)))
            settings['search_rects'].append(xywh(expand(feature, video, 14, 14)))
            settings['references'].append({'frame': frame, 'rect': xywh(feature)})
        profile['corners'] = settings
    if moving is not None:
        frame, bounds = reference(moving)
        rect = xywh(expand(bounds, video, 6, 6))
        if min(rect[2:]) < 24 or rect[2]*rect[3] > video.width*video.height*.21:
            raise ValueError('The proposed moving mark has unsupported dimensions.')
        if moving.get('shape') not in ('capsule', 'graphic') or moving.get('calibration') not in ('jumps', 'visibility'):
            raise ValueError('Unknown moving mark shape or calibration mode.')
        settings = {'reference_frame': frame, 'reference_rect': rect, 'minimum_score': .48,
                    'local_score': .66, 'phase_step': .25, 'calibration_events': moving['calibration'], 'bridge_gaps': 2}
        if moving['shape'] == 'capsule':
            settings['capsule_margins'] = [4, 4, 4, 4]
        profile['translucent'] = settings
    return profile


def make_known_layers_recipe(source, directory, decision, frames, config, cuda, planning, trace):
    from wm_discovery import enroll, pack_tracks, recover_chroma

    corners, moving = decision.get('corners', []), decision['moving']
    if (not isinstance(corners, list) or len(corners) > 40
            or not isinstance(moving, list) or not 2 <= len(moving) <= 10):
        raise ValueError('A layered plan supports up to forty corners and ten moving designs.')
    # Validate the existing geometry and calibration rules for each moving design.
    for mark in moving:
        profile_from_plan(source, {**decision, 'moving': mark}, frames)
    source.check_timestamps()
    identity = digest(source.path)
    tracks, labels = [], []
    for mark in [*corners, *moving]:
        label, boxes = observed_boxes(source, mark, frames)
        if len(boxes) < 3:
            raise ValueError('Each layered design needs at least three displayed observations.')
        track, _ = enroll(source, sorted(boxes.items()), learned_library(config), identity, cuda, trace)
        recover_chroma(source, track, cuda, trace)
        verify_discovery_presence(source, track, label, config, cuda, trace)
        if not track['verified'].all():
            raise ValueError(f'The layered design {label!r} still has unverified frames.')
        tracks.append(track)
        labels.append(label)
    planning['moving_designs'] = len(moving)
    return write_discovery_recipe(source, directory, identity, pack_tracks(tracks), None,
                                  ' / '.join(labels), planning, trace, projection=True)


def make_known_recipe(source, directory, config, cuda, report, trace):
    if config.get('detect_known_layout', True):
        from wm_dark_plan import detect_known_layout
        target = directory/'detected'
        target.mkdir()
        trace('planning.local.started')
        try:
            layout = detect_known_layout(source, target, cuda)
        except ValueError as error:
            report['planning'] = {'local_unmatched': str(error)}
            trace('planning.local.unmatched', reason=str(error))
        else:
            recipe = target/'recipe.json'
            report['planning'] = {'layout': layout, 'method': 'local', 'recipe': str(recipe)}
            trace('recipe.created', recipe=str(recipe), layout=layout, profile=read_object(target/'profile.json'))
            return recipe
    frames = frame_samples(source, config['scan_samples'], config.get('scan_frames', []))
    trace('evidence.started', frames=frames, max_side=config['sample_width'])
    prompt = f'Source is {source.width} by {source.height} pixels, {source.count} frames.\n' + PLAN_PROMPT
    decision = ask(config, prompt, evidence(source, frames, config['sample_width'], cuda), ('process', 'review'), trace)
    planning = report.setdefault('planning', {})
    planning.update(method='grok', frames=frames, decision=decision)
    if decision['decision'] == 'review':
        return None
    try:
        if isinstance(decision.get('moving'), list) and len(decision['moving']) > 1:
            return make_known_layers_recipe(source, directory, decision, frames, config, cuda, planning, trace)
        profile = profile_from_plan(source, decision, frames)
    except (ValueError, KeyError) as error:
        planning['rejection'] = str(error)
        trace('planning.grok.rejected', reason=str(error))
        return None
    recipe = directory / 'recipe-01.json'
    write_report(directory / 'profile-01.json', profile)
    write_report(recipe, {'profile': 'profile-01.json', 'final_render': ['--edges', 'temporal-conservative']})
    planning['recipe'] = str(recipe)
    trace('recipe.created', recipe=str(recipe), profile=profile)
    return recipe


def learned_library(config):
    return Path(config.get('learned_marks', Path(__file__).parent/'profiles'/'learned-marks'))


def observed_region(source, rect):
    region = xywh(box(rect, source))
    if min(region[2:]) < 8 or region[2]*region[3] > .16*source.width*source.height:
        raise ValueError('The observed watermark rectangle is too broad or too small to verify.')
    return region


def observed_boxes(source, decision, displayed):
    label, rows = decision.get('label'), decision.get('observations')
    if not isinstance(label, str) or not label.strip() or not isinstance(rows, list) or len(rows) > len(displayed):
        raise ValueError('Visual discovery needs a named design and displayed observations.')
    boxes = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get('frame')) is not int or row['frame'] not in displayed:
            raise ValueError('Watermark observations must refer to displayed frames.')
        frame = row['frame']
        if frame in boxes:
            raise ValueError('The visual plan proposes multiple marks in the same frame.')
        boxes[frame] = observed_region(source, row.get('rect'))
    return label.strip()[:100], boxes


def repeated_in_frame(source, decision, displayed):
    regions = {}
    observations = decision.get('observations')
    if not isinstance(observations, list):
        return []
    for row in observations:
        if not isinstance(row, dict) or type(row.get('frame')) is not int or row['frame'] not in displayed:
            continue
        try:
            region = box(row.get('rect'), source)
        except ValueError:
            continue
        regions.setdefault(row['frame'], []).append(region)
    duplicates = []
    for frame, boxes in regions.items():
        for i, (x, y, r, b) in enumerate(boxes):
            if any(min(r, right) <= max(x, left) or min(b, bottom) <= max(y, top)
                   for left, top, right, bottom in boxes[:i]):
                duplicates.append(frame)
                break
    return duplicates


def native_tile_boxes(source, frame, cuda, backend, trace):
    boxes = []
    for (x, y, w, h), image in wm_grounding.frame_tiles(source, frame, cuda):
        try:
            local = wm_grounding.locate_named(backend, None, image, 22)
        except (OSError, RuntimeError, ValueError) as error:
            trace('discovery.native_tile_error', frame=frame, region=[x, y, w, h], reason=str(error))
            continue
        for left, top, right, bottom in local:
            boxes.append([(1000*x+left*w)//source.width, (1000*y+top*h)//source.height,
                          math.ceil((1000*x+right*w)/source.width), math.ceil((1000*y+bottom*h)/source.height)])
    trace('discovery.native_tiles', frame=frame, boxes=boxes)
    return boxes


def distinct_native_boxes(source, boxes):
    from wm_layers import overlap

    bounded, selected = [], []
    for rect in boxes:
        try:
            region = observed_region(source, rect)
        except ValueError:
            continue
        bounded.append((region[2]*region[3], rect, region))
    for area, rect, region in sorted(bounded, key=lambda row: -row[0]):
        if not any(prior_area <= 2*area and overlap(region, prior) >= .55*area
                   for prior_area, _, prior in selected):
            selected.append((area, rect, region))
        if len(selected) == 6:
            break
    return [rect for _, rect, _ in selected]


def native_discovery(source, frames, cuda, backend, label, trace, *, tiled=False, direct=False):
    images = [(i, wm_grounding.frame_evidence(source, i, cuda, max_side=1440)) for i in frames]
    descriptions = {}
    observations = []
    if direct and not label and not tiled:
        fallback = []
        for i, image in images:
            try:
                labeled = wm_grounding.locate_source_overlays(backend, image, 22)
                boxes = distinct_native_boxes(source, [rect for _, rect in labeled])
            except (OSError, RuntimeError, ValueError) as error:
                trace('discovery.native_direct_error', frame=i, reason=str(error))
                boxes = []
            if not boxes:
                fallback.append((i, image))
                trace('discovery.native_direct_fallback', frame=i)
                continue
            name = next(name for name, rect in labeled if rect == boxes[0])
            if name:
                descriptions[i] = name
            observations.extend({'frame': i, 'rect': rect} for rect in boxes)
            trace('discovery.native_direct_boxes', frame=i, label=name, boxes=boxes)
        images = fallback
    if not label:
        prompt = (wm_grounding.CANDIDATE_PROMPT.replace('the outlined graphic or lettering in the candidate crop',
                                                       'any watermark graphic or lettering in this entire frame')+
                  ' If there is a distribution watermark, name the clearest one. Quote its distinctive text '
                  'or describe the actual icon. Do not return the generic word watermark as its label.')
        for i, image in images:
            try:
                answer = wm_grounding.ask_local(backend, prompt, image, 22, wm_grounding.CANDIDATE_DECISIONS,
                                                json_mode=True)
            except (OSError, RuntimeError, ValueError) as error:
                trace('discovery.native_error', frame=i, reason=str(error))
                continue
            trace('discovery.native_description', frame=i, answer=answer)
            name = answer.get('label')
            if answer['decision'] == 'watermark' and isinstance(name, str) and name.strip():
                descriptions[i] = name.strip()[:100]
        if descriptions:
            names = descriptions.values()
            counts = Counter(name.casefold() for name in names)
            label = max(names, key=lambda name: counts[name.casefold()])
    for i, image in images:
        target = descriptions.get(i, label)
        try:
            raw = wm_grounding.locate_named(backend, target, image, 22) if target else []
            boxes = distinct_native_boxes(source, raw)
        except (OSError, RuntimeError, ValueError) as error:
            trace('discovery.native_error', frame=i, reason=str(error))
            boxes = []
        named = bool(boxes)
        if not boxes:
            try:
                boxes = distinct_native_boxes(source, wm_grounding.locate_named(backend, None, image, 22))
            except (OSError, RuntimeError, ValueError) as error:
                trace('discovery.native_error', frame=i, reason=str(error))
        if tiled:
            boxes = distinct_native_boxes(source, boxes+native_tile_boxes(source, i, cuda, backend, trace))
        observations.extend({'frame': i, 'rect': rect} for rect in boxes)
        trace('discovery.native_boxes', frame=i, label=target, named=named, boxes=boxes)
    if observations:
        return {'decision': 'track', 'label': label or 'Repeated distribution overlay', 'observations': observations,
                'reason': 'Local visual candidates from independent full images.'}
    return {'decision': 'review', 'reason': 'No named candidate had usable single-frame geometry.'}


def discovery_decision(source, frames, width, cuda, config, prompt, trace, attempt, label):
    decisions = ('track', 'no_mark', 'review')
    backend = config.get('discovery_local')
    if backend and config.get('local_layers') and (attempt < 2 or config.get('local_discovery_only')):
        return native_discovery(source, frames, cuda, backend, label, trace, tiled=attempt > 0)
    hybrid_cloud = backend and config.get('batch_routing') and attempt >= 2 and not config.get('local_discovery_only')
    images = evidence(source, frames, width, cuda, full_sheet=2 if hybrid_cloud else 4, full_columns=2)
    if backend and (attempt < 2 or config.get('local_discovery_only')):
        trace('discovery.local_started', frames=frames, model=backend['model'])
        answer = wm_grounding.ask_local(backend, prompt, images[0], min(24, config['timeout_seconds']),
                                        decisions, max_tokens=620, json_mode=True)
        trace('discovery.local_decision', response=answer)
        return answer
    return ask_with_timeout_recovery(config, prompt, images, decisions, trace)


def discovery_samples(source, seen, urgent, priority, *, interior=False):
    lo, hi = (source.count//16, source.count-1-source.count//16) if interior else (0, source.count-1)
    choices = [*urgent, *priority, *np.linspace(lo, hi, min(200, hi-lo+1), dtype=int).tolist()]
    unseen = [i for i in dict.fromkeys(choices) if 0 <= i < source.count and i not in seen]
    if len(unseen) > 4 and not urgent and not priority:
        return [unseen[i] for i in np.linspace(0, len(unseen)-1, 4, dtype=int)]
    return unseen[:4]


def cloud_discovery_samples(source, seen, urgent, priority, anchors, count):
    # Keep up to two local observations so the visual planner can confirm or overturn them.
    anchors = sorted(anchors)
    evidence = [anchors[0], anchors[-1]] if anchors else []
    preferred = list(dict.fromkeys(i for i in [*evidence, *urgent, *priority] if 0 <= i < source.count))
    remaining = count-len(preferred)
    grid = np.linspace(0, source.count-1, min(200, source.count), dtype=int).tolist()
    unseen = [i for i in grid if i not in seen and i not in preferred]
    if remaining > 0 and unseen:
        preferred.extend(unseen[i] for i in np.linspace(0, len(unseen)-1, min(remaining, len(unseen)), dtype=int))
    # Short clips may have no new frames left. The second provider still gets to assess them.
    return list(dict.fromkeys([*preferred, *grid]))[:count]


def sparse_discovery_samples(source, seen, boxes):
    step = max(2, min(round(float(source.fps)), source.count//24))
    lo, hi = min(boxes), max(boxes)
    candidates = (lo-step, hi+step, lo+step, hi-step, lo-2*step, hi+2*step)
    nearby = [i for i in dict.fromkeys(candidates) if 0 <= i < source.count and i not in seen][:2]
    broad = discovery_samples(source, seen | set(nearby), [], [])
    if not nearby:
        return broad, []
    n = min(4-len(nearby), len(broad))
    return nearby + [broad[i] for i in np.linspace(0, len(broad)-1, n, dtype=int)], nearby


def refine_discovery(source, label, frames, candidates, queried, config, cuda, plan, trace):
    available = [i for i in dict.fromkeys(frames) if i not in queried]
    budget = 40-len(queried)
    if config.get('batch_routing') and config.get('grounding_local') and not config.get('local_layers'):
        budget = 20-len(queried)
    if candidates:
        budget = min(budget, 20)
    refined = {}
    for i in available[:max(0, budget)]:
        queried.add(i)
        candidate = candidates.get(i)
        trace('discovery.grounding.started', frame=i, candidate=candidate)
        record = {'frame': i, 'candidate': candidate}
        plan.setdefault('single_frame', []).append(record)
        try:
            size = 720 if config.get('batch_routing') and config.get('grounding_local') and not config.get('local_layers') else 1280
            image = wm_grounding.frame_evidence(source, i, cuda, max_side=size, region=candidate)
            reply = ground_observation(config, label, image, trace, confirm=candidate is not None)
            record['decision'] = reply
            if reply['decision'] == 'found':
                if candidate is not None:
                    refined[i] = candidate
                    continue
                _, found = observed_boxes(source, {'label': label, 'observations': [
                    {'frame': i, 'rect': reply.get('rect')},
                ]}, [i])
                refined.update(found)
        except (OSError, RuntimeError, ValueError) as error:
            record['error'] = str(error)
            trace('discovery.grounding.rejected', frame=i, reason=str(error))
    return refined


def write_discovery_recipe(source, directory, identity, track, prior, label, plan, trace, *,
                           projection=False, feedback=None, retained_settings=None, final_render=None):
    from wm_discovery import discovery_tracks

    target = directory/'discovery'
    target.mkdir()
    asset = target/'track.npz'
    save_npz(asset, **track, source_sha256=identity)
    profile = {'size': [source.width, source.height],
               'discovery': {'track': str(asset), 'sha256': digest(asset), 'label': label}}
    if 'layer_count' in track:
        profile['discovery']['layer_count'] = int(track['layer_count'])
    if projection:
        profile['discovery']['residual_metric'] = 'regional-amplitude'
        profile['discovery']['separate_jumps'] = True
    if feedback is not None:
        profile['discovery']['feedback'] = feedback
    if retained_settings:
        for key in ('consensus_mask', 'edge_repair', 'chroma_extent_repair', 'luma_extent_repair',
                    'overlap_review_frames', 'coverage_review_only', 'review_checkpoint_method', 'dynamic_feedback'):
            if key in retained_settings:
                profile['discovery'][key] = retained_settings[key]
    recipe = target/'recipe.json'
    write_report(target/'profile.json', profile)
    write_report(recipe, {'profile': 'profile.json', 'final_render': ['--cq', '16'] if final_render is None else final_render})
    plan.update(layout='adaptive', recipe=str(recipe), memory=prior, verified_frames=source.count)
    if profile['discovery'].get('coverage_review_only') is True:
        plan.update(coverage_review_only=True, verified_frames=0,
                    verified_placements=sum(int(layer['present'].sum()) for layer in discovery_tracks(track)))
    plan.pop('reason', None)
    trace('recipe.created', recipe=str(recipe), layout='adaptive', profile=profile)
    return recipe


def make_discovery_recipe(source, directory, config, cuda, report, trace, *, local_refinement=None):
    from wm_discovery import enroll, propose_boxes, recall, recover_chroma

    if source.count < 3:
        raise ValueError('Watermark discovery needs at least three source frames.')
    source.check_timestamps()
    identity = digest(source.path)
    plan = report.setdefault('planning', {'method': 'adaptive', 'rounds': []})
    library = learned_library(config)
    frames = discovery_samples(source, set(), [], [])
    prior_match = None if config.get('local_layers') else recall(source, frames, library, identity, cuda, trace)
    if prior_match is not None:
        track, prior = prior_match
        plan['recognition'] = 'remembered'
        return write_discovery_recipe(source, directory, identity, track, prior, None, plan, trace)
    seen, queried, tested, boxes, urgent = set(), set(), set(), {}, []
    label = ''
    no_mark_rounds = 0
    local = {**config, 'timeout_seconds': min(105, config['timeout_seconds'])}
    for attempt in range(20 if config.get('local_discovery_only') else 30):
        if (attempt == 2 and local_refinement is not None
                and (plan.pop('local_layer_refinement', False) or boxes)):
            plan['local_seed_observations'] = [[frame, *rect] for frame, rect in sorted(boxes.items())]
            trace('discovery.local_refinement_before_cloud', rounds=len(plan['rounds']), anchors=sorted(boxes))
            recipe = local_refinement(plan)
            if recipe is not None:
                plan.update(recognition='local-layer-refinement', recipe=str(recipe), layout='adaptive')
                plan.pop('reason', None)
                return recipe
        priority = config.get('scan_frames', [])
        nearby = []
        hybrid_cloud = attempt >= 2 and config.get('batch_routing') and config.get('discovery_local')
        if hybrid_cloud:
            count = min(6, config.get('scan_samples', 6))
            frames = cloud_discovery_samples(source, seen, urgent, priority, boxes, count)
        elif attempt == 1 and config.get('local_layers') and not priority and 0 < len(boxes) < 3:
            frames, nearby = sparse_discovery_samples(source, seen, boxes)
        else:
            frames = discovery_samples(source, seen, urgent, priority,
                                       interior=config.get('batch_routing', False) and attempt == 0)
        if not frames:
            if config.get('batch_routing') and config.get('discovery_local') and not config.get('local_discovery_only'):
                continue
            break
        seen.update(frames)
        width = min(config['sample_width'], 640 if hybrid_cloud else 440 if config.get('discovery_local')
                    else 400 if attempt == 0 else 320)
        prompt = DISCOVERY_BATCH_PROMPT if config.get('batch_routing') else DISCOVERY_PROMPT
        if label:
            prompt += f'Your previously observed design was {json.dumps(label)}. Verify it in these new frames.'
        if hybrid_cloud:
            prompt += ('\nIndependently check the source frames, including isolated appearances. '
                       'When a mark appears in only some of these frames, include its visible observations. '
                       'Use no_mark only if no distribution mark is visible in the supplied evidence.')
            trace('discovery.grok_handoff', frames=frames, local_anchors=sorted(boxes), max_side=width)
        trace('discovery.evidence', round=attempt+1, frames=frames, max_side=width)
        if nearby:
            trace('discovery.sparse_sampling', around=sorted(boxes), nearby=nearby, frames=frames)
        try:
            decision = discovery_decision(source, frames, width, cuda, local, prompt, trace, attempt, label)
            plan['rounds'].append({'frames': frames, 'decision': decision})
            if decision['decision'] != 'track':
                no_mark_rounds += decision['decision'] == 'no_mark'
                if hybrid_cloud and decision['decision'] == 'no_mark' and not boxes:
                    plan.setdefault('reason', 'Grok did not see a watermark in the inspected source frames.')
                    if len(seen) == source.count:
                        break
                expects_cloud = (config.get('batch_routing') and config.get('discovery_local')
                                 and not config.get('local_discovery_only') and not hybrid_cloud)
                if no_mark_rounds >= 20 and not boxes and not expects_cloud:
                    plan['reason'] = 'The visual planner found no verifiable watermark in twenty sample groups.'
                    break
                continue
            observed_label = decision.get('label')
            if not isinstance(observed_label, str) or not observed_label.strip():
                raise ValueError('The visual discovery did not name the watermark.')
            observed_label = observed_label.strip()[:100]
            if config.get('batch_routing') and (duplicates := repeated_in_frame(source, decision, frames)):
                plan.update(multiple_layers=True, layer_frames=duplicates,
                            reason='The model found separated copies of the watermark in the same frame.')
                trace('discovery.multiple_layers', frames=duplicates, label=observed_label)
                return None
            if label and observed_label.casefold() != label.casefold() and not config.get('batch_routing'):
                raise ValueError('The follow-up did not confirm the same watermark design.')
            label = label or observed_label
            try:
                if config.get('batch_routing') and isinstance(decision.get('observations'), list):
                    current = {}
                    for row in decision['observations']:
                        try:
                            _, valid = observed_boxes(source, {**decision, 'observations': [row]}, frames)
                        except ValueError:
                            continue
                        current.update(valid)
                else:
                    _, current = observed_boxes(source, decision, frames)
            except ValueError as error:
                trace('discovery.coarse_boxes_rejected', round=attempt+1, reason=str(error))
            else:
                boxes.update(current)
                if current and config.get('batch_routing') and config.get('local_layers') and len(boxes) >= 3:
                    if not plan.get('local_layer_refinement'):
                        trace('discovery.local_refinement_ready', round=attempt+1, frames=sorted(boxes))
                    plan['local_layer_refinement'] = True
        except (RuntimeError, ValueError) as error:
            plan['reason'] = str(error)
            trace('discovery.unresolved', round=attempt+1, reason=str(error))
            continue
        for mode in ('coarse', 'candidate', 'single'):
            if mode != 'coarse':
                if len(queried) >= 40:
                    break
                candidates = propose_boxes(source, sorted(boxes.items())[:80], cuda) if mode == 'candidate' else {}
                if mode == 'candidate' and not candidates:
                    continue
                targets = [*urgent, *frames]
                if candidates:
                    targets = sorted(candidates, key=lambda i: sum(abs(a-b) for a, b in zip(candidates[i], boxes[i])),
                                     reverse=True)
                refined = refine_discovery(source, label, targets, candidates, queried, local, cuda, plan, trace)
                if not refined:
                    continue
                boxes.update(refined)
            try:
                if len(boxes) < 3:
                    raise ValueError('At least two training observations and a separate validation frame are needed.')
                observations = sorted(boxes.items())
                if len(observations) > 80:
                    observations = [observations[i] for i in np.linspace(0, len(observations)-1, 80, dtype=int)]
                fingerprint = tuple((i, tuple(rect)) for i, rect in observations)
                if fingerprint in tested:
                    continue
                tested.add(fingerprint)
                track, prior = enroll(source, observations, library, identity, cuda, trace,
                                      tolerate_outlier=config.get('batch_routing', False))
                if config.get('batch_routing'):
                    recover_chroma(source, track, cuda, trace)
                    verify_discovery_presence(source, track, label, config, cuda, trace)
            except (RuntimeError, ValueError) as error:
                plan['reason'] = str(error)
                trace('discovery.unresolved', round=attempt+1, mode=mode, reason=str(error))
                urgent = []
                continue
            missing = np.flatnonzero(~track['verified'])
            plan['max_verified'] = max(plan.get('max_verified', 0), source.count-len(missing))
            trace('discovery.coverage', round=attempt+1, verified=source.count-len(missing), total=source.count,
                  memory=prior, mode=mode)
            if len(missing):
                urgent = [int(missing[i]) for i in np.linspace(0, len(missing)-1, min(4, len(missing)), dtype=int)]
                plan['reason'] = f'{len(missing)} frames lack a verified location.'
                continue
            if config.get('local_layers'):
                classification, target, evidence_frames = confirm_discovery_target(source, track, config, cuda, trace)
                if classification != 'confirmed':
                    plan['reason'] = ('The mask omits part of a confirmed watermark.' if classification == 'partial'
                                      else 'The tracked target could not be confirmed as a distribution watermark.')
                    boxes.clear()
                    label, urgent = (target, evidence_frames) if classification == 'partial' else ('', [])
                    break
                label = target
            plan['recognition'] = 'visual'
            return write_discovery_recipe(source, directory, identity, track, prior, label, plan, trace)
    plan.setdefault('reason', 'Visual samples did not establish a watermark track that can be safely repaired.')
    return None


def make_recipe(source, directory, config, cuda, report, trace, *, local_refinement=None):
    if config.get('recognition', 'adaptive') == 'known':
        return make_known_recipe(source, directory, config, cuda, report, trace)
    return make_discovery_recipe(source, directory, config, cuda, report, trace, local_refinement=local_refinement)


def review_inputs(source, attempt, profile, max_samples, previous_frame, priority=()):
    audit = read_object(attempt / 'watermark-audit.json')
    frames = [] if previous_frame is None else [previous_frame]
    frames.extend(i for i in priority if 0 <= i < source.count)
    frames.extend(audit.get('discovery_worst_frames', [])[:2])
    frames.extend(audit.get('discovery_absence_samples', []))
    if scan := profile.get('minute_color_scan'):
        frames.extend(scan['review_frames'])
    if profile.get('translucent', {}).get('verify_paired_motion'):
        for layer in audit.get('white_moving_layers', []):
            frames.extend(item['frame'] for item in layer.get('worst_frames', [])[:1])
    frames.extend(item['frame'] for item in audit.get('url_worst_frames', [])[:max_samples//3])
    if profile.get('translucent'):
        frames.append(profile['translucent']['reference_frame'])
    if corner := profile.get('corners'):
        frames.extend(r['frame'] for r in corner.get('references', []))
        if 'reference_frame' in corner:
            frames.append(corner['reference_frame'])
    frames.extend(frame_samples(source, max_samples))
    return list(dict.fromkeys(frames))[:max_samples], audit


def numeric_findings(attempt, profile, audit, max_url_p95):
    verification = read_object(attempt / 'verification.json')
    findings = []
    if profile.get('discovery', {}).get('coverage_review_only') is True:
        findings.append('Coverage is incomplete: this candidate edits confirmed placements only and requires review.')
    if not all(verification.get(key) is True for key in (
            'same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok')):
        findings.append('Frame, audio or decode verification failed.')
    if profile.get('corners') and audit.get('corner_frames_at_original_detection_threshold') != 0:
        findings.append('Corner mark detection failed or still found a mark.')
    white_moving = bool(profile.get('translucent', {}).get('white_plate'))
    if profile.get('translucent') and not white_moving:
        p95 = audit.get('url_relative_luma_template_response', {}).get('p95')
        if p95 is None or p95 > max_url_p95:
            findings.append('The moving mark lacks reliable measurements or exceeds the residual limit.')
    for kind, count in (('moving', 1+len(profile.get('translucent', {}).get('white_plate', {}).get('variants', []))
                        +len(profile.get('extra_white', []))
                        if white_moving else 0), ('stationary', len(profile.get('stationary', [])))):
        if not count:
            continue
        measurements = audit.get(f'white_{kind}_layers', [])
        if len(measurements) != count or any(
                row.get('before_median', 0) < .15 or row.get('absolute_median_residual_ratio', 1) > .15
                for row in measurements):
            findings.append(f'The white {kind} layers lack reliable measurements or exceed the residual limit.')
    if profile.get('translucent', {}).get('continuous_motion') and (
            audit.get('translucent_tracked_frames') != verification.get('frames')):
        findings.append('The continuous moving overlay was not tracked across the full video.')
    if profile.get('discovery') and not audit.get('discovery_passed'):
        findings.append('The discovered moving mark did not pass the full-source residual and scene protection checks.')
    return findings


def uncovered_overlay_regions(source, track_path, required):
    from wm_discovery import discovery_tracks

    with np.load(track_path, allow_pickle=False) as saved:
        layers = discovery_tracks(dict(saved))
    return uncovered_track_regions(source, layers, required)


def uncovered_track_regions(source, layers, required):
    if not required:
        return []
    supports = []
    for layer in layers:
        rows, columns = np.nonzero(layer['mask'])
        supports.append((layer, (columns.min(), rows.min(), columns.max()+1, rows.max()+1)))
    missing = []
    for region in required:
        frame, x, y, w, h = region
        x, y, right, bottom = expand((x, y, x+w, y+h), source, 0, 0)
        if not 0 <= frame < source.count or x >= right or y >= bottom:
            raise ValueError('A confirmed overlay region is outside the source.')
        covered = np.zeros((bottom-y, right-x), bool)
        for layer, (left, top, local_right, local_bottom) in supports:
            if 'present' in layer and not layer['present'][frame]:
                continue
            u, v = layer['positions'][frame]
            a, b = max(x, u+left), max(y, v+top)
            c, d = min(right, u+local_right), min(bottom, v+local_bottom)
            if a < c and b < d:
                covered[b-y:d-y, a-x:c-x] = True
        if covered.mean() < .78:
            missing.append(tuple(region))
    return missing


def overlay_coverage_findings(source, profile, track_path, required):
    if not required:
        return []
    if not profile.get('discovery'):
        return ['The repair does not provide tracks for the previously confirmed overlays.']
    missing = uncovered_overlay_regions(source, track_path, required)
    if not missing:
        return []
    frames = ', '.join(map(str, sorted({row[0] for row in missing})[:6]))
    return [f'The repair omits {len(missing)} previously confirmed overlay regions on frames {frames}.']


def recipe_coverage_findings(source, recipe, required):
    if not required:
        return []
    profile = read_object((recipe.parent/read_object(recipe)['profile']).resolve(strict=True))
    return overlay_coverage_findings(source, profile, profile.get('discovery', {}).get('track'), required)


def relocated_recipe(path):
    recipe = read_object(path)
    for key in ('profile', 'calibration_pairs'):
        if key in recipe:
            recipe[key] = str((path.parent / recipe[key]).resolve(strict=True))
    for key in ('shared_render', 'final_render', 'reference_render'):
        if key in recipe:
            recipe[key] = [arg.replace('{recipe}', str(path.parent)) for arg in recipe[key]]
    for step in recipe.get('patches', []):
        step['config'] = str((path.parent / step['config']).resolve(strict=True))
        step['args'] = [arg.replace('{recipe}', str(path.parent)) for arg in step.get('args', [])]
    return recipe


def review_track(attempt):
    corrected = attempt/'track-recovered.npz'
    return corrected if corrected.is_file() else attempt/'track.npz'


def corner_overlap_regions(source, profile, attempt):
    original = attempt/'track.npz'
    corrected = review_track(attempt)
    corner = profile.get('corners')
    if corrected == original or not corner:
        return []
    settings = profile['translucent']
    with np.load(original, allow_pickle=False) as before, np.load(corrected, allow_pickle=False) as after:
        recovered = np.flatnonzero((before['positions'][:, 0] < 0) & (after['positions'][:, 0] >= 0)
                                   & (after['scores'] < .34))
        if not recovered.size:
            return []
        x, y = (after['positions'][recovered]-motion_padding(settings)).T
        w, h = settings['reference_rect'][2:]
        active = corner_activity(corner, before['corners'][recovered])
        cuts = before['cuts']
    candidates = []
    for flags, polygon in zip(active.T, corner['masks'], strict=True):
        xs, ys = zip(*polygon)
        if min(ys) != 0:
            continue
        left, right, bottom = min(xs), max(xs)+1, max(ys)+1
        intersects = (x < right)&(x+w > left)&(y < bottom)&(y+h > 0)
        frames = recovered[flags & intersects]
        shots = np.searchsorted(cuts, frames, side='right')
        for shot in np.split(frames, np.flatnonzero(np.diff(shots))+1):
            if not shot.size:
                continue
            frame = int(shot[len(shot)//2])
            bounds = (left//2*2, 0, (right+1)//2*2, (bottom+1)//2*2)
            rect = xywh(expand(bounds, source, 20, 24))
            candidates.append((-len(shot), frame, rect))
    return [[frame, *rect] for _, frame, rect in sorted(candidates)[:2]]


def corner_overlap_point(attempt, profile, track, frame, crop):
    corner = profile.get('corners', {})
    mark = logo_rect(profile, track, frame)
    if not mark or not corner:
        raise ValueError('The crop does not contain a tracked corner overlap.')
    with np.load(attempt/'track.npz', allow_pickle=False) as original:
        if original['positions'][frame, 0] >= 0:
            raise ValueError('A point-free corner repair needs an overlap confirmed by tracking recovery.')
        active = corner_activity(corner, original['corners'][frame])
    x, y, w, h = mark
    a, b, c, d = crop
    points = []
    for flag, polygon in zip(active, corner['masks'], strict=True):
        xs, ys = zip(*polygon)
        if flag and min(ys) == 0:
            left, right = max(x, a, min(xs)), min(x+w, c, max(xs)+1)
            top, bottom = max(y, b, 0), min(y+h, d, max(ys)+1)
            if left < right and top < bottom:
                points.append(((left+right)/2, (top+bottom)/2))
    if len(points) != 1:
        raise ValueError('The crop must overlap exactly one active top corner and the recovered moving mark.')
    return points[0]


def paired_review_regions(source, profile, track, frames, audit):
    layers = [(profile, track), *((p, t) for p, t, _ in extra_tracks(profile, track))]
    responses = [{item['frame']: abs(item['after_response']) for item in layer.get('worst_frames', [])}
                 for layer in audit.get('white_moving_layers', [])]
    regions = []
    for (settings, points), scores in zip(layers, responses):
        used = {row[0] for row in regions}
        for i in sorted(frames, key=lambda n: -scores.get(n, 0)):
            rect = logo_rect(settings, points, i)
            if rect is None or i in used:
                continue
            x, y, w, h = rect
            region = xywh(expand((x, y, x+w, y+h), source, 30, 24))
            if min(region[2:]) >= 64:
                regions.append([i, *region])
                break
    return regions


def colored_risk_regions(source, profile, track, frames, audit):
    regions = []
    used_shots = set()
    for row in audit.get('url_worst_frames', []):
        frame = row['frame']
        shot = np.searchsorted(track['cuts'], frame, side='right')
        rect = logo_rect(profile, track, frame)
        if frame not in frames or shot in used_shots or rect is None:
            continue
        x, y, w, h = rect
        regions.append([frame, *xywh(expand((x, y, x+w, y+h), source, 22, 18))])
        used_shots.add(shot)
        if len(regions) == 2:
            break
    return regions


def tracked_patch_job(source, profile, track, focus, radius, target):
    def overlap(rect):
        if rect is None:
            return 0
        x, y, w, h = rect
        a, b, c, d = target
        return max(0, min(x+w, c)-max(x, a))*max(0, min(y+h, d)-max(y, b))

    layers = [(profile, track), *((p, t) for p, t, _ in extra_tracks(profile, track))]
    matches = [overlap(logo_rect(p, t, focus)) for p, t in layers]
    layer = max(range(len(layers)), key=lambda i: (matches[i], layers[i][1]['scores'][focus]))
    if not matches[layer]:
        raise ValueError('The visible defect does not overlap a tracked moving mark.')
    settings, points = layers[layer]
    edges = np.r_[0, track['cuts'], source.count]
    shot = np.searchsorted(edges, focus, side='right')-1
    radius = min(radius, 44)
    lo, hi = max(int(edges[shot]), focus-radius), min(int(edges[shot+1]), focus+radius+1)
    marks = [rect for i in range(lo, hi) if (rect := logo_rect(settings, points, i)) is not None]
    bounds = (min(x for x, _, _, _ in marks), min(y for _, y, _, _ in marks),
              max(x+w for x, _, w, _ in marks), max(y+h for _, y, _, h in marks))
    bounds = expand(bounds, source, 8, 8)
    region = xywh(bounds)
    check_rect(region, source.width, source.height)
    return {'frames': [lo, hi], 'layer': layer, 'region': region,
            'context': xywh(expand(bounds, source, 36, 40)), 'scale': .76, 'method': 'model',
            'fade_frames': min(5, focus-lo, hi-focus-1)}


def check_patch_window(plan, lo, hi):
    render = next(step['args'] for step in plan if step['name'] == 'render')
    for i, flag in enumerate(render[:-1]):
        if flag in ('--animation-patch', '--caption-patch'):
            with np.load(render[i+1], allow_pickle=False) as patch:
                start, stop = map(int, patch['frames'])
            if max(lo, start) < min(hi, stop):
                raise ValueError('The proposed local repair overlaps an existing patch.')
    return render


def crop_target(regions, frame):
    crops = [row[1:] for row in regions if row[0] == frame]
    if len(crops) == 1:
        x, y, w, h = crops[0]
        return x, y, x+w, y+h
    return None


def strong_colored_plate(settings):
    return settings.get('white_plate', {}).get('opacity_reference', {}).get('gain', 0) >= .4


def protect_caption_band(source, render, job):
    band = int(render[render.index('--caption-band')+1]) if '--caption-band' in render else (source.height*5//12)*2
    band = band//2*2
    if 0 <= band < source.height:
        job['protect'] = [[0, band, source.width, source.height-band]]


def multi_patch_jobs(source, profile, track, plan, frames, regions, requested, radius):
    settings = profile.get('translucent', {})
    colored = strong_colored_plate(settings)
    if not (settings.get('verify_paired_motion') or colored):
        raise ValueError('Multi-shot repair requires verified paired tracks or a strong calibrated colored mark.')
    if (not isinstance(requested, list) or not 2 <= len(requested) <= 30
            or any(type(i) is not int or i not in frames for i in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('A multi-shot repair must name two to thirty distinct displayed frames.')
    jobs = {}
    for frame in sorted(requested):
        target = crop_target(regions, frame)
        if target is None:
            raise ValueError(f'Frame {frame} needs exactly one displayed crop.')
        job = tracked_patch_job(source, profile, track, frame, radius, target)
        lo, hi = job['frames']
        render = check_patch_window(plan, lo, hi)
        if colored:
            protect_caption_band(source, render, job)
        if any(max(lo, other['frames'][0]) < min(hi, other['frames'][1]) for other in jobs.values()):
            raise ValueError('Multi-shot repair windows must not overlap.')
        jobs[f'frame-{frame}'] = job
    return jobs


def repair_recipe(source, directory, number, previous, attempt, plan, frames, action, radius, regions=(),
                  *, comparisons=(), cuda=True, trace=None, local_backend=None):
    recipe = relocated_recipe(previous)
    profile = read_object(Path(recipe['profile']))
    if not isinstance(action, dict):
        raise ValueError('Grok did not supply a repair action.')
    next_path = directory / f'recipe-{number:02d}.json'
    if action.get('kind') in ('discovery-detail', 'discovery-edge', 'discovery-luma'):
        focus = action.get('frame')
        settings = profile.get('discovery')
        if not settings or type(focus) is not int or focus not in frames or not 0 <= focus < source.count:
            raise ValueError('A discovery retry needs a displayed watermark frame.')
        edge = action['kind'] == 'discovery-edge'
        with np.load(attempt/'track.npz', allow_pickle=False) as saved:
            track = dict(saved)
        extended = None
        if 'matte_positions' in track:
            raise ValueError('A model-detail retry does not apply to the calibrated transparent layer.')
        if action.get('defect') == 'residue':
            displayed_layers = {index for frame, index in comparisons if frame == focus}
            selected = action.get('layer')
            if selected is None and len(displayed_layers) == 1:
                selected = next(iter(displayed_layers))
            if type(selected) is int and selected in displayed_layers:
                from wm_discovery import LUMA_EXTENT_GROUPS, discovery_tracks, luma_extent_repairs
                from wm_discovery_render import (broad_chroma_extent_retry, chroma_extent_retry, luma_extent_retry,
                                                 registered_luma_extent_retry)

                if 0 <= selected < len(discovery_tracks(track)):
                    luma_repairs = luma_extent_repairs(settings)
                    methods = ('luma',) if action['kind'] == 'discovery-luma' else ('chroma', 'luma')
                    for method in methods:
                        setting = f'{method}_extent_repair'
                        if method == 'chroma' and (setting in settings or luma_repairs):
                            continue
                        if method == 'luma' and len(luma_repairs) == len(LUMA_EXTENT_GROUPS):
                            continue
                        try:
                            candidate = Video.open(attempt/'result.mp4')
                            if method == 'chroma':
                                extended = broad_chroma_extent_retry(source, candidate, track, settings, selected,
                                                                     focus, cuda, trace)
                                if extended is None:
                                    extended = chroma_extent_retry(source, candidate, track, settings, selected, cuda, trace)
                            else:
                                extended = luma_extent_retry(source, candidate, track, settings, selected, focus, cuda, trace)
                                if extended is None and local_backend is not None:
                                    extended = registered_luma_extent_retry(source, candidate, track, settings, selected,
                                                                            focus, cuda, trace, local_backend)
                        except (OSError, RuntimeError, ValueError, cv2.error) as error:
                            if trace:
                                trace(f'repair.{method}_extent_rejected', frame=focus, layer=selected, reason=str(error))
                        if extended is None:
                            continue
                        name = 'discovery-extent' if method == 'chroma' else 'discovery-luma-extent'
                        asset = directory/f'{name}-{number:02d}.npz'
                        save_npz(asset, replace=True, **extended)
                        settings.update(track=str(asset), sha256=digest(asset))
                        addition = {'layer': selected, 'frame': focus}
                        settings[setting] = addition if method == 'chroma' else [*luma_repairs, addition]
                        if method == 'chroma':
                            settings['consensus_mask'] = False
                        if trace:
                            trace(f'repair.{method}_extent_scheduled', frame=focus, layer=selected, track=str(asset))
                        break
        if extended is None and action['kind'] == 'discovery-luma':
            raise ValueError('Independent source frames did not confirm the reported luminance fragment.')
        if extended is None and edge:
            from wm_discovery import discovery_tracks
            from wm_discovery_render import edge_residue

            layer = action.get('layer')
            layers = discovery_tracks(track)
            if (not can_retry_edge(recipe, settings) or action.get('defect') != 'residue'
                    or type(layer) is not int or not 0 <= layer < len(layers) or (focus, layer) not in comparisons):
                raise ValueError('An edge retry needs a reported residue in a displayed layer and an unused mask option.')
            evidence = edge_residue(source, Video.open(attempt/'result.mp4'), layers[layer], focus, cuda)
            if trace:
                trace('repair.edge_evidence', layer=layer, frame=focus, **evidence)
            if not evidence['eligible']:
                raise ValueError('Independent frames did not support an edge correction for the reported residue.')
            settings['edge_repair'] = {'frame': focus, 'layer': layer}
        if extended is None and not edge:
            if not can_retry_discovery(recipe):
                raise ValueError('A detail retry needs unused model options.')
            if 'luma_extent_repair' not in settings:
                settings['consensus_mask'] = action.get('defect') == 'damage' and 'chroma_extent_repair' not in settings
            recipe.setdefault('final_render', []).extend(['--model-scale', '1', '--raft-iterations', '12'])
        settings['separate_jumps'] = True
        detail_profile = directory/f'discovery-profile-{number:02d}.json'
        write_report(detail_profile, profile)
        recipe['profile'] = str(detail_profile)
        write_report(next_path, recipe)
        return next_path, (focus,)
    if action.get('kind') == 'local-patches':
        requested = action.get('frames')
        with np.load(review_track(attempt), allow_pickle=False) as track:
            jobs = multi_patch_jobs(source, profile, dict(track), plan, frames, regions, requested, radius)
        options = recipe.get('final_render', [])
        if '--edges' in options and options[options.index('--edges')+1] == 'off':
            options[options.index('--edges')+1] = 'white-flat'
        patch_config = directory/f'local-{number:02d}.json'
        write_report(patch_config, {'jobs': jobs})
        output = f'grok-{number:02d}-{{job}}'
        recipe.setdefault('patches', []).append({'command': 'local-patch', 'name': 'Grok local {job}',
                                                  'config': str(patch_config), 'jobs': 'all',
                                                  'output': f'patches/{output}.npz',
                                                  'args': ['--report', f'{{work}}/patches/{output}.json'],
                                                  'attach': 'animation'})
        write_report(next_path, recipe)
        return next_path, tuple(sorted(requested))
    if action.get('kind') == 'local-patch':
        focus = action.get('frame')
        if type(focus) is not int or focus not in frames or not profile.get('translucent'):
            raise ValueError('A local repair needs a displayed frame with a tracked moving mark.')
        point = action.get('point')
        settings = profile['translucent']
        crop = crop_target(regions, focus)
        options = recipe.get('final_render', [])
        paired = settings.get('verify_paired_motion', False)
        colored = strong_colored_plate(settings)
        if point is None and crop is not None and (paired or colored or '--recover-corner-overlap' in options):
            target = crop
        elif isinstance(point, list) and len(point) == 2 and all(type(v) is int and 0 <= v <= 1000 for v in point):
            px, py = point[0]*source.width/1000, point[1]*source.height/1000
            target = (px, py, px+1, py+1)
        else:
            raise ValueError('A local repair needs the visible defect point in normalized coordinates.')
        with np.load(review_track(attempt), allow_pickle=False) as track:
            if paired or (colored and point is None):
                job = tracked_patch_job(source, profile, dict(track), focus, radius, target)
                if '--edges' in options and options[options.index('--edges')+1] == 'off':
                    options[options.index('--edges')+1] = 'white-flat'
                else:
                    render = check_patch_window(plan, *job['frames'])
                    if colored:
                        protect_caption_band(source, render, job)
                    patch_config = directory/f'local-{number:02d}.json'
                    write_report(patch_config, job)
                    recipe.setdefault('patches', []).append({'command': 'local-patch', 'config': str(patch_config),
                                                              'output': f'patches/grok-{number:02d}.npz',
                                                              'args': ['--report', f'{{work}}/patches/grok-{number:02d}.json'],
                                                              'attach': 'animation'})
                write_report(next_path, recipe)
                return next_path, (focus,)
            x, y = map(int, track['positions'][focus])
            px, py = corner_overlap_point(attempt, profile, track, focus, target) if point is None else target[:2]
            pad_x, pad_y = motion_padding(settings)
            w, h = settings['reference_rect'][2:]
            if x < 0 or not (x-pad_x <= px < x-pad_x+w and y-pad_y <= py < y-pad_y+h):
                raise ValueError('The visible defect point does not overlap the tracked moving mark in this frame.')
            edges = np.r_[0, track['cuts'], source.count]
            shot = np.searchsorted(edges, focus, side='right') - 1
            lo, hi = max(int(edges[shot]), focus-radius), min(int(edges[shot+1]), focus+radius+1)
            positions = track['positions'][lo:hi]
            active = positions[:, 0] >= 0
            x, y = (positions[active]-[pad_x, pad_y]).T
            if np.any((x < 0) | (y < 0) | (x+w > source.width) | (y+h > source.height)):
                raise ValueError('A local repair needs its tracked area inside the frame.')
        with np.load(attempt/'track.npz', allow_pickle=False) as original_track:
            under_corner = original_track['positions'][focus, 0] < 0
            if under_corner:
                corner = profile.get('corners', {})
                active = corner_activity(corner, original_track['corners'][focus])
                shot_edges = np.r_[0, original_track['cuts'], source.count]
                shot = np.searchsorted(shot_edges, focus, side='right')-1
                lo, hi = map(int, shot_edges[shot:shot+2])
        if under_corner:
            options = recipe.get('final_render', [])
            if '--recover-corner-overlap' not in options or hi-lo > 240:
                raise ValueError('The proposed corner overlap cannot be resolved in one bounded shot.')
            found = []
            for i, (flag, polygon) in enumerate(zip(active, corner['masks'], strict=True)):
                xs, ys = zip(*polygon)
                if flag and min(ys) == 0 and min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys):
                    found.append(i)
            if len(found) != 1:
                raise ValueError('The overlap repair point must be inside one active top corner.')
            x, y, w, h = corner['regions'][found[0]]
            if y != 0:
                raise ValueError('Texture repair requires a region on the top edge.')
            for step in recipe.get('patches', []):
                if step['command'] == 'region-patch':
                    previous_patch = read_object(Path(step['config']))
                    if previous_patch.get('cover_corner') and previous_patch.get('method') == 'texture-edge':
                        start, stop = previous_patch['frames']
                        if max(lo, start) < min(hi, stop):
                            raise ValueError('This shot already has a corner overlap repair.')
            region = [x, y, w, min(source.height, h+54)]
            patch_config = directory/f'corner-overlap-{number:02d}.json'
            write_report(patch_config, {'frames': [lo, hi], 'region': region, 'bounds': [[lo, x, y, x, y]],
                                        'cover_corner': True, 'recover_corner_overlap': True,
                                        'method': 'texture-edge', 'feather': 10})
            recipe.setdefault('patches', []).append({'command': 'region-patch', 'name': f'corner overlap texture {lo}',
                                                      'config': str(patch_config),
                                                      'output': f'patches/grok-overlap-{number:02d}.npz', 'attach': 'animation'})
            write_report(next_path, recipe)
            return next_path, (focus,)
        render = check_patch_window(plan, lo, hi)
        patch_config = directory / f'local-{number:02d}.json'
        settings = {'frames': [lo, hi]}
        if colored:
            with np.load(review_track(attempt), allow_pickle=False) as track:
                settings = tracked_patch_job(source, profile, dict(track), focus, radius, (px, py, px+1, py+1))
        protect_caption_band(source, render, settings)
        write_report(patch_config, settings)
        recipe.setdefault('patches', []).append({'command': 'local-patch', 'config': str(patch_config),
                                                  'output': f'patches/grok-{number:02d}.npz',
                                                  'args': ['--report', f'{{work}}/patches/grok-{number:02d}.json'],
                                                  'attach': 'animation'})
    elif action.get('kind') in ('corner-method', 'corner-mask'):
        index, method, focus = action.get('index'), action.get('method'), action.get('frame')
        corner = profile.get('corners', {})
        count = len(corner.get('regions', []))
        if type(index) is not int or not 0 <= index < count:
            raise ValueError('The proposed corner index is outside the profile.')
        if type(focus) is not int or focus not in frames:
            raise ValueError('A corner repair needs a displayed frame number.')
        with np.load(attempt/'track.npz', allow_pickle=False) as track:
            if not corner_activity(corner, track['corners'][focus])[index]:
                raise ValueError('The proposed corner was inactive in the reported frame.')
        methods = corner.get('methods', ['model']*count)
        if action['kind'] == 'corner-method':
            if (method != 'extend-top' or methods[index] == method or corner['regions'][index][1] != 0
                    or min(y for _, y in corner['masks'][index]) != 0):
                raise ValueError('The corner cannot use the requested top-edge method.')
            methods[index] = method
            corner['methods'] = methods
        else:
            xs, ys = zip(*corner['masks'][index])
            left, top = min(xs), min(ys)
            right, bottom = max(xs)+1, max(ys)+1
            rectangle = {(left, top), (right-1, top), (right-1, bottom-1), (left, bottom-1)}
            if set(zip(xs, ys)) != rectangle or any(v % 2 for v in (left, top, right, bottom)):
                raise ValueError('Automatic expansion requires a rectangular, aligned corner mask.')
            w, h = right-left, bottom-top
            bounds = expand((left, top, right, bottom), source, max(8, w//8*2), max(6, h//4*2))
            x, y, r, b = bounds
            if ((r-x)*(b-y) > source.width*source.height*.20 or b-y > source.height*.40
                    or bounds == (left, top, right, bottom)):
                raise ValueError('Expanding the corner would cover too much scene content.')
            corner['masks'][index] = [[x, y], [r-1, y], [r-1, b-1], [x, b-1]]
            corner['regions'][index] = xywh(expand(bounds, source, 82, 82))
        profile_path = directory / f'profile-{number:02d}.json'
        write_report(profile_path, profile)
        recipe['profile'] = str(profile_path)
    else:
        raise ValueError('The proposed repair is outside the available actions.')
    write_report(next_path, recipe)
    return next_path, (focus,)


def publish_result(directory, attempt, report, trace):
    if report.get('planning', {}).get('coverage_review_only') is True:
        raise ValueError('An incomplete-coverage review candidate cannot be published as complete.')
    candidate = attempt/'result.mp4'
    result = directory/'result.mp4'
    temporary = directory/'result.part'
    if result.exists():
        if not os.path.samefile(candidate, result) and len(set(fingerprints([candidate, result]).values())) != 1:
            raise FileExistsError(result)
    else:
        try:
            os.link(candidate, result)
        except OSError:
            shutil.copyfile(candidate, temporary)
            temporary.replace(result)
    report.update(status='complete', result=str(result))
    report.pop('error', None)
    trace('result.published', path=str(result), bytes=result.stat().st_size)


def learn_from_release(attempt, recipe, config, trace):
    profile = read_object((recipe.parent/read_object(recipe)['profile']).resolve(strict=True))
    if not profile.get('discovery'):
        return
    from wm_discovery import remember
    try:
        for path, added in remember(learned_library(config), attempt):
            trace('discovery.memory.accepted', prototype=str(path), new=added)
    except (OSError, ValueError) as error:
        trace('discovery.memory.error', reason=str(error))


def job_trace(events_path):
    started = monotonic()
    lock = Lock()

    def trace(event, **details):
        row = {'utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
               'elapsed_seconds': round(monotonic()-started, 3), 'event': event, **details}
        with lock, events_path.open('a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(row, ensure_ascii=False) + '\n')
    return trace


def can_retry_discovery(recipe):
    options = recipe.get('final_render', [])
    return all(flag not in options for flag in ('--model-scale', '--raft-iterations'))


def can_retry_edge(recipe, settings):
    options = recipe.get('final_render', [])
    return ('edge_repair' not in settings and not settings.get('consensus_mask')
            and all(flag in options for flag in ('--model-scale', '--raft-iterations')))


def local_untracked_watermark(source, candidate, frames, config, cuda, trace, active_crops):
    from wm_layers import build_layer, confirm_layer, overlap, proposals_for, visible_box
    from wm_track import highpass

    backend = config['discovery_local']
    frames = frame_samples(candidate, min(7, candidate.count), frames)
    decision = native_discovery(candidate, frames, cuda, backend, '', trace)

    def unchanged_untracked(frame, bounds):
        rect = xywh(bounds)
        if (min(rect[2:]) < 8 or rect[2]*rect[3] > .18*source.width*source.height
                or any(overlap(rect, active[1:]) for active in active_crops(frame))):
            return False
        return all(item['p99'] <= 4 and item['maximum'] <= 12
                   for item in reported_region_delta(source, candidate, frame, bounds, cuda))

    rows = {}
    for observation in decision.get('observations', []):
        frame = observation['frame']
        bounds = box(observation['rect'], source)
        if unchanged_untracked(frame, bounds):
            rows.setdefault(frame, []).append({'rect': bounds})
    if len(rows) < 3:
        return None
    observations = proposals_for(source, rows)
    signals = {i: highpass(source.planes(get_frame(source, i, cuda))[0], 2.2) for i in rows}
    layer = build_layer(signals, observations)
    if layer is None:
        return None
    fallback = partial(classify_layer, source, config=config, cuda=cuda, trace=trace)
    classification = confirm_layer(source, layer, signals, backend, cuda, trace, fallback)
    if classification not in ('confirmed', 'partial'):
        return None
    anchor = max(layer.matches, key=lambda item: layer.matches[item][2])
    x, y, w, h = visible_box(layer.motif, *layer.matches[anchor][:2])
    bounds = expand((x, y, x+w, y+h), source, 0, 0)
    if not unchanged_untracked(anchor.frame, bounds):
        return None
    hint = {'frame': anchor.frame, 'rect': xywh(bounds), 'label': layer.label,
            'observations': [list(region) for region in layer.reviewed_regions]}
    trace('evidence.local_remaining_watermark', hint=hint, classification=classification)
    return hint


def review_discovery(source, candidate, attempt, recipe, profile, config, cuda, trace, record, persist,
                     frames, audit, focus, repairs_left):
    from wm_discovery import LUMA_EXTENT_GROUPS, discovery_tracks, luma_extent_repairs

    previous = [row[0] for row in record.get('comparison_regions', [])]
    with np.load(attempt/'track.npz', allow_pickle=False) as track:
        model_repair = 'matte_positions' not in track
        layers = discovery_tracks(dict(track))
    supports = [layer.get('present', np.ones(source.count, bool)) for layer in layers]

    def crop_for(layer, i):
        h, w = layer['mask'].shape
        x, y = map(int, layer['positions'][i])
        bounds = (x//2*2, y//2*2, (x+w+1)//2*2, (y+h+1)//2*2)
        return [i, *xywh(expand(bounds, source, 18, 16))]

    def active_crops(i):
        return [crop_for(layer, i) for layer, present in zip(layers, supports, strict=True) if present[i]]

    summaries = audit.get('discovery_layers', [audit])
    regions, comparison_layers = [], []
    edge_repair = profile['discovery'].get('edge_repair', {})
    extent_repair = profile['discovery'].get('chroma_extent_repair', {})
    luma_repairs = luma_extent_repairs(profile['discovery'])
    luma_checks = audit.get('discovery_luma_extents') or [audit.get('discovery_luma_extent', {})]
    feedback = profile['discovery'].get('feedback', {})
    required = record.get('required_comparisons', [])
    for index, (layer, present, summary) in enumerate(zip(layers, supports, summaries, strict=True)):
        worst = summary['discovery_worst_frames']
        priority = [worst[0], *focus, *previous, *worst] if len(layers) > 1 else [*focus, *previous, *worst]
        limit = 1 if len(layers) > 1 else 2
        if index == edge_repair.get('layer'):
            priority = [edge_repair['frame'], *priority]
            limit = 2
        if index == extent_repair.get('layer'):
            samples = audit.get('discovery_chroma_extent', {}).get('samples', [])
            strongest = [max(samples, key=lambda row: row['ratio'])['frame']] if samples else []
            priority = [extent_repair['frame'], *strongest, *priority]
            limit = 2
        repair_frames = [frame for repair, checked in zip(luma_repairs, luma_checks) if repair['layer'] == index
                         for frame in (repair['frame'], checked.get('worst_frame', repair['frame']))]
        if repair_frames:
            priority = [*repair_frames, *priority]
            limit = max(limit, len(set(repair_frames)) + (index == extent_repair.get('layer')))
        if index >= feedback.get('first_layer', len(layers)):
            priority = [feedback['frame'], *priority]
        pending = [frame for frame, required_layer in required if required_layer == index]
        if any(not present[i] for i in pending):
            raise ValueError('A required repair comparison is missing its tracked source layer.')
        priority = [*pending, *priority]
        limit = max(limit, len(set(pending+repair_frames)) + bool(repair_frames and index == extent_repair.get('layer')))
        visible = list(dict.fromkeys(i for i in priority if present[i]))
        for i in visible[:limit]:
            regions.append(crop_for(layer, i))
            comparison_layers.append(index)
    if not set(map(tuple, required)).issubset((region[0], index)
                                              for region, index in zip(regions, comparison_layers, strict=True)):
        raise ValueError('A required comparison is outside the candidate review scope.')
    record['comparison_regions'] = regions
    record['comparison_layers'] = comparison_layers
    persist()
    config = {**config, 'review_images_per_call': 1, 'timeout_seconds': min(config['timeout_seconds'], 115)}
    plan = read_object(recipe) if repairs_left and model_repair else None
    detail = plan is not None and can_retry_discovery(plan)
    edge = (plan is not None and config.get('batch_routing') and not record['review_only_regions']
            and can_retry_edge(plan, profile['discovery']))
    luma = (plan is not None and config.get('batch_routing') and not record['review_only_regions']
            and len(luma_repairs) < len(LUMA_EXTENT_GROUPS))
    late_repair = 'discovery-edge' if edge else 'discovery-luma' if luma else None
    retry = ''
    if detail:
        retry = (' Use repair for a confirmed target with visible residue or a repair artifact. Return '
                 '{"decision":"repair","reason":"explain the visible defect",'
                 '"repair":{"kind":"discovery-detail","frame":123,"defect":"residue"}} with the actual displayed '
                 'frame number. Set defect to "damage" for damage to scene content and to "residue" for remaining '
                 'watermark pixels or a mix of both.')
    elif late_repair:
        retry = (' For clear watermark residue use repair with the actual displayed frame and the action '
                 f'{{"kind":"{late_repair}","frame":123,"defect":"residue"}}. '
                 'For scene damage or uncertainty use review.')
    choices = 'accept, repair or review' if retry else 'accept or review'
    prompt = ('Compare the labeled Source and Restored crops. '
              'The change map locates edits. Judge visible damage from the two original crops. '
              'Identify any subtitle or intentional credit in Source and check that it remains readable. '
              'Check that the visible watermark is gone and that scene details and subtitles stay intact. '
              'An unreadable small code or suffix can stay unreadable in the explanation. '
              'Verify the visible graphic and check that all its visible strokes, including small text, are gone. '
              f'Return only JSON with "decision" ({choices}) and "reason" in Chinese. '
              'Use review for uncertainty about the target or context. No tools.') + retry
    if label := profile['discovery']['label']:
        prompt += (f' The earlier discovery described the target as {json.dumps(label, ensure_ascii=False)}. '
                   'Use this approximate description to find the graphic. Verify from Source whether it is a distribution overlay.')
    trace('evidence.comparison', phase='review', regions=regions)
    review_trace = partial(trace, phase='review')
    pairs = comparison_evidence(source, candidate, regions, min(384, config['sample_width']), cuda, show_changes=True)
    review_issue = None
    if config.get('batch_routing') and not record['review_only_regions']:
        prompt += (' For review, also report "issue" as "context" only if the crop alone cannot show whether '
                   'the original scene is flat or textured. For visible residue, damage, or uncertain target '
                   'identity use "issue" as "defect" or "identity". For a defect, include "defect" as "residue" '
                   'or "damage" using the same distinction. Use identity when the visible graphic '
                   'could be scene content, a subtitle, an intentional credit, or a different overlay.')

        def review_issue(initial, offset):
            frame, x, y, w, h = regions[offset]
            if late_repair and not detail:
                action = initial.get('repair', initial.get('action'))
                explicit = (initial['decision'] == 'repair' and isinstance(action, dict)
                            and action.get('kind') == late_repair and action.get('frame') == frame
                            and action.get('defect') == 'residue')
                residue = (initial['decision'] == 'review' and initial.get('issue') == 'defect'
                           and initial.get('defect') == 'residue')
                if explicit or residue:
                    return {**initial, 'decision': 'repair',
                            'repair': {'kind': late_repair, 'frame': frame,
                                       'defect': 'residue', 'layer': comparison_layers[offset]}}
                if initial['decision'] == 'repair':
                    return {'decision': 'review', 'reason': initial['reason']}
            if initial['decision'] != 'review':
                return initial
            if initial.get('issue') == 'defect' and detail:
                trace('evidence.close_defect_repair', phase='review', frame=frame, initial=initial)
                return {**initial, 'decision': 'repair',
                        'repair': {'kind': 'discovery-detail', 'frame': frame, 'defect': initial.get('defect')}}
            if initial.get('issue') != 'context':
                return initial
            surrounding = [frame, *xywh(expand((x, y, x+w, y+h), source, max(24, w//2), max(24, h)))]
            trace('evidence.close_context_recheck', phase='review', frame=frame, region=surrounding, initial=initial)
            images = comparison_evidence(source, candidate, [surrounding], 640, cuda, lift_shadows=True)
            images += comparison_evidence(source, candidate, [[frame, 0, 0, source.width, source.height]], 640, cuda)
            context_prompt = ('Both images pair matching Source and Restored views of the same frame: wider crops '
                              'first, then the full frame. Judge the local repair where the Source crop shows '
                              'the overlay. Check visible scene edges, texture and tone at and around the repair '
                              'boundary, and look for watermark residue or smearing. An opaque overlay conceals '
                              'the original pixels beneath it. Use the visible surrounding scene in both views '
                              'to judge continuity, and explain any visible defect or missing context. A labeled '
                              'shadow lift changes both crops equally for inspection. Accept only if the watermark '
                              'is gone and the visible details connect naturally. Use review for visible damage, '
                              'residue, or insufficient surrounding evidence. Return JSON with decision accept or '
                              'review and a brief reason in Chinese. No tools.')
            return review_visual({**config, 'review_images_per_call': 2}, context_prompt, images, review_trace,
                                 keep_prompt_on_retry=True)

    approved = record.setdefault('approved_comparisons', {}) if config.get('batch_routing') else None
    issues = record['comparison_issues'] = []
    comparison = review_comparisons(config, prompt, pairs, review_trace, review_issue,
                                    approved=approved, persist=persist, issues=issues)
    record['source_comparison'] = comparison
    persist()
    if comparison['decision'] != 'accept':
        return comparison

    if (model_repair and config.get('batch_routing') and config.get('local_layers')
            and not record['review_only_regions']):
        from wm_discovery_render import edge_residue

        consensus = profile['discovery'].get('consensus_mask', False)
        for index, (layer, summary) in enumerate(zip(layers, summaries, strict=True)):
            frame = summary['discovery_worst_frames'][0]
            proof = edge_residue(source, candidate, layer, frame, cuda, consensus=consensus)
            matches = [item for item in proof.get('samples', [])[1:] if item['matched']]
            if not proof['eligible'] and len(matches) >= 3:
                frame = max(matches, key=lambda item: item['edge'])['frame']
                proof = edge_residue(source, candidate, layer, frame, cuda, consensus=consensus)
            trace('evidence.edge_release_check', phase='review', layer=index, frame=frame, **proof)
            if not proof['eligible']:
                continue
            result = {'decision': 'review', 'reason': '本机在多个独立帧中检出相同的水印边缘残影。'}
            if detail or edge:
                regions.append(crop_for(layer, frame))
                comparison_layers.append(index)
                result.update(decision='repair', repair={'kind': 'discovery-detail' if detail else 'discovery-edge',
                                                         'frame': frame, 'defect': 'residue', 'layer': index})
            return result

    if (config.get('batch_routing') and config.get('local_layers') and not record['review_only_regions']):
        try:
            hint = local_untracked_watermark(source, candidate, frames, config, cuda, review_trace, active_crops)
        except (OSError, RuntimeError, ValueError) as error:
            review_trace('evidence.local_sweep_unavailable', reason=str(error))
        else:
            if hint is not None:
                return {'decision': 'review', 'reason': f'已确认成片仍有发行水印 {hint["label"]}。',
                        'watermark_hint': hint}

    def remember_context_hints(answers):
        if not config.get('batch_routing') or record['review_only_regions']:
            return
        saved = list(record.get('context_watermark_hints', ()))
        updated = saved.copy()
        for answer in answers:
            for hint in review_watermark_hints(answer, source, frames):
                if hint not in updated and len(updated) < 3:
                    updated.append(hint)
        if updated != saved:
            record['context_watermark_hints'] = updated
            persist()
            review_trace('evidence.additional_context_candidates', hints=updated)

    def check_context(initial, _offset):
        if initial.get('concerns') or initial.get('findings'):
            remember_context_hints([initial])
            return {'decision': 'review', 'reason': initial['reason']}
        i = initial.get('frame')
        area = initial.get('area')
        if (area not in ('target', 'elsewhere') or type(i) is not int or i not in frames or record['review_only_regions']
                or any(row[0] == i for row in record['review_regions'])):
            return {'decision': 'review', 'reason': initial['reason']}
        try:
            x, y, right, bottom = expand(box(initial.get('rect'), source), source, 0, 0)
        except ValueError:
            return {'decision': 'review', 'reason': initial['reason']}
        active = active_crops(i)
        inside = any(u <= x and v <= y and right <= u+w and bottom <= v+h for _, u, v, w, h in active)
        if area == 'elsewhere' or not inside:
            def intersects(bounds):
                a, b, c, d = bounds
                return any(max(a, u) < min(c, u+w) and max(b, v) < min(d, v+h) for _, u, v, w, h in active)

            if (not config.get('batch_routing') or intersects((x, y, right, bottom))
                    or not active and initial.get('finding') != 'watermark'):
                return {'decision': 'review', 'reason': initial['reason']}
            bounds = x, y, right, bottom
            delta = reported_region_delta(source, candidate, i, bounds, cuda)
            review_trace('evidence.target_geometry_disagrees', frame=i, reported=list(bounds), active=active,
                         delta=delta, initial=initial)
            if any(row['p99'] > 4 or row['maximum'] > 12 for row in delta):
                return {'decision': 'review', 'reason': initial['reason']}
            rect = xywh(bounds)
            label = initial.get('label')
            if (initial.get('finding') == 'watermark' and isinstance(label, str) and label.strip()
                    and min(rect[2:]) >= 8 and rect[2]*rect[3] <= .18*source.width*source.height):
                record['provisional_watermark_hint'] = {'frame': i, 'rect': rect, 'label': label.strip()[:100]}
                persist()
                review_trace('evidence.additional_overlay_candidate', **record['provisional_watermark_hint'])
            full = comparison_evidence(source, candidate, [[i, 0, 0, source.width, source.height]], 960, cuda)
            full_prompt = ('Compare the entire matching Source and Restored frame. An output-only review identified a '
                           'possible new defect. Check whether the structure is already in Source, whether any part of '
                           'the scene or subtitles changed visibly, and whether any distribution watermark is still '
                           'visible in Restored. Use review for a defect or uncertainty. Return only JSON with '
                           'decision accept or review and a brief reason. For review, give finding as watermark for a '
                           'remaining distribution mark, damage for new scene damage, or uncertain. For watermark, '
                           'give its visible label and rect [left,top,right,bottom] normalized to 0–1000 within the '
                           'individual Restored image. Preserve editorial text. No tools.')
            full_prompt += f' The previous observation was {json.dumps(initial["reason"], ensure_ascii=False)}.'
            verified = review_visual(config, full_prompt, full, review_trace, keep_prompt_on_retry=True)
            if verified['decision'] == 'accept' or verified.get('finding') == 'damage':
                record.pop('provisional_watermark_hint', None)
                persist()
            if verified['decision'] != 'accept':
                report = verified if verified.get('finding') == 'watermark' else initial
                if report.get('finding') == 'watermark' and verified.get('finding') != 'damage':
                    try:
                        proposed = box(report.get('rect'), source)
                    except ValueError:
                        proposed = bounds
                    rect = xywh(proposed)
                    if not intersects(proposed) and min(rect[2:]) >= 8 and rect[2]*rect[3] <= .18*source.width*source.height:
                        label = report.get('label')
                        hint = {'frame': i, 'rect': rect, 'label': label.strip()[:100] if isinstance(label, str) else ''}
                        review_trace('evidence.missed_watermark', **hint)
                        return {**verified, 'watermark_hint': hint}
                return verified
            if active:
                crops = comparison_evidence(source, candidate, active, 640, cuda, show_changes=True)
                target = review_comparisons(config, prompt, crops, review_trace, None)
                if target['decision'] != 'accept':
                    return target
            remaining = [frame for frame in frames if frame != i]
            if remaining:
                images = evidence(candidate, remaining, width, cuda, full_sheet=4, full_columns=2)
                return review_visual(config, context_prompt, images, review_trace, keep_prompt_on_retry=True)
            return verified
        trace('evidence.focused_recheck', phase='review', frame=i, regions=active, initial=initial)
        direct = ('The images show only restored output. Check the actual pixels for legible watermark remnants, '
                  'logo outlines, or repair seams. Distinguish intentional subtitles and scene content. '
                  f'Return JSON with decision {"accept, repair or review" if detail else "accept or review"} '
                  'and a short Chinese reason. Use review if the pixels are insufficient. No tools.')
        if detail:
            direct += retry
        zoom = evidence(candidate, [i], 560, cuda, active, only_regions=True)
        focused = review_visual(config, direct, zoom, review_trace, keep_prompt_on_retry=True)
        if focused['decision'] != 'accept':
            return focused
        if backend := config.get('discovery_local'):
            local_prompt = ('This image shows only restored output. Check the actual pixels for a remaining '
                            'distribution watermark, a partial graphic, or a repair seam. Keep intentional subtitles '
                            'and scene text. Return JSON with decision "accept" if clear, "review" for any defect or '
                            'uncertainty, and a brief reason.')
            for image in zoom:
                try:
                    local = wm_grounding.ask_local(backend, local_prompt, image, 22, ('accept', 'review'), json_mode=True)
                except (RuntimeError, ValueError) as error:
                    local = {'decision': 'review', 'reason': str(error)}
                review_trace('evidence.local_recheck', frame=i, answer=local)
                if local['decision'] != 'accept':
                    return local
        unpaired = [region for region in active if region not in regions]
        if unpaired:
            extra = comparison_evidence(source, candidate, unpaired, 520, cuda, show_changes=True)
            return review_visual(config, prompt, extra, review_trace, keep_prompt_on_retry=True)
        return focused

    wide = record['review_regions']
    width = min(440, config['sample_width'])
    trace('evidence.started', phase='review', frames=frames, regions=wide, output_only=True, max_side=width)
    context_prompt = ('All frames and crops show only restored output. Check for visible watermark remnants and '
                      'damage to scene details or subtitles. Return JSON with decision "accept" if clean or "review" '
                      'for a visible concern or uncertainty, and a short Chinese reason. For a specific concern, '
                      'include the displayed integer "frame" and "area" ("target" for the repaired mark or '
                      '"elsewhere" for anything else). For a concern in a full frame, include the tight rectangle '
                      '"rect" [left,top,right,bottom] with integers 0 to 1000 relative to that individual frame. '
                      'For review, give "finding" as watermark, damage or uncertain. For watermark give its visible '
                      '"label". Preserve editorial text and scene content. No tools.')
    pending = record.setdefault('pending_context', {}) if config.get('batch_routing') else None

    def split_context():
        trace('evidence.context_split', phase='review', frames=frames, frames_per_sheet=2)
        images = evidence(candidate, frames, width, cuda, wide, full_sheet=2, full_columns=2)
        reported = []
        verdict = review_comparisons(config, context_prompt, images, review_trace, check_context,
                                     approved=record.setdefault('approved_context', {}), persist=persist,
                                     reported=reported, pending=pending)
        remember_context_hints(reported)
        return verdict

    can_split = config.get('batch_routing') and not record['review_only_regions'] and len(frames) > 2
    if can_split and record.get('split_context_review'):
        verdict = split_context()
    else:
        images = evidence(candidate, frames, width, cuda, wide, record['review_only_regions'], full_sheet=4,
                          full_columns=2)
        try:
            verdict = review_visual(config, context_prompt, images, review_trace, keep_prompt_on_retry=True,
                                    review_issue=check_context, pending=pending, persist=persist)
        except wm_grok.PromptTimeout:
            if not can_split:
                raise
            record['split_context_review'] = True
            persist()
            verdict = split_context()
    if pending:
        pending.clear()
        persist()
    if verdict['decision'] == 'accept':
        record.pop('provisional_watermark_hint', None)
        record.pop('context_watermark_hints', None)
    return verdict


def review_candidate(source, attempt, recipe, config, cuda, trace, record, persist, *,
                     repairs_left=0, focus_frames=(), focus_crops=()):
    record['review_method'] = review_method(config)
    profile = read_object((recipe.parent / read_object(recipe)['profile']).resolve(strict=True))
    discovered = bool(profile.get('discovery'))
    configured_regions = config.get('review_regions', [])
    overlap_regions = [] if configured_regions else corner_overlap_regions(source, profile, attempt)
    regions = configured_regions or overlap_regions
    feedback = profile.get('discovery', {}).get('feedback')
    extent = profile.get('discovery', {}).get('chroma_extent_repair')
    overlap_checks = profile.get('discovery', {}).get('overlap_review_frames', [])
    priority = [*focus_frames, *overlap_checks, *([feedback['frame']] if feedback else []), *([extent['frame']] if extent else []),
                *config.get('review_frames', []), *(row[0] for row in regions)]
    maximum = config['review_samples']
    frames, audit = review_inputs(source, attempt, profile, maximum, None, priority)
    if overlap_checks:
        frames = list(dict.fromkeys([*frames, *overlap_checks]))
    findings = numeric_findings(attempt, profile, audit, config['max_url_p95'])
    findings += overlay_coverage_findings(source, profile, attempt/'track.npz', config.get('required_overlay_regions'))
    trace('candidate.checked', frames=frames, findings=findings,
          verification=read_object(attempt/'verification.json'), audit=audit)
    if discovered and not config.get('review_only_regions'):
        from wm_quality import review_output
        verification = read_object(attempt/'verification.json')
        if all(verification.get(key) is True for key in (
                'same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok')):
            candidate = Video.open(attempt/'result.mp4')
            record.update(frames=frames, local_findings=findings, candidate=str(candidate.path),
                          review_regions=regions, review_only_regions=False)
            whole = review_output(source, candidate, attempt, config, cuda, trace, record, persist, priority=priority)
            if whole['decision'] != 'accept':
                decision = {'decision': 'review', 'reason': whole['reason']}
                if whole['hints']:
                    decision['watermark_hint'] = whole['hints'][0]
                record['decision'] = decision
                persist()
                return decision
    if findings and config.get('batch_routing') and not repairs_left:
        decision = {'decision': 'review', 'reason': ' / '.join(findings)}
        record.update(frames=frames, local_findings=findings, decision=decision, candidate=str(attempt/'result.mp4'))
        persist()
        return decision
    if discovered:
        candidate = Video.open(attempt/'result.mp4')
        only_regions = config.get('review_only_regions', False)
        if only_regions and not any(region[0] in frames for region in regions):
            raise ValueError('No configured review region was selected for this candidate.')
        record.update(frames=frames, local_findings=findings, candidate=str(candidate.path),
                      review_regions=regions, review_only_regions=only_regions)
        if findings and config.get('batch_routing'):
            decision = {'decision': 'review', 'reason': ' / '.join(findings)}
            if repairs_left and len(findings) == 1 and not audit.get('discovery_passed') and can_retry_discovery(read_object(recipe)):
                with np.load(attempt/'track.npz', allow_pickle=False) as track:
                    if 'matte_positions' not in track:
                        decision.update(decision='repair', repair={'kind': 'discovery-detail',
                                                                  'frame': audit['discovery_worst_frames'][0]})
            record['decision'] = decision
            persist()
            return decision
        with wm_grok.rendered_candidate_review(not findings):
            decision = review_discovery(source, candidate, attempt, recipe, profile, config, cuda, trace, record,
                                        persist, frames, audit, focus_frames, repairs_left)
        record['decision'] = decision
        persist()
        return decision
    signals = {'local_findings': findings, 'corners': profile.get('corners', {}).get('masks', []),
               'moving_mark': bool(profile.get('translucent')), 'repairs_left': repairs_left,
               'source_size': [source.width, source.height]}
    settings = profile.get('translucent', {})
    paired = settings.get('verify_paired_motion', False)
    colored = strong_colored_plate(settings)
    crop_frame_actions = paired or colored or bool(overlap_regions)
    signals['crop_frame_actions'] = crop_frame_actions
    with np.load(review_track(attempt), allow_pickle=False) as track:
        if corner := profile.get('corners'):
            active = corner_activity(corner, track['corners'][frames])
            signals['active_source_corners'] = {i: np.flatnonzero(row).tolist() for i, row in zip(frames, active, strict=True)}
        moving = [(profile, track, 'primary'), *extra_tracks(profile, track)] if profile.get('translucent') else []
        signals['source_moving_boxes_normalized'] = {}
        for layer, points, label in moving:
            settings = layer['translucent']
            w, h = settings['reference_rect'][2:]
            px, py = motion_padding(settings)
            signals['source_moving_boxes_normalized'][label] = {
                i: [max(0, round(1000*(left-px)/source.width)), max(0, round(1000*(top-py)/source.height)),
                    min(1000, round(1000*(left-px+w)/source.width)), min(1000, round(1000*(top-py+h)/source.height))]
                for i, (left, top) in zip(frames, points['positions'][frames].tolist(), strict=True) if left >= 0}
        if paired and not regions:
            regions = paired_review_regions(source, profile, dict(track), frames, audit)
        if colored and not regions:
            regions = colored_risk_regions(source, profile, track, frames, audit)
    if focus_crops and not configured_regions:
        regions = [*focus_crops, *(row for row in regions if row[0] not in focus_frames)]
    signals['review_regions'] = regions
    only_regions = config.get('review_only_regions', False)
    record.update(frames=frames, local_findings=findings, candidate=str(attempt/'result.mp4'),
                  review_regions=regions, review_only_regions=only_regions)
    persist()
    if only_regions and not any(region[0] in frames for region in regions):
        raise ValueError('No configured review region was selected for this candidate.')
    allow_crop_repair = crop_frame_actions and repairs_left > 0
    prompt = CROP_REVIEW_PROMPT if only_regions else REVIEW_PROMPT + '\nLocal evidence: ' + json.dumps(signals)
    print(f'Grok is reviewing {attempt.name}.', flush=True)
    width = config['sample_width']
    candidate = Video.open(attempt/'result.mp4')
    review_trace = partial(trace, phase='review')
    trace('evidence.started', phase='review', frames=frames, regions=regions,
          only_regions=only_regions, max_side=width)
    with wm_grok.rendered_candidate_review(not findings):
        decision = review_visual(config, prompt, evidence(candidate, frames, width, cuda, regions,
                                 only_regions), review_trace,
                                 allow_crop_repair=allow_crop_repair,
                                 allow_batch_repair=(paired or colored) and allow_crop_repair)
    record['decision'] = decision
    persist()
    return decision


def finish_attempts(source, directory, recipe, config, cuda, trace, report, persist, *,
                    reuse=None, render_gate=None, verified_plan=None):
    first_attempt = len(report['attempts']) + (1 if verified_plan is None else 0)
    review_only = verified_plan is not None and render_gate is None
    last_attempt = first_attempt if review_only else max(first_attempt, config['max_repairs'] + 1)
    focus_frames, focus_crops = (), []
    required_comparisons = []
    for number in range(first_attempt, last_attempt + 1):
        attempt = directory / f'attempt-{number:02d}'
        attempt_trace = partial(trace, attempt=number)
        if verified_plan is not None:
            plan, verified_plan = verified_plan, None
            record = report['attempts'][-1]
        else:
            attempt_trace('attempt.started', directory=str(attempt), recipe=str(recipe))
            plan = build_plan(source.path, attempt, recipe, not cuda)
            attempt.mkdir()
            log = {'source': str(source.path), 'recipe': str(recipe),
                   'inputs': fingerprints(input_files(source.path, recipe, attempt, plan)), 'stages': plan}
            record = {'directory': str(attempt), 'recipe': str(recipe)}
            if required_comparisons:
                record['required_comparisons'] = required_comparisons
            report['status'] = 'running'
            report['attempts'].append(record)
            persist()
            attempt_trace('attempt.inputs', inputs=log['inputs'])
            with processing_slot(render_gate), lock_job(attempt):
                first = reuse_prepared(attempt, log, reuse, attempt_trace) if reuse else 0
                execute_plan(attempt, log, first, attempt_trace, render_slots=config.get('render_workers', 1))
        reuse = attempt
        decision = review_candidate(source, attempt, recipe, config, cuda, attempt_trace, record, persist,
                                    repairs_left=last_attempt-number, focus_frames=focus_frames, focus_crops=focus_crops)
        if decision['decision'] == 'accept' and not record['local_findings']:
            publish_result(directory, attempt, report, attempt_trace)
            learn_from_release(attempt, recipe, config, attempt_trace)
            return report
        if decision['decision'] != 'repair' or number == last_attempt:
            report['status'] = 'needs_review'
            return report
        regions = record['review_regions']
        displayed = [*record['frames'], *(row[0] for row in record.get('comparison_regions', []))]
        comparisons = [(region[0], layer) for region, layer in zip(
            record.get('comparison_regions', []), record.get('comparison_layers', []), strict=True)]
        issues = record.get('comparison_issues', [])
        actions = [decision.get('repair')]
        for issue in issues:
            verdict = issue.get('verdict', {})
            if verdict.get('decision') == 'repair' and verdict.get('repair') not in actions:
                actions.append(verdict.get('repair'))
        errors = []
        for action in actions:
            try:
                next_recipe, next_focus = repair_recipe(source, directory, number+1, recipe, attempt, plan, displayed,
                                                        action, config['repair_radius_frames'], regions,
                                                        comparisons=comparisons, cuda=cuda, trace=attempt_trace,
                                                        local_backend=config.get('discovery_local'))
                missing = recipe_coverage_findings(source, next_recipe, config.get('required_overlay_regions'))
                if missing:
                    raise ValueError(' / '.join(missing))
            except ValueError as error:
                errors.append(str(error))
                attempt_trace('repair.rejected', action=action, reason=str(error))
            else:
                recipe, focus_frames = next_recipe, next_focus
                break
        else:
            report.update(status='needs_review', error=' / '.join(dict.fromkeys(errors)))
            return report
        pending = [comparisons[row['comparison']] for row in issues]
        required_comparisons = list(dict.fromkeys([*map(tuple, record.get('required_comparisons', [])), *pending]))
        focus_frames = tuple(dict.fromkeys([*focus_frames, *(i for i, _ in required_comparisons)]))
        focus_crops = [row for row in regions if row[0] in focus_frames]
        attempt_trace('repair.scheduled', action=action, recipe=str(recipe), focus=focus_frames[0], focus_frames=focus_frames)
    return report


def export_review_candidate(directory, report, trace):
    for record in reversed(report.get('attempts', [])):
        attempt = Path(record['directory'])
        candidate, verification = attempt/'result.mp4', attempt/'verification.json'
        if not candidate.is_file() or not verification.is_file():
            continue
        checks = read_object(verification)
        if not all(checks.get(key) is True for key in ('same_video_timing_and_size',
                                                     'same_audio_payload_and_timing', 'full_decode_ok')):
            continue
        target = directory/'needs-review.mp4'
        identity = digest(candidate)
        if target.exists() and digest(target) != identity:
            target.unlink()
        if not target.exists():
            try:
                os.link(candidate, target)
            except OSError:
                shutil.copyfile(candidate, target)
        report.update(review_candidate=str(target), review_candidate_sha256=identity)
        trace('candidate.exported_for_review', candidate=str(target), sha256=identity,
              local_findings=record.get('local_findings'), visual_decision=record.get('decision'))
        break


def run_job(path, directory, recipe, config, cuda, reuse=None, *, render_gate=None, local_refinement=None):
    with source_runtime(config):
        return _run_job(path, directory, recipe, config, cuda, reuse,
                        render_gate=render_gate, local_refinement=local_refinement)


def _run_job(path, directory, recipe, config, cuda, reuse=None, *, render_gate=None, local_refinement=None):
    directory.mkdir(parents=True)
    status_path = directory / 'director.json'
    events_path = directory / 'events.jsonl'
    trace = job_trace(events_path)

    report = {'source': str(path), 'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
              'status': 'running', 'attempts': [], 'events': str(events_path)}
    if config.get('required_overlay_regions'):
        report['required_overlay_regions'] = config['required_overlay_regions']
    if recipe is not None:
        report['planning'] = {'method': 'recipe', 'recipe': str(recipe)}
    persist = partial(write_report, status_path, report)
    logged_config = dict(config)
    for name in ('discovery_local', 'grounding_local'):
        if name in logged_config:
            logged_config[name] = {key: value for key, value in logged_config[name].items()
                                   if key not in ('evidence_cache', 'read_pixels')}
    trace('job.started', source=str(path), recipe=str(recipe) if recipe else None,
          cache=str(reuse) if reuse else None, cuda_decode=cuda, config=logged_config)
    persist()
    try:
        source = Video.open(path)
        trace('source.probed', size=[source.width, source.height], fps=str(source.fps),
              frames=source.count, offset=str(source.offset), matrix=source.matrix)
        if recipe is None:
            recipe = make_recipe(source, directory, config, cuda, report, partial(trace, phase='planning'),
                                 local_refinement=local_refinement)
            persist()
            if recipe is None:
                report['status'] = 'needs_review'
                return report
        candidate_profile = read_object(recipe.parent/read_object(recipe)['profile'])
        coverage_review_only = candidate_profile.get('discovery', {}).get('coverage_review_only') is True
        if coverage_review_only:
            config = {**config, 'max_repairs': 0}
            report.setdefault('planning', {})['coverage_review_only'] = True
            trace('candidate.coverage_review_only', reason='Uncertain placements are left unchanged.')
        if (required := config.get('required_overlay_regions')) and not coverage_review_only:
            findings = recipe_coverage_findings(source, recipe, required)
            if findings:
                report['status'] = 'needs_review'
                report.setdefault('planning', {})['rejection'] = ' / '.join(findings)
                trace('candidate.coverage_rejected', required=len(required), findings=findings)
                return report
        with processing_phase():
            return finish_attempts(source, directory, recipe, config, cuda, trace, report, persist,
                                   reuse=reuse, render_gate=render_gate)
    except wm_grok.SourceReviewDeferred as error:
        report.update(status='failed', error=str(error))
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        report.update({'status': 'needs_review' if isinstance(error, WorkflowLimit) else 'failed', 'error': str(error),
                       'budget_exhausted': isinstance(error, WorkflowLimit)})
        trace('job.error', error_type=type(error).__name__, error=str(error))
        print(f'{path.name}: {error}', file=sys.stderr, flush=True)
        return report
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    finally:
        if report['status'] != 'complete':
            try:
                export_review_candidate(directory, report, trace)
            except (OSError, ValueError) as error:
                trace('candidate.review_export_failed', error=str(error))
        persist()
        trace('job.finished', status=report['status'], result=report.get('result'), error=report.get('error'))


def recheck_job(path, directory, config, cuda, *, render_gate=None):
    status_path = directory/'director.json'
    with lock_job(directory):
        report = read_object(status_path)
        if Path(report['source']).resolve() != path:
            raise ValueError('This job belongs to a different source video.')
        if report['status'] not in ('failed', 'needs_review', 'interrupted', 'rechecking') or not report['attempts']:
            raise ValueError('A visual recheck needs a stopped job with a completed local candidate.')
        record = report['attempts'][-1]
        attempt = (directory/f'attempt-{len(report["attempts"]):02d}').resolve(strict=True)
        if attempt.parent != directory or attempt != Path(record['directory']).resolve():
            raise ValueError('The recorded attempt is outside the job directory.')
        recipe = Path(record['recipe']).resolve(strict=True)
        plan = build_plan(path, attempt, recipe, not cuda)
        state, finished = resume_state(attempt, plan, fingerprints(input_files(path, recipe, attempt, plan)))
        if finished != len(plan) or state['status'] != 'complete':
            raise ValueError('The candidate needs all local stages and artifact checks before a visual recheck.')
        trace = partial(job_trace(directory/'events.jsonl'), mode='recheck', attempt=len(report['attempts']))
        persist = partial(write_report, status_path, report)
        config = dict(config)
        config['required_overlay_regions'] = [list(row) for row in dict.fromkeys(
            tuple(row) for rows in (report.get('required_overlay_regions', []), config.get('required_overlay_regions', []))
            for row in rows)]
        if config['required_overlay_regions']:
            report['required_overlay_regions'] = config['required_overlay_regions']
        for old, new in (('frames', 'review_frames'), ('review_regions', 'review_regions'),
                         ('review_only_regions', 'review_only_regions')):
            if old in record:
                config[new] = record[old]
        config['review_samples'] = max(config['review_samples'], len(config.get('review_frames', [])))
        if ((report.get('error') or '').startswith('Grok timed out during ')
                and record.get('source_comparison', {}).get('decision') == 'accept'):
            record['split_context_review'] = True
        report.update(status='rechecking', model=config['model'], reasoning_effort=config['reasoning_effort'])
        report.pop('error', None)
        record.pop('decision', None)
        trace('recheck.started', candidate=str(attempt/'result.mp4'), recipe=str(recipe),
              model=config['model'], reasoning_effort=config['reasoning_effort'])
        persist()
        try:
            finish_attempts(Video.open(path), directory, recipe, config, cuda, trace, report, persist,
                            render_gate=render_gate, verified_plan=plan)
        except wm_grok.SourceReviewDeferred as error:
            report.update(status='failed', error=str(error))
            raise
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            report.update(status='failed', error=str(error))
            trace('recheck.error', error_type=type(error).__name__, error=str(error))
            print(f'{path.name}: {error}', file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
            raise
        finally:
            persist()
            trace('recheck.finished', status=report['status'], result=report.get('result'), error=report.get('error'))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='An MP4 file or a directory of MP4 files.')
    parser.add_argument('job', type=Path, help='New job directory.')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--recipe', type=Path, help='Existing repair recipe. Omit to plan from source frames.')
    parser.add_argument('--reuse-prepared', type=Path, help='Reuse matching intermediate stages from an earlier attempt.')
    parser.add_argument('--detect-known-layout', action='store_true',
                        help='Select the legacy layout recognizer instead of adaptive discovery.')
    parser.add_argument('--recheck', action='store_true', help='Retry visual acceptance of a locally verified candidate.')
    parser.add_argument('--cpu-decode', action='store_true')
    args = parser.parse_args()
    source, job = args.source.resolve(strict=True), args.job.resolve()
    config = load_settings(args.config.resolve(strict=True))
    if args.detect_known_layout:
        config['detect_known_layout'] = True
        config['recognition'] = 'known'
    recipe = args.recipe.resolve(strict=True) if args.recipe else None
    reuse = args.reuse_prepared.resolve(strict=True) if args.reuse_prepared else None
    if args.recheck:
        if not source.is_file() or recipe or reuse:
            raise ValueError('Choose the original MP4 and its existing job directory for a visual recheck.')
        report = recheck_job(source, job, config, not args.cpu_decode)
        return 0 if report['status'] == 'complete' else 2
    if job.exists():
        raise ValueError('Choose a new director job directory.')
    if source.is_file():
        if source.suffix.lower() != '.mp4':
            raise ValueError('Supply an MP4 source.')
        report = run_job(source, job, recipe, config, not args.cpu_decode, reuse)
        return 0 if report['status'] == 'complete' else 2
    if reuse:
        raise ValueError('A prepared attempt can only be reused for a single MP4 source.')
    sources = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
    if not sources:
        raise ValueError('The source directory has no MP4 files.')
    job.mkdir(parents=True)
    results = []
    for index, path in enumerate(sources, 1):
        item = job / f'{index:03d}-{path.stem}'
        report = run_job(path, item, recipe, config, not args.cpu_decode)
        planning = report.get('planning', {})
        last = report['attempts'][-1] if report['attempts'] else {}
        reason = (report.get('error') or planning.get('rejection') or planning.get('reason')
                  or ' / '.join(last.get('local_findings', []))
                  or last.get('decision', {}).get('reason') or planning.get('decision', {}).get('reason'))
        results.append({'source': str(path), 'status': report['status'], 'report': str(item/'director.json'),
                        'result': report.get('result'), 'layout': planning.get('layout'), 'reason': reason})
        write_report(job / 'batch.json', {'total': len(sources), 'jobs': results})
    return 0 if all(r['status'] == 'complete' for r in results) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
