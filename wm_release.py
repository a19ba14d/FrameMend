"""Recheck later bounding-box evidence against an already verified full render."""

from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np

from wm_director import ask_with_timeout_recovery, expand, review_method, xywh
from wm_discovery import patch
from wm_discovery_render import layer_masks, projected_residual, put_mask
from wm_grounding import frame_evidence, preview_residue
from wm_layers import load_layer_evidence, visible_box
from wm_media import Video, digest, get_frame
from wm_track import highpass, load_track
from workflow import fingerprints, input_files, read_object, resume_state, write_object


METHOD = 'same-source-exact-release-v1'
DECISIONS = ('accept', 'review', 'uncertain')
PROMPT = (
    'Compare each Original and Restored crop with the same frame number. Identify the original '
    'distribution watermark from its visible shape or wording. Check the restored crop both for '
    'remaining watermark fragments and for new blur, smeared edges or other damage to nearby scene '
    'details. Preserve the original scene softness where present. Return only JSON with decision '
    'accept only when every pair is clear and the nearby scene is preserved, otherwise review or '
    'uncertain. Include reason and checks, one object per pair with the exact frame number as frame, '
    'decision, and an explanation grounded in the visible pixels. No tools.'
)


def accepted_pairs(answer, frames):
    if (not isinstance(answer, dict) or answer.get('decision') != 'accept'
            or not isinstance(answer.get('reason'), str) or not answer['reason'].strip()):
        return False
    checks = answer.get('checks')
    if not isinstance(checks, list) or len(checks) != len(frames):
        return False
    return (all(isinstance(check, dict) and check.get('decision') == 'accept'
                and isinstance(check.get('explanation'), str) and check['explanation'].strip()
                for check in checks)
            and {str(check.get('frame')) for check in checks} == {str(frame) for frame in frames})


def verified_attempt(source, report_path, report, source_hash, output_hash):
    last = report['attempts'][-1]
    attempt = Path(last['directory']).resolve()
    recipe = Path(last['recipe']).resolve()
    output = Path(report['result']).resolve()
    if (report['status'] != 'complete' or attempt.parent != report_path.resolve().parent or last.get('local_findings') != []
            or last.get('source_comparison', {}).get('decision') != 'accept'
            or last.get('decision', {}).get('decision') != 'accept'
            or Path(report['source']).resolve() != source.path.resolve() or digest(output) != output_hash):
        raise ValueError('The full candidate needs accepted source and final comparisons.')
    run = read_object(attempt/'run.json')
    stages = run['stages']
    names = ('result.mp4', 'track.npz', 'verification.json', 'watermark-audit.json')
    if not {str(attempt/name) for name in names}.issubset(path for stage in stages for path in stage['outputs']):
        raise ValueError('The candidate has incomplete full-video evidence.')
    inputs = fingerprints(input_files(source.path, recipe, attempt, stages))
    verified, finished = resume_state(attempt, stages, inputs)
    technical = read_object(attempt/'verification.json')
    audit = read_object(attempt/'watermark-audit.json')
    if (inputs[str(source.path)] != source_hash or verified['status'] != 'complete' or finished != len(stages)
            or next((stage['artifacts'].get(str(attempt/'result.mp4')) for stage in stages
                     if stage['name'] == 'render'), None) != output_hash
            or not all(technical.get(key) is True for key in ('same_video_timing_and_size',
                        'same_audio_payload_and_timing', 'full_decode_ok'))
            or not source.count == technical['frames'] == audit['frames_audited'] == audit['discovery_verified_frames']
            or audit.get('discovery_passed') is not True
            or any(layer.get('discovery_passed') is not True for layer in audit.get('discovery_layers', []))):
        raise ValueError('The candidate did not verify every layer, frame and stream.')
    return attempt, (recipe.parent/read_object(recipe)['profile']).resolve(), output


def review_regions(source, output, track, settings, regions, checkpoints, identity, config, cuda, trace):
    backend = config.get('discovery_local')
    if not backend:
        return None
    anchors = {region: [] for region in regions}
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            continue
        try:
            layers, _ = load_layer_evidence(
                checkpoint, source, identity,
                on_rejected=lambda **details: trace('release.motif_layer_rejected', path=str(checkpoint), **details))
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
            trace('release.motif_evidence_rejected', path=str(checkpoint), reason=str(error))
            continue
        for layer in layers:
            for observation, (x, y, score) in layer.matches.items():
                region = (observation.frame, *visible_box(layer.motif, x, y))
                if region in anchors and score >= .70:
                    anchors[region].append((checkpoint, layer, x, y, score))
    if any(not matches for matches in anchors.values()):
        trace('release.exact_shape_missing', regions=[region for region in anchors if not anchors[region]])
        return None
    consensus = settings.get('consensus_mask', False) and 'matte_positions' not in track
    masks = list(layer_masks(source, track, cuda, consensus, settings.get('edge_repair', {}).get('layer')))
    groups, records = [], []
    for region, matches in anchors.items():
        frame, left, top, width, height = region
        canvas = np.zeros((source.height, source.width), np.uint8)
        for layer, _, mask in masks:
            if 'present' not in layer or layer['present'][frame]:
                put_mask(canvas, mask, *map(int, layer['positions'][frame]))
        before, after = (highpass(video.planes(get_frame(video, frame, cuda))[0], 2.2)
                         for video in (source, output))
        pixels = []
        for checkpoint, layer, x, y, score in matches:
            motif = layer.motif
            edited = patch(canvas, (x, y, motif.width, motif.height)) > 0
            covered = int(np.count_nonzero(edited & (motif.mask > 0)))
            residual = float(projected_residual(motif, before, after, x, y, regional=True))
            pixels.append({'checkpoint': str(checkpoint), 'label': layer.label, 'anchor_score': score,
                           'projected_residue': residual, 'covered_pixels': covered})
        trace('release.exact_pixels', region=region, evidence=pixels)
        if any(not 0 <= row['projected_residue'] <= .35 or not row['covered_pixels'] for row in pixels):
            return None
        bounds = (left, top, left+width, top+height)
        crop = xywh(expand(bounds, source, 0, 0))
        local = preview_residue(backend, frame_evidence(output, frame, cuda, 800, region=crop), 22)
        trace('release.local_comparison', region=region, answer=local)
        if local.get('decision') != 'none':
            return None
        crop = xywh(expand(bounds, source, 12, 12))
        if not groups or len(groups[-1]) == 4 or frame in groups[-1]:
            groups.append({})
        groups[-1][frame] = (region, crop)
        records.append({'region': list(region), 'pixels': pixels, 'local': local})
    comparisons = []
    for group in groups:
        images = [(f'{name} frame {frame}', frame_evidence(video, frame, cuda, 800, region=crop)[1])
                  for frame, (_, crop) in group.items() for name, video in (('Original', source), ('Restored', output))]
        answer = ask_with_timeout_recovery(config, PROMPT, images, DECISIONS, trace)
        trace('release.independent_comparison', regions=group, answer=answer)
        if not accepted_pairs(answer, group):
            return None
        comparisons.append({'regions': [list(region) for region, _ in group.values()], 'answer': answer})
    return {'regions': records, 'comparisons': comparisons}


def matching_proof(proof, binding):
    if proof.get('binding') != binding:
        return False
    evidence = proof['evidence']
    if sorted(row['region'] for row in evidence['regions']) != binding['regions']:
        return False
    if any(row['local']['decision'] != 'none' or not row['pixels']
           or any(p['anchor_score'] < .70 or not 0 <= p['projected_residue'] <= .35 or p['covered_pixels'] <= 0
                  for p in row['pixels']) for row in evidence['regions']):
        return False
    comparisons = evidence['comparisons']
    return (sorted(region for check in comparisons for region in check['regions']) == binding['regions']
            and all(accepted_pairs(check['answer'], [region[0] for region in check['regions']]) for check in comparisons))


def exact_coverage_accepted(source, report_path, report, regions, identity, output_hash, proof_path, *,
                            config=None, cuda=True, trace=None, checkpoints=()):
    regions = sorted(set(regions))
    binding = {'method': METHOD, 'source_sha256': identity, 'result_sha256': output_hash,
               'regions': list(map(list, regions))}
    if not regions or (config is None and not proof_path.is_file()):
        return False
    try:
        attempt, profile_path, output = verified_attempt(source, report_path, report, identity, output_hash)
        binding['track_sha256'] = digest(attempt/'track.npz')
        method = review_method(config) if config is not None else None
        if proof_path.is_file():
            proof = read_object(proof_path)
            if (method is None or proof.get('review') == method) and matching_proof(proof, binding):
                return True
        if config is None:
            return False
        track = load_track(source, profile_path, attempt/'track.npz')
        settings = read_object(profile_path)['discovery']
        evidence = review_regions(source, Video.open(output), track, settings, regions, checkpoints, identity,
                                  config, cuda, trace)
        if evidence is None:
            return False
        write_object(proof_path, {'binding': binding, 'review': method, 'evidence': evidence})
        trace('release.exact_coverage_accepted', proof=str(proof_path), regions=len(regions))
        return True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, BadZipFile, cv2.error) as error:
        if trace:
            trace('release.exact_coverage_rejected', reason=str(error), report=str(report_path))
        return False
