import json
from pathlib import Path

import cv2
import numpy as np

from wm_caption import CaptionReferences
from wm_matte import Matte
from wm_media import Video, get_frame, plane_rects, read_frames
from wm_temporal import TemporalEdges
from wm_track import load_profile, load_track


def audit_captions(source, baseline, output, profile_path, workdir, band, indices, report_path, cuda=True):
    video, prior, result = map(Video.open, (source, baseline, output))
    for version in (prior, result):
        if (version.width, version.height, version.count, version.fps) != (video.width, video.height, video.count, video.fps):
            raise ValueError('Caption audit requires frame-aligned versions of the same source.')
    profile = load_profile(video, profile_path)
    track = load_track(video, profile_path, Path(workdir) / 'track.npz')
    matte = Matte(profile, track, Path(workdir) / 'matte.npz', clean_edges=True)
    edges = TemporalEdges(video, matte, track, rim=True)
    captions = CaptionReferences(video, matte, band, edges)
    checks = []
    for i in sorted(set(indices)):
        if not 0 <= i < video.count:
            raise ValueError(f'Caption audit frame {i} is out of bounds.')
        rect = captions.caption_rect(i)
        if rect is None:
            continue
        shot = np.searchsorted(edges.cuts, i, side='right') - 1
        lo, hi = map(int, edges.cuts[shot:shot+2])
        production = set(captions.candidate_frames(i, lo, hi))
        held_out = [i+d for d in (-24, -18, -11, -8, 8, 11, 18, 24)
                    if lo <= i+d < hi and i+d not in production]
        if len(held_out) < 2:
            continue
        frames = list(read_frames(video, min(i, *held_out), max(i, *held_out)+1, cuda))
        matte.restore(video, [f for _, f in frames], frames[0][0], spatial=False)
        donors, validity = captions.collect(i, dict(frames), rect, held_out, planes=True)
        captions.clean_old(i+1)
        if len(donors) < 2:
            continue
        expected, strokes, votes = captions.consensus([parts[0] for parts in donors], validity, details=True)
        support = edges.mask(i, rect, edges.repair_shapes[0]).astype(bool)
        mask = strokes & support
        count = int(mask.sum())
        if not count:
            continue
        errors, nearby_errors, chroma_errors = [], [], []
        chroma, chroma_values, chroma_valid = captions.chroma_reference(donors, validity, mask & (votes >= 3))
        chroma_count = 0 if chroma_valid is None else int(chroma_valid.sum())
        around = cv2.dilate(strokes.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        nearby = around & support & ~strokes & (votes >= 3) & (expected > 68) & (expected < 188)
        for version in (prior, result):
            restored = plane_rects(version.planes(get_frame(version, i, cuda)), rect)
            delta = np.abs(restored[0]-expected)
            errors.append(float(delta[mask].mean()))
            nearby_errors.append(float(delta[nearby].mean()) if nearby.any() else None)
            if chroma_count:
                color = np.stack([p[chroma][chroma_valid] for p in restored[1:]], axis=1)
                chroma_errors.append(float(np.abs(color-chroma_values[chroma_valid]).mean()))
        checks.append(dict(frame=i, reference_pixels=count, baseline_luma_mae=errors[0], refined_luma_mae=errors[1],
                           nearby_reference_pixels=int(nearby.sum()), nearby_baseline_luma_mae=nearby_errors[0],
                           nearby_refined_luma_mae=nearby_errors[1], chroma_reference_pixels=chroma_count,
                           chroma_baseline_mae=chroma_errors[0] if chroma_errors else None,
                           chroma_refined_mae=chroma_errors[1] if chroma_errors else None))
    if not checks:
        raise ValueError('No held-out subtitle strokes were found in the requested frames.')
    total = sum(row['reference_pixels'] for row in checks)
    report = dict(reference_pixels=total, frames=checks)
    for key in ('baseline_luma_mae', 'refined_luma_mae'):
        report[key] = sum(row[key]*row['reference_pixels'] for row in checks) / total
    for area, metric in (('nearby', 'luma_mae'), ('chroma', 'mae')):
        label = f'{area}_reference_pixels'
        samples = [row for row in checks if row[label]]
        if samples:
            count = sum(row[label] for row in samples)
            report[label] = count
            for comparison in ('baseline', 'refined'):
                key = f'{area}_{comparison}_{metric}'
                report[key] = sum(row[key]*row[label] for row in samples) / count
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
