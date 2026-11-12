"""Scan a video for the fixed screen positions of a repeated animation."""

import json
from pathlib import Path

import cv2
import numpy as np

from wm_media import Video, plane_rects, read_frames, yuv_to_rgb
from wm_opening import brand_colors
from wm_track import check_rect


def signature(source, starts, ages, rect, cuda):
    anchor_frames = []
    for start in starts:
        anchor_frames.append({i-start: tuple(p.copy() for p in plane_rects(source.planes(frame), rect))
                              for i, frame in read_frames(source, start+ages[0], start+ages[-1]+1, cuda)
                              if i-start in ages})
    samples = []
    for age in ages:
        first, second = (anchors[age] for anchors in anchor_frames)
        a, b = [yuv_to_rgb(parts, source.matrix) for parts in (first, second)]
        cyan_a, yellow_a, _ = brand_colors(a)
        cyan_b, yellow_b, _ = brand_colors(b)
        same_color = (cyan_a & cyan_b) | (yellow_a & yellow_b)
        close = np.abs(a.astype(np.int16)-b).max(axis=2) < 26
        mask = cv2.erode((same_color & close).astype(np.uint8), np.ones((5, 5), np.uint8))
        y, x = np.where(mask)
        if len(y) < 40:
            raise ValueError(f'No reliable shared animation color at age {age}.')
        take = np.linspace(0, len(y)-1, min(2300, len(y)), dtype=int)
        y, x = y[take], x[take]
        coords = [(y//s, x//s) for s in (1, 2, 2)]
        values = [(p[index].astype(np.float32)+q[index])/2 for p, q, index in zip(first, second, coords, strict=True)]
        samples.append((coords, values))
    return samples


def scan_signature(video, rect, ages, samples, cuda):
    distances = np.empty((len(ages), video.count), np.float32)
    for i, frame in read_frames(video, cuda=cuda):
        planes = plane_rects(video.planes(frame), rect)
        for j, (coords, values) in enumerate(samples):
            distances[j, i] = sum(float(np.abs(p[index]-value).mean()) for p, index, value in
                                  zip(planes, coords, values, strict=True))/3
    last = video.count-ages[-1]
    return np.stack([distances[j, age:last+age] for j, age in enumerate(ages)]).mean(axis=0)


def audit_animation(source, output, starts, ages, region, report_path, threshold=6, cuda=True):
    source, output = Video.open(source), Video.open(output)
    if (source.width, source.height, source.count, source.fps, source.matrix) != (
            output.width, output.height, output.count, output.fps, output.matrix):
        raise ValueError('Animation audit requires aligned source and output frames.')
    rect = check_rect(region, source.width, source.height)
    ages = sorted(set(ages))
    if (len(ages) < 3 or ages[0] < 0 or len(set(starts)) != 2
            or any(i < 0 or i+ages[-1] >= source.count for i in starts) or not 0 < threshold <= 20):
        raise ValueError('Choose two reference starts, at least three ages and a valid distance threshold.')
    samples = signature(source, starts, ages, rect, cuda)
    before = scan_signature(source, rect, ages, samples, cuda)
    print(f'Scanned {source.count} source frames for animation anchors.', flush=True)
    after = scan_signature(output, rect, ages, samples, cuda)
    occurrences = np.flatnonzero(before <= threshold).tolist()
    matches = np.flatnonzero(after <= threshold).tolist()
    report = dict(frames_per_video=source.count, reference_starts=starts, ages=ages,
                  max_mean_yuv_error=threshold, source_matches=occurrences, output_matches=matches,
                  source_occurrences=[dict(frame=i, seconds=float(i/source.fps),
                                           source_error=round(float(before[i]), 4),
                                           output_error=round(float(after[i]), 4)) for i in occurrences])
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
