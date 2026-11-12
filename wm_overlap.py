"""Locate a known overlay from its exposed pixels when another verified overlay occludes it."""

import cv2
import numpy as np

from wm_discovery import patch


def exposed_match(motif, signal, candidates, occluders, *, prepared=False):
    """Require a distinctive exposed fragment; never treat interpolation alone as evidence."""
    support = motif.weights > 0
    total = float(motif.weights.sum())
    if not total:
        return None
    matches = {}
    for x, y, original_score in candidates[:8]:
        if original_score < .25 or x < 0 or y < 0:
            continue
        if x + motif.width > signal.shape[1] or y + motif.height > signal.shape[0]:
            continue
        pixels = signal[y:y+motif.height, x:x+motif.width]
        for mask, u, v in occluders:
            if x >= u+mask.shape[1] or u >= x+motif.width or y >= v+mask.shape[0] or v >= y+motif.height:
                continue
            # Hide the occluding graphic's edges too; those are not evidence for the target.
            cover = patch(mask if prepared else cv2.dilate(mask, np.ones((5, 5), np.uint8)),
                          (x-u, y-v, motif.width, motif.height)) > 0
            occluded = float(np.sum(motif.weights*cover)/total)
            if not .15 <= occluded <= .85:
                continue
            exposed = support & ~cover
            if np.count_nonzero(exposed) < max(96, .15*np.count_nonzero(support)):
                continue
            weights = motif.weights*exposed
            reference_power = float(np.sum(motif.signal**2*weights))
            observed_power = float(np.sum(pixels**2*weights))
            if min(reference_power, observed_power) < 4*weights.sum():
                continue
            score = float(np.sum(motif.signal*pixels*weights)/np.sqrt(reference_power*observed_power))
            point = (int(x), int(y))
            if score > matches.get(point, (0., 0.))[0]:
                matches[point] = score, occluded
    ranked = sorted(matches.items(), key=lambda row: row[1][0], reverse=True)
    if not ranked or ranked[0][1][0] < .80:
        return None
    (x, y), (score, occluded) = ranked[0]
    competitors = [value[0] for (u, v), value in ranked[1:] if max(abs(x-u), abs(y-v)) > 3]
    if competitors and score-max(competitors) < .12:
        return None
    return x, y, score, occluded


def coherent_overlaps(candidates, cuts, *, max_step=8):
    """Require at least three adjacent, smoothly moving observations in one shot."""
    groups, current = [], []
    for frame, point in sorted(candidates.items()):
        if current:
            previous, prior = current[-1]
            if (frame != previous+1 or frame in cuts
                    or max(abs(point[0]-prior[0]), abs(point[1]-prior[1])) > max_step):
                groups.append(current)
                current = []
        current.append((frame, point))
    groups.append(current)
    return {frame: point for group in groups if len(group) >= 3 for frame, point in group}


def review_samples(frames, maximum=12):
    frames = np.asarray(sorted(set(frames)), dtype=int)
    if not len(frames):
        return []
    runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1)+1)
    selected = list(dict.fromkeys(int(run[k]) for run in runs for k in (0, len(run)//2, -1)))
    return [selected[k] for k in np.linspace(0, len(selected)-1, min(maximum, len(selected)), dtype=int)]
