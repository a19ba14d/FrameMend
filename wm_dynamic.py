"""Bounded per-frame masks for a visually identified short animated overlay.

These masks remain review-only. A stable-template failure must not silently turn a
changing brand animation into an unedited interval.
"""

import cv2
import numpy as np

from wm_discovery import MAX_TRACKS, discovery_tracks, pack_tracks
from wm_media import get_frame, read_frames, yuv_to_rgb
from wm_runtime import check_budget
from wm_track import highpass


def animated_group(hints, fps):
    # Reviewers sometimes split an icon, lettering and URL belonging to one mark.
    rows = []
    for row in sorted(hints, key=lambda row: row['frame']):
        x, y, w, h = row['rect']
        merged = False
        for prior in reversed(rows):
            if prior['frame'] != row['frame']:
                break
            a, b, c, d = prior['rect']
            if max(x, a)-min(x+w, a+c) <= 24 and max(y, b)-min(y+h, b+d) <= 24:
                left, top = min(x, a), min(y, b)
                prior['rect'] = [left, top, max(x+w, a+c)-left, max(y+h, b+d)-top]
                prior['motion'] = 'animated' if 'animated' in (row.get('motion'), prior.get('motion')) else 'unknown'
                merged = True
                break
        if not merged:
            rows.append(dict(row))
    groups = []
    for row in rows:
        x, y, w, h = row['rect']
        match = None
        for group in reversed(groups):
            last = group[-1]
            gap = row['frame']-last['frame']
            a, b, c, d = last['rect']
            distance = np.hypot(x+w/2-a-c/2, y+h/2-b-d/2)
            if (0 < gap <= fps*.8 and distance <= max(90, 2.5*max(w, h, c, d))
                    and w*h >= .16*c*d):
                match = group
                break
        if match is None:
            match = []
            groups.append(match)
        match.append(row)
    eligible = [group for group in groups if len(group) >= 3
                and group[-1]['frame']-group[0]['frame'] <= fps*8
                and any(row.get('motion') == 'animated' for row in group)]
    return max(eligible, key=lambda group: sum(row['rect'][2]*row['rect'][3] for row in group), default=[])


def segment_overlay(rgb, bounds):
    """Separate bright/colored graphic pixels from their local background.

    A supplied rectangle is only a search region. It is never used as a solid
    erase mask. Insufficient foreground/background separation is a rejection.
    """
    height, width = rgb.shape[:2]
    x, y, w, h = map(int, bounds)
    margin_x, margin_y = max(6, round(w*.10)), max(6, round(h*.08))
    x, y, w, h = x-margin_x, y-margin_y, w+2*margin_x, h+2*margin_y
    x, y, right, bottom = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if min(right-x, bottom-y) < 8:
        return None
    pad = 12
    left, top, end, foot = max(0, x-pad), max(0, y-pad), min(width, right+pad), min(height, bottom+pad)
    crop = rgb[top:foot, left:end]
    inside = np.zeros(crop.shape[:2], np.uint8)
    inside[y-top:bottom-top, x-left:right-left] = 1
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[..., 1], hsv[..., 2]
    outside = value[inside == 0]
    if len(outside) < 32:
        return None
    bright_floor = max(200, min(235, float(np.quantile(outside, .95))+24))
    colored = (saturation > 105) & (value > 135)
    bright = (saturation < 85) & (value > bright_floor)
    number, components, stats, _ = cv2.connectedComponentsWithStats((colored & (inside != 0)).astype(np.uint8))
    if number < 2:
        return None
    largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 20:
        return None
    _, cy, _, ch, _ = stats[largest]
    band = np.zeros_like(inside)
    band[max(0, cy-max(8, round(ch*.4))):min(len(band), cy+ch+max(8, round(ch*.12)))] = 1
    inside &= band
    seeds = (colored | bright) & (inside != 0)
    if seeds.sum() < 32:
        return None
    labels = np.full(inside.shape, cv2.GC_BGD, np.uint8)
    labels[inside != 0] = cv2.GC_PR_BGD
    nearby = cv2.dilate(seeds.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
    labels[nearby & (inside != 0)] = cv2.GC_PR_FGD
    labels[seeds] = cv2.GC_FGD
    cv2.grabCut(crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK)
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    # Keep components supported by actual color/brightness seeds.
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 4 or not seeds[components == i].any():
            mask[components == i] = 0
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if mask.sum() < 32 or mask.sum() > .16*height*width:
        return None
    return left, top, mask


def dynamic_tracks(video, candidate, hints, retained, cuda, trace, *, clean_frames=()):
    group = animated_group(hints, float(video.fps))
    if not group:
        raise ValueError('No bounded, repeatedly observed animated watermark was found.')
    anchor_frames = sorted({row['frame'] for row in group})
    pad = max(1, round(float(video.fps)*.5))
    start, stop = max(0, anchor_frames[0]-pad), min(video.count, anchor_frames[-1]+pad+1)
    before = [i for i in clean_frames if start <= i < anchor_frames[0]]
    after = [i for i in clean_frames if anchor_frames[-1] < i < stop]
    start = max(before)+1 if before else start
    stop = min(after) if after else stop
    prior = discovery_tracks(retained)
    if stop-start+len(prior) > MAX_TRACKS:
        raise ValueError('The animated overlay exceeds the per-frame repair capacity.')
    # Check that output detections are present in the original before planning edits.
    confirmed = []
    for row in group:
        i = row['frame']
        raw = get_frame(video, i, cuda)
        rgb = yuv_to_rgb(video.planes(raw), video.matrix)
        found = segment_overlay(rgb, row['rect'])
        if found is None:
            continue
        x, y, mask = found
        original = rgb[y:y+mask.shape[0], x:x+mask.shape[1]].astype(np.int16)
        output = yuv_to_rgb(candidate.planes(get_frame(candidate, i, cuda)), candidate.matrix)
        delta = np.abs(original-output[y:y+mask.shape[0], x:x+mask.shape[1]])[mask > 0]
        if float(np.mean(delta)) <= 12:
            confirmed.append(row)
    if len(confirmed) < 3:
        raise ValueError('The animated detection lacks three unchanged source confirmations.')
    cuts = retained['cuts']
    tracks, rejected, evidence = [], [], []
    for i, raw in read_frames(video, start, stop, cuda):
        check_budget()
        neighbours = sorted(confirmed, key=lambda row: abs(row['frame']-i))[:2]
        # Union of neighbouring observations accommodates rapid scale/position changes.
        left = min(row['rect'][0] for row in neighbours)-10
        top = min(row['rect'][1] for row in neighbours)-10
        right = max(row['rect'][0]+row['rect'][2] for row in neighbours)+10
        bottom = max(row['rect'][1]+row['rect'][3] for row in neighbours)+10
        rgb = yuv_to_rgb(video.planes(raw), video.matrix)
        found = segment_overlay(rgb, (left, top, right-left, bottom-top))
        if found is None:
            rejected.append(i)
            continue
        x, y, mask = found
        a, b, w, h = cv2.boundingRect(mask)
        x, y = x+a, y+b
        mask = mask[b:b+h, a:a+w]
        if min(w, h) < 8:
            rejected.append(i)
            continue
        signal = highpass(video.planes(raw)[0], 2.2)[y:y+h, x:x+w]
        points = np.zeros((video.count, 2), np.int32)
        points[i] = x, y
        active = np.zeros(video.count, bool)
        active[i] = True
        scores = active.astype(np.float32)
        tracks.append({'positions': points, 'scores': scores, 'present': active, 'verified': active.copy(),
                       'signal': signal.astype(np.float32), 'weights': mask.astype(np.float32), 'mask': mask,
                       'cuts': cuts})
        evidence.append({'frame': i, 'rect': [x, y, w, h], 'pixels': int(mask.sum())})
    if not tracks:
        raise ValueError('No source-confirmed animated mask can be rendered.')
    unresolved = sorted(set(anchor_frames).difference(row['frame'] for row in evidence))
    trace('feedback.dynamic_masks', interval=[start, stop], anchors=anchor_frames, masks=evidence, rejected=rejected,
          unresolved_anchors=unresolved)
    return tracks, {'frames': [row['frame'] for row in evidence], 'first_layer': len(prior),
                    'anchors': anchor_frames, 'unverified_frames': rejected, 'unresolved_anchors': unresolved,
                    'method': 'source-confirmed-local-segmentation'}
