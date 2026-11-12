"""Fit continuously moving translucent marks from templates over black."""

import cv2
import numpy as np

from wm_matte import fit_fields, frame_variants, logo_rect, phase_offsets, save_calibrations
from wm_media import get_frame, plane_rects, read_frames, yuv_to_rgb
from wm_track import check_rect


BLACK = (16, 128, 128)


def template_seed(video, template, size, frames, cuda):
    rect = check_rect(template['rect'], video.width, video.height)
    if rect[2:] != size:
        raise ValueError('Flat templates must have the same size as the tracking reference.')
    index = template['frame']
    if type(index) is not int or not 0 <= index < video.count:
        raise ValueError('Flat template frame is outside the source.')
    source_opacity = float(template['source_opacity'])
    opacity = float(template['opacity'])
    if not 0 < source_opacity <= 1 or not 0 < opacity <= .8:
        raise ValueError('Flat templates require source opacity in (0, 1] and moving opacity in (0, 0.8].')
    dx, dy = template.get('shift', (0, 0))
    if index not in frames:
        frames[index] = video.planes(get_frame(video, index, cuda))
    planes = plane_rects(frames[index], rect)
    peak = yuv_to_rgb(planes, video.matrix).max(axis=2).astype(np.float32) / source_opacity
    peak = .68 * peak + .32 * cv2.medianBlur(peak, 3)
    local_peak = cv2.dilate(peak, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    body = cv2.dilate(np.uint8(peak > 30), np.ones((5, 5), np.uint8))
    coverage = np.clip((peak - 4) / (np.maximum(local_peak * .98, 162) - 4), 0, 1) * body
    alpha = opacity * coverage
    shift = np.float32([[1, 0, dx], [0, 1, dy]])
    support = cv2.warpAffine(cv2.dilate(body, np.ones((5, 5), np.uint8)), shift, size)
    seed = []
    for plane, black in zip(planes, BLACK, strict=True):
        tint = (plane.astype(np.float32) - black) * (opacity / source_opacity)
        tint = cv2.resize(tint, size, interpolation=cv2.INTER_CUBIC) * body
        fields = np.stack((-alpha, (tint + alpha * (black - 128)) / 128), axis=-1)
        seed.append(np.moveaxis(cv2.warpAffine(fields, shift, size), -1, 0).copy())
    return np.stack(seed), support


def flat_sample(parts, support, shift):
    # Use nearby clean pixels only where their local variation supports a flat plate.
    clean, confidence = [], []
    for plane, scale in zip(parts, (1, 2, 2), strict=True):
        dx, dy = shift / scale
        height, width = plane.shape
        mask = cv2.resize(support.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        mask = (cv2.warpAffine(mask, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height)) > .04).astype(np.uint8)
        prediction = cv2.inpaint(plane, mask, 3, cv2.INPAINT_NS)
        good = (1-mask).astype(np.float32)
        values = plane.astype(np.float32)
        size = 39 if scale == 1 else 21
        mass = cv2.boxFilter(good, -1, (size, size))
        normalizer = np.maximum(mass, .03)
        mean = cv2.boxFilter(values * good, -1, (size, size)) / normalizer
        variance = cv2.boxFilter(values**2 * good, -1, (size, size)) / normalizer - mean**2
        weight = np.minimum(mass * 3, 1) / (1 + (np.maximum(variance, 0) / (48 if scale == 1 else 19))**2)
        clean.append(prediction)
        confidence.append(weight)
    return tuple(p.copy() for p in parts), clean, confidence


def calibrate_flat(video, profile, track, output, templates, cuda):
    if not isinstance(templates, list) or templates[0].get('ranges'):
        raise ValueError('Use a default flat template followed by templates with frame ranges.')
    variants = frame_variants(video.count, [template.get('ranges', []) for template in templates[1:]])

    size = tuple(profile['translucent']['reference_rect'][2:])
    anchors = {}
    seeds = [template_seed(video, template, size, anchors, cuda) for template in templates]
    rows, phases = [[] for _ in seeds], [[] for _ in seeds]
    shift = phase_offsets(profile, track)
    selected = np.zeros(video.count, bool)
    for variant in range(len(seeds)):
        eligible = np.flatnonzero((variants == variant) & (track['positions'][:, 0] >= 0))
        if len(eligible) < 3:
            raise ValueError('Each flat template needs at least three tracked frames.')
        stride = max(1, (len(eligible)+127)//128)
        selected[eligible[::stride]] = True
    for i, frame in read_frames(video, cuda=cuda):
        if selected[i]:
            variant = variants[i]
            parts = plane_rects(video.planes(frame), logo_rect(profile, track, i))
            rows[variant].append(flat_sample(parts, seeds[variant][1], shift[i]))
            phases[variant].append(shift[i])

    fields = []
    for variant, (seed, support) in enumerate(seeds):
        print(f'Fitting flat template {variant+1}/{len(seeds)} on {len(rows[variant])} sampled frames.', flush=True)
        fields.append(fit_fields(rows[variant], np.asarray(phases[variant]), support, seed))
    save_calibrations(output, fields, variants, track)
    print(f'Saved {output}', flush=True)
