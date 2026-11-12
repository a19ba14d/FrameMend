"""Calibrate transparent overlays from dark plates and aligned samples."""

import cv2
import numpy as np

from wm_media import get_frame, plane_rects, read_frames, yuv_to_rgb
from wm_track import check_rect, crop, highpass, template_peak


WHITE = (235, 128, 128)


class WhiteEdgeAlignment:
    def __init__(self, alpha):
        solid = (alpha > alpha.max()*.4).astype(np.uint8)
        components = cv2.connectedComponentsWithStats(solid, 8)[2][1:]
        if not len(components):
            raise ValueError('White edge alignment needs a measurable opaque shape.')
        x, y, width, height, _ = components[np.argmax(components[:, 4])]
        left, top = max(0, x-8), max(0, y-8)
        right, bottom = x+width+8, y+height+8
        self.offset = left, top
        self.template = highpass(-np.log1p(-alpha[top:bottom, left:right]), 2.2)

    def align(self, luma, rect, prior):
        x, y = rect[0]+self.offset[0], rect[1]+self.offset[1]
        h, w = self.template.shape
        left, top = max(0, x-7), max(0, y-7)
        if x < -5 or y < -5 or x+w > luma.shape[1]+5 or y+h > luma.shape[0]+5:
            return None
        sample = luma[top:y+h+7, left:x+w+7].astype(np.float32)
        contrast = -np.log(np.clip((WHITE[0]-sample)/219, .045, 1.1))
        signal = highpass(contrast, 2.2).clip(-.22, .22)
        saturated = cv2.dilate((sample > 213).astype(np.uint8), np.ones((7, 7), np.uint8))
        signal[saturated > 0] = 0
        dx, dy, score = template_peak(signal, self.template)
        position = np.float32([dx+left-x, dy+top-y])
        distance = np.linalg.norm(position-prior)
        if score >= .34 and .18 <= distance <= 4:
            return position
        return None


def plate_alpha(video, settings, anchors, cuda):
    rect = check_rect(settings['rect'], video.width, video.height)
    index = settings['frame']
    floor = float(settings.get('background_luma', 16))
    ceiling = float(settings['max_alpha'])
    if not 0 <= index < video.count or not 0 <= floor < 64 or not .025 <= ceiling <= .65:
        raise ValueError('Invalid white plate frame, background luma or opacity limit.')
    if index not in anchors:
        anchors[index] = video.planes(get_frame(video, index, cuda))[0]
    signal = crop(anchors[index], rect).astype(np.float32) - floor
    alpha = np.clip(signal / (WHITE[0] - floor), 0, ceiling)
    core = (alpha >= ceiling * .28).astype(np.uint8)
    support = cv2.dilate(core, np.ones((5, 5), np.uint8)).astype(bool)
    alpha *= support & (signal >= 1.5)
    if not alpha.any():
        raise ValueError('The dark plate does not contain a measurable white overlay.')
    print(f'White plate frame {index}, rect {rect}, {np.count_nonzero(alpha)} pixels, '
          f'median core opacity {np.median(alpha[core > 0]):.4f}', flush=True)
    return alpha


def aligned_alpha(samples, ceiling):
    # In log distance from white, opacity contributes a fixed additive signal.
    dark = [-np.log(np.clip((WHITE[0] - patch) / 219, .035, 1.1)) for patch in samples]
    responses = [patch - cv2.GaussianBlur(patch, (0, 0), 7) for patch in dark]
    center = np.median(responses, axis=0)
    support = (center > .18 * ceiling).astype(np.uint8)
    core = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(core, np.ones((5, 5), np.uint8))
    if not mask.any():
        raise ValueError('No common text was found in the aligned white overlay samples.')
    fitted = []
    for patch in dark:
        clean = cv2.inpaint(patch, mask, 4, cv2.INPAINT_NS)
        fitted.append(patch - clean)
    strength = np.median(fitted, axis=0)
    alpha = np.clip(-np.expm1(-strength), 0, ceiling)
    alpha *= mask & ((center > .035 * ceiling) | (core > 0))
    return alpha.astype(np.float32)


def field_model(alpha):
    return [np.stack((-alpha, alpha * ((white - 128) / 128))) for white in WHITE]


def plate_model(video, settings, anchors, cuda):
    reference = settings.get('opacity_reference')
    if not reference:
        return field_model(plate_alpha(video, settings, anchors, cuda))
    rect = check_rect(settings['rect'], video.width, video.height)
    opaque_rect = check_rect(reference['rect'], video.width, video.height)
    w, h = rect[2:]
    dx, dy = reference['offset']
    gain = float(reference['gain'])
    if not (0 < gain <= .65 and np.isfinite([dx, dy]).all()
            and 0 <= settings['frame'] < video.count and 0 <= reference['frame'] < video.count):
        raise ValueError('The colored plate needs a valid opaque reference and measured opacity.')
    parts = plane_rects(video.planes(get_frame(video, settings['frame'], cuda)), rect)
    opaque = plane_rects(video.planes(get_frame(video, reference['frame'], cuda)), opaque_rect)
    strength = yuv_to_rgb(opaque, video.matrix).max(axis=2).astype(np.float32)
    placed = cv2.warpAffine(strength, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    measured = yuv_to_rgb(parts, video.matrix).max(axis=2).astype(np.float32)
    footprint = cv2.dilate((placed > 18).astype(np.uint8), np.ones((3, 3), np.uint8))
    local_peak = cv2.dilate(placed, np.ones((5, 5), np.uint8))
    opacity = np.minimum(gain, measured/np.maximum(local_peak, 150))*footprint
    if not np.any(opacity > gain*.65):
        raise ValueError('The dark and opaque references have no shared foreground.')
    models = []
    for plane, nominal in zip(parts, (16, 128, 128), strict=True):
        size = plane.shape[::-1]
        alpha = cv2.resize(opacity, size, interpolation=cv2.INTER_AREA)
        support = cv2.resize(footprint.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        background = float(np.median(plane[~support])) if (~support).any() else nominal
        # Store the premultiplied foreground from the observed dark pixels.
        tint = (plane.astype(np.float32)-background+alpha*(background-128))*support/128
        models.append(np.stack([cv2.resize(field, (w, h), interpolation=cv2.INTER_NEAREST)
                                for field in (-alpha, tint)]))
    print(f'Colored dark plate frame {settings["frame"]}, rect {rect}, '
          f'measured opacity {gain:.4f}, {np.count_nonzero(opacity)} pixels.', flush=True)
    return models


def calibrate_white(video, profile, track, output, cuda):
    from wm_matte import frame_variants, phase_offsets, save_calibrations

    anchors = {}
    stationary = [plate_alpha(video, layer, anchors, cuda) for layer in profile.get('stationary', [])]
    models, variants = [], None
    if moving := profile.get('translucent'):
        settings = moving['white_plate']
        shape = moving['reference_rect'][2:]
        if settings['rect'][2:] != shape:
            raise ValueError('The white plate and moving tracking rectangle must have the same size.')
        default = plate_model(video, settings, anchors, cuda)
        models.append(default)
        if alternatives := settings.get('variants', []):
            variants = frame_variants(video.count, [item['ranges'] for item in alternatives])
            rects = [None if 'plate' in item else check_rect(item['rect'], *shape) for item in alternatives]
            positions, scores = track['positions'], track['scores']
            samples, selected = [[] for _ in alternatives], np.zeros(video.count, dtype=bool)
            for i in range(1, len(alternatives) + 1):
                if rects[i-1] is None:
                    continue
                eligible = np.flatnonzero((variants == i) & (positions[:, 0] >= 0) & (scores >= .55))
                if len(eligible) < 12:
                    raise ValueError('A white variant needs at least 12 confidently aligned frames.')
                selected[eligible[::max(1, (len(eligible)+159)//160)]] = True
            offsets = phase_offsets(profile, track)
            fixed = FixedWhiteMatte(profile['stationary'], stationary, video.count) if stationary else None
            frames = read_frames(video, cuda=cuda) if selected.any() else ()
            for i, frame in frames:
                if not selected[i]:
                    continue
                planes = video.planes(frame)
                if fixed:
                    fixed.restore(i, planes)
                j = variants[i] - 1
                x, y, w, h = rects[j]
                px, py = positions[i]
                dx, dy = offsets[i]
                matrix = np.float32([[1, 0, -(px+x+dx)], [0, 1, -(py+y+dy)]])
                samples[j].append(cv2.warpAffine(planes[0], matrix, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32))
            for j, (rect, group) in enumerate(zip(rects, samples, strict=True), 1):
                if rect is None:
                    plate = alternatives[j-1]['plate']
                    if plate['rect'][2:] != shape:
                        raise ValueError('Alternate white plates must use the same template size.')
                    models.append(plate_model(video, plate, anchors, cuda))
                    continue
                alpha = -default[0][0].copy()
                learned = aligned_alpha(group, float(settings['max_alpha']))
                crop(alpha, rect)[:] = learned
                models.append(field_model(alpha))
                print(f'White variant {j}, {len(group)} aligned samples, rect {rect}, '
                      f'{np.count_nonzero(learned)} learned pixels.', flush=True)
    additional = []
    for layer in profile.get('extra_white', []):
        plate = layer['white_plate']
        if plate['rect'][2:] != layer['reference_rect'][2:] or plate.get('variants'):
            raise ValueError('An additional layer needs one matching white plate.')
        additional.append(plate_model(video, plate, anchors, cuda))
    save_calibrations(output, models, variants, track, stationary=stationary, additional=additional)
    print(f'Calibrated {len(stationary)} stationary and {len(models)+len(additional)} moving white mattes. Saved {output}', flush=True)


class FixedWhiteMatte:
    def __init__(self, settings, alphas, count):
        from wm_matte import frame_variants

        gain = 1 / (1 - np.minimum(np.arange(1025)[:, None] / 1024, .65))
        codes = np.arange(256)[None]
        self.tables = [np.rint(white + (codes-white)*gain).clip(0, 255).astype(np.uint8) for white in WHITE]
        self.layers = []
        for item, alpha in zip(settings, alphas, strict=True):
            rect = tuple(item['rect'])
            active = frame_variants(count, [item['ranges']]) > 0 if 'ranges' in item else np.ones(count, bool)
            maps = []
            for scale in (1, 2, 2):
                resized = cv2.resize(alpha, (rect[2]//scale, rect[3]//scale), interpolation=cv2.INTER_AREA)
                level = np.rint(resized * 1024).astype(np.uint16)
                coords = np.where(level > 0)
                maps.append((coords, level[coords]))
            self.layers.append((rect, active, maps))

    def restore(self, index, planes):
        for rect, active, maps in self.layers:
            if active[index]:
                for dst, table, (coords, levels) in zip(plane_rects(planes, rect), self.tables, maps, strict=True):
                    dst[coords] = table[levels, dst[coords]]
