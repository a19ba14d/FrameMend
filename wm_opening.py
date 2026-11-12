import json
from pathlib import Path

import cv2
import numpy as np

from wm_inpaint import ProPainter, composite, feather
from wm_matte import affine_between, exclusion
from wm_media import digest, plane_rects, read_frames, yuv_to_rgb
from wm_repair import dilate
from wm_track import check_rect, save_npz


def rect_mask(rect, region):
    x, y, w, h = rect
    ox, oy, rw, rh = region
    mask = np.zeros((rh, rw), dtype=np.uint8)
    mask[y-oy:y-oy+h, x-ox:x-ox+w] = 1
    return mask


def fill_shapes(mask, minimum_area=5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    cv2.drawContours(result, [c for c in contours if cv2.contourArea(c) >= minimum_area], -1, 1, -1)
    return result


def brand_colors(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    cyan = (g-r > 38) & (b-r > 22)
    yellow = (r-b > 48) & (g-b > 46) & (r > 168) & (g > 156)
    white = (np.minimum(np.minimum(r, g), b) > 175) & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 30)
    return cyan, yellow, white


def wordmark_mask(rgb, support):
    colors = (np.logical_or.reduce(brand_colors(rgb)) * support).astype(np.uint8)
    colors = cv2.morphologyEx(colors, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return dilate(fill_shapes(colors), 14)


def wipe_mask(rgb):
    cyan, yellow, _ = brand_colors(rgb)
    colors = (cyan | yellow).astype(np.uint8)
    contours, _ = cv2.findContours(colors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = np.zeros_like(colors)
    for contour in contours:
        area = cv2.contourArea(contour)
        arc = cv2.arcLength(contour, True)
        _, _, w, h = cv2.boundingRect(contour)
        circularity = 4 * np.pi * area / max(1, arc * arc)
        if area > 16000 or (area > 4 and circularity > 0.63 and 0.70 <= w/h <= 1.43):
            cv2.drawContours(dots, [contour], -1, 1, -1)
    return dilate(dots, 5)


def aligned_closed_frame(video, reference, target, ref_mask, target_mask, rect):
    a, b = video.planes(reference)[0], video.planes(target)[0]
    transform = affine_between(a, b, ref_mask, target_mask)
    x, y, w, h = rect
    rgb = yuv_to_rgb(plane_rects(video.planes(reference), rect), video.matrix)
    transform[:, 2] += transform[:, :2] @ (x, y) - (x, y)
    return cv2.warpAffine(rgb, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


class Opening:
    def __init__(self, video, profile, track, config, path):
        self.video, self.profile, self.track = video, profile, track
        self.config, self.path = Path(config), Path(path)
        self.settings = json.loads(self.config.read_text(encoding='utf-8-sig'))
        self.rect = check_rect(self.settings['region'], video.width, video.height)
        self.rgb, self.alpha = None, None
        self.corner = None

    def prepare(self, matte, cuda=True, model_dir=None, corners=None):
        if self.path.exists():
            self.load()
            return
        settings = self.settings
        first = settings['clean_closed_frame']
        patch_stop, transition, stop = (settings[k] for k in ('reference_patch_stop', 'transition_start', 'stop'))
        refs = settings['clean_open_frames']
        if not 0 <= first < patch_stop < transition < stop <= min(refs) <= max(refs) < self.video.count:
            raise ValueError('Invalid opening frame ranges.')
        corner_settings = settings.get('corner')
        read_stop = max(stop, corner_settings['stop']) if corner_settings else stop
        items = [f for _, f in read_frames(self.video, 0, read_stop, cuda)]
        for i in range(0, read_stop, 20):
            if matte:
                matte.restore(self.video, items[i:i+20], i)
        raw_rgb = np.stack([yuv_to_rgb(plane_rects(self.video.planes(f), self.rect), self.video.matrix)
                            for f in items[:stop]])
        early_rect = settings['early_support']
        support = rect_mask(early_rect, self.rect)
        x, y, w, h = early_rect
        early_reference = items[first]
        reference_mask = exclusion(self.video, self.profile, self.track, first)
        reference_mask[y:y+h, x:x+w] = 0
        alpha = np.zeros(raw_rgb.shape[:3], dtype=np.float32)
        restored = raw_rgb.copy()
        for i in range(first+1, patch_stop):
            mask = exclusion(self.video, self.profile, self.track, i)
            mask[y:y+h, x:x+w] = 0
            plate = aligned_closed_frame(self.video, early_reference, items[i], reference_mask, mask, self.rect)
            delta = np.abs(raw_rgb[i].astype(np.int16) - plate).max(axis=2)
            seeds = ((delta > 15) * support).astype(np.uint8)
            seeds = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            mask = dilate(support if i < 13 else fill_shapes(seeds, 3), 10)
            alpha[i] = feather(mask, 7)
            restored[i] = plate
        # Combine tight wordmark masks with real closed and open eye frames as temporal context.
        masks = []
        start = max(first + 1, patch_stop - 4)
        rgb = []
        for i in range(start, transition):
            if i < patch_stop:
                mask = np.zeros_like(support)
                plate = np.rint(raw_rgb[i] * (1-alpha[i, ..., None]) + restored[i] * alpha[i, ..., None]).astype(np.uint8)
            else:
                rect = next(k[2:] for k in settings['support_ranges'] if k[0] <= i < k[1])
                mask = wordmark_mask(raw_rgb[i], rect_mask(rect, self.rect))
                if i < 39:
                    # The animated wordmark briefly emits separate dark rays near its top edge.
                    mask = dilate(mask, 8)
                mask[:, :max(0, settings.get('protect_left', 0)-self.rect[0])] = 0
                plate = raw_rgb[i]
            masks.append(mask)
            rgb.append(plate)
        open_plate = None
        for i, f in read_frames(self.video, min(refs), max(refs)+1, cuda):
            if i not in refs:
                continue
            if matte:
                matte.restore(self.video, [f], i)
            if open_plate is None:
                open_plate = f
            rgb.append(yuv_to_rgb(plane_rects(self.video.planes(f), self.rect), self.video.matrix))
            masks.append(np.zeros_like(support))
        kwargs = {'model_dir': model_dir} if model_dir else {}
        model = ProPainter(scale=settings.get('scale', 0.5), **kwargs)
        predictions = model.predict(np.asarray(rgb), np.asarray(masks))
        for i in range(patch_stop, transition):
            restored[i] = predictions[i-start]
            alpha[i] = feather(masks[i-start], 10)
        icon = settings.get('open_plate_icon')
        if icon:
            first_open, end_open, right = icon
            ref_mask = exclusion(self.video, self.profile, self.track, min(refs))
            local_x = np.arange(self.rect[0], self.rect[0]+self.rect[2])
            weight = np.clip((right-local_x)/90, 0, 0.94)[None, :, None]
            for i in range(first_open, end_open):
                mask = exclusion(self.video, self.profile, self.track, i)
                scene_roi = mask[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2]]
                scene_roi[masks[i-start] > 0] = 0
                plate = aligned_closed_frame(self.video, open_plate, items[i], ref_mask, mask, self.rect)
                blend = weight * min(1.0, (i-first_open+1) / 10)
                restored[i] = np.rint(restored[i]*(1-blend) + plate*blend).astype(np.uint8)
        previous = transition - 1
        plate = np.rint(raw_rgb[previous] * (1-alpha[previous, ..., None])
                        + restored[previous] * alpha[previous, ..., None]).astype(np.uint8)
        # Preserve the original wipe dots over the last visible pieces of the brand graphic.
        footprint = dilate(masks[previous-start], 7)
        for i in range(transition, stop):
            restored[i] = plate
            alpha[i] = feather(footprint, 10) * (1 - wipe_mask(raw_rgb[i]))
        extras = {}
        if corner_settings:
            if not corners or not corners.jobs:
                raise ValueError('Opening corner override requires a prepared corner cache.')
            extras = self.make_corner(items, corners, corner_settings)
        save_npz(self.path, rgb=restored, alpha=np.rint(alpha * 255).astype(np.uint8), **extras,
                 source_sha256=self.track['source_sha256'], config_sha256=digest(self.config))
        self.load()
        print(f'Opening branding repair cached through frame {stop - 1}.', flush=True)

    def load(self):
        with np.load(self.path, allow_pickle=False) as cached:
            if (str(cached['source_sha256']) != str(self.track['source_sha256'])
                    or str(cached['config_sha256']) != digest(self.config)):
                raise ValueError('Opening cache belongs to another source or configuration.')
            self.rgb, self.alpha = cached['rgb'], cached['alpha']
            if 'corner_rgb' in cached:
                self.corner = (tuple(cached['corner_rect'].tolist()), cached['corner_rgb'], cached['corner_alpha'])

    def make_corner(self, items, corners, settings):
        job = corners.jobs[0]
        rect = job.patch_rect
        x, y, w, h = rect
        base_index = settings['plate_frame']
        base = items[base_index].copy()
        corners.restore(base_index, base)
        reference = exclusion(self.video, self.profile, self.track, base_index)
        caption_rect = settings['caption_rect']
        caption_mask = rect_mask(caption_rect, rect)
        caption_y = plane_rects(self.video.planes(items[settings['caption_frame']]), rect)[0]
        caption = dilate(((caption_y > 180) & caption_mask.astype(bool)).astype(np.uint8), 6) & caption_mask
        shifted = {i: (j, dx) for i, j, dx in settings['shifted_wipe']}
        results, weights = [], []
        for i, frame in enumerate(items[:settings['stop']]):
            mask = exclusion(self.video, self.profile, self.track, i)
            plate = aligned_closed_frame(self.video, base, frame, reference, mask, rect)
            alpha = job.alpha.copy()
            if i >= settings['caption_start']:
                alpha *= 1 - caption
            lo, hi = settings['solid_wipe']
            if lo <= i < hi:
                source = yuv_to_rgb(plane_rects(self.video.planes(frame), (0, 184, 420, 38)), self.video.matrix)
                plate[:] = np.median(source.reshape(-1, 3), axis=0).astype(np.uint8)
            elif i in shifted:
                ref, dx = shifted[i]
                sample = yuv_to_rgb(plane_rects(self.video.planes(items[ref]), (x+dx, y, w, h)), self.video.matrix)
                cyan, yellow, _ = brand_colors(sample)
                graphic = (cyan | yellow).astype(np.uint8)
                coverage = dilate(fill_shapes(graphic, 2), 4).astype(bool)
                plate[coverage] = sample[coverage]
            results.append(plate)
            weights.append(np.rint(alpha * 255).astype(np.uint8))
        return dict(corner_rect=np.asarray(rect), corner_rgb=np.asarray(results), corner_alpha=np.asarray(weights))

    def restore_corner(self, index, frame):
        if self.corner is None or index >= len(self.corner[1]):
            return False
        rect, rgb, alpha = self.corner
        roi = plane_rects(self.video.planes(frame), rect)
        composite(roi, rgb[index], alpha[index].astype(np.float32) / 255, self.video.matrix)
        return True

    def restore(self, index, frame):
        if index >= len(self.rgb) or not self.alpha[index].any():
            return
        roi = plane_rects(self.video.planes(frame), self.rect)
        composite(roi, self.rgb[index], self.alpha[index].astype(np.float32) / 255, self.video.matrix)
