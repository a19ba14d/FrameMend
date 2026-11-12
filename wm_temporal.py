import itertools
from functools import lru_cache

import cv2
import numpy as np

from wm_edge_retouch import polish_texture
from wm_inpaint import composite
from wm_matte import logo_rect, repair_ringing
from wm_media import plane_rects, read_frames, yuv_to_rgb


class CapsuleRim:
    """Reconstruct the thin capsule border from both sides of its normal."""

    def __init__(self, width, height, margins):
        self.width, self.height = width, height
        left, top, right, bottom = margins
        support_radius = (height - top - bottom) // 2
        self.ends = left + support_radius, width - right - support_radius - 1
        self.center = top + support_radius - 0.5
        self.radius = support_radius - 5.5

    @lru_cache(maxsize=192)
    def samples(self, dx, dy, scale):
        dx, dy = float(dx), float(dy)
        y, x = np.mgrid[:self.height//scale, :self.width//scale].astype(np.float32)
        px = x * scale + (scale - 1) / 2 - dx
        py = y * scale + (scale - 1) / 2 - dy
        nx = px - np.clip(px, *self.ends)
        ny = py - self.center
        length = np.sqrt(nx**2 + ny**2)
        distance = length - self.radius
        nx /= np.maximum(length, 1)
        ny /= np.maximum(length, 1)
        band = np.clip((6.5 - np.abs(distance)) / 2, 0, 1)
        maps = []
        for normal in (-9, -8, 8, 9):
            step = (normal - distance) / scale
            maps.append((x + nx * step, y + ny * step))
        return maps, band, np.clip((distance + 8) / 16, 0, 1)

    def restore(self, planes, shift):
        for channel, plane in enumerate(planes):
            maps, band, fraction = self.samples(*map(float, shift), 1 if channel == 0 else 2)
            inner0, inner1, outer0, outer1 = [
                cv2.remap(plane, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                for mx, my in maps
            ]
            inner = (inner0 + inner1) * 0.5
            outer = (outer0 + outer1) * 0.5
            spread = np.abs(inner - outer) + np.abs(inner0 - inner1) + np.abs(outer0 - outer1)
            tolerance = 32 if channel == 0 else 22
            predicted = inner + fraction * (outer - inner)
            delta = predicted - plane
            # A subtitle stroke or scene edge is a poor interpolation target.
            weight = band * np.clip(1 - (spread - 5) / tolerance, 0, 1)
            weight *= np.clip(1 - (np.abs(delta) - 17) / tolerance, 0, 1)
            plane[:] = np.rint(plane + delta * weight).astype(np.uint8)

    def align(self, source, shift, fields):
        maps, band, fraction = self.samples(*map(float, shift), 1)
        y, x = np.nonzero((band > 0.85) & ((np.indices(band.shape).sum(axis=0) % 3) == 0))
        inner_x, inner_y = (axis[y, x][None] for axis in maps[1])
        outer_x, outer_y = (axis[y, x][None] for axis in maps[2])
        inner = cv2.remap(source, inner_x, inner_y, cv2.INTER_LINEAR).astype(np.float32)
        outer = cv2.remap(source, outer_x, outer_y, cv2.INTER_LINEAR).astype(np.float32)
        # Both samples are outside the high-contrast rim. Estimate the unmarked
        # inner sample using the calibrated matte before testing the phase.
        d, k = [cv2.remap(f, inner_x - shift[0], inner_y - shift[1], cv2.INTER_LINEAR) for f in fields]
        inner = (inner - 128 * (1 + k)) / (1 + d) + 128
        confidence = np.clip(1 - np.abs(inner - outer) / 18, 0, 1)
        base = inner + fraction[y, x][None] * (outer - inner)
        observed = source[y, x][None]
        shift = shift.copy()
        for axis in (1, 0):
            weight = confidence.copy()
            if axis == 0:
                weight *= ((x - shift[0] < self.ends[0] - 6) | (x - shift[0] > self.ends[1] + 6))[None]
            if weight.sum() < 60:
                continue
            scores = []
            candidates = (0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0)
            for step in candidates:
                dx, dy = shift
                mx = (x.astype(np.float32) - dx - (step if axis == 0 else 0))[None]
                my = (y.astype(np.float32) - dy - (step if axis == 1 else 0))[None]
                d, k = [cv2.remap(f, mx, my, cv2.INTER_LINEAR) for f in fields]
                error = np.abs(observed - (base + d * (base - 128) + 128 * k))
                scores.append(float((np.minimum(error, 21) * weight).sum() / weight.sum()) + abs(step)*0.6)
            best = int(np.argmin(scores))
            if scores[0] - scores[best] > 0.5:
                shift[axis] += candidates[best]
        return shift


class TemporalEdges:
    def __init__(self, video, matte, track, rim=False, conservative=False):
        self.video, self.matte = video, matte
        self.track = track
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.conservative = conservative
        self.repair_shapes = matte.edge_shapes if conservative else [
            np.maximum(m, matte.foreground_shape) for m in matte.edge_shapes]
        self.reference_mask = cv2.dilate(self.repair_shapes[1], np.ones((3, 3), np.uint8))
        self.smooth_edges = (cv2.dilate(np.maximum.reduce(matte.edge_shapes), np.ones((5, 5), np.uint8))
                             if conservative else None)
        self.cuts = np.r_[0, track['cuts'], video.count]
        self.flat_fallback = conservative or bool(matte.profile['translucent'].get('flat_templates'))
        margins = matte.profile['translucent'].get('capsule_margins')
        self.rim = CapsuleRim(matte.width, matte.height, margins) if rim and margins else None

    def mask(self, index, rect, shape, scale=1):
        x, y, w, h = rect
        position = self.track['positions'][index]
        if position[0] < 0:
            return np.zeros((h//scale, w//scale), np.uint8)
        dx, dy = position + self.matte.shifts[index] - (x, y)
        grid = np.float32([[1/scale, 0, dx/scale], [0, 1/scale, dy/scale]])
        return cv2.warpAffine(shape, grid, (w//scale, h//scale), flags=cv2.INTER_NEAREST)

    def predict(self, target, reference, ref_index, context, roi, safe):
        a, b = target[0], reference[0]
        w, h = context[2:]
        shape = (w//2, h//2)
        small = [cv2.resize(p, shape, interpolation=cv2.INTER_AREA) for p in (a, b)]
        flow = cv2.resize(self.flow.calc(*small, None), (w, h), interpolation=cv2.INTER_LINEAR) * 2
        x, y, rw, rh = roi
        dx, dy = x-context[0], y-context[1]
        yy, xx = np.mgrid[dy:dy+rh, dx:dx+rw].astype(np.float32)
        local = flow[dy:dy+rh, dx:dx+rw]
        mx, my = xx + local[..., 0], yy + local[..., 1]
        ref_safe = 1 - self.mask(ref_index, context, self.reference_mask)
        ref_weight = cv2.remap(ref_safe.astype(np.float32), mx, my, cv2.INTER_LINEAR)
        aligned_y = cv2.remap(b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        luma = a[dy:dy+rh, dx:dx+rw]
        error = np.abs(luma.astype(np.float32) - aligned_y)
        known = ref_weight * safe
        local_error = cv2.boxFilter(error * known, -1, (15, 15)) / np.maximum(cv2.boxFilter(known, -1, (15, 15)), 0.20)
        motion = np.sqrt(np.square(local).sum(axis=2))
        confidence = ref_weight / (1 + (local_error / 6.5) ** 2) / (1 + (motion / 32) ** 2)
        confidence *= np.clip((43 - error) / 25, 0, 1)
        result = [aligned_y]
        chroma_maps = [cv2.resize(p, (rw//2, rh//2), interpolation=cv2.INTER_AREA) * 0.5 - 0.25 for p in (mx, my)]
        for plane in reference[1:]:
            result.append(cv2.remap(plane, *chroma_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        return result, confidence

    def apply(self, frame_index, neighbors):
        source = neighbors[frame_index]
        roi = logo_rect(self.matte.profile, self.track, frame_index)
        if roi is None:
            return source.copy()
        x, y, w, h = roi
        x0, y0 = max(0, x-24), max(0, y-24)
        context = (x0, y0, min(self.video.width, x+w+24)-x0, min(self.video.height, y+h+24)-y0)
        parts = plane_rects(self.video.planes(source), context)
        safe = 1 - self.mask(frame_index, roi, self.repair_shapes[0])
        shot = np.searchsorted(self.cuts, frame_index, side='right') - 1
        lo, hi = self.cuts[shot:shot+2]
        predictions = []
        for j in (max(lo, frame_index-5), min(hi-1, frame_index+5)):
            if j == frame_index or j not in neighbors:
                continue
            reference = plane_rects(self.video.planes(neighbors[j]), context)
            predictions.append(self.predict(parts, reference, j, context, roi, safe))
        out = source.copy()
        dest = plane_rects(self.video.planes(out), roi)
        for channel, plane in enumerate(dest):
            scale = 1 if channel == 0 else 2
            edge = self.mask(frame_index, roi, self.matte.edge_shapes[channel], scale)
            repair = self.mask(frame_index, roi, self.repair_shapes[channel], scale)
            if not predictions and not self.flat_fallback:
                plane[:] = repair_ringing(plane, edge, channel)
                continue
            if predictions:
                weights = [conf if scale == 1 else cv2.resize(conf, (w//2, h//2), interpolation=cv2.INTER_AREA)
                           for _, conf in predictions]
                total = sum(weights)
                prediction = sum(planes[channel] * wt for (planes, _), wt in zip(predictions, weights, strict=True))
                target = prediction / np.maximum(total, 0.01)
                floor, span = (0.26, 0.70) if self.conservative else (0.16, 0.58)
                trusted = np.clip((total - floor) / span, 0, 1) * repair
                limits = (15, 11, 11) if self.conservative else (25, 17, 17)
                limit = limits[channel]
                delta = np.clip(target - plane, -limit, limit)
                temporal = np.clip(np.rint(plane + delta * trusted), 0, 255).astype(np.uint8)
            else:
                trusted, temporal = 0, plane
            # Whole strokes can be interpolated in flat areas. Across hair, eyes or
            # subtitles, restrict the spatial fallback to the codec-ringing edges.
            stable = (1 - repair).astype(np.float32)
            sample = temporal.astype(np.float32)
            kernel = (25, 25) if channel == 0 else (13, 13)
            mass = cv2.boxFilter(stable, -1, kernel)
            safe_mass = np.maximum(mass, 0.1)
            mean = cv2.boxFilter(sample * stable, -1, kernel) / safe_mass
            variance = cv2.boxFilter(sample**2 * stable, -1, kernel) / safe_mass - mean**2
            flat = (mass > 0.20) & (variance < (32 if channel == 0 else 16))
            fallback = repair * flat
            if not self.flat_fallback:
                fallback |= edge
            fallback = (fallback * (trusted < 0.68)).astype(np.uint8)
            plane[:] = repair_ringing(temporal, fallback, channel) if fallback.any() else temporal
        if self.smooth_edges is not None:
            edge = self.mask(frame_index, roi, self.smooth_edges).astype(np.float32)
            strength = cv2.GaussianBlur(edge, (5, 5), .95)
            image = yuv_to_rgb(dest, self.video.matrix)
            cleaned = polish_texture(image, strength*1.48)
            composite(dest, np.rint(cleaned).clip(0, 255).astype(np.uint8),
                      (strength > 0).astype(np.float32), self.video.matrix)
        if self.rim:
            self.rim.restore(dest, self.matte.shifts[frame_index])
        return out


def temporal_frames(video, matte, track, start, stop, cuda, rim=False, caption_band=None, caption_context=False,
                    conservative=False):
    radius, step = 5, 18
    refiner = TemporalEdges(video, matte, track, rim, conservative)
    captions = None
    if caption_band is not None:
        from wm_caption import CAPTION_RADIUS, CaptionReferences
        captions = CaptionReferences(video, matte, caption_band, refiner, caption_context)
        radius = CAPTION_RADIUS
    frames = iter(read_frames(video, max(0, start-radius), min(video.count, stop+radius), cuda))
    fields = matte.fields[0].cpu().numpy() if refiner.rim else None
    neighbors = {}
    i = start
    while batch := list(itertools.islice(frames, step)):
        if refiner.rim:
            for index, frame in batch:
                rect = logo_rect(matte.profile, track, index)
                if rect:
                    patch = plane_rects(video.planes(frame), rect)[0]
                    matte.shifts[index] = refiner.rim.align(patch, matte.shifts[index], fields)
        matte.restore(video, [f for _, f in batch], batch[0][0], spatial=False)
        neighbors.update(batch)
        last = batch[-1][0]
        end = min(stop, last-radius+1 if last < stop+radius-1 and last < video.count-1 else last+1)
        while i < end:
            restored = refiner.apply(i, neighbors)
            if captions:
                shot = np.searchsorted(refiner.cuts, i, side='right') - 1
                captions.apply(i, restored, neighbors, *refiner.cuts[shot:shot+2])
            yield i, restored
            i += 1
        for old in tuple(neighbors):
            if old < i-radius:
                del neighbors[old]
        if captions:
            captions.clean_old(i-radius)
    if i != stop:
        raise RuntimeError(f'Temporal cleanup stopped at frame {i}, expected {stop}.')
