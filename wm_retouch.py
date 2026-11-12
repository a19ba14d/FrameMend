import json
from pathlib import Path

import cv2
import numpy as np

from wm_matte import logo_rect
from wm_media import plane_rects
from wm_track import check_rect
from wm_edge_retouch import fit_rising_edge, repair_along_edge


class Retouch:
    def __init__(self, video, config, matte=None):
        self.video = video
        self.matte = matte
        self.by_frame = {}
        self.stripes = {}
        self.aligned_edges = {}
        self.white_plates = set()
        entries = json.loads(Path(config).read_text(encoding='utf-8-sig'))
        for entry in entries:
            start, stop = entry['frames']
            if not 0 <= start < stop <= video.count:
                raise ValueError(f'Invalid retouch interval {start}:{stop}.')
            mode = entry['mode']
            if mode == 'white-plate':
                if matte is None or not matte.edge_shapes:
                    raise ValueError('white-plate retouch requires a translucent matte with edge masks.')
                self.white_plates.update(range(start, stop))
                continue
            if mode in ('stripe', 'aligned-edge'):
                if matte is None or not matte.edge_shapes:
                    raise ValueError(f'{mode} retouch requires a translucent matte with edge masks.')
                lo, hi = entry['rows']
                if not 4 <= lo < hi <= matte.height-5 or hi-lo < 12:
                    raise ValueError('Invalid stripe sample rows.')
                for i in range(start, stop):
                    if mode == 'stripe':
                        self.stripes[i] = (lo, hi, entry.get('min_luma', 160))
                    else:
                        self.aligned_edges[i] = (lo, hi)
                continue
            if mode not in ('dark', 'cyan'):
                raise ValueError(f'Unknown retouch mode {mode}.')
            regions = [check_rect(rect, video.width, video.height) for rect in entry['regions']]
            for i in range(start, stop):
                self.by_frame.setdefault(i, []).extend((mode, rect) for rect in regions)

    def apply(self, index, frame):
        if index in self.stripes:
            self.apply_stripe(index, frame, self.stripes[index])
        if index in self.aligned_edges:
            self.apply_edge(index, frame, self.aligned_edges[index])
        if index in self.white_plates:
            self.apply_white_plate(index, frame)
        entries = self.by_frame.get(index)
        if not entries:
            return
        planes = self.video.planes(frame)
        for mode, rect in entries:
            y, u, v = plane_rects(planes, rect)
            if mode == 'dark':
                blackhat = cv2.morphologyEx(y, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
                seeds = ((y < 85) & (blackhat > 24)).astype(np.uint8)
            else:
                cyan = ((u > 133) & (v < 107)).astype(np.uint8)
                seeds = cv2.resize(cyan, y.shape[::-1], interpolation=cv2.INTER_NEAREST)
            if not seeds.any():
                continue
            diameter = 17 if mode == 'cyan' else 9
            mask = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
            chroma = cv2.resize(mask, u.shape[::-1], interpolation=cv2.INTER_NEAREST)
            for channel, support in ((y, mask), (u, chroma), (v, chroma)):
                channel[:] = cv2.inpaint(channel, support, 3, cv2.INPAINT_NS)

    def support(self, index, plane, channel):
        scale = 1 if channel == 0 else 2
        dx, dy = self.matte.shifts[index]
        shape = np.maximum(self.matte.foreground_shape, self.matte.edge_shapes[channel])
        transform = np.float32([[1/scale, 0, dx/scale], [0, 1/scale, dy/scale]])
        return cv2.warpAffine(shape, transform, plane.shape[::-1], flags=cv2.INTER_NEAREST)

    def apply_white_plate(self, index, frame):
        rect = logo_rect(self.matte.profile, self.matte.track, index)
        if rect is None:
            return
        y, u, v = plane_rects(self.video.planes(frame), rect)
        # Leave a safety margin around ink and the edge of the painted card.
        interior = cv2.erode((y >= 207).astype(np.uint8), np.ones((7, 7), np.uint8))
        support = self.support(index, y, 0)
        clear = (interior > 0) & (support == 0) & (y >= 229)
        if clear.sum() < 32:
            return
        guards = (interior > 0, cv2.resize(interior.astype(np.float32), u.shape[::-1],
                                          interpolation=cv2.INTER_AREA) == 1)
        clear_chroma = cv2.resize(clear.astype(np.float32), u.shape[::-1], interpolation=cv2.INTER_AREA) == 1
        for channel, plane in enumerate((y, u, v)):
            reference = clear if channel == 0 else clear_chroma
            if reference.sum() < 8:
                continue
            value = float(np.median(plane[reference]))
            active = guards[int(channel > 0)] & (self.support(index, plane, channel) > 0)
            delta = value - plane.astype(np.float32)
            blend = np.clip(1 - (np.abs(delta) - 9) / 19, 0, 1) * active
            plane[:] = np.rint(plane + blend * delta).astype(np.uint8)

    def apply_edge(self, index, frame, rows):
        rect = logo_rect(self.matte.profile, self.matte.track, index)
        if rect is None:
            return
        x, y, width, height = rect
        x0, y0 = max(0, x-110), max(0, y-30)
        context = (x0, y0, x+width-x0, min(self.video.height, y+height+30)-y0)
        planes = self.video.planes(frame)
        surroundings = plane_rects(planes, context)
        curve = fit_rising_edge(surroundings[0], y-y0+rows[0]-3, y-y0+rows[1]+17)
        if curve is None:
            return
        for j, (dst, src) in enumerate(zip(plane_rects(planes, rect), surroundings, strict=True)):
            scale = 1 if j == 0 else 2
            repair_along_edge(dst, src, self.support(index, dst, j),
                              (x-x0)//scale, (y-y0)//scale, curve, rows, scale)

    def apply_stripe(self, index, frame, parameters):
        rect = logo_rect(self.matte.profile, self.matte.track, index)
        if rect is None:
            return
        lo, hi, min_luma = parameters
        dy = self.matte.shifts[index, 1]
        planes = plane_rects(self.video.planes(frame), rect)
        ylo, yhi = (round(r + float(dy)) for r in (lo, hi))
        light = np.minimum(*[np.median(planes[0][r-1:r+2], axis=0) for r in (ylo, yhi)])
        light = np.clip((light-min_luma)/18, 0, 1).astype(np.float32)[None]
        for j, plane in enumerate(planes):
            scale = 1 if j == 0 else 2
            support = self.support(index, plane, j)
            brightness = light if scale == 1 else cv2.resize(light, (plane.shape[1], 1), interpolation=cv2.INTER_AREA)
            low, high = (round((r + float(dy))/scale) for r in (lo, hi))
            retouch_stripe(plane, support, low, high, brightness, j)


def retouch_stripe(plane, support, low, high, brightness, channel):
    raw = plane.astype(np.float32)
    upper, lower = raw[low-1:low+2], raw[high-1:high+2]
    top, bottom = np.median(upper, axis=0), np.median(lower, axis=0)
    spread = np.abs(top-bottom) + np.ptp(upper, axis=0) + np.ptp(lower, axis=0)
    row = np.arange(len(plane), dtype=np.float32)[:, None]
    estimate = top + (bottom-top) * np.clip((row-low)/(high-low), 0, 1)
    agreement = np.clip(1-(spread-(4 if channel == 0 else 2))/(16 if channel == 0 else 9), 0, 1)
    support = cv2.GaussianBlur(cv2.dilate(support.astype(np.float32), np.ones((5, 5), np.uint8)), (5, 5), 1.2)
    feather = np.clip(np.minimum(row-low, high-row)/(4 if channel == 0 else 2), 0, 1)
    delta = estimate-raw
    trust = np.clip(1-(np.abs(delta)-5)/(17 if channel == 0 else 9), 0, 1)
    plane[:] = np.rint(raw + delta * support * feather * agreement * trust * brightness).astype(np.uint8)
