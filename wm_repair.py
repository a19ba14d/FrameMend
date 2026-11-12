import json
import os
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from wm_inpaint import ProPainter, composite, corner_jobs, feather, inpaint_roi
from wm_media import digest, plane_rects, read_frames, yuv_to_rgb


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    out_start: int
    out_stop: int
    in_start: int
    in_stop: int

    def weight(self, index):
        return min(1.0, (index - self.out_start + 1) / (self.start - self.out_start + 1),
                   (self.out_stop - index) / (self.out_stop - self.stop + 1))


def windows(count, cuts, core, overlap):
    edges = [0, *map(int, cuts), count]
    for lo, hi in zip(edges, edges[1:]):
        for start in range(lo, hi, core):
            stop = min(start + core, hi)
            yield Window(start, stop, max(lo, start-overlap), min(hi, stop+overlap),
                         max(lo, start-2*overlap), min(hi, stop+2*overlap))


def save_array(path, array):
    temp = path.with_name(f'{path.name}.{os.getpid()}.part')
    try:
        with temp.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def dilate(mask, radius):
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))


class CornerJob:
    def __init__(self, rect, base, active, dilation, method='model'):
        self.rect, self.active, self.method = rect, active, method
        self.mask = dilate(base, dilation)
        y, x = np.where(self.mask)
        x0, y0 = max(0, x.min()-4) // 2 * 2, max(0, y.min()-4) // 2 * 2
        x1, y1 = min(rect[2], (x.max()+6)//2*2), min(rect[3], (y.max()+6)//2*2)
        self.slice = np.s_[y0:y1, x0:x1]
        self.patch_rect = (rect[0]+x0, rect[1]+y0, x1-x0, y1-y0)
        self.alpha = feather(self.mask, max(1, dilation - 2))[self.slice].copy()
        self.ring = (self.mask * (1 - dilate(base, 3))).astype(np.float32)[self.slice].copy()
        outside = dilate(self.mask, 20) & (1 - dilate(self.mask, 4))
        rows, columns = np.where(outside)
        self.dark_samples = [(rows[::4]//s, columns[::4]//s) for s in (1, 2, 2)]
        if method == 'extend-top':
            if rect[1] != 0 or not base[0].any() or self.mask[-2:].any():
                raise ValueError('extend-top needs a top-edge mask and unmasked rows below it.')
            rows, columns = np.indices(self.mask.shape, dtype=np.float32)
            boundary = (self.mask * rows).max(axis=0) + 1
            # Reflect a short band of visible structure rather than repeating a single scanline.
            self.map_x = columns[self.slice].copy()
            self.map_y = (boundary + np.maximum(boundary - rows, 0) * .30)[self.slice].copy()

    def extend_top(self, planes, matrix):
        rgb = yuv_to_rgb(plane_rects(planes, self.rect), matrix)
        return cv2.remap(rgb, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def dark_plate(self, planes, matrix):
        context = plane_rects(planes, self.rect)
        luma = context[0][self.dark_samples[0]]
        if len(luma) < 80 or luma.mean() > 36 or np.ptp(luma) > 26:
            return None
        low, high = np.percentile(luma, (5, 95))
        if high > 42 or high-low > 11:
            return None
        plate = tuple(np.full((2//s, 2//s), round(float(np.median(plane[coords]))), np.uint8)
                      for plane, coords, s in zip(context, self.dark_samples, (1, 2, 2), strict=True))
        color = yuv_to_rgb(plate, matrix)[0, 0]
        return np.broadcast_to(color, (*self.alpha.shape, 3))


class Corners:
    @classmethod
    def from_cache(cls, video, profile, track, directory, matte_path):
        state = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        options = {key: state[key] for key in ('scale', 'core', 'overlap', 'dilation', 'raft_iterations')}
        options['context_matte_path'] = matte_path if state.get('context_matte_sha256') else None
        options['model_dir'] = Path(state['model_dir']) if state.get('model_dir') else None
        return cls(video, profile, track, directory, phases=bool(state.get('phase_detail')), **options)

    def __init__(self, video, profile, track, directory, *, scale=0.625, core=56, overlap=6, dilation=14,
                 raft_iterations=6, context_matte_path=None, model_dir=None, phases=False):
        if core < 2 * overlap or overlap < 1 or not 0.25 <= scale <= 1 or dilation < 5 or raft_iterations < 1:
            raise ValueError('Invalid corner window, scale, dilation or flow iterations.')
        self.video, self.directory = video, Path(directory)
        jobs = corner_jobs(video, profile, track)
        methods = profile['corners'].get('methods', ['model'] * len(jobs))
        if not isinstance(methods, list) or len(methods) != len(jobs) or any(
                method not in ('model', 'extend-top') for method in methods):
            raise ValueError('Choose one method per corner region: model or extend-top.')
        self.jobs = [CornerJob(*job, dilation, method) for job, method in zip(jobs, methods, strict=True)]
        self.windows = list(windows(video.count, track['cuts'], core, overlap))
        self.window_starts = np.asarray([w.start for w in self.windows])
        self.scale, self.raft_iterations, self.model_dir = scale, raft_iterations, model_dir
        self.phases = phases
        self.dark_fallback_frames = []
        self.directory.mkdir(parents=True, exist_ok=True)
        info = dict(version=1, source_sha256=str(track['source_sha256']), profile_sha256=str(track['profile_sha256']),
                    cuts=track['cuts'].tolist(), scale=scale, core=core, overlap=overlap, dilation=dilation,
                    raft_iterations=raft_iterations, context_matte_sha256=digest(context_matte_path) if context_matte_path else None,
                    model_dir=str(model_dir.resolve()) if model_dir else None)
        if phases:
            info['phase_detail'] = 1
        manifest = self.directory / 'manifest.json'
        if not manifest.exists():
            try:
                with manifest.open('x', encoding='utf-8') as stream:
                    json.dump(info, stream, indent=2)
            except FileExistsError:
                pass
        if json.loads(manifest.read_text(encoding='utf-8')) != info:
            raise ValueError('Corner cache settings differ. Use a new work folder.')

    def _path(self, window, side):
        return self.directory / f'{window.start:07d}-{window.stop:07d}-{side}.npy'

    def pending(self, start, stop):
        for window in self.windows:
            if window.out_stop <= start or window.out_start >= stop:
                continue
            missing = [(side, job) for side, job in enumerate(self.jobs)
                       if job.method == 'model' and job.active[window.out_start:window.out_stop].any()
                       and not self._path(window, side).exists()]
            if missing:
                yield window, missing

    def decoded_windows(self, pending, matte, cuda):
        # Group touching reads and keep only the previous window's overlapping frames.
        boundaries = [0]
        for i in range(1, len(pending)):
            if pending[i][0].in_start > pending[i-1][0].in_stop:
                boundaries.append(i)
        boundaries.append(len(pending))
        for lo, hi in zip(boundaries, boundaries[1:]):
            if lo == hi:
                continue
            first = end = pending[lo][0].in_start
            items = []
            with closing(read_frames(self.video, first, pending[hi-1][0].in_stop, cuda)) as frames:
                for window, jobs in pending[lo:hi]:
                    fresh = [frame for _, frame in islice(frames, window.in_stop-end)]
                    if matte and fresh:
                        matte.restore(self.video, fresh, end)
                    items = items[window.in_start-first:] + fresh
                    first, end = window.in_start, window.in_stop
                    yield window, jobs, items
                # Let the decoder check FFmpeg's exit status after its last frame.
                next(frames, None)

    def prepare(self, start, stop, matte, cuda=True):
        model = None
        processed = 0
        pending = list(self.pending(start, stop))
        for window, missing, items in self.decoded_windows(pending, matte, cuda):
            if model is None:
                kwargs = {'model_dir': self.model_dir} if self.model_dir else {}
                model = ProPainter(scale=self.scale, raft_iterations=self.raft_iterations, **kwargs)
            for side, job in missing:
                active = job.active[window.in_start:window.in_stop]
                masks = job.mask[None] * active[:, None, None]
                direction = None
                if self.phases:
                    x, y, width, height = job.rect
                    direction = (1 if 2*y+height <= self.video.height else -1,
                                 1 if 2*x+width <= self.video.width else -1)
                _, predictions = inpaint_roi(self.video, model, items, job.rect, masks, direction)
                lo, hi = window.out_start-window.in_start, window.out_stop-window.in_start
                patch = predictions[lo:hi, job.slice[0], job.slice[1]].copy()
                save_array(self._path(window, side), patch)
            processed += window.stop - window.start
            if processed >= 240:
                print(f'Corner repair cached through frame {window.stop}/{self.video.count}', flush=True)
                processed = 0
        print('Corner cache ready.', flush=True)

    @lru_cache(maxsize=12)
    def _load(self, path):
        return np.load(path, allow_pickle=False, mmap_mode='r')

    def restore(self, index, frame):
        if not self.jobs:
            return
        mid = np.searchsorted(self.window_starts, index, side='right') - 1
        candidates = self.windows[max(0, mid-1):mid+2]
        planes = self.video.planes(frame)
        for side, job in enumerate(self.jobs):
            if not job.active[index]:
                continue
            original = plane_rects(planes, job.patch_rect)
            dark = job.dark_plate(planes, self.video.matrix) if job.method == 'model' else None
            if dark is not None:
                composite(original, dark, job.alpha, self.video.matrix)
                self.dark_fallback_frames.append(index)
                continue
            if job.method == 'extend-top':
                result = job.extend_top(planes, self.video.matrix)
            else:
                patches = [(self._load(self._path(w, side))[index-w.out_start], w.weight(index)) for w in candidates
                           if w.out_start <= index < w.out_stop]
                if len(patches) == 1:
                    result = patches[0][0]
                else:
                    result = sum(p.astype(np.float32) * weight for p, weight in patches) / sum(w for _, w in patches)
            composite(original, result, job.alpha, self.video.matrix, job.ring)
