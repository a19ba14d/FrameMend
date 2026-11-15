"""Remove recurring watermarks from a video locally, without a vision-language model.

The planner reads on-screen text with local OCR, keeps text that recurs at a fixed
position across shots (corner badges) or recurs with the same wording at changing
positions (jumping watermarks), learns each mark's appearance from many frames,
tracks it through the whole video and hands the result to the existing scan,
render, verify and audit stages. A second OCR pass over the restored video decides
whether to accept the result or retry with a wider mask.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import faulthandler
from functools import cached_property, partial
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import select
import shutil
import struct
import subprocess
import sys
import time
from time import monotonic, perf_counter
import unicodedata

import cv2
import numpy as np

from wm_discovery import Motif, consensus_fields, pack_tracks, patch, peaks, response
from wm_media import (TOOLS, Video, clipped_plane, digest, encode, executable, get_frame, read_frames, read_selected,
                      yuv_to_rgb)
from wm_track import highpass, save_npz
from workflow import build_plan, execute_plan, fingerprints, input_files, lock_job, read_object, write_object


OCR_DIRS = (TOOLS/'vision'/'ocr', TOOLS/'detection'/'python')
HIGHPASS_SIGMA = 2.2
OCR_MAX_SIDE = 1280
CUT_THRESHOLD = 30.
CUT_MINIMUM = 12.


# ----------------------------------------------------------------------------
# Local OCR

def text_key(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if c.isalnum() or c in '.@')


def similar(a, b):
    if not a or not b:
        return 0.
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


class OCR:
    def __init__(self, threads=None):
        for path in OCR_DIRS:
            if path.is_dir() and str(path) not in sys.path:
                sys.path.append(str(path))
        from rapidocr import RapidOCR

        threads = threads or max(2, min(8, (os.cpu_count() or 4)//4))
        self.engine = RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads': threads,
                                       'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                                       'Global.use_cls': False, 'Global.log_level': 'error'})

    def read(self, bgr, minimum=.5):
        """Return [(key, text, (x, y, w, h), score, line height)] in the coordinates of ``bgr``.

        The line height is the box height for level text; for text set at a slant (a diagonal watermark) the box
        around it is far taller than its letters, and the height is that of the slanted line instead.
        """
        rows = []
        for key, text, quad, score in self.read_quads(bgr, minimum):
            x, y, w, h = cv2.boundingRect(quad)
            if w >= 6 and h >= 6:
                rows.append((key, text, (int(x), int(y), int(w), int(h)), score, line_height(quad, h)))
        return rows

    def read_quads(self, bgr, minimum=.5):
        """Return [(key, text, 4x2 corner array, score)] in the coordinates of ``bgr``."""
        height, width = bgr.shape[:2]
        scale = min(1., OCR_MAX_SIDE/max(height, width))
        image = bgr if scale == 1 else cv2.resize(bgr, (round(width*scale), round(height*scale)),
                                                  interpolation=cv2.INTER_AREA)
        result = self.engine(image, use_cls=False, text_score=minimum)
        if result.boxes is None:
            return []
        rows = []
        for quad, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            key = text_key(text)
            if key:
                rows.append((key, text, (np.asarray(quad, np.float32)/scale).astype(np.float32), float(score)))
        return rows


OCR_WORKER_TRIES = 3    # workers a request is given: one that dies or fails is replaced and the request made again
OCR_WORKER_WAIT = 600.  # seconds a request may take before its worker is taken for hung


class OCRWorker:
    """OCR's interface, with the engine in a process of its own. The engine's native code (onnxruntime and the
    threads it keeps) now and then corrupted the interpreter it ran in - objects freed while still in use, which
    surfaced later as errors in plain Python code (difflib comparing readings) or brought the process down. In a
    process of its own it cannot touch the job's state: a worker that dies, hangs or fails a request is replaced by
    a fresh one and the request made again (OCR gives the same answer for the same picture)."""

    def __init__(self, threads=None):
        self.threads, self.process = threads, None

    def start(self):
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--ocr-worker']
        if self.threads:
            command.append(str(self.threads))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()   # the worker's cue to finish
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
        process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, method, *args):
        request = pickle.dumps((method, args), protocol=pickle.HIGHEST_PROTOCOL)
        failure = None
        for _ in range(OCR_WORKER_TRIES):
            if self.process is None:
                self.start()
            try:
                send_message(self.process.stdin, request)
                status, result = pickle.loads(receive_message(self.process.stdout, OCR_WORKER_WAIT))
            except (OSError, EOFError, pickle.UnpicklingError, struct.error) as error:
                status, result = 'error', f'the worker stopped ({type(error).__name__}: {error})'
            if status == 'ok':
                return result
            failure = result
            self.process.kill()
            self.close()
        raise RuntimeError(f'OCR failed in {OCR_WORKER_TRIES} workers: {failure}')

    def read(self, bgr, minimum=.5):
        return self.call('read', np.ascontiguousarray(bgr), minimum)

    def read_quads(self, bgr, minimum=.5):
        return self.call('read_quads', np.ascontiguousarray(bgr), minimum)


def send_message(stream, payload):
    """Write a length-prefixed message to an unbuffered pipe, all of it."""
    data = memoryview(struct.pack('<Q', len(payload))+payload)
    while data:
        data = data[stream.write(data):]


def receive_message(stream, wait=None):
    """Read a length-prefixed message from an unbuffered pipe; EOFError where it closes first, TimeoutError where
    nothing comes for wait seconds."""
    def exact(size):
        data = bytearray()
        while len(data) < size:
            if wait is not None and not select.select([stream], [], [], wait)[0]:
                raise TimeoutError(f'nothing came for {wait:.0f} seconds')
            chunk = stream.read(min(size-len(data), 1 << 22))
            if not chunk:
                raise EOFError('the pipe closed')
            data += chunk
        return bytes(data)
    size, = struct.unpack('<Q', exact(8))
    return exact(size)


def serve_ocr(threads=None):
    """The worker's side of OCRWorker: requests on standard input, answers on what was standard output. A request
    that fails is answered with the error and the worker ends, as its state may be what failed."""
    answers = os.fdopen(os.dup(1), 'wb', buffering=0)
    os.dup2(2, 1)       # anything else written to standard output (a native library's notes) goes to the log
    sys.stdout = sys.stderr
    requests = os.fdopen(os.dup(0), 'rb', buffering=0)
    ocr = OCR(threads)
    while True:
        try:
            method, args = pickle.loads(receive_message(requests))
        except EOFError:
            return 0
        try:
            answer = ('ok', getattr(ocr, method)(*args))
        except Exception as error:
            answer = ('error', f'{type(error).__name__}: {error}')
        send_message(answers, pickle.dumps(answer, protocol=pickle.HIGHEST_PROTOCOL))
        if answer[0] != 'ok':
            return 1


TILTED = 10.    # degrees off level from which a text line's box height no longer measures its letters


def line_height(quad, height):
    """Height of the text line a detector quad (top-left, top-right, bottom-right, bottom-left) outlines."""
    dx, dy = quad[1]-quad[0]
    tilt = abs(math.degrees(math.atan2(float(dy), float(dx))))
    if min(tilt, 180-tilt) < TILTED:
        return float(height)
    sides = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    return float(min(sides[0]+sides[2], sides[1]+sides[3])/2)


# ----------------------------------------------------------------------------
# Geometry helpers

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax+aw, bx+bw)-max(ax, bx)
    h = min(ay+ah, by+bh)-max(ay, by)
    if w <= 0 or h <= 0:
        return 0.
    inter = w*h
    return inter/(aw*ah+bw*bh-inter)


def overlap_share(a, b):
    """Intersection area relative to the smaller rectangle."""
    w = min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])
    h = min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1])
    return max(0, w)*max(0, h)/max(1, min(a[2]*a[3], b[2]*b[3]))


def union_rect(rects):
    rects = list(rects)
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0]+r[2] for r in rects)
    y1 = max(r[1]+r[3] for r in rects)
    return x0, y0, x1-x0, y1-y0


def center(rect):
    x, y, w, h = rect
    return x+w/2, y+h/2


def median_rect(rects):
    return tuple(int(round(v)) for v in np.median(np.asarray(rects, np.float64), axis=0))


def grow(rect, dx, dy):
    x, y, w, h = rect
    return x-dx, y-dy, w+2*dx, h+2*dy


def runs(mask):
    """Yield (start, stop) of True runs."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    for start, stop in zip(edges[::2], edges[1::2]):
        yield int(start), int(stop)


def fill_gaps(mask, limit):
    mask = np.asarray(mask, bool).copy()
    spans = list(runs(mask))
    for (_, stop), (start, _) in zip(spans, spans[1:]):
        if start-stop <= limit:
            mask[stop:start] = True
    return mask


def widen(mask, frames):
    if frames <= 0:
        return mask
    kernel = np.ones(2*frames+1, np.uint8)
    return np.convolve(mask.astype(np.uint8), kernel, mode='same') > 0


# ----------------------------------------------------------------------------
# Data gathering

@dataclass
class Reading:
    frame: int
    key: str
    text: str
    rect: tuple
    score: float
    shot: int
    line: float = None      # height of the letters, which a slanted line's box overstates

    @property
    def height(self):
        return self.rect[3]

    @property
    def thickness(self):
        return self.line or float(self.rect[3])


@dataclass
class Samples:
    video: Video
    cuts: np.ndarray
    shots: np.ndarray
    frames: list
    raws: dict
    readings: list
    transient: np.ndarray = None   # pixels covered by non-watermark text in two or more samples
    overlay: tuple = None          # (scene frames, median high-pass picture, significance) from overlay_consensus
    scene_keys: list = field(default_factory=list)  # text of marks handed to the scene logos (within_one_shot)

    @cached_property
    def signals(self):
        return {i: highpass(self.video.planes(self.raws[i])[0], HIGHPASS_SIGMA) for i in self.frames}

    def bgr(self, i):
        return yuv_to_rgb(self.video.planes(self.raws[i]), self.video.matrix)[..., ::-1]

    def chroma(self, i, plane):
        key = (i, plane)
        cache = self.__dict__.setdefault('_chroma', {})
        if key not in cache:
            half = self.video.planes(self.raws[i])[plane].astype(np.float32)
            full = cv2.resize(half, (self.video.width, self.video.height), interpolation=cv2.INTER_LINEAR)
            cache[key] = highpass(full, 2*HIGHPASS_SIGMA)
        return cache[key]


def detect_cuts(video, cuda, trace):
    width = 64
    height = max(2, round(width*video.height/video.width/2)*2)
    previous = None
    differences = np.zeros(video.count, np.float32)
    for i, raw in read_frames(video, cuda=cuda, size=(width, height)):
        current = raw[:width*height].reshape(height, width).astype(np.int16)
        if previous is not None:
            differences[i] = np.abs(current-previous).mean()
        previous = current
    # A cut is a large isolated jump; sustained motion raises neighbouring differences as well.
    padded = np.pad(differences, 3, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, 7)
    neighbours = np.median(np.delete(windows, 3, axis=1), axis=1)
    isolated = (differences >= CUT_MINIMUM) & (differences >= 3*np.maximum(neighbours, 1.))
    cuts = np.flatnonzero((differences > CUT_THRESHOLD) | isolated).astype(np.int32)
    cuts = cuts[cuts > 0]
    shots = np.zeros(video.count, np.int32)
    shots[cuts] = 1
    shots = np.cumsum(shots).astype(np.int32)
    trace('plan.cuts', cuts=len(cuts), shots=int(shots[-1])+1)
    return cuts, shots


def gather_samples(video, cuda, ocr, count, trace):
    cuts, shots = detect_cuts(video, cuda, trace)
    step = max(1, video.count//count)
    frames, raws, readings = [], {}, []
    started = perf_counter()
    for i, raw in read_frames(video, cuda=cuda, step=step):
        frames.append(i)
        raws[i] = raw
        bgr = yuv_to_rgb(video.planes(raw), video.matrix)[..., ::-1].copy()
        for key, text, rect, score, line in ocr.read(bgr):
            readings.append(Reading(i, key, text, rect, score, int(shots[i]), line))
    trace('plan.ocr', frames=len(frames), readings=len(readings), seconds=round(perf_counter()-started, 2))
    return Samples(video, cuts, shots, frames, raws, readings)


# ----------------------------------------------------------------------------
# Candidate grouping

@dataclass(eq=False)
class Group:
    readings: list = field(default_factory=list)
    parent: 'Group' = None     # for another state of a badge: the group of the badge's usual state

    @property
    def key(self):
        keys = [r.key for r in self.readings]
        return max(keys, key=lambda k: (sum(similar(k, other) for other in keys), len(k)))

    @property
    def rect(self):
        return median_rect([r.rect for r in self.readings])

    @property
    def frames(self):
        return sorted({r.frame for r in self.readings})

    @property
    def shots(self):
        return {r.shot for r in self.readings}

    @property
    def height(self):
        return float(np.median([r.rect[3] for r in self.readings]))

    @property
    def thickness(self):
        """Height of the letters: the box height unless the line is slanted."""
        return float(np.median([r.thickness for r in self.readings]))

    @property
    def tilted(self):
        return self.thickness < .67*self.height


def recurring(group, sampled, share, shots):
    """Seen in several shots, or for a large part of the video."""
    frames = group.frames
    covered = (frames[-1]-frames[0])/max(1, sampled[-1]-sampled[0])
    return len(group.shots) >= 2 or covered >= share


VARIANT_HELD = .6      # share of another state's sampled frames next to another sample of that state
VARIANT_SPREAD = 4.    # another state's width differs by this many times the readings' width deviation
VARIANT_SHIFT = 2.     # or its place by this many times the readings' position deviation


def static_groups(readings, sampled, shots):
    sample_count = len(sampled)
    groups = []
    for reading in sorted(readings, key=lambda r: (r.frame, r.rect)):
        best, best_overlap = None, .5
        for group in groups:
            overlap = iou(reading.rect, group.rect)
            if overlap >= best_overlap and similar(reading.key, group.key) >= .5 and not rescaled(reading, group):
                best, best_overlap = group, overlap
        if best is None:
            groups.append(Group([reading]))
        else:
            best.readings.append(reading)

    def steady(group):
        rects = np.asarray([r.rect for r in group.readings], np.float64)
        jitter = np.median(np.abs(rects[:, :2]-np.median(rects[:, :2], axis=0)), axis=0)
        if jitter.max() > max(3., .15*group.height):
            return False
        key = group.key
        return np.mean([similar(r.key, key) >= .6 for r in group.readings]) >= .7
    accepted = []
    minimum = max(3, math.ceil(.15*sample_count))
    for group in groups:
        if len(group.frames) >= minimum and recurring(group, sampled, .5, shots) and steady(group):
            accepted.append(group)
    # A badge shown for a while at another size (smaller during the opening, say) or a little aside reads as the
    # same words next to where it usually is; that state is part of the watermark however briefly it is shown.
    # Such a state holds for a stretch of samples; readings at another width scattered among the usual state's
    # are the same badge read with or without its icon.
    order = {frame: k for k, frame in enumerate(sampled)}

    def stretch(group):
        frames = set(group.frames)
        held = [f for f in frames if f in order and ({sampled[k] for k in (order[f]-1, order[f]+1)
                                                       if 0 <= k < sample_count} & frames)]
        return len(held) >= VARIANT_HELD*len(frames)
    variants = []
    for group in groups:
        if any(group is g for g in accepted) or len(group.frames) < 3 or not steady(group) or not stretch(group):
            continue
        for other in accepted:
            pad = other.thickness
            if (similar(group.key, other.key) >= .9 and distinct(group, other)
                    and len(set(group.frames) & set(other.frames)) <= max(1, .2*len(group.frames))
                    and iou(grow(group.rect, pad, pad), grow(other.rect, pad, pad)) > 0):
                group.parent = other
                variants.append(group)
                break
    return accepted+variants


def deviation(values):
    """Median absolute deviation of numbers, or the length of the per-axis deviations of points."""
    values = np.asarray(values, np.float64)
    return float(np.linalg.norm(np.atleast_1d(np.median(np.abs(values-np.median(values, axis=0)), axis=0))))


def distinct(group, other):
    """Another size or place than the other group's, by more than readings of either vary from frame to frame
    (OCR reads one badge with or without its icon, a box a few pixels wider or narrower)."""
    widths = [[r.rect[2] for r in g.readings] for g in (group, other)]
    centres = [[center(r.rect) for r in g.readings] for g in (group, other)]
    gap = abs(np.median(widths[0])-np.median(widths[1]))
    shift = math.dist(np.median(centres[0], axis=0), np.median(centres[1], axis=0))
    resized = gap >= max(.1*np.median(widths[1]), VARIANT_SPREAD*max(1., deviation(widths[0])+deviation(widths[1])))
    moved = shift >= max(.5*other.thickness,
                         VARIANT_SHIFT*max(1., deviation(centres[0])+deviation(centres[1])))
    return resized or moved


def rescaled(reading, group):
    """The group's words shown at another size: the badge in another state rather than a reading of this one."""
    if reading.thickness < .67*reading.height or group.tilted:
        return False    # the box around slanted lettering varies with the slant read, not with the badge's size
    width = group.rect[2]
    return similar(reading.key, group.key) >= .9 and abs(reading.rect[2]-width) > max(8, .1*width)


def moving_groups(readings, sampled, shots):
    sample_count = len(sampled)
    families = []
    for reading in sorted(readings, key=lambda r: -r.score):
        for group in families:
            h = group.height
            w = float(np.median([r.rect[2] for r in group.readings]))
            if (similar(reading.key, group.key) >= .65 and abs(reading.rect[3]-h) <= .3*h
                    and abs(reading.rect[2]-w) <= .35*w):
                group.readings.append(reading)
                break
        else:
            families.append(Group([reading]))
    accepted = []
    minimum = max(3, math.ceil(.12*sample_count))
    # A web address that keeps coming back at different places is a watermark even when it is only
    # on screen for a few seconds at a time, such as a marquee crossing the picture now and then.
    brief = max(3, math.ceil(.05*sample_count))
    for group in families:
        needed = brief if BRAND.search(group.key) else minimum
        if len(group.frames) < needed or not recurring(group, sampled, .6, shots) or len(group.key) < 2:
            continue
        positions = []
        for r in group.readings:
            c = center(r.rect)
            if all(math.dist(c, p) > .6*group.height for p in positions):
                positions.append(c)
        if len(positions) < 2:
            continue
        # One reading per frame: keep the most confident.
        best = {}
        for r in group.readings:
            if r.frame not in best or r.score > best[r.frame].score:
                best[r.frame] = r
        group.readings = sorted(best.values(), key=lambda r: r.frame)
        accepted.append(group)
    return sorted(accepted, key=lambda g: -len(g.frames))


def follows_scene(samples, group):
    """Count sample pairs where the text moves together with the picture around it.

    Text painted into the scene moves with the camera; an overlay moves on its own or not at all.
    Returns (pairs that follow the background, pairs where the text moved).
    """
    lh = group.height
    readings = sorted(group.readings, key=lambda r: r.frame)
    follows = informative = 0
    for a, b in zip(readings, readings[1:]):
        if a.shot != b.shot:
            continue
        ca, cb = center(a.rect), center(b.rect)
        moved = np.subtract(cb, ca)
        if np.hypot(*moved) < 4:
            continue
        informative += 1
        # Median optical flow of the picture around the first placement, excluding both text boxes.
        region = grow(union_rect([a.rect, b.rect]), int(2*lh), int(2*lh))
        first, second = (patch(samples.video.planes(samples.raws[r.frame])[0], region) for r in (a, b))
        flow = cv2.calcOpticalFlowFarneback(first, second, None, .5, 4, 21, 3, 5, 1.1, 0)
        background = np.ones(first.shape, bool)
        for other in (a, b):
            x, y, w, h = grow(other.rect, 4, 4)
            background[max(0, y-region[1]):max(0, y+h-region[1]), max(0, x-region[0]):max(0, x+w-region[0])] = False
        textured = np.abs(cv2.Laplacian(first, cv2.CV_32F)) > 4
        usable = background & textured
        if usable.sum() < 200:
            informative -= 1
            continue
        dx, dy = np.median(flow[usable], axis=0)
        if np.hypot(dx-moved[0], dy-moved[1]) <= max(3., .35*np.hypot(*moved)):
            follows += 1
    return follows, informative


BRAND = re.compile(r'(https?|www\.|@|[a-z0-9]{2,}\.(com|cn|net|ai|top|cc|tv|vip|xyz|fun|org|app|io|me|co|info|site|'
                   r'club|live|shop|pro|icu|in|la|ink|video|link))')


def brand_like(key, static_keys):
    return bool(BRAND.search(key)) or any(similar(key, other) >= .6 for other in static_keys)


@dataclass
class Mark:
    kind: str                 # 'static', 'moving' or 'faint'
    label: str
    keys: list
    line_height: float
    observations: list        # [(frame, (x, y, w, h))] union rect of the text lines
    lines: list               # [(dx, dy, w, h)] text lines relative to the observation rect
    parent: str = None        # another state of the badge with this label, shown at another size
    tilted: bool = False      # set at a slant: its lines' boxes are mostly picture around the letters


def merge_static(groups):
    marks = []
    owner = {}      # id(group) -> label of the mark it went into
    pending = list(groups)
    while pending:
        cluster = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for group in list(pending):
                for member in cluster:
                    pad = .8*min(group.height, member.height)
                    # Lines of one badge appear together; alternating badge states stay separate layers.
                    shared = len(set(group.frames) & set(member.frames))
                    together = shared >= .6*min(len(group.frames), len(member.frames))
                    if together and iou(grow(group.rect, pad, pad), grow(member.rect, pad, pad)) > 0:
                        cluster.append(group)
                        pending.remove(group)
                        changed = True
                        break
        rect = union_rect(g.rect for g in cluster)
        frames = sorted({i for g in cluster for i in g.frames})
        lines = [(g.rect[0]-rect[0], g.rect[1]-rect[1], g.rect[2], g.rect[3]) for g in cluster]
        keys = [g.key for g in sorted(cluster, key=lambda g: g.rect[1])]
        label = ' '.join(keys)
        # Labels name layers across attempts, so another state of the same badge needs a name of its own.
        taken = {m.label for m in marks}
        if label in taken:
            label = next(f'{label} #{n}' for n in range(2, len(marks)+2) if f'{label} #{n}' not in taken)
        parent = next((owner[id(g.parent)] for g in cluster if g.parent is not None and id(g.parent) in owner), None)
        marks.append(Mark('static', label, keys, float(np.median([g.thickness for g in cluster])),
                          [(i, rect) for i in frames], lines, parent, any(g.tilted for g in cluster)))
        owner.update((id(g), label) for g in cluster)
    return marks


def merge_moving(groups):
    marks = []
    used = set()
    for a_index, a in enumerate(groups):
        if a_index in used:
            continue
        used.add(a_index)
        members = [(a, (0., 0.))]
        a_by_frame = {r.frame: r for r in a.readings}
        for b_index, b in enumerate(groups):
            if b_index in used:
                continue
            offsets = []
            for r in b.readings:
                if r.frame in a_by_frame:
                    ca, cb = center(a_by_frame[r.frame].rect), center(r.rect)
                    offsets.append((cb[0]-ca[0], cb[1]-ca[1]))
            if len(offsets) < 2:
                continue
            offsets = np.asarray(offsets)
            typical = np.median(offsets, axis=0)
            tolerance = .35*max(a.height, b.height)
            agree = np.mean(np.linalg.norm(offsets-typical, axis=1) <= tolerance)
            if agree >= .6 and np.linalg.norm(typical) <= 3*max(a.height, b.height):
                members.append((b, tuple(typical)))
                used.add(b_index)
        # Line boxes in a frame-independent frame of reference centered on the primary line.
        sizes = []
        for group, (ox, oy) in members:
            w = float(np.median([r.rect[2] for r in group.readings]))
            h = group.height
            sizes.append((ox-w/2, oy-h/2, w, h))
        left = min(s[0] for s in sizes)
        top = min(s[1] for s in sizes)
        right = max(s[0]+s[2] for s in sizes)
        bottom = max(s[1]+s[3] for s in sizes)
        width, height = int(round(right-left)), int(round(bottom-top))
        lines = [(int(round(s[0]-left)), int(round(s[1]-top)), int(round(s[2])), int(round(s[3]))) for s in sizes]
        observations = {}
        for group, (ox, oy) in members:
            typical = float(np.median([r.rect[2] for r in group.readings]))
            for r in group.readings:
                if r.frame in observations and group is not a:
                    continue
                if abs(r.rect[2]-typical) > .15*typical:
                    continue  # cut by the frame edge or misread: its centre is not the line centre
                cx, cy = center(r.rect)
                cx, cy = cx-ox, cy-oy  # primary line center
                observations[r.frame] = (int(round(cx+left)), int(round(cy+top)), width, height)
        keys = [g.key for g, _ in members]
        marks.append(Mark('moving', ' '.join(keys), keys, a.thickness,
                          sorted(observations.items()), lines, tilted=a.tilted))
    return marks


# ----------------------------------------------------------------------------
# Appearance learning

def per_shot(observations, shots, limit=2):
    counts, kept = {}, []
    for i, rect in observations:
        key = (int(shots[i]), rect[0]//8, rect[1]//8)
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0)+1
            kept.append((i, rect))
    return kept


def diverse(samples, observations, limit=16, minimum=6.):
    """Pick observations whose surroundings differ, taking shots in turn.

    A consensus over frames with the same background would mistake that background for the watermark.
    """
    by_shot = {}
    for i, rect in observations:
        by_shot.setdefault(int(samples.shots[i]), []).append((i, rect))
    queues = [list(reversed(items[::max(1, len(items)//8)] + items)) for items in by_shot.values()]
    chosen, crops, seen = [], [], set()
    while queues and len(chosen) < limit:
        for queue in list(queues):
            if not queue:
                queues.remove(queue)
                continue
            i, rect = queue.pop()
            if i in seen:
                continue
            seen.add(i)
            pad = rect[3]//2
            crop = patch(samples.video.planes(samples.raws[i])[0], grow(rect, pad, pad)).astype(np.float32)
            if all(np.abs(crop-other).mean() > minimum for other in crops):
                chosen.append((i, rect))
                crops.append(crop)
    return chosen


def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return np.maximum(mask, holes.astype(np.uint8))


def close_holes(mask, share):
    """The mask with its holes filled, each hole at most `share` of the mask's own pixels."""
    count, holes, stats, _ = cv2.connectedComponentsWithStats((fill_holes(mask) > mask).astype(np.uint8),
                                                               connectivity=4)
    small = 1+np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] <= share*np.count_nonzero(mask))
    return np.maximum(mask, np.isin(holes, small).astype(mask.dtype))


def enclosed_bays(part, outline, mouth):
    """The bays of a part within its convex outline that it wraps round: those whose mouth (where they meet the
    outline's edge) is at most `mouth` times as long as their walls (where they meet the part). A ring with a gap,
    or a notch deeper than it is wide, has them; a banner's shallow bays and the inside of a lone arc do not."""
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    count, bays = cv2.connectedComponents(((outline > 0) & (part == 0)).astype(np.uint8), connectivity=4)
    rim = (outline > 0) & (cv2.erode(outline, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
    near = cv2.dilate(part, cross) > 0
    filled = np.zeros_like(part)
    for b in range(1, count):
        bay = bays == b
        if np.count_nonzero(bay & rim) <= mouth*np.count_nonzero(bay & near):
            filled[bay] = 1
    return filled


def laid_over(mask, other, reach):
    """`other` moved by at most `reach` pixels to where it overlaps `mask` most (of equal overlaps, the smallest move),
    and that overlap as a share of the smaller of the two."""
    x, y, w, h = cv2.boundingRect(other.astype(np.uint8))
    if not w or not h or not mask.any():
        return other, 0.
    rows, columns = mask.shape
    x0, y0, x1, y1 = max(0, x-reach), max(0, y-reach), min(columns, x+w+reach), min(rows, y+h+reach)
    piece = other[y:y+h, x:x+w].astype(np.float32)
    overlap = cv2.matchTemplate(mask[y0:y1, x0:x1].astype(np.float32), piece, cv2.TM_CCORR)
    best = float(overlap.max())
    rows_, columns_ = np.nonzero(overlap >= best-.5)
    nearest = int(np.argmin((columns_+x0-x)**2+(rows_+y0-y)**2))
    bx, by = int(columns_[nearest]), int(rows_[nearest])
    moved = np.zeros_like(other)
    moved[y0+by:y0+by+h, x0+bx:x0+bx+w] = other[y:y+h, x:x+w]
    return moved, best/min(np.count_nonzero(mask), np.count_nonzero(other))


def ellipse(size):
    size = max(1, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


@dataclass
class Layer:
    mark: Mark
    motif: Motif
    origin: tuple          # motif top-left relative to the observation rect
    base: tuple = None     # fixed motif position for static marks
    positive: float = 0.
    positions: np.ndarray = None
    scores: np.ndarray = None
    others: np.ndarray = None
    present: np.ndarray = None
    located: np.ndarray = None  # placements as scanned, before presence decisions edit them
    lead: np.ndarray = None     # frames a fixed badge builds up or winds down over, joined after the animations


def build_mask(support, mark, origin, dilation, extra=None, exclude=None):
    lh = mark.line_height
    mask = support.copy()
    if extra is not None:
        mask |= extra
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 12:
            mask[labels == label] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(.3*lh))
    lines = np.zeros_like(mask)
    for dx, dy, w, h in mark.lines:
        x, y = dx-origin[0], dy-origin[1]
        cv2.rectangle(lines, (int(x), int(y)), (int(x+w-1), int(y+h-1)), 1, cv2.FILLED)
    if (mark.kind == 'static' or mark.tilted) and mask.any():
        # Text boxes fill gaps such as changing digits, but only where the badge itself has evidence;
        # OCR sometimes joins a badge line with nearby scene text into one long box, and the box of a slanted
        # line is mostly picture around its letters.
        reach = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(1.2*lh)), ellipse(.3*lh))
        lines &= reach
    mask = fill_holes(mask | lines)
    if dilation > 0:
        mask = cv2.dilate(mask, ellipse(2*dilation+1))
    if exclude is not None and np.any(mask & exclude):
        mask[exclude > 0] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            small = np.flatnonzero(areas < max(64, .1*areas.max()))+1
            mask[np.isin(labels, small)] = 0
    return mask.astype(np.uint8)


def consensus_motif(aligned, mark, origin, dilation, extra=None, exclude=None, visible=None):
    """visible: for each aligned patch, which of its pixels lie within the picture (None: all). A patch reaching past
    the picture's edge is zero there, which is no evidence either way."""
    if visible is not None and all(seen.all() for seen in visible):
        visible = None
    signal, weights, support = consensus_fields(np.stack(aligned).astype(np.float32),
                                                None if visible is None else np.stack(visible))
    mask = build_mask(support, mark, origin, dilation, extra, exclude)
    return Motif(signal.astype(np.float32), weights.astype(np.float32), mask)


def inside(video, rect):
    """Which pixels of a rect lie within the picture."""
    x, y, w, h = rect
    seen = np.zeros((h, w), bool)
    seen[max(0, -y):max(0, min(h, video.height-y)), max(0, -x):max(0, min(w, video.width-x))] = True
    return seen


def stable_overlay(samples, frames, rect, lh):
    """Pixels next to the text whose colour stays the same over unrelated backgrounds.

    Opaque icons beside a corner badge's text have little texture for the luma consensus,
    yet they keep exactly the same colour in every scene, unlike the picture behind them.
    Returns (region, mask, scenes) in frame coordinates or (region, None, scenes).
    """
    video = samples.video
    x, y, w, h = rect
    left, top = max(0, int(x-2.2*lh)), max(0, int(y-1.2*lh))
    right, bottom = min(video.width, int(x+w+2.2*lh)), min(video.height, int(y+h+1.2*lh))
    region = (left, top, right-left, bottom-top)
    seed = np.zeros((bottom-top, right-left), np.uint8)
    seed[max(0, y-top):max(0, y+h-top), max(0, x-left):max(0, x+w-left)] = 1
    # A component that runs into the search border continues into the scene; at the picture's edge it does not.
    open_sides = (left > 0, top > 0, right < video.width, bottom < video.height)
    keep, scenes, _ = steady_pixels((samples.bgr(i)[top:bottom, left:right] for i in frames), seed, lh, open_sides)
    return region, keep, scenes


STEADY_CHAIN = .8       # share of a part's pixels that must keep one colour for it to join the overlay through others
STEADY_LINK = .7        # ...across a gap of up to half this many line heights from a part already kept


def steady_pixels(pictures, seed, lh, open_sides, least=0, around=0.):
    """The overlay's pixels next to the seed (its lettering) in crops of it over unrelated backgrounds: those that keep
    one colour, near the seed (or near other such parts that are) and not running into an open side of the crops
    (left, top, right, bottom).
    Pixels darker than least (in their brightest channel) do not count: backgrounds that all come from dark scenes
    keep one colour there as well. Crops can differ only where something passes in front (another overlay over a
    sign in the scene); unless a share `around` of the picture around the seed changes between them, the seed
    is no overlay and nothing is returned. Returns (mask or None, number of unrelated backgrounds, share of the
    picture around the seed that changes or None)."""
    background = seed == 0
    crops = []
    for picture in pictures:
        crop = picture.astype(np.int16)
        # Only unrelated backgrounds tell an overlay apart from a still part of the scene.
        if all(np.abs(crop[background]-other[background]).mean() > 12 for other in crops):
            crops.append(crop)
    if len(crops) < 4:
        return None, len(crops), None
    height, width = seed.shape
    stack = np.stack(crops).astype(np.float32)
    median = np.median(stack, axis=0)
    deviation = np.median(np.abs(stack-median), axis=0).max(axis=-1)
    ring = background & ~(cv2.dilate(seed, ellipse(.35*lh)) > 0)
    changing = float((deviation[ring] > 6).mean()) if ring.any() else 0.
    if changing < around:
        return None, len(crops), changing
    # Animated icons (spinning, pulsing) cover their full extent only part of the time: a pixel whose
    # colour comes back in a good share of unrelated scenes belongs to the overlay as well.
    # Only next to the steady core, since some backgrounds (sky, walls) also repeat between scenes.
    recurrence = np.zeros(deviation.shape, np.int32)
    for crop in stack:
        recurrence = np.maximum(recurrence, (np.abs(stack-crop).max(axis=-1) <= 12).sum(axis=0))
    lit = median.max(axis=-1) >= least
    core = cv2.morphologyEx(((deviation <= 6) & lit).astype(np.uint8), cv2.MORPH_OPEN, ellipse(3)) | seed
    near = cv2.dilate(core, ellipse(.6*lh)) > 0
    steady = ((deviation <= 6) | ((recurrence >= max(3, .4*len(crops))) & near)) & lit
    stable = cv2.morphologyEx(steady.astype(np.uint8), cv2.MORPH_OPEN, ellipse(3))
    reach = ellipse(.35*lh)
    touch = cv2.dilate(seed, reach) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(stable, connectivity=8)

    def closed(label):
        cx, cy, cw, ch, _ = stats[label]
        return not ((cx == 0 and open_sides[0]) or (cy == 0 and open_sides[1])
                    or (cx+cw == width and open_sides[2]) or (cy+ch == height and open_sides[3]))
    keep = np.zeros_like(stable)
    for label in np.unique(labels[touch & (stable > 0)]):
        if label and closed(label):
            keep[labels == label] = 1
    # A badge's frame (corner brackets, a bar beside it) can stand clear of its lettering, joined to it only through
    # other parts of the badge: a part that keeps one colour itself (not only coming back in some scenes), across
    # a gap of up to a third of a line height from a part already kept, is kept too.
    parts = [label for label in range(1, count) if closed(label) and not keep[labels == label].any()
             and float((deviation[labels == label] <= 6).mean()) >= STEADY_CHAIN]
    link = ellipse(STEADY_LINK*lh)
    while parts and keep.any():
        grown = cv2.dilate(keep, link) > 0
        joined = [label for label in parts if grown[labels == label].any()]
        if not joined:
            break
        for label in joined:
            keep[labels == label] = 1
            parts.remove(label)
    return (keep if keep.any() else None), len(crops), changing


def learn_static(samples, mark, dilation, trace):
    rect = mark.observations[0][1]
    lh = mark.line_height
    mx, my = int(.9*lh)+8, int(.45*lh)+8
    crop = grow(rect, mx, my)
    # Opaque icons sit beside level badges. Around slanted lettering the boxes are mostly picture, which in a
    # video of dark scenes keeps one colour often enough to pass for an icon.
    region, stable, scenes = (None, None, 0) if mark.tilted else \
        stable_overlay(samples, [i for i, _ in mark.observations], rect, lh)
    if stable is not None:
        sx, sy, sw, sh = cv2.boundingRect(stable)
        crop = union_rect([crop, (region[0]+sx-6, region[1]+sy-6, sw+12, sh+12)])
    observations = diverse(samples, mark.observations)
    # A translucent part of a badge (the glass disc around its icon) can reach beyond the margin given its lettering
    # and its opaque icon: where the mask runs into the canvas's edge, the canvas widens.
    for widening in range(STATIC_WIDEN+1):
        motif = static_motif(samples, mark, rect, crop, region, stable, observations, dilation)
        wider = widened(crop, motif.mask, int(.5*lh)+8, samples.video)
        if wider == crop or widening == STATIC_WIDEN:
            break
        crop = wider
    origin = (crop[0]-rect[0], crop[1]-rect[1])
    scores = [motif.local(samples.signals[i], crop[:2], 0, 0)[2] for i, _ in mark.observations]
    layer = Layer(mark, motif, origin, base=crop[:2], positive=float(np.median(scores)))
    trace('plan.static_learned', label=mark.label, position=list(crop[:2]), size=[motif.width, motif.height],
          mask_pixels=int(motif.mask.sum()), positive=round(layer.positive, 3), observations=len(observations),
          stable_pixels=int(stable.sum()) if stable is not None else 0, stable_scenes=scenes, widened=widening)
    return layer


STATIC_WIDEN = 3        # times a badge's canvas widens where its mask runs into the canvas's edge


def widened(crop, mask, step, video):
    """The crop grown by step on each side where the mask runs into its edge, short of the picture's edge."""
    x, y, w, h = crop
    x0 = min(x, max(0, x-step)) if mask[:, :2].any() else x
    y0 = min(y, max(0, y-step)) if mask[:2].any() else y
    x1 = max(x+w, min(video.width, x+w+step)) if mask[:, -2:].any() else x+w
    y1 = max(y+h, min(video.height, y+h+step)) if mask[-2:].any() else y+h
    return x0, y0, x1-x0, y1-y0


def static_motif(samples, mark, rect, crop, region, stable, observations, dilation):
    """A badge's motif learned over a crop of the picture."""
    origin = (crop[0]-rect[0], crop[1]-rect[1])
    extra = None
    if stable is not None:
        extra = patch(stable, (crop[0]-region[0], crop[1]-region[1], crop[2], crop[3]))
    # Subtitles and other passing text near a badge must keep their pixels. Pixels that keep one colour over
    # unrelated scenes are the badge's own, whatever OCR read over them (it reads icons as stray letters), and so is
    # what they enclose: a ticker running along a bar of the badge changes in every sample, like a caption, but
    # inside the badge's frame.
    exclude = patch(samples.transient, crop) if samples.transient is not None else None
    if exclude is not None and extra is not None:
        exclude = exclude & (fill_holes(extra) == 0).astype(np.uint8)
    aligned = [patch(samples.signals[i], crop) for i, _ in observations]
    motif = consensus_motif(aligned, mark, origin, dilation, extra, exclude)
    # Re-estimate from every sampled frame that clearly shows the mark.
    for _ in range(2):
        scored = []
        for i in samples.frames:
            _, _, score = motif.local(samples.signals[i], crop[:2], 0, 0)
            scored.append((score, i))
        confirmed = [(i, rect) for score, i in scored if score >= .45]
        confirmed = diverse(samples, confirmed)
        if len(confirmed) < 3:
            break
        motif = consensus_motif([patch(samples.signals[i], crop) for i, _ in confirmed], mark, origin,
                                dilation, extra, exclude)
    return motif


VARIANT_MATCH = .6      # the parent's motif, rescaled, must match the other state this well to stand in for it


def learn_variant(samples, mark, parent, dilation, parent_dilation, trace):
    """Another state of a learned badge, shown at another size: the parent's motif rescaled onto it.

    A state shown briefly (smaller during the opening, say) is seen over too few scenes to tell an opaque icon
    from the picture behind it, while the usual state has been seen over many. Falls back to learning the
    state on its own when the rescaled badge does not match it.
    """
    if len(mark.keys) != len(parent.mark.keys) or any(similar(a, b) < .9 for a, b in zip(mark.keys, parent.mark.keys)):
        return learn_static(samples, mark, dilation, trace)     # other lines as well: not the same badge resized
    rect = mark.observations[0][1]
    known = parent.mark.observations[0][1]
    frames = [i for i, _ in mark.observations]
    estimate = rect[2]/known[2]
    radius = max(6, int(.3*mark.line_height))
    best = None
    for scale in estimate*np.arange(.94, 1.07, .02):
        size = (round(parent.motif.width*scale), round(parent.motif.height*scale))
        signal = cv2.resize(parent.motif.signal, size, interpolation=cv2.INTER_AREA)
        weights = np.clip(cv2.resize(parent.motif.weights, size, interpolation=cv2.INTER_AREA), 0, 1)
        guess = (round(rect[0]+scale*(parent.base[0]-known[0])), round(rect[1]+scale*(parent.base[1]-known[1])))
        total = None
        for i in frames:
            region = patch(samples.signals[i], (guess[0]-radius, guess[1]-radius, size[0]+2*radius,
                                                size[1]+2*radius))
            scores = response(region, signal, weights)
            total = scores if total is None else total+scores
        _, value, _, (a, b) = cv2.minMaxLoc(total)
        if best is None or value > best[0]:
            best = (value, scale, (guess[0]-radius+a, guess[1]-radius+b), signal.astype(np.float32),
                    weights.astype(np.float32))
    value, scale, base, signal, weights = best
    match = value/len(frames)
    if match < VARIANT_MATCH:
        trace('plan.variant_unmatched', label=mark.label, parent=parent.mark.label, match=round(match, 3))
        return learn_static(samples, mark, dilation, trace)
    # The parent's mask carries its own dilation, rescaled with it; add what this state's dilation lacks.
    size = signal.shape[::-1]
    mask = (cv2.resize(parent.motif.mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= .5)
    mask = mask.astype(np.uint8)
    extra = dilation-scale*parent_dilation
    if extra >= .5:
        mask = cv2.dilate(mask, ellipse(2*round(extra)+1))
    motif = Motif(signal, weights, mask)
    layer = Layer(mark, motif, (base[0]-rect[0], base[1]-rect[1]), base=base, positive=float(match))
    trace('plan.static_learned', label=mark.label, position=list(base), size=[motif.width, motif.height],
          mask_pixels=int(motif.mask.sum()), positive=round(match, 3), parent=parent.mark.label,
          scale=round(float(scale), 3))
    return layer


def chroma_support(samples, placements, size, near):
    """Colour edges that agree across aligned placements and lie close to the luma mask.

    Translucent tinted icons barely change luma, so their outline only shows in chroma.
    """
    if len(placements) < 4:
        return None
    width, height = size
    support = np.zeros((height, width), np.uint8)
    visible = np.stack([inside(samples.video, (x, y, width, height)) for _, x, y in placements])
    if visible.all():
        visible = None
    for plane in (1, 2):
        aligned = [patch(samples.chroma(i, plane), (x, y, width, height)) for i, x, y in placements]
        _, _, mask = consensus_fields(np.stack(aligned).astype(np.float32), visible)
        support |= mask
    if not support.any():
        return None
    count, labels = cv2.connectedComponents(support, connectivity=8)
    touching = np.unique(labels[(near > 0) & (support > 0)])
    keep = np.isin(labels, touching[touching > 0]).astype(np.uint8)
    return keep if keep.any() else None


def trimmed(motif, origin, margin=4):
    """Crop a motif to its mask so that the template does not reach far beyond the watermark."""
    x, y, w, h = cv2.boundingRect(motif.mask)
    left, top = max(0, x-margin), max(0, y-margin)
    right, bottom = min(motif.width, x+w+margin), min(motif.height, y+h+margin)
    view = np.s_[top:bottom, left:right]
    return (Motif(motif.signal[view].copy(), motif.weights[view].copy(), motif.mask[view].copy()),
            (origin[0]+left, origin[1]+top))


def learn_moving(samples, mark, dilation, trace):
    lh = mark.line_height
    mx, my = int(3.*lh)+8, int(.8*lh)+8
    origin = (-mx, -my)
    observations = diverse(samples, mark.observations)
    if len(observations) < 3:
        raise ValueError(f'Too few complete observations for {mark.label}.')
    # Align OCR boxes to one observation's text region; try a few references and keep the best agreement.
    _, (_, _, rw, rh) = observations[0]
    search = int(.4*lh)+8
    best = None
    for reference in sorted({0, len(observations)//2, len(observations)-1}):
        ri, (rx, ry, _, _) = observations[reference]
        template = patch(samples.signals[ri], (rx, ry, rw, rh))
        placements = []
        for i, (x, y, w, h) in observations:
            region = patch(samples.signals[i], (x-search, y-search, rw+2*search, rh+2*search))
            result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (dx, dy) = cv2.minMaxLoc(result)
            placements.append((i, x-search+dx, y-search+dy, score))
        agreement = sum(score >= .3 for *_, score in placements)
        if best is None or agreement > best[0]:
            best = agreement, placements
    placements = best[1]
    size = (rw+2*mx, rh+2*my)

    def stacked(spots):
        # A mark seen at the picture's edge (entering, leaving, bouncing) tells nothing of its part beyond it.
        return ([patch(samples.signals[i], (u, v, *size)) for i, u, v in spots],
                [inside(samples.video, (u, v, *size)) for _, u, v in spots])

    aligned, visible = stacked([(i, x-mx, y-my) for i, x, y, score in placements if score >= .2])
    if len(aligned) < 3:
        raise ValueError(f'Too few aligned observations for {mark.label}.')
    motif = consensus_motif(aligned, mark, origin, dilation, visible=visible)
    # Refine alignment against the consensus itself.
    final = [(i, x-mx, y-my) for i, x, y, score in placements if score >= .2]
    for _ in range(2):
        refined = []
        for i, x, y, _ in placements:
            u, v, score = motif.local(samples.signals[i], (x-mx, y-my), 6, 6)
            if score >= .3:
                refined.append((i, u, v))
        if len(refined) < 3:
            break
        final = refined
        aligned, visible = stacked(final)
        motif = consensus_motif(aligned, mark, origin, dilation, visible=visible)
    near = cv2.dilate(motif.mask, ellipse(.5*lh))
    extra = chroma_support(samples, final, size, near)
    if extra is not None:
        aligned, visible = stacked(final)
        motif = consensus_motif(aligned, mark, origin, dilation, extra, visible=visible)
    # Flat parts of a badge (a filled bar under the text, the inside of an icon) have texture only along their
    # outline, and an outline broken towards the outside leaves a notch that filling holes cannot close.
    closed = fill_holes(cv2.morphologyEx(motif.mask, cv2.MORPH_CLOSE, ellipse(lh)))
    motif = Motif(motif.signal, motif.weights, closed)
    wide = origin
    motif, origin = trimmed(motif, origin)
    shift = (origin[0]-wide[0], origin[1]-wide[1])
    scores = [motif.local(samples.signals[i], (x-mx+shift[0], y-my+shift[1]), 6, 6)[2]
              for i, x, y, _ in placements]
    layer = Layer(mark, motif, origin, positive=float(np.median(scores)))
    trace('plan.moving_learned', label=mark.label, size=[motif.width, motif.height],
          mask_pixels=int(motif.mask.sum()), positive=round(layer.positive, 3), observations=len(placements),
          chroma_pixels=int(extra.sum()) if extra is not None else 0)
    return layer


# ----------------------------------------------------------------------------
# Faint full-picture overlays
#
# Large translucent watermarks (a huge site name across the middle, a tilted brand line) are too faint
# for OCR on any single frame. Over many unrelated scenes the picture averages away while the overlay
# stays, so the median high-pass picture shows them, and OCR on its significance map reads them.

FAINT_ANGLES = (0, -30, 30, -45, 45)


def overlay_consensus(samples):
    """Median high-pass picture over unrelated scenes and the significance of every pixel of it."""
    video = samples.video
    size = (64, max(2, round(64*video.height/video.width)))
    small = {i: cv2.resize(video.planes(samples.raws[i])[0], size, interpolation=cv2.INTER_AREA).astype(np.float32)
             for i in samples.frames}
    chosen = []
    for i in samples.frames:
        if all(np.abs(small[i]-small[j]).mean() > 12 for j in chosen):
            chosen.append(i)
    if len(chosen) < 8:
        return chosen, None, None
    median = np.zeros((video.height, video.width), np.float32)
    significance = np.zeros_like(median)
    for top in range(0, video.height, 128):
        stack = np.stack([samples.signals[i][top:top+128] for i in chosen])
        m = np.median(stack, axis=0)
        spread = np.median(np.abs(stack-m), axis=0)
        median[top:top+128] = m
        significance[top:top+128] = m/(1.25*1.4826*spread/math.sqrt(len(chosen))+.3)
    return chosen, median, significance


def overlay_picture(significance):
    return np.clip(128+20*np.clip(cv2.GaussianBlur(significance, (0, 0), .8), -6, 6), 0, 255).astype(np.uint8)


def read_rotated(ocr, picture, angles=FAINT_ANGLES, minimum=.7):
    """OCR a grey picture at a few rotations: [(key, text, polygon in picture coordinates, line height, score)]."""
    height, width = picture.shape
    found = []
    for angle in angles:
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.)
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        size = (int(height*s+width*c), int(height*c+width*s))
        matrix[0, 2] += size[0]/2-width/2
        matrix[1, 2] += size[1]/2-height/2
        rotated = cv2.warpAffine(picture, matrix, size, borderValue=128) if angle else picture
        inverse = cv2.invertAffineTransform(matrix)
        for key, text, quad, score in ocr.read_quads(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR), minimum):
            polygon = (quad.astype(np.float64)@inverse[:, :2].T+inverse[:, 2]).astype(np.float32)
            sides = np.linalg.norm(np.roll(polygon, -1, axis=0)-polygon, axis=1)
            height = float(min(sides[0]+sides[2], sides[1]+sides[3])/2)
            # Text read at a rotation where it is not level comes with a skewed box.
            dx, dy = quad[1]-quad[0]
            tilt = abs(math.degrees(math.atan2(float(dy), float(dx))))
            if height >= 6:
                found.append((key, text, polygon, height, float(score), min(tilt, 180-tilt)))
    return found


def polygon_share(polygon, mask):
    """Share of a polygon's area that lies on a 0/1 mask."""
    region = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(region, [np.round(polygon).astype(np.int32)], 1)
    area = int(region.sum())
    return float((region & mask).sum())/area if area else 1.


def polygon_rect(polygon):
    x, y, w, h = cv2.boundingRect(np.round(polygon).astype(np.int32))
    return int(x), int(y), int(w), int(h)


def covered_share(rect, others):
    """Largest share of ``rect`` that one of ``others`` covers."""
    area = max(1, rect[2]*rect[3])
    best = 0.
    for other in others:
        w = min(rect[0]+rect[2], other[0]+other[2])-max(rect[0], other[0])
        h = min(rect[1]+rect[3], other[1]+other[3])-max(rect[1], other[1])
        best = max(best, max(0, w)*max(0, h)/area)
    return best


def known_marks(marks, shape):
    """Where OCR marks rest: their pixels belong to their own layers, not to a faint overlay."""
    known = np.zeros(shape, np.uint8)
    for mark in marks:
        pad = int(.5*mark.line_height)
        rects = [rect for _, rect in mark.observations]
        if mark.kind == 'moving':
            # Only where a moving mark rests for a large share of the video does it show in the consensus.
            rects = [r for r in rects if sum(iou(r, o) >= .5 for o in rects) >= .3*len(rects)]
        for rect in rects:
            x, y, w, h = grow(rect, pad, pad)
            known[max(0, y):max(0, y+h), max(0, x):max(0, x+w)] = 1
    return known


def faint_marks(samples, marks, ocr, trace):
    chosen, median, significance = samples.overlay
    if median is None:
        trace('plan.faint', scenes=len(chosen), marks=[], reason='too few unrelated scenes')
        return []
    readings = read_rotated(ocr, overlay_picture(significance))
    known = known_marks(marks, median.shape)
    # The same text read at several rotations: keep the most complete reading.
    kept, seen = [], []
    for reading in sorted(readings, key=lambda r: (r[5] > 8, -len(r[0])*r[4])):
        key, text, polygon, height, score, _ = reading
        rect = polygon_rect(polygon)
        if len(key) < 2:
            continue
        # Compare with every earlier reading, including those already explained by OCR marks. A duplicate lies
        # mostly inside the earlier reading: a huge overlay line that merely contains a small URL is not one.
        duplicate = any(covered_share(rect, [polygon_rect(other[2])]) >= .3
                        and (similar(key, other[0]) >= .5 or key in other[0] or other[0] in key) for other in seen)
        seen.append(reading)
        if not duplicate and polygon_share(polygon, known) < .5:
            kept.append(reading)
    # Neighbouring lines (a name above an address) form one overlay.
    clusters = []
    for reading in kept:
        rect = grow(polygon_rect(reading[2]), int(.6*reading[3]), int(.6*reading[3]))
        joined = [c for c in clusters if any(covered_share(rect, [r]) > 0 for r in c['rects'])]
        cluster = {'readings': [reading], 'rects': [rect]}
        for other in joined:
            clusters.remove(other)
            cluster['readings'] += other['readings']
            cluster['rects'] += other['rects']
        clusters.append(cluster)
    found = []
    for cluster in clusters:
        members = cluster['readings']
        mark = Mark('faint', ' '.join(r[1] for r in members), [r[0] for r in members],
                    float(np.median([r[3] for r in members])), [], [])
        mark.polygons = [r[2] for r in members]
        found.append(mark)
    for mark in found:
        mark.known = known
    trace('plan.faint', scenes=len(chosen), texts=[r[1] for r in kept], marks=[m.label for m in found])
    return found


def part_of_overlay(mark, overlays, trace):
    """A few clear OCR readings of a faint overlay's own text, taken for a moving mark.

    They rest in one place inside the overlay and read like it; a real moving mark that only passes through the
    overlay is seen elsewhere too.
    """
    if mark.kind != 'moving':
        return False
    rects = [rect for _, rect in mark.observations]
    resting = [r for r in rects if sum(iou(r, o) >= .5 for o in rects) >= .5*len(rects)]
    if len(resting) < .5*len(rects):
        return False
    for overlay in overlays:
        inside = [r for r in rects if covered_share(r, [polygon_rect(p) for p in overlay.polygons]) >= .6]
        alike = any(similar(key, other) >= .5 or key in other or other in key
                    for key in mark.keys for other in overlay.keys)
        if alike and len(inside) >= .5*len(rects):
            trace('plan.overlay_part', label=mark.label, overlay=overlay.label)
            return True
    return False


def learn_faint(samples, mark, dilation, trace):
    chosen, median, significance = samples.overlay
    lh = mark.line_height
    region = np.zeros(median.shape, np.uint8)
    for polygon in mark.polygons:
        cv2.fillPoly(region, [np.round(polygon).astype(np.int32)], 1)
    # Marks found by OCR have layers of their own; keep their pixels out of the faint overlay.
    known = cv2.dilate(mark.known, ellipse(9)) > 0
    near = (cv2.dilate(region, ellipse(min(.3*lh, 24))) > 0) & ~known
    # Large letters have thick strokes: average over a stroke's width to lift them out of the noise.
    sigma = max(1.2, .03*lh)
    strength = cv2.GaussianBlur(np.abs(significance), (0, 0), sigma)
    # Significance depends on how many unrelated scenes the video has; compare with the area around the text.
    ring = (cv2.dilate(region, ellipse(min(.9*lh, 72))) > 0) & ~near & ~known
    background = float(np.percentile(strength[ring], 90)) if ring.sum() >= 200 else 1.
    threshold = float(np.clip(1.6*background, .35, 4.))
    strokes = ((strength >= threshold) & near).astype(np.uint8)
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, ellipse(min(7, .06*lh)))
    touching = cv2.dilate(region, ellipse(min(.15*lh, 12))) > 0
    # The high-pass filter rings along the frame border identically in every scene; a component lying mostly
    # in that band is the filter, not the overlay (letters that touch the border keep most of their pixels inside).
    band = max(8, round(3*HIGHPASS_SIGMA))
    border = np.ones(strokes.shape, bool)
    border[band:-band, band:-band] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    keep = np.zeros_like(strokes)
    for label in np.unique(labels[touching & (strokes > 0)]):
        area = stats[label, cv2.CC_STAT_AREA]
        if not label or area < 20:
            continue
        component = labels == label
        if (component & border).sum() > .5*area:
            continue
        keep[component] = 1
    if keep.sum() < 32:
        raise ValueError(f'Too little consistent evidence for the faint overlay {mark.label}.')
    margin = 2*dilation+8
    crop = grow(cv2.boundingRect(keep), margin, margin)
    x, y = int(crop[0]), int(crop[1])
    mask = patch(keep, crop)
    if dilation > 0:
        mask = cv2.dilate(mask, ellipse(2*dilation+1))
    signal = patch(median, crop).astype(np.float32)
    weights = (np.minimum(np.abs(patch(significance, crop))/8, 1)*patch(keep, crop)).astype(np.float32)
    motif = Motif(signal, weights, mask.astype(np.uint8))
    scores = [motif.local(samples.signals[i], (x, y), 0, 0)[2] for i in chosen]
    layer = Layer(mark, motif, (0, 0), base=(x, y), positive=float(np.median(scores)))
    # Unblending needs the strokes themselves and how strongly they stand out from the surrounding picture.
    layer.keep = patch(keep, crop)
    layer.levels = (background, threshold, sigma)
    trace('plan.faint_learned', label=mark.label, position=[x, y], size=[motif.width, motif.height],
          mask_pixels=int(mask.sum()), positive=round(layer.positive, 3), threshold=round(threshold, 2))
    return layer


def decide_faint(video, layer):
    """Faint overlays match weakly in any one frame; decide on the score averaged over a second."""
    fps = float(video.fps)
    window = max(1, round(fps))
    smooth = np.convolve(layer.scores, np.ones(window)/window, mode='same')
    threshold = max(.06, .5*layer.positive)
    present = fill_gaps(smooth >= threshold, round(2*fps))
    present = widen(present, round(.5*fps))
    layer.positions[:] = layer.base
    layer.present = present
    return threshold


# ----------------------------------------------------------------------------
# Unblending faint overlays
#
# A translucent overlay composites every pixel as I = (1 - alpha)*B + alpha*C. Averaged over many unrelated
# scenes the picture becomes smooth while the overlay stays put, so the mean picture minus its large-scale median
# shows the letters filled in: D = alpha*(C - mean B). An overlay's text has one opacity; with video white as its
# colour (video black for a dark overlay), that opacity and a soft letter matte follow from D inside the strokes
# the consensus high-pass found. Inverting the blend restores the picture under an overlay far too large to
# repaint, and keeps its detail.
#
# The mean picture shows the letters only as sharp as they are on average. An encoder keeps a faint overlay sharp
# in a plain picture and smooths it over a busy one, so frames plain around the overlay (a fade to black, a title
# card) give the sharp matte where there are any: there a pixel is the plain level plus alpha (colour - level).
# Each frame is then unblended with that matte blurred as far as the frame's own picture of the overlay shows.

UNBLEND_FRAMES = 500            # frames averaged into the mean picture
UNBLEND_LIGHT, UNBLEND_DARK = 235., 16.
UNBLEND_ACCEPT = .35            # share of the overlay's consensus strength the inverse blend must remove to be used
UNBLEND_CLEAN = .75             # below this share traces may remain: the result goes to review
UNBLEND_SHOWN = .5              # median strength of an overlay's letters, as fitted, from which a frame shows it
UNBLEND_SHOWN_SPAN = .5         # seconds that median is taken over
UNBLEND_SHOWN_WEAK = .1         # share of the video's median power of the drawn matte below which a frame tells nothing
UNBLEND_REFIT = .5              # seconds of scanned frames found not to show the overlay from which it is fitted again
UNBLEND_CHECKED = 6             # scenes showing the overlay, at least, that its removal is checked on
UNBLEND_STROKE = .3             # line heights: width of the disk, wider than any stroke, that the strokes' top-hat opens by
UNBLEND_STROKE_SHARE = .25      # share of the top-hat's 90th percentile in the letters from which a pixel is a stroke
UNBLEND_SIDE_SOFT = 8.          # luma levels past a one-sided overlay's colour over which its change of colour fades out
UNBLEND_LAW_FRAMES = 120        # frames whose stroke pixels, against the picture's level under them, tell the blend
UNBLEND_LAW_BIN = 200           # stroke pixels a level bin needs to count in the blend law
UNBLEND_LAW_BETTER = .5         # share of the white (black) overlay's misfit below which a fitted blend law is taken
UNBLEND_CALM = 4.               # |pixel - its blur| at most this all around a stroke pixel: the level inpainted under it
                                # is trusted (and weighs fully in the regressed matte)
UNBLEND_LEAST_SPAN = 20.        # least distance of the level under a pixel from the overlay's colour to tell its alpha
UNBLEND_BLURS = (0, .5, .8, 1.1, 1.5, 2.)  # blurs of the matte each frame chooses from
UNBLEND_KEEP = 1.02             # a frame keeps the previous frame's blur unless another fits better by more than this
UNBLEND_PLAIN = .6              # median |pixel - its 5x5 mean| around the overlay at most this: a plain frame
UNBLEND_PLAIN_KEPT = 48         # plain frames kept, spread over the video (many are plain without the overlay: black
                                # frames around a part it is not on)...
UNBLEND_PLAIN_USED = 16         # ...of which at most this many that show it give the matte
UNBLEND_PLAIN_SPAN = 60         # least distance between a plain frame's level and the overlay's colour to show it
UNBLEND_PLAIN_AGREE = .35       # a plain frame's opacity within this share of the mean picture's
# One blend law and colour fit most overlays' letters, but not every part of some: an emblem drawn nearly opaque
# over a dark picture (whose picture under it is lost) or with a law of its own keeps much of its strength after
# unblending while the letters beside it go. Such parts are repaired as well, in the frames the overlay shows in.
UNBLEND_PART_LEAST = 100        # letter pixels of a part, at least, for its own removal to be judged
UNBLEND_PART_SHOWN = .25        # share of the overlay's strength over the background a part must show with to be judged
PLANE_SCALES = (1, 2, 2)


def law_span(mode, span):
    """The part of the distance from a level to an overlay's colour that the overlay moves it by, per unit alpha."""
    if mode == 'lighten':
        return np.maximum(span, 0)
    if mode == 'darken':
        return np.minimum(span, 0)
    return span


def blend_law(levels, rises, light):
    """How an overlay blends, from its strokes' rise over the picture's level against that level (medians in bins of
    the level B): 'normal', a rise of a(C - B) towards its colour C; or 'lighten' / 'darken', the same only where it
    lifts a darker picture / darkens a lighter one (a grey overlay drawn to brighten, which leaves a picture lighter
    than itself as it is). A clip's levels often span too little to tell the colour, so the white (black) normal
    overlay is kept unless a law fits far better. Returns (mode, opacity, colour, misfit, the white's misfit) or None."""
    bins = []
    for lo in range(0, 256, 8):
        sel = (levels >= lo) & (levels < lo+8)
        if sel.sum() >= UNBLEND_LAW_BIN:
            bins.append((lo+3.5, float(np.median(rises[sel])), math.sqrt(float(sel.sum()))))
    if len(bins) < 2:
        return None
    b, d, weight = (np.array(v) for v in zip(*bins))

    def fit(mode, colour):
        s = law_span(mode, colour-b)
        power = float((weight*s*s).sum())
        if power <= 0:
            return None
        a = float((weight*s*d).sum())/power
        if not 0 < a < 1:
            return None
        return float((weight*np.square(d-a*s)).sum()/weight.sum()), a

    plain_colour = UNBLEND_LIGHT if light else UNBLEND_DARK
    plain = fit('normal', plain_colour)
    best = None
    for mode in ('normal', 'lighten', 'darken'):
        for colour in np.arange(8., 248.):
            found = fit(mode, colour)
            if found is not None and (best is None or found[0] < best[3]):
                best = (mode, found[1], float(colour), found[0])
    if best is not None and (plain is None or best[3] <= UNBLEND_LAW_BETTER*plain[0]):
        return (*best, None if plain is None else plain[0])
    if plain is None:
        return None
    return 'normal', plain[1], plain_colour, plain[0], plain[0]


@dataclass
class Unblend:
    label: str
    rect: tuple         # luma crop (x, y, w, h): even-aligned and inside the frame
    variants: list      # per matte blur, per plane (slope, intercept) at plane resolution
    present: np.ndarray
    check: dict
    colour: float = UNBLEND_LIGHT
    zone: np.ndarray = None         # luma crop pixels that judge a frame's blur
    patterns: list = None           # per blur, the matte's high-pass at those pixels
    last: int = None
    chosen: list = None             # frames unblended with each blur
    mode: str = 'normal'            # 'lighten' / 'darken': it lifts only a picture darker than its colour / darkens
                                    # only one lighter (blend_law)
    poor: np.ndarray = None         # luma crop pixels of the parts the inverse blend leaves (part_removals)

    @property
    def fields(self):
        return self.variants[0]

    def span(self, level):
        """How far the overlay's colour lies from a level, as far as it can move that level."""
        return law_span(self.mode, self.colour-level)

    def restore_luma(self, luma, k):
        """Float luma unblended with the matte of blur k; a one-sided overlay leaves the picture past its colour."""
        slope, intercept = self.variants[k][0]
        restored = (luma-intercept)/slope
        if self.mode == 'lighten':
            restored = np.minimum(restored, luma)
        elif self.mode == 'darken':
            restored = np.maximum(restored, luma)
        return restored

    def choose(self, luma):
        """The blur whose matte, drawn over the frame's own smooth level, best matches the frame's high-pass over
        the letters. Fitting the source rather than judging the unblended result keeps the texture an unblend
        amplifies out of the choice."""
        if len(self.variants) == 1:
            return 0
        crop = luma.astype(np.float32)
        hp = highpass(crop, 1.5)[self.zone]
        span = self.span(cv2.GaussianBlur(crop, (0, 0), 6))[self.zone]
        errors =[float(np.square(hp-span*pattern).mean()) for pattern in self.patterns]
        best = int(np.argmin(errors))
        if self.last is not None and errors[self.last] <= UNBLEND_KEEP*errors[best]:
            best = self.last
        self.last = best
        return best

    def amplitude(self, luma):
        """How strongly the overlay's letters show in a frame: the least-squares scale of the matte drawn over the
        frame's own smooth level (as choose fits it, with the blur that fits best) in the frame's high-pass over the
        letters. About 1 where the overlay shows as fitted and about 0 where it is not there. Returns (scale, power of
        the drawn matte): where the picture behind is near the overlay's colour its letters hardly show either way."""
        crop = luma.astype(np.float32)
        hp = highpass(crop, 1.5)[self.zone]
        span = self.span(cv2.GaussianBlur(crop, (0, 0), 6))[self.zone]
        best = None
        for pattern in self.patterns:
            model = span*pattern
            power = float((model*model).mean())
            if power <= 0:
                continue
            scale = float((hp*model).mean())/power
            error = float(np.square(hp-scale*model).mean())
            if best is None or error < best[2]:
                best = (scale, power, error)
        return (np.nan, 0.) if best is None else best[:2]

    def apply(self, planes):
        """Unblend writable YUV planes in place."""
        x, y, w, h = self.rect
        luma = planes[0][y:y+h, x:x+w].astype(np.float32)
        k = self.choose(luma)
        if self.chosen is not None:
            self.chosen[k] += 1
        # A one-sided overlay changed the colour only where it changed the luma.
        weight = None
        if self.mode != 'normal':
            weight = np.clip((self.colour-luma if self.mode == 'lighten' else luma-self.colour)/UNBLEND_SIDE_SOFT, 0, 1)
        for p, (plane, scale, (slope, intercept)) in enumerate(zip(planes, PLANE_SCALES, self.variants[k])):
            view = plane[y//scale:(y+h)//scale, x//scale:(x+w)//scale]
            if p == 0:
                restored = self.restore_luma(luma, k)
            else:
                original = view.astype(np.float32)
                restored = (original-intercept)/slope
                if weight is not None:
                    w_small = cv2.resize(weight, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_AREA)
                    restored = original+w_small*(restored-original)
            np.copyto(view, np.clip(np.rint(restored), 0, 255).astype(np.uint8))


def overlay_strength(signals, sigma):
    """Blurred |significance| of the median high-pass picture over unrelated scenes, as in overlay_consensus."""
    stack = np.stack(signals)
    median = np.median(stack, axis=0)
    spread = np.median(np.abs(stack-median), axis=0)
    significance = median/(1.25*1.4826*spread/math.sqrt(len(stack))+.3)
    return cv2.GaussianBlur(np.abs(significance), (0, 0), sigma)


def overlay_presence(video, layer):
    """Frames to unblend. Unlike a repair mask this must not reach past the overlay: unblending a frame without
    it would print its negative. An overlay found in nearly every frame is taken to be in all of them."""
    fps = float(video.fps)
    window = max(1, round(fps))
    smooth = np.convolve(layer.scores, np.ones(window)/window, mode='same')
    present = fill_gaps(smooth >= max(.06, .5*layer.positive), round(2*fps))
    return np.ones(video.count, bool) if present.mean() >= .9 else present


def mean_picture(video, present, cuda, place=None):
    """Mean planes over frames showing the overlay, their count, and the luma crops at place (x, y, w, h) of the
    plainest of them."""
    step = max(1, video.count//UNBLEND_FRAMES)
    sums, count, plain = None, 0, []
    candidates, stride = 0, 1
    for i, raw in read_frames(video, cuda=cuda, step=step):
        if not present[i]:
            continue
        planes = video.planes(raw)
        if sums is None:
            sums = [np.zeros(p.shape, np.float64) for p in planes]
        for total, plane in zip(sums, planes):
            total += plane
        count += 1
        if place is not None:
            x, y, w, h = place
            crop = planes[0][y:y+h, x:x+w]
            rough = float(np.median(np.abs(crop.astype(np.float32)-cv2.blur(crop, (5, 5)).astype(np.float32))))
            if rough <= UNBLEND_PLAIN:
                if candidates % stride == 0:
                    plain.append((rough, i, crop.copy()))
                    if len(plain) > UNBLEND_PLAIN_KEPT:
                        plain, stride = plain[::2], 2*stride
                candidates += 1
    if count < 60:
        raise ValueError('Too few frames show the overlay to average it.')
    return [total/count for total in sums], count, plain


def plain_matte(plain, place, hole, near, gate, core, colour, opacity):
    """The overlay's alpha at place read off plain frames, with the frames used; None without two plain frames
    whose picture of the overlay agrees with the mean picture's opacity."""
    x, y, w, h = place
    hole, gate, core = hole[y:y+h, x:x+w], gate[y:y+h, x:x+w], core[y:y+h, x:x+w]
    ring = near[y:y+h, x:x+w] & (hole == 0)
    if ring.sum() < 500 or core.sum() < 100:
        return None, []
    seen, used = [], []
    for _, i, crop in plain:
        luma = crop.astype(np.float64)
        level = without_letters(luma, hole)
        if np.percentile(np.abs(luma-cv2.GaussianBlur(luma, (0, 0), 2))[ring], 90) > 2:
            continue
        span = colour-level
        if np.median(np.abs(span[gate])) < UNBLEND_PLAIN_SPAN:
            continue
        alpha = np.where(np.abs(span) >= 30, (luma-level)/np.where(np.abs(span) >= 30, span, 1), np.nan)
        if abs(float(np.nanmedian(alpha[core]))/opacity-1) > UNBLEND_PLAIN_AGREE:
            continue
        seen.append(alpha)
        used.append(i)
        if len(seen) >= UNBLEND_PLAIN_USED:
            break
    if len(seen) < 2:
        return None, used
    stack = np.stack(seen)
    alpha = np.zeros(stack.shape[1:])
    counted = np.isfinite(stack).any(axis=0)
    alpha[counted] = np.nanmedian(stack[:, counted], axis=0)
    return (np.clip(alpha, 0, 1.5*opacity)*gate).astype(np.float32), used


def without_letters(mean, hole):
    """The mean picture with the letters inpainted out of it: the background they lie on."""
    filled = cv2.inpaint(np.clip(np.rint(mean), 0, 255).astype(np.uint8), hole, 5, cv2.INPAINT_TELEA)
    return np.where(hole > 0, cv2.GaussianBlur(filled.astype(np.float64), (0, 0), 3), mean)


def overlay_opacity(ratio, gate, label):
    """One opacity for the whole overlay: the median inside the letters, with the letters drawn at half of it."""
    opacity = .08
    for _ in range(4):
        inside = (cv2.GaussianBlur(ratio/opacity, (0, 0), 1.2) >= .5) & gate
        core = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if core.sum() < 100:
            raise ValueError(f'The faint overlay {label} does not show in the mean picture.')
        opacity = float(np.median(ratio[core]))
        if not .015 <= opacity <= .5:
            raise ValueError(f'The faint overlay {label} has an implausible opacity {opacity:.3f}.')
    return opacity, core


def fit_overlay(video, samples, layer, exclude, cuda, trace, present=None):
    """Opacity, colour and letter matte of a faint overlay over the frames present (by default the scan's), checked
    on the scenes it was found in."""
    mark = layer.mark
    present = overlay_presence(video, layer) if present is None else present
    shape = (video.height, video.width)
    lh = mark.line_height
    keep = np.zeros(shape, np.uint8)
    view, section = clipped_plane(keep, (*layer.base, layer.keep.shape[1], layer.keep.shape[0]))
    view[:] = layer.keep[section]
    region = np.zeros(shape, np.uint8)
    for polygon in mark.polygons:
        cv2.fillPoly(region, [np.round(polygon).astype(np.int32)], 1)
    # Other marks' pixels are repainted with their own layers (their frames and brackets reach past their text).
    near = (cv2.dilate(region, ellipse(min(.3*lh, 24))) > 0) & ~(cv2.dilate(mark.known, ellipse(9)) > 0) & ~exclude
    # The consensus found the strokes' edges; closing over a stroke width covers their insides too.
    letters = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, ellipse(max(3, round(.12*lh)))) > 0
    gate = (cv2.dilate(letters.astype(np.uint8), ellipse(3)) > 0) & near
    if not gate.any():
        raise ValueError(f'The faint overlay {mark.label} has no letters left to unblend.')
    x, y, w, h = grow(cv2.boundingRect(gate.astype(np.uint8)), 12, 12)
    place = (max(0, x), max(0, y), min(video.width, x+w)-max(0, x), min(video.height, y+h)-max(0, y))
    means, count, plain = mean_picture(video, present, cuda, place)
    mean = means[0]
    background = cv2.medianBlur(np.clip(np.rint(mean), 0, 255).astype(np.uint8), 61).astype(np.float64)
    light = float(np.median((mean-background)[letters & near])) >= 0
    colour = UNBLEND_LIGHT if light else UNBLEND_DARK
    # Large letters fill much of any window a median would take, raising it and understating the opacity; the
    # background under them is inpainted from around them instead. With the whole gate as the hole, strokes wider
    # than the median's reach keep no level of their own inside (they would read as no overlay at all); but a gate
    # much wider than the strokes leaves the inpainting to guess the picture across it, and a background shading
    # under the letters comes out as false letters and gaps. The strokes alone, found by a top-hat (the mean picture
    # less its opening by a disk wider than a stroke), leave it only their own width to bridge, but miss the insides
    # of strokes wider than the disk (an emblem's bars run to a fifth of the line height). Both are fitted; the one
    # that removes more of the overlay is kept. The consensus' closing bridges only thin strokes, so the gate keeps
    # holes inside wide ones: the strokes are looked for inside the gate closed over the same disk.
    disk = ellipse(max(3, round(UNBLEND_STROKE*lh)))
    hat = cv2.morphologyEx(mean.astype(np.float32), cv2.MORPH_TOPHAT if light else cv2.MORPH_BLACKHAT, disk)
    hat = cv2.GaussianBlur(hat, (0, 0), 1.)
    search = (cv2.morphologyEx(gate.astype(np.uint8), cv2.MORPH_CLOSE, disk) > 0) & near
    strokes = (hat >= UNBLEND_STROKE_SHARE*float(np.percentile(hat[gate], 90))) & search
    holes = {'gate': cv2.dilate(gate.astype(np.uint8), ellipse(5))}
    pieces = None   # the overlay's glyphs (its strokes within a few pixels of each other), judged one by one
    if strokes.sum() >= 100:
        holes['strokes'] = cv2.dilate(strokes.astype(np.uint8), ellipse(3))
        pieces = cv2.connectedComponents(cv2.dilate(strokes.astype(np.uint8), ellipse(5)), connectivity=8)[1]
    fits, failure = [], None
    for name, hole in holes.items():
        try:
            unblend = blend_fit(video, samples, layer, present, means, count, plain, place, letters, near, gate,
                                hole, light, pieces)
        except ValueError as error:
            failure = error
            continue
        unblend.check['hole'] = name
        fits.append(unblend)
    # And fitted frame by frame, with the overlay's own blend law and colour, under the strokes alone.
    hole = holes.get('strokes', holes['gate'])
    core = cv2.erode((strokes if 'strokes' in holes else gate).astype(np.uint8), ellipse(2)) > 0
    try:
        unblend = regress_fit(video, samples, layer, present, place, letters, near, hole, core, light, cuda, pieces)
    except ValueError as error:
        failure = error
    else:
        unblend.check['hole'] = 'regressed'
        fits.append(unblend)
    if not fits:
        raise failure
    unblend = max(fits, key=lambda u: u.check['removed'])
    trace('plan.unblend_fit', label=mark.label, rect=list(unblend.rect), **unblend.check,
          tried={u.check['hole']: u.check['removed'] for u in fits})
    return unblend


def blend_fit(video, samples, layer, present, means, count, plain, place, letters, near, gate, hole, light,
              pieces=None):
    """The inverse blend of a faint overlay with the background under its letters inpainted over hole, checked on
    the sampled scenes showing it (fit_overlay)."""
    mark = layer.mark
    shape = (video.height, video.width)
    mean = means[0]
    colour = UNBLEND_LIGHT if light else UNBLEND_DARK
    background = without_letters(mean, hole)
    room = colour-background
    room = np.maximum(room, 30) if light else np.minimum(room, -30)
    ratio = (mean-background)/room
    opacity, core = overlay_opacity(ratio, gate, mark.label)
    deviations = [(mean-background, background)]
    for chroma, scale in zip(means[1:], PLANE_SCALES[1:]):
        small = cv2.resize(hole, (chroma.shape[1], chroma.shape[0]), interpolation=cv2.INTER_NEAREST)
        base = without_letters(chroma, small)
        deviations.append((chroma-base, base))
    # The overlay's colour: the chroma deviations average over the same scenes as the smooth mean matte.
    matte = np.clip(cv2.GaussianBlur(ratio/opacity, (0, 0), 1.2), 0, 1)*gate
    smooth = (opacity*cv2.GaussianBlur(matte.astype(np.float32), (0, 0), .7)).astype(np.float32)
    colours = [colour]
    for p, scale in enumerate(PLANE_SCALES[1:], 1):
        a = cv2.resize(smooth, (shape[1]//scale, shape[0]//scale), interpolation=cv2.INTER_AREA)
        chroma, base = deviations[p]
        reliable = a >= .5*opacity
        c = 128.
        if reliable.sum() >= 50:
            c = float(np.clip(np.median((base+chroma/np.maximum(a, 1e-3))[reliable]), 16, 240))
        colours.append(round(c, 1))
    # The sharp matte: off plain frames where there are any, else the mean picture's own, unsmoothed.
    sharp, used = plain_matte(plain, place, hole, near, gate, core, colour, opacity)
    if sharp is not None:
        alpha = np.zeros(shape, np.float32)
        alpha[place[1]:place[1]+place[3], place[0]:place[0]+place[2]] = sharp
        source = 'plain'
    else:
        alpha = (opacity*np.clip(ratio/opacity, 0, 1)*gate).astype(np.float32)
        source = 'mean'
    unblend, check = checked_unblend(video, samples, layer, present, alpha, colours, 'normal', letters, near,
                                     pieces)
    unblend.check = {'opacity': round(opacity, 4), 'colours': colours, 'frames': count,
                     'present': int(present.sum()), 'letters': int(core.sum()), 'matte': source, 'mode': 'normal',
                     'plain_frames': [int(i) for i in used], **check}
    return unblend


def regress_fit(video, samples, layer, present, place, letters, near, hole, core, light, cuda, pieces=None):
    """The inverse blend of a faint overlay fitted frame by frame: each frame's picture under the strokes inpainted
    from around them, the blend law (blend_law) from the strokes' rise over it, and each pixel's alpha the
    least-squares fit of its rise to that law over all the frames showing the overlay, a frame weighing the less the
    busier the picture around. The mean picture's matte takes the overlay to be white (black) and the scenes behind
    to average out; a grey overlay that lifts only the darker picture, or a clip of a few long scenes, defeats it."""
    mark = layer.mark
    shape = (video.height, video.width)
    x, y, w, h = place
    hole_c = np.ascontiguousarray(hole[y:y+h, x:x+w]).astype(np.uint8)
    core_c = core[y:y+h, x:x+w]
    if core_c.sum() < 100:
        raise ValueError(f'The faint overlay {mark.label} shows too few stroke pixels to fit.')

    def frames(limit):
        for i, raw in read_frames(video, cuda=cuda, step=max(1, video.count//limit)):
            if present[i]:
                planes = video.planes(raw)
                crop = planes[0][y:y+h, x:x+w].astype(np.float32)
                level = without_letters(crop, hole_c).astype(np.float32)
                rough = np.abs(crop-cv2.GaussianBlur(crop, (0, 0), 2))
                rough[hole_c > 0] = 0
                yield planes, crop, level, cv2.dilate(rough, np.ones((15, 15), np.uint8))

    levels, rises = [], []
    for _, crop, level, rough in frames(UNBLEND_LAW_FRAMES):
        calm = core_c & (rough <= UNBLEND_CALM)
        levels.append(level[calm])
        rises.append((crop-level)[calm])
    law = blend_law(np.concatenate(levels), np.concatenate(rises), light) if levels else None
    if law is None:
        raise ValueError(f'The faint overlay {mark.label} lies on too little calm picture to tell its blend.')
    mode, opacity, colour, misfit, plain_misfit = law
    num, den = np.zeros((h, w)), np.zeros((h, w))
    tints, count = [[], []], 0
    for planes, crop, level, rough in frames(UNBLEND_FRAMES):
        count += 1
        span = law_span(mode, colour-level)
        use = (hole_c > 0) & (np.abs(span) >= UNBLEND_LEAST_SPAN)
        weight = np.where(use, 1/(1+np.square(rough/UNBLEND_CALM)), 0.)
        num += weight*(crop-level)*span
        den += weight*span*span
        # The overlay's colour from the steady middle of its strokes, wherever it shows there.
        sure = (core_c & use & (rough <= UNBLEND_CALM)).astype(np.float32)
        for p in (1, 2):
            plane = planes[p][y//2:(y+h)//2, x//2:(x+w)//2].astype(np.float32)
            size = (plane.shape[1], plane.shape[0])
            base = without_letters(plane, cv2.resize(hole_c, size, interpolation=cv2.INTER_NEAREST))
            where = cv2.resize(sure, size, interpolation=cv2.INTER_AREA) >= .99
            tints[p-1].append((base+(plane-base)/opacity)[where])
    alpha_c = np.where(den >= 3*UNBLEND_LEAST_SPAN**2, num/np.maximum(den, 1e-9), 0.)
    alpha = np.zeros(shape, np.float32)
    alpha[y:y+h, x:x+w] = np.clip(alpha_c, 0, 1.5*opacity)*(hole_c > 0)
    colours = [colour]
    for tint in tints:
        values = np.concatenate(tint) if tint else np.zeros(0)
        colours.append(round(float(np.clip(np.median(values), 16, 240)), 1) if len(values) >= 50 else 128.)
    unblend, check = checked_unblend(video, samples, layer, present, alpha, colours, mode, letters, near, pieces)
    unblend.check = {'opacity': round(opacity, 4), 'colours': colours, 'frames': count,
                     'present': int(present.sum()), 'letters': int(core.sum()), 'matte': 'regressed', 'mode': mode,
                     'law_misfit': round(misfit, 3),
                     'white_misfit': None if plain_misfit is None else round(plain_misfit, 3), **check}
    return unblend


def checked_unblend(video, samples, layer, present, alpha, colours, mode, letters, near, pieces=None):
    """The unblend of a letter matte (alpha over the whole frame) in its colour per plane, with a check on the
    sampled scenes showing the overlay: its consensus strength inside the letters must mostly go, also glyph by
    glyph (pieces: labels of the overlay's glyphs at place, the frame's coordinates). Returns the unblend and the
    check's numbers."""
    shape = (video.height, video.width)
    alphas = [cv2.GaussianBlur(alpha, (0, 0), s) if s else alpha for s in UNBLEND_BLURS]
    x, y, w, h = grow(cv2.boundingRect((alphas[-1] > 1e-3).astype(np.uint8)), 2, 2)
    left, top = max(0, x)//2*2, max(0, y)//2*2
    right = min(video.width, -(-(x+w)//2)*2)
    bottom = min(video.height, -(-(y+h)//2)*2)
    rect = (left, top, right-left, bottom-top)
    variants = []
    for a_full in alphas:
        fields = []
        for p, scale in enumerate(PLANE_SCALES):
            a = a_full
            if scale > 1:
                a = cv2.resize(a_full, (shape[1]//scale, shape[0]//scale), interpolation=cv2.INTER_AREA)
            crop = np.ascontiguousarray(a[top//scale:bottom//scale, left//scale:right//scale])
            fields.append((1-crop, crop*colours[p]))
        variants.append(fields)
    zone = ((cv2.dilate(letters.astype(np.uint8), ellipse(4)) > 0) & near)[top:bottom, left:right]
    patterns = [highpass(a[top:bottom, left:right], 1.5)[zone] for a in alphas]
    unblend = Unblend(layer.mark.label, rect, variants, present, {}, colours[0], zone, patterns, mode=mode)
    background_level, _, sigma = layer.levels
    # Scenes without the overlay show none of it before and its negative after: only those showing it tell.
    chosen = [i for i in samples.overlay[0] if present[i]]
    chosen = chosen if len(chosen) >= UNBLEND_CHECKED else samples.overlay[0]
    lumas = [video.planes(samples.raws[i])[0][top:bottom, left:right].astype(np.float32) for i in chosen]
    before = overlay_strength([highpass(l, HIGHPASS_SIGMA) for l in lumas], sigma)
    restored, blurs = [], [0]*len(variants)
    for l in lumas:
        unblend.last = None
        k = unblend.choose(l)
        blurs[k] += 1
        restored.append(highpass(np.clip(np.rint(unblend.restore_luma(l, k)), 0, 255), HIGHPASS_SIGMA))
    unblend.last = None
    after = overlay_strength(restored, sigma)
    where = letters[top:bottom, left:right] & near[top:bottom, left:right]
    removed = (before[where].mean()-after[where].mean())/max(before[where].mean()-background_level, 1e-3)
    parts, rest = [], None
    if pieces is not None:
        parts, unblend.poor, rest = part_removals(before, after, where, pieces[top:bottom, left:right],
                                                  background_level)
    return unblend, {'blurs': blurs, 'strength_before': round(float(before[where].mean()), 3),
                     'strength_after': round(float(after[where].mean()), 3), 'removed': round(float(removed), 3),
                     'parts': parts, 'removed_rest': rest}


def part_removals(before, after, where, pieces, background):
    """The unblend check's removal per glyph of an overlay (pieces: labelled glyphs, 0 elsewhere) over its letters
    (where) from its strength before and after; the glyphs that show clearly and of which less than UNBLEND_CLEAN
    goes are left to repair. Returns the glyphs' summaries, the mask of those left (None if none) and the removal
    over the rest of the letters (None if nothing is left, or nothing else)."""
    shown = float(before[where].mean())-background
    poor = np.zeros(where.shape, bool)
    parts = []
    for part in np.unique(pieces[where]):
        glyph = pieces == part
        sel = where & glyph
        if not part or sel.sum() < UNBLEND_PART_LEAST:
            continue
        b, a = float(before[sel].mean()), float(after[sel].mean())
        if b-background < UNBLEND_PART_SHOWN*shown:
            continue
        removed = (b-a)/max(b-background, 1e-3)
        parts.append({'box': [int(v) for v in cv2.boundingRect(glyph.astype(np.uint8))], 'pixels': int(sel.sum()),
                      'removed': round(removed, 3)})
        if removed < UNBLEND_CLEAN:
            poor |= glyph
    rest = where & ~poor
    if not poor.any() or rest.sum() < UNBLEND_PART_LEAST:
        return parts, (poor if poor.any() else None), None
    removed = (before[rest].mean()-after[rest].mean())/max(before[rest].mean()-background, 1e-3)
    return parts, poor, round(float(removed), 3)


def shown_frames(video, unblend, cuda):
    """Frames showing a fitted faint overlay, judged by its own letters: the strength they show with in each frame
    (Unblend.amplitude), its median over half a second at least half the fitted strength. The scan's match score
    this replaces drops wherever a busy picture hides the overlay's few strokes, and unblending must reach exactly
    the frames that show it: a frame left out keeps the overlay, and one without it gets its negative printed.
    Frames whose picture is too near the overlay's colour to tell keep the scan's verdict. Returns the presence and
    the per-frame strengths."""
    x, y, w, h = unblend.rect
    scales = np.full(video.count, np.nan, np.float32)
    powers = np.zeros(video.count, np.float32)
    for i, raw in read_frames(video, cuda=cuda):
        scales[i], powers[i] = unblend.amplitude(video.planes(raw)[0][y:y+h, x:x+w])
    told = powers >= UNBLEND_SHOWN_WEAK*float(np.median(powers[powers > 0])) if np.any(powers > 0) else powers > 0
    scales[~told] = np.nan
    window = max(1, round(UNBLEND_SHOWN_SPAN*float(video.fps))) | 1
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(scales, window//2, mode='edge'), window)
    counted = np.isfinite(windows).sum(axis=1)
    smooth = np.full(video.count, np.nan, np.float32)
    enough = counted >= max(1, window//3)
    if enough.any():
        smooth[enough] = np.nanmedian(windows[enough], axis=1)
    present = unblend.present.copy()
    known = np.isfinite(smooth)
    present[known] = smooth[known] >= UNBLEND_SHOWN
    if present.mean() >= .9:
        present[:] = True
    return present, scales


def write_unblended(video, unblends, output, cuda):
    for unblend in unblends:
        unblend.last, unblend.chosen = None, [0]*len(unblend.variants)
    with encode(video, output, 0, video.count, 12) as stream:
        for i, raw in read_frames(video, cuda=cuda):
            planes = video.planes(raw)
            for unblend in unblends:
                if unblend.present[i]:
                    unblend.apply(planes)
                else:
                    unblend.last = None
            stream.write(raw)
    return Video.open(output)


def fit_unblends(video, samples, layers, cuda, trace):
    """Fit the inverse blend of each faint overlay and keep those that remove enough of it, with the frames that show
    it judged by its fitted letters; each faint layer takes those frames as its presence (a reading under an overlay
    it shows in is its lettering, not a logo of the scene)."""
    unblends = []
    exclude = np.zeros((video.height, video.width), np.uint8)
    for layer in layers:
        if layer.mark.kind == 'static' and layer.base is not None:
            view, section = clipped_plane(exclude, (*layer.base, layer.motif.width, layer.motif.height))
            np.maximum(view, layer.motif.mask[section], out=view)
    exclude = cv2.dilate(exclude, ellipse(5)) > 0
    for layer in layers:
        if layer.mark.kind != 'faint':
            continue
        try:
            unblend = fit_overlay(video, samples, layer, exclude, cuda, trace)
        except ValueError as error:
            trace('plan.unblend_rejected', label=layer.mark.label, reason=str(error))
            continue
        scanned = unblend.present
        present, scales = shown_frames(video, unblend, cuda)
        # Stretches the scan took in without the overlay thin the mean picture's letters, and their plain frames
        # (which show none of it) give no matte: fitted again over the frames its letters show in.
        dropped = int((scanned & ~present).sum())
        if dropped >= UNBLEND_REFIT*float(video.fps):
            try:
                refit = fit_overlay(video, samples, layer, exclude, cuda, trace, present)
            except ValueError as error:
                trace('plan.unblend_refit', label=layer.mark.label, dropped=dropped, error=str(error))
            else:
                trace('plan.unblend_refit', label=layer.mark.label, dropped=dropped,
                      removed=refit.check['removed'], was=unblend.check['removed'])
                unblend = refit
                present, scales = shown_frames(video, unblend, cuda)
        rest = unblend.check.get('removed_rest')
        if unblend.check['removed'] < UNBLEND_ACCEPT and (rest is None or rest < UNBLEND_ACCEPT):
            trace('plan.unblend_rejected', label=layer.mark.label, reason='too little of the overlay removed')
            continue
        changed = present != scanned
        told = np.isfinite(scales)
        trace('plan.unblend_shown', label=layer.mark.label, present=int(present.sum()),
              scan_present=int(scanned.sum()), added=int((present & changed).sum()),
              removed=int((~present & changed).sum()), untold=int((~told).sum()),
              strength_median=round(float(np.median(scales[told])), 3) if told.any() else None,
              strength_p05=round(float(np.percentile(scales[told], 5)), 3) if told.any() else None,
              absent=[[a, b] for a, b in runs(~present)][:20])
        unblend.present = present
        unblend.check['present'] = int(present.sum())
        layer.present = present.copy()
        unblends.append(unblend)
    return unblends


def unblend_overlays(video, directory, unblends, cuda, trace):
    """Remove fitted faint overlays by inverse blending; returns the video to continue with and the overlays."""
    if not unblends:
        return video, []
    result = write_unblended(video, unblends, directory/'unblended.mp4', cuda)
    trace('plan.unblended', labels=[u.label for u in unblends], path=str(result.path),
          blurs={u.label: u.chosen for u in unblends}, sigmas=list(UNBLEND_BLURS))
    return result, unblends


def remnant_label(label):
    return f'{label} (parts left)'


def remnant_layer(video, unblend, faint, settings):
    """The repair layer over the parts of an overlay its inverse blend leaves (Unblend.poor): the faint layer's own
    picture of the overlay, masked to those parts grown like any mask, at its place in the frames the overlay was
    unblended in."""
    label = remnant_label(unblend.label)
    left, top, w, h = unblend.rect
    full = np.zeros((video.height, video.width), np.uint8)
    full[top:top+h, left:left+w] = unblend.poor
    growth = mask_growth(settings, label)
    if growth > 0:
        full = cv2.dilate(full, ellipse(2*growth+1))
    x, y = faint.base
    mask = patch(full, (x, y, faint.motif.width, faint.motif.height))
    if np.count_nonzero(mask) < 32:
        return None
    count = video.count
    positions = np.tile(np.array(faint.base, np.int32), (count, 1))
    present = unblend.present.copy()
    mark = Mark('faint', label, faint.mark.keys, faint.mark.line_height, faint.mark.observations, faint.mark.lines)
    motif = Motif(faint.motif.signal, faint.motif.weights, np.ascontiguousarray(mask, np.uint8))
    return Layer(mark, motif, faint.origin, faint.base, faint.positive, positions, present.astype(np.float32),
                 np.full(count, -1., np.float32), present, positions.copy())


def unblend_removal(unblend, repaired):
    """The share of an unblended overlay removed: of its letters, or with the parts it left repaired (repaired),
    of the rest (all of it where the parts left are all there is)."""
    if not repaired or unblend.poor is None:
        return unblend.check['removed']
    rest = unblend.check.get('removed_rest')
    return 1. if rest is None else rest


# ----------------------------------------------------------------------------
# Whole-video tracking

@dataclass
class Shadow:
    """Where a moving motif would match a fixed badge's own picture.

    `zone[b, a]` is set when the badge alone explains a placement of the moving motif centred at (a, b) relative
    to the badge's top-left; `small` is the same at half resolution. A moving copy of a corner badge would
    otherwise lock onto the sharper badge itself, yet it may still pass over or next to the badge."""
    zone: np.ndarray
    small: np.ndarray


def shadow(moving, fixed):
    px, py = moving.width//2, moving.height//2
    padded = cv2.copyMakeBorder(fixed.signal*fixed.weights, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
    scores = response(padded, moving.signal, moving.weights)
    best = float(scores.max())
    if best < .2:
        return None
    zone = cv2.dilate((scores >= max(.15, .5*best)).astype(np.uint8), ellipse(7))
    small = cv2.resize(zone.astype(np.float32), ((zone.shape[1]+1)//2, (zone.shape[0]+1)//2),
                       interpolation=cv2.INTER_AREA)
    return Shadow(zone, cv2.dilate((small > 0).astype(np.uint8), ellipse(3)))


def in_shadow(avoid, x, y):
    for left, top, cast in avoid:
        a, b = x-left, y-top
        if 0 <= b < cast.zone.shape[0] and 0 <= a < cast.zone.shape[1] and cast.zone[b, a]:
            return True
    return False


def locate(motif, image, avoid=()):
    """Best and runner-up placement of a motif anywhere, including half outside the picture.

    `avoid` lists (left, top, Shadow) for fixed badges showing in this picture: placements the badge itself
    explains are skipped."""
    px, py = motif.width//2, motif.height//2
    padded = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=0)

    def blank(scores, scale):
        # A response index is the placement's top-left in the padded picture, i.e. its centre in the picture.
        for left, top, cast in avoid:
            zone = cast.zone if scale == 1 else cast.small
            view, section = clipped_plane(scores, (left//scale, top//scale, zone.shape[1], zone.shape[0]))
            view[zone[section] > 0] = -1
        return scores
    if padded.size <= 700000:
        scores = blank(response(padded, motif.signal, motif.weights), 1)
        ranked = list(peaks(scores, motif.width, motif.height, 2))
        (x, y, score), (_, _, other) = ranked[0], ranked[1]
        return x-px, y-py, score, other
    small = cv2.resize(padded, (padded.shape[1]//2, padded.shape[0]//2), interpolation=cv2.INTER_AREA)
    scores = blank(response(small, motif.small_signal, motif.small_weights), 2)
    guesses = peaks(scores, motif.width//2, motif.height//2, 4)
    ranked = sorted((motif.local(image, (2*x-px, 2*y-py), 5, 5) for x, y, _ in guesses), key=lambda r: -r[2])
    ranked = [r for r in ranked if not in_shadow(avoid, r[0]+px, r[1]+py)] or [(*ranked[0][:2], 0.)]
    x, y, score = ranked[0]
    other = next((s for a, b, s in ranked[1:] if abs(a-x) > motif.width//3 or abs(b-y) > motif.height//3), -1.)
    return x, y, score, other


SCAN_THREADS = max(1, min(8, (os.cpu_count() or 2)//2))  # threads matching the layers in the frames of a scan


def scan_layers(video, layers, cuda, trace):
    count = video.count
    for layer in layers:
        layer.positions = np.zeros((count, 2), np.int32)
        layer.scores = np.zeros(count, np.float32)
        layer.others = np.full(count, -1., np.float32)
    started = perf_counter()
    # Fixed layers first: where one shows, a moving layer must not take the badge's own picture for itself.
    layers = sorted(layers, key=lambda layer: layer.base is None)
    shadows = {id(moving): [(id(fixed), cast) for fixed in layers
                            if fixed.base is not None and fixed.mark.kind == 'static'
                            and (cast := shadow(moving.motif, fixed.motif)) is not None]
               for moving in layers if moving.base is None}
    def placed(raw):
        """Each layer's (x, y, score, runner-up) in one frame, which no other frame bears on."""
        signal = highpass(video.planes(raw)[0], HIGHPASS_SIGMA)
        shown, found = {}, []
        for layer in layers:
            if layer.base is not None:
                x, y, score = layer.motif.local(signal, layer.base, 2, 2)
                other = -1.
                if layer.mark.kind == 'static' and score >= max(.22, .5*layer.positive):
                    shown[id(layer)] = x, y
            else:
                avoid = [(*shown[key], cast) for key, cast in shadows[id(layer)] if key in shown]
                x, y, score, other = locate(layer.motif, signal, avoid)
            found.append((x, y, score, other))
        return found

    # Decoding is far faster than matching: the frames are matched on several threads (the filters and correlations
    # run outside the interpreter's lock), a bounded number of them waiting at a time.
    waiting = deque()

    def settle():
        i, job = waiting.popleft()
        for layer, (x, y, score, other) in zip(layers, job.result()):
            layer.positions[i] = x, y
            layer.scores[i] = score
            layer.others[i] = other
        if (i+1) % 600 == 0:
            print(f'Tracked {i+1}/{count} frames ({(i+1)/(perf_counter()-started):.1f} fps)', flush=True)
    with ThreadPoolExecutor(SCAN_THREADS) as pool:
        for i, raw in read_frames(video, cuda=cuda):
            waiting.append((i, pool.submit(placed, raw)))
            if len(waiting) >= 4*SCAN_THREADS:
                settle()
        while waiting:
            settle()
    for layer in layers:
        layer.located = layer.positions.copy()
    trace('plan.scanned', frames=count, seconds=round(perf_counter()-started, 2))


def reuse_scan(layer, old):
    """Carry an earlier attempt's scan over to a layer relearned with a wider mask, instead of scanning again.

    A wider mask leaves the template and the support of its weights unchanged: fixed layers match in place exactly
    as before, and a moving layer's crop only grows, which moves its placements by the change of origin."""
    if old is None or old.located is None:
        return False
    if layer.base is not None:
        layer.positions = np.tile(np.asarray(layer.base, np.int32), (len(old.scores), 1))
    else:
        layer.positions = old.located+np.subtract(layer.origin, old.origin).astype(np.int32)
        layer.positive = old.positive   # the thresholds must stay on the scale of the reused scores
    layer.scores, layer.others = old.scores.copy(), old.others.copy()
    layer.located = layer.positions.copy()
    return True


STATIC_LEAD = 1.          # seconds right next to a fixed badge's showing over which it may build up or wind down...
STATIC_LEAD_LEVEL = .2    # ...matching at least this share of its own picture's score


def decide_static(video, layer, threshold_scale):
    fps = float(video.fps)
    threshold = max(.22, .5*layer.positive*threshold_scale)
    present = layer.scores >= threshold
    present = fill_gaps(present, round(.5*fps))
    for start, stop in list(runs(present)):
        if stop-start < 3 and layer.scores[start:stop].max() < 1.3*threshold:
            present[start:stop] = False
    # Badges animate in and out: around a confident stretch, the partial states still match moderately.
    weak = layer.scores >= max(.15, .3*layer.positive*threshold_scale)
    for start, stop in list(runs(weak)):
        if present[start:stop].any():
            present[start:stop] = True
    # Some build up in parts, the lettering shown a moment before its plate (or the plate before the line below it
    # that the picture was learned with), and match only a little then. A short stretch right before or after a
    # showing that still matches that much is the badge too: repainted, and not left to bleed into the frames beside.
    # It joins the showing once the animations are learned, which count their steps from where the showing starts.
    lead = (layer.scores >= max(.15, STATIC_LEAD_LEVEL*layer.positive*threshold_scale)) & ~present
    extended = present.copy()
    for start, stop in list(runs(lead)):
        if stop-start <= STATIC_LEAD*fps and ((start > 0 and present[start-1]) or
                                              (stop < len(present) and present[stop])):
            extended[start:stop] = True
    present = widen(present, 2)
    layer.lead = widen(extended, 2) & ~present
    layer.positions[:] = layer.base
    layer.present = present
    return threshold


def visible_share(motif, position, video):
    """Share of the mask that lies inside the picture at this placement."""
    x, y = (int(v) for v in position)
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(motif.width, video.width-x), min(motif.height, video.height-y)
    if right <= left or bottom <= top:
        return 0.
    return float(motif.mask[top:bottom, left:right].sum())/max(1, int(motif.mask.sum()))


def clipped_local(motif, image, guess, radius_x, radius_y):
    """Like Motif.local, but scored only on the part of the motif inside the picture at the guess, so that a
    mark half across the border scores as well as its visible part matches."""
    x, y = guess
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(motif.width, image.shape[1]-x), min(motif.height, image.shape[0]-y)
    if right-left < 8 or bottom-top < 8:
        return x, y, 0.
    weights = np.ascontiguousarray(motif.weights[top:bottom, left:right])
    if np.count_nonzero(weights) < 32:
        return x, y, 0.
    signal = np.ascontiguousarray(motif.signal[top:bottom, left:right])
    region = patch(image, (x+left-radius_x, y+top-radius_y, right-left+2*radius_x, bottom-top+2*radius_y))
    scores = response(region, signal, weights)
    _, _, _, (a, b) = cv2.minMaxLoc(scores)
    return x-radius_x+a, y-radius_y+b, float(scores[b, a])


def end_velocity(positions, start, stop):
    """Average motion per frame over a few frames, in pixels."""
    if stop-start < 2:
        return np.zeros(2)
    return (positions[stop-1]-positions[start]).astype(np.float64)/(stop-1-start)


def steady(positions, start, stop):
    """Snap near-still stretches to their median so a resting mark does not jitter by a pixel."""
    current = start
    for i in range(start+1, stop+1):
        if i == stop or np.abs(positions[i]-positions[current]).max() > 3:
            if i-current >= 3:
                positions[current:i] = np.median(positions[current:i], axis=0).round()
            current = i


# Many marks glide along one scripted loop again and again. Where the loop is known from frames kept elsewhere, a
# frame it passes over a busy or covered part of the picture still matches the motif there clearly better than
# anywhere off the path (about a quarter against a twentieth), though too weakly to be kept on its own.
REPEAT_AGREE = 2       # pixels by which a gliding mark may differ from its own position one loop later
REPEAT_WINDOW = .25    # seconds on each side of a frame over which scores along the repeated path are averaged
REPEAT_GAP = 1.        # seconds of weak frames inside a repeated stretch that are kept with the frames around them


def path_period(present, positions, fps):
    """Frames after which a gliding mark retraces its path, or None.

    Only frames where the mark moves count: a mark resting in one place agrees with itself after any delay."""
    count = len(present)
    moving = np.zeros(count, bool)
    moving[:-1] = present[:-1] & present[1:] & (np.abs(np.diff(positions, axis=0)).max(axis=1) >= 1)
    least = round(2*fps)
    best, period = 0, None
    for shift in range(least, count-least):
        both = moving[:-shift] & present[shift:]
        pairs = int(both.sum())
        if pairs < max(least, best):
            continue
        agree = int((both & (np.abs(positions[:-shift]-positions[shift:]).max(axis=1) <= REPEAT_AGREE)).sum())
        if agree > best and agree >= .8*pairs:
            best, period = agree, shift
    return period if best >= least else None


def repeat_path(video, layer, present, positions, low, cuda, trace):
    """Keep the frames where a gliding mark, absent so far, matches along the place its loop puts it then."""
    fps = float(video.fps)
    period = path_period(present, positions, fps)
    if period is None:
        return
    count = video.count
    guesses = {}
    for i in np.flatnonzero(~present):
        for k in sorted((k for k in range(-(count//period), count//period+1) if k), key=abs):
            j = i+k*period
            if 0 <= j < count and present[j]:
                guesses[int(i)] = tuple(int(v) for v in positions[j])
                break
    wanted = np.zeros(count, bool)
    wanted[list(guesses)] = True
    found = {}
    for start, stop in runs(wanted):
        for i, raw in read_frames(video, start, stop, cuda):
            guess = guesses[i]
            shown = visible_share(layer.motif, guess, video)
            if shown < .25:
                found[i] = (*guess, np.nan)
                continue
            signal = highpass(video.planes(raw)[0], HIGHPASS_SIGMA)
            local = layer.motif.local if shown >= .97 else partial(clipped_local, layer.motif)
            found[i] = local(signal, guess, 3, 3)
    threshold, half = .5*low, max(2, round(REPEAT_WINDOW*fps))
    recovered = 0
    for start, stop in runs(wanted):
        scores = np.array([found[i][2] for i in range(start, stop)])
        kept = np.zeros(stop-start, bool)
        for k in range(stop-start):
            window = scores[max(0, k-half):k+half+1]
            kept[k] = np.isfinite(window).any() and np.nanmean(window) >= threshold
        kept = fill_gaps(kept, round(REPEAT_GAP*fps))
        for k in np.flatnonzero(kept):
            i = start+int(k)
            x, y, score = found[i]
            present[i] = True
            positions[i] = (x, y) if score >= threshold else guesses[i]
        recovered += int(kept.sum())
    trace('plan.moving_repeat', label=layer.mark.label, period=period, examined=len(guesses), recovered=recovered)


TRACE_OUTLIERS = .1    # share of a gap's frames whose best placement may jump off the path leading through it


def continuous_gaps(present, positions, located, scores, step, low):
    """Keep gaps between kept frames whose best placements lead continuously from one end to the other.

    The mark passes over a busier part of the picture and matches too weakly to be kept, yet each frame's best
    placement anywhere is still where it is: a path without jumps that meets both ends. A slow glide also defeats
    the motion carried out of each end. A mark that is really gone leaves best placements scattered over the picture.
    Returns the number of frames kept."""
    kept = 0
    spans = list(runs(present))
    for (_, stop), (start, _) in zip(spans, spans[1:]):
        path = np.vstack([positions[stop-1:stop], located[stop:start], positions[start:start+1]]).astype(np.float64)
        smooth = np.array([np.median(path[max(0, k-2):k+3], axis=0) for k in range(len(path))])
        off = np.abs(path-smooth).max(axis=1) > step
        off[0] = off[-1] = False
        if off[1:-1].mean() > TRACE_OUTLIERS or np.median(scores[stop:start]) < .5*low:
            continue
        good = np.flatnonzero(~off)
        if np.any(np.abs(np.diff(path[good], axis=0)).max(axis=1) > step*np.diff(good)):
            continue
        for axis in range(2):
            path[:, axis] = np.interp(np.arange(len(path)), good, path[good, axis])
        positions[stop:start] = np.round(path[1:-1])
        present[stop:start] = True
        kept += start-stop
    return kept


def decide_moving(video, layer, cuda, threshold_scale, trace):
    """Keep confident placements, then follow each stretch outward into weaker or partly hidden frames.

    A mark may rest and jump (holds) or glide continuously, like a marquee crossing the picture; both are
    stretches whose position changes by at most a few pixels between neighbouring frames.
    """
    fps = float(video.fps)
    high = max(.28, .5*layer.positive*threshold_scale)
    low = max(.16, .3*layer.positive*threshold_scale)
    step = max(6., .5*layer.mark.line_height)
    located = layer.positions
    confident = (layer.scores >= high) & (layer.scores-layer.others >= .05)
    stretches = []
    for start, stop in runs(confident):
        current = start
        for i in range(start+1, stop+1):
            if i == stop or np.abs(located[i]-located[i-1]).max() > step:
                stretches.append((current, i))
                current = i
    stretches = [(a, b) for a, b in stretches if b-a >= 2 or layer.scores[a:b].max() >= 1.3*high]
    present = np.zeros(video.count, bool)
    positions = located.copy()
    for start, stop in stretches:
        present[start:stop] = True
        steady(positions, start, stop)
    # Predict where the mark is just before and after each stretch, carrying its motion forward.
    limit = round(2*fps)
    pending = {}   # frame -> [(predicted position, radius, side key)]
    ends = []
    for index, (start, stop) in enumerate(stretches):
        lo = stretches[index-1][1] if index else 0
        hi = stretches[index+1][0] if index+1 < len(stretches) else video.count
        for side, edge, direction in ((('before', index), start, -1), (('after', index), stop-1, 1)):
            window = (start, min(stop, start+8)) if direction < 0 else (max(start, stop-8), stop)
            velocity = end_velocity(positions, *window)
            moving = np.abs(velocity).max() >= .5
            # A gliding mark may need longer than the usual reach to cross the border completely.
            reach = 3*limit if moving else limit
            frames = range(start-1, max(lo, start-reach)-1, -1) if direction < 0 else range(stop, min(hi, stop+reach))
            ends.append((side, list(frames), moving, visible_share(layer.motif, positions[edge], video)))
            for i in frames:
                distance = abs(i-edge)
                predicted = positions[edge]+velocity*(i-edge)
                radius = min(10, 3+int(.15*distance)) if moving else 2
                pending.setdefault(i, []).append((predicted, radius, side))
    found = {}
    if pending:
        for start, stop in runs(np.isin(np.arange(video.count), sorted(pending))):
            for i, raw in read_frames(video, start, stop, cuda):
                signal = highpass(video.planes(raw)[0], HIGHPASS_SIGMA)
                for predicted, radius, side in pending[i]:
                    guess = tuple(int(round(v)) for v in predicted)
                    shown = visible_share(layer.motif, guess, video)
                    if shown >= .97:
                        x, y, score = layer.motif.local(signal, guess, radius, radius)
                    elif shown >= .25:
                        x, y, score = clipped_local(layer.motif, signal, guess, radius, radius)
                    else:
                        x, y, score = *guess, 0.
                    found[i, side] = (x, y, score, guess)
    recovered = leaving = 0
    tolerance = max(2, round(.2*fps))
    for side, frames, moving, edge_shown in ends:
        misses = []
        for i in frames:
            if present[i]:
                break
            x, y, score, guess = found[i, side]
            shown = visible_share(layer.motif, guess, video)
            if score >= low:
                for j, position in misses:
                    present[j] = True
                    positions[j] = position
                recovered += len(misses)+1
                misses = []
                present[i] = True
                positions[i] = x, y
            elif not moving:
                continue
            elif shown == 0:
                break
            elif shown < .6 or (shown < min(.97, edge_shown-.02) and score >= .5*low):
                # Entering or leaving the picture: further across the border than where it was last seen, what
                # remains matches only weakly, and once most of it is gone too little remains to match at all; the
                # motion is known. While most of it is inside, a mark that no longer matches at all has not left
                # but vanished in place (the next copy appears elsewhere), so it needs that weak match.
                for j, position in misses+[(i, guess)]:
                    present[j] = True
                    positions[j] = position
                leaving += len(misses)+1
                misses = []
            else:
                misses.append((i, guess))
                if len(misses) > tolerance:
                    break
    if pending:
        trace('plan.moving_rescored', label=layer.mark.label, examined=len(pending), recovered=recovered,
              leaving=leaving)
    repeat_path(video, layer, present, positions, low, cuda, trace)
    traced = continuous_gaps(present, positions, located, layer.scores, step, low)
    if traced:
        trace('plan.moving_traced', label=layer.mark.label, frames=traced)
    # Short unexplained gaps between consistent placements are kept as faint marks.
    spans = list(runs(present))
    for (before, stop), (start, after) in zip(spans, spans[1:]):
        gap = start-stop
        if gap > round(.5*fps):
            continue
        jump = np.abs(positions[start]-positions[stop-1]).max()
        # Bridge a resting mark, or a gliding one whose motion carries it across the gap; never a jump.
        velocity = end_velocity(positions, max(before, stop-8), stop)
        carried = np.abs(positions[stop-1]+velocity*(gap+1)-positions[start]).max()
        if jump <= 3 or (np.abs(velocity).max() >= .5 and carried <= step):
            for k, i in enumerate(range(stop, start), 1):
                positions[i] = np.round(positions[stop-1]+(positions[start]-positions[stop-1])*k/(gap+1))
            present[stop:start] = True
    layer.present = present
    layer.positions = positions
    return high, low, len(stretches)


# ----------------------------------------------------------------------------
# Recipe

def inside_window(mass, starts, size):
    """The run of `size` of an axis of a mark larger than the picture to keep: of the stretches of it the frames
    show (starts: each frame's first shown index, the frame `size` long), the run showing the most of its mask."""
    total = np.concatenate([[0.], np.cumsum(mass)])
    span = len(mass)
    first = max(0, int(starts.min()))
    last = min(span, int(starts.max())+size)
    if last-first <= size:
        return first, last
    best, chosen = -1., first
    for start in range(first, last-size+1):
        lows = np.clip(np.maximum(start, starts), 0, span)
        highs = np.clip(np.minimum(start+size, starts+size), 0, span)
        shown = float(np.where(highs > lows, total[highs]-total[lows], 0.).sum())
        if shown > best:
            best, chosen = shown, start
    return chosen, chosen+size


def track_dict(layer, video):
    positions = layer.positions.astype(np.int32)
    signal, weights, mask = layer.motif.signal, layer.motif.weights, layer.motif.mask
    h, w = mask.shape
    if (w > video.width or h > video.height) and layer.present.any():
        # A mark grown past the picture (the large steps of a zoom): the track keeps the part of it the frames show,
        # as no frame can show more than the picture's size of it.
        x, y = positions[layer.present].T.astype(np.int64)
        left, right = inside_window(mask.sum(axis=0).astype(np.float64), -x, video.width) if w > video.width             else (0, w)
        top, bottom = inside_window(mask.sum(axis=1).astype(np.float64), -y, video.height) if h > video.height             else (0, h)
        signal, weights, mask = (a[top:bottom, left:right] for a in (signal, weights, mask))
        positions = positions+np.array([left, top], np.int32)
    return dict(positions=positions, scores=layer.scores.astype(np.float32),
                present=layer.present.astype(bool), verified=np.ones(video.count, bool),
                signal=np.ascontiguousarray(signal, np.float32), weights=np.ascontiguousarray(weights, np.float32),
                mask=np.ascontiguousarray(mask, np.uint8))


def write_recipe(video, directory, layers, cuts, identity, settings):
    target = directory
    target.mkdir(parents=True)
    tracks = [track_dict(layer, video) for layer in layers]
    tracks[0]['cuts'] = np.asarray(cuts, np.int32)
    track = pack_tracks(tracks) if len(tracks) > 1 else tracks[0]
    if len(tracks) > 1:
        track['cuts'] = np.asarray(cuts, np.int32)
    asset = target/'track.npz'
    save_npz(asset, **track, source_sha256=identity)
    profile = {'size': [video.width, video.height],
               'discovery': {'track': str(asset), 'sha256': digest(asset),
                             'label': ' | '.join(layer.mark.label for layer in layers),
                             'layer_count': len(layers), 'separate_jumps': True, 'area_limit': MARK_AREA_LIMIT}}
    write_object(target/'profile.json', profile)
    recipe = target/'recipe.json'
    # The OCR check decides acceptance; the pixel audit is optional evidence and costs a full extra decode.
    write_object(recipe, {'profile': 'profile.json', 'final_render': settings.get('final_render', ['--cq', '16']),
                          'audit': bool(settings.get('audit'))})
    return recipe


def preview_layers(samples, layers, directory):
    """Write one image per layer: the sampled frame with the mask outline at the tracked position. Animation steps
    show only in a few frames each and get no image."""
    for number, layer in enumerate(layers):
        if layer.mark.kind == 'part':
            continue
        shown = [i for i in samples.frames if layer.present[i]][:4]
        tiles = []
        for i in shown:
            image = samples.bgr(i).copy()
            x, y = (int(v) for v in layer.positions[i])
            mask = np.zeros(image.shape[:2], np.uint8)
            region = patch(layer.motif.mask, (-x, -y, image.shape[1], image.shape[0]))
            mask[:] = region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            pad = int(layer.mark.line_height*2)
            left, top = max(0, x-pad), max(0, y-pad)
            right = min(image.shape[1], x+layer.motif.width+pad)
            bottom = min(image.shape[0], y+layer.motif.height+pad)
            tiles.append(cv2.resize(image[top:bottom, left:right], (480, max(1, round(480*(bottom-top)/(right-left))))))
        if tiles:
            cv2.imwrite(str(directory/f'layer{number}.jpg'), np.concatenate(tiles, axis=0))


# ----------------------------------------------------------------------------
# Acceptance

RESIDUAL_INSIDE = .6    # share of a residual's box inside a repaired region for that region's mark to have left it
STAYING = ('static', 'hold', 'faint')   # layers whose mark keeps one place over the frames they repair


def layer_place(layer):
    """Where a layer that stays in place sits: the median of its placements over the frames it is present in."""
    shown = layer.positions[layer.present] if layer.present.any() else layer.positions
    return tuple(float(v) for v in np.median(shown, axis=0))


EDGE_LOOKS = (.15, .5)  # seconds before a fixed mark's showing and after it whose frames are checked as well


def edge_frames(video, layers, taken, limit):
    """Frames just outside where marks that stay in place were repainted: one that builds up or winds down there
    and matched too little to be followed shows in full in the result, and evenly spaced frames rarely land on it."""
    wanted = set()
    for layer in layers:
        if layer.mark.kind not in STAYING:
            continue
        for start, stop in runs(layer.present):
            for look in EDGE_LOOKS:
                for i in (start-round(look*float(video.fps)), stop-1+round(look*float(video.fps))):
                    if 0 <= i < video.count and not layer.present[i]:
                        wanted.add(i)
    wanted = sorted(wanted-set(taken))
    if len(wanted) > limit:
        wanted = [wanted[round(k*(len(wanted)-1)/max(1, limit-1))] for k in range(limit)]
    return sorted(set(wanted))


def residual_check(source, candidate, keys, layers, cuda, ocr, count, trace):
    """OCR the same frames of the source and the restored video and compare watermark text: evenly spaced ones, and
    as many again just outside where the marks that stay in place were repainted."""
    step = max(1, source.count//count)
    offset = min(step//2, source.count-1)
    findings, checked, confirmed = [], 0, 0

    def frames():
        yield from zip(read_frames(source, offset, cuda=cuda, step=step),
                       read_frames(candidate, offset, cuda=cuda, step=step))
        # In ascending order, as edge_frames gives them: one decoder for each run of them, not one for each frame.
        edges = edge_frames(source, layers, range(offset, source.count, step), count)
        yield from zip(read_selected(source, edges, cuda), read_selected(candidate, edges, cuda))
    pairs = frames()
    for (i, raw), (_, restored) in pairs:
        checked += 1
        before = [row for row in ocr.read(yuv_to_rgb(source.planes(raw), source.matrix)[..., ::-1].copy())
                  if max(similar(row[0], k) for k in keys) >= .6]
        after = [row for row in ocr.read(yuv_to_rgb(candidate.planes(restored), candidate.matrix)[..., ::-1].copy())
                 if max(similar(row[0], k) for k in keys) >= .6]
        confirmed += bool(before)
        signal = None
        for key, text, rect, score, line in after:
            owner, cause, covering = None, 'foreign', set()
            for index, layer in enumerate(layers):
                if layer.present[i]:
                    x, y = layer.positions[i]
                    region = (int(x), int(y), layer.motif.width, layer.motif.height)
                    if iou(rect, region) > 0:
                        covering.add(index)
                        # Lettering far larger or smaller than the mark's own lines, or mostly outside its region, is
                        # the scene showing next to or behind it (a sign the moving mark passes over), not the mark
                        # left behind.
                        if layer.mark.kind not in ('static', 'moving') or \
                                (.5 <= line/max(1., layer.mark.line_height) <= 2.
                                 and overlap_share(rect, region) >= RESIDUAL_INSIDE):
                            owner, cause = index, 'inside_mask'
            if owner is None:
                # Text outside every repaired region: a missed watermark if a learned motif matches the source there.
                # A layer repaired around this text already removed the mark that matches where it was repaired.
                signal = highpass(source.planes(raw)[0], HIGHPASS_SIGMA) if signal is None else signal
                cx, cy = center(rect)
                for index, layer in enumerate(layers):
                    if layer.mark.kind == 'scene':
                        continue    # a logo in the scene carries no picture of itself to match
                    motif = layer.motif
                    guess = (int(cx-motif.width/2), int(cy-motif.height/2))
                    reach = (motif.width//2, motif.height//2)
                    x, y, match = motif.local(signal, guess, *reach)
                    if visible_share(motif, guess, source) < .97:
                        # Half across the border: judge by the part that is in the picture.
                        clipped = clipped_local(motif, signal, guess, *reach)
                        if clipped[2] > match:
                            x, y, match = clipped
                    if index in covering and \
                            math.dist((x, y), layer.positions[i]) <= max(4., .25*layer.mark.line_height):
                        continue
                    # A mark that stays in place is missed only where it stays: its picture matching weakly elsewhere
                    # is chance (scene lettering that shows once the mark over it is repaired).
                    if layer.mark.kind in STAYING and \
                            math.dist((x, y), layer_place(layer)) > max(8., .5*layer.mark.line_height):
                        continue
                    if match >= .3:
                        owner, cause = index, 'missed'
                        break
            findings.append({'frame': int(i), 'text': text, 'rect': list(rect), 'score': round(score, 3),
                             'layer': owner, 'cause': cause})
    trace('accept.ocr', checked=checked, source_frames_with_marks=confirmed, residuals=len(findings),
          causes={c: sum(f['cause'] == c for f in findings) for c in ('inside_mask', 'missed', 'foreign')})
    return findings, checked, confirmed


# ----------------------------------------------------------------------------
# Job

def job_trace(path):
    stream = path.open('a', encoding='utf-8')

    def trace(event, **details):
        row = {'t': round(monotonic(), 3), 'event': event, **details}
        stream.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
        stream.flush()
    return trace


def normalized_source(path, directory, trace):
    try:
        video = Video.open(path)
        video.check_timestamps()
        return video
    except ValueError as error:
        trace('source.normalizing', reason=str(error))
    info = json.loads(subprocess.check_output([executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
                                               '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
                                               '-of', 'json', str(path)]))
    rate = info['streams'][0].get('avg_frame_rate') or info['streams'][0]['r_frame_rate']
    target = directory/'source.mp4'
    subprocess.run([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(path),
                    '-map', '0:v:0', '-map', '0:a?', '-c:a', 'copy',
                    '-vf', f'fps={rate},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,setsar=1',
                    '-color_range', 'tv', '-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '14',
                    '-b:v', '0', '-fps_mode', 'cfr', str(target)], check=True)
    video = Video.open(target)
    video.check_timestamps()
    return video


def plan_marks(video, cuda, ocr, settings, trace):
    count = settings['samples'] or int(np.clip(round(video.count/float(video.fps)/1.2), 24, 96))
    samples = gather_samples(video, cuda, ocr, count, trace)
    shots = int(samples.shots[-1])+1
    static = static_groups(samples.readings, samples.frames, shots)
    taken = {id(r) for g in static for r in g.readings}
    moving = moving_groups([r for r in samples.readings if id(r) not in taken], samples.frames, shots)
    kept = []
    for group in moving:
        follows, informative = follows_scene(samples, group)
        trace('plan.moving_candidate', label=group.key, follows=follows, informative=informative)
        if informative >= 2 and follows >= .6*informative:
            trace('plan.scene_text', label=group.key, reason='moves with the picture')
        else:
            kept.append(group)
    static_keys = [g.key for g in static]
    watermark_keys = static_keys+[g.key for g in kept]

    def related(key):
        return any(similar(key, k) >= .4 or (min(len(key), len(k)) >= 2 and (key in k or k in key))
                   for k in watermark_keys)
    static_rects = [g.rect for g in static]

    def protected(reading):
        """Text whose pixels a badge mask must leave alone: subtitles and other passing captions."""
        if related(reading.key) or BRAND.search(reading.key):
            return False
        x, y, w, h = reading.rect
        if len(reading.key) < 2 or h > 1.5*w:
            return False    # a lone letter or a tall box: OCR reading an icon or picture, not a caption line
        if x <= 1 or x+w >= video.width-1:
            return False    # cut off by the frame edge: a mark entering or leaving, not a caption
        # Animated badges pass through intermediate states that OCR reads as other words.
        return all(overlap_share(reading.rect, rect) < .5 for rect in static_rects)
    coverage = np.zeros((video.height, video.width), np.uint8)
    for frame in samples.frames:
        seen = np.zeros_like(coverage)
        for r in samples.readings:
            if r.frame == frame and protected(r):
                x, y, w, h = r.rect
                seen[max(0, y):max(0, y+h), max(0, x):max(0, x+w)] = 1
        coverage += seen
    samples.transient = (coverage >= 2).astype(np.uint8)
    marks = []
    for mark in merge_static(static)+merge_moving(kept):
        if mark.kind == 'moving' and not any(brand_like(k, static_keys) for k in mark.keys):
            trace('plan.scene_text', label=mark.label, reason='no address, handle or badge text')
            continue
        if within_one_shot(mark, samples):
            # Followed through its shot with the picture instead (scene_logos).
            trace('plan.scene_text', label=mark.label, reason='moves within one shot only')
            samples.scene_keys += mark.keys
            continue
        marks.append(mark)
    samples.overlay = overlay_consensus(samples)
    faint = faint_marks(samples, marks, ocr, trace)
    plain = [m for m in marks if not part_of_overlay(m, faint, trace)]
    if len(plain) < len(marks):
        for mark in faint:
            mark.known = known_marks(plain, mark.known.shape)
    marks = plain+faint
    trace('plan.marks', marks=[{'kind': m.kind, 'label': m.label, 'observations': len(m.observations),
                                'line_height': m.line_height} for m in marks])
    return samples, marks


def mask_growth(settings, label):
    """Mask dilation for one layer: the attempt's base plus what earlier attempts added where text was left."""
    return settings.get('dilation', 4)+settings.get('grown', {}).get(label, 0)


def learn_layers(samples, marks, settings, trace):
    layers = []
    for mark in marks:
        dilation = mask_growth(settings, mark.label)
        parent = next((layer for layer in layers if layer.mark.label == mark.parent), None)
        try:
            if mark.kind == 'static' and parent is not None:
                layers.append(learn_variant(samples, mark, parent, dilation, mask_growth(settings, parent.mark.label),
                                            trace))
            elif mark.kind == 'static':
                layers.append(learn_static(samples, mark, dilation, trace))
            elif mark.kind == 'faint':
                layers.append(learn_faint(samples, mark, dilation, trace))
            else:
                layers.append(learn_moving(samples, mark, dilation, trace))
        except ValueError as error:
            trace('plan.mark_rejected', label=mark.label, reason=str(error))
    return layers


def decide_presence(video, layers, cuda, settings, trace):
    for layer in layers:
        if layer.mark.kind == 'faint':
            threshold = decide_faint(video, layer)
            trace('plan.faint_presence', label=layer.mark.label, threshold=round(threshold, 3),
                  present=int(layer.present.sum()), frames=video.count)
        elif layer.base is not None:
            threshold = decide_static(video, layer, settings['threshold_scale'])
            trace('plan.static_presence', label=layer.mark.label, threshold=round(threshold, 3),
                  present=int(layer.present.sum()), frames=video.count)
        else:
            high, low, holds = decide_moving(video, layer, cuda, settings['threshold_scale'], trace)
            trace('plan.moving_presence', label=layer.mark.label, high=round(high, 3), low=round(low, 3),
                  holds=holds, present=int(layer.present.sum()), frames=video.count)
    bridge_alternating(video, [layer for layer in layers if layer.mark.kind == 'static'], trace)
    moving = [layer for layer in layers if layer.mark.kind == 'moving']
    layers = [layer for layer in layers if layer.present.any()]
    parts = follow_animations(video, [layer for layer in layers if layer.mark.kind != 'faint'], cuda, settings, trace)
    # After the animations: those follow the layers' own presence, which gives way to the turned holds.
    holds = turned_holds(video, moving, cuda, settings, trace)
    for layer in layers:
        if layer.lead is not None and layer.lead.any():
            layer.present |= layer.lead
            trace('plan.static_lead', label=layer.mark.label, frames=int(layer.lead.sum()))
    return [layer for layer in layers if layer.present.any()]+parts+holds


def bridge_alternating(video, layers, trace):
    """Cover the switch between badge states that share a corner.

    A badge that alternates between two looks animates from one to the other, and in between matches neither.
    Short gaps where no state is present, next to one of them, get the neighbouring states' masks.
    """
    def rect(layer):
        x, y = layer.base
        return x, y, layer.motif.width, layer.motif.height
    families = []
    for layer in layers:
        family = next((f for f in families if any(overlap_share(rect(layer), rect(o)) >= .3 for o in f)), None)
        if family is None:
            families.append([layer])
        else:
            family.append(layer)
    limit = round(float(video.fps))
    for family in families:
        if len(family) < 2:
            continue
        shown = np.any([layer.present for layer in family], axis=0)
        bridged = 0
        for start, stop in runs(~shown):
            if stop-start > limit:
                continue
            for layer in family:
                if (start > 0 and layer.present[start-1]) or (stop < video.count and layer.present[stop]):
                    layer.present[start:stop] = True
            bridged += stop-start
        trace('plan.badge_bridged', labels=[layer.mark.label for layer in family], frames=bridged)


# ----------------------------------------------------------------------------
# Animated entrances and exits
#
# Badges often animate in: the logo waits beside its place, rolls in, and the lettering unfolds after it; some leave
# the reverse way, and a mark that travels folds into its logo, flies to its next place and unfolds there. No motif of
# the finished mark matches those frames, but a brand plays the same animation every time, so the frames the same
# number of steps before each entrance agree with each other where the animation is, while the scenes behind them
# differ. Their consensus at each step is the animation's mask at that step, and each entrance is checked against the
# consensus of the others before the mask is used on it. A flight does not take the same path every time, so each
# entrance is followed frame by frame with the step before, and the steps are learned where the entrances were
# followed to. Exits are handled the same way, counting steps after the mark was last shown. Entrances at different
# places are learned apart: a mark may fly in at one place and unfold out of another badge at the next.

ENTRY_GAP = .5        # seconds a mark must be gone before its return counts as an entrance
ENTRY_REACH = 3.      # seconds searched before each entrance and after each exit
ENTRY_MEMBERS = 3     # least number of entrances to compare
ENTRY_LEARN = 12      # entrances the consensus is learned from at most; any others are only checked against it
ENTRY_SCALE = 2       # the consensus is found at 1/ENTRY_SCALE of full resolution
ENTRY_SHIFT = 1.5     # seconds by which an entrance may be detected early or late: a slow unfolding crosses the
                      # detection threshold sooner over a plain background than over a busy one (over a bright sky, a
                      # badge was found only 35 frames after it had flown in)
ENTRY_DRIFT = .3      # seconds by which an entrance's timing may drift from its shift over the animation: an upload
                      # converted to another frame rate repeats (or drops) frames, not at the same places every time
ENTRY_SKIP = .05      # cost of a repeated or dropped frame in an entrance's timing, against a step's match score
ENTRY_SPEED = .02     # share of the picture's width an animation may travel from one frame to the next
ENTRY_TIE = .03       # an animation is followed to another place only where it matches clearly better
ENTRY_MATCH = .5      # score at which an entrance shows a step learned from the others
ENTRY_AREA = 40       # least pixels of a step's mask, at the reduced resolution
ENTRY_TAIL = 6        # last steps of the reach: what agrees there is the scene around the mark, not its animation
ENTRY_PATIENCE = .25  # seconds of steps an entrance may go unmatched inside its animation (motion blur, overlaps)
ENTRY_SLACK = 1.3     # consecutive steps share one layer while their combined mask grows no larger than this
ENTRY_INSIDE = 1.2    # seconds inside the showing searched for an animation playing on around the mark
ENTRY_COLOUR = 14     # YUV distance within which an entrance's pixel agrees with the others' at a step
ENTRY_AGREE = .75     # share of the entrances whose colours must agree for a pixel to join a step's mask...
ENTRY_CHANGED = .5    # ...and share in which the pixel differs from the scene there before the animation
ENTRY_WIDEN = True    # steps beyond the edge widened by colour (False only to compare)
ENTRY_HULL = 2.5      # a step's lone part beyond the edge takes its convex outline where that is at most this many
ENTRY_HULL_SOLID = 1.5  # times its area and at least this many (a part filling most of its outline is whole)...
ENTRY_HULL_SCATTER = 5.  # ...and a cluster of several parts where at most this many times theirs...
ENTRY_HULL_PART = .04    # ...counting the parts at least this share of the cluster's largest (specks keep out)
ENTRY_HULL_GAP = .025    # share of the picture's height within which parts reach out to join one cluster
ENTRY_NOTCH = .5         # notches in a step's parts narrower than this share of that reach are closed...
ENTRY_HOLE = .5          # ...and holes inside a step or an animation layer, each at most this share of its pixels
ENTRY_MOUTH = .4         # bays a part wraps round, their mouth at most this share of their walls, are filled
ENTRY_AROUND = 3         # a step gains what steps this near on both sides cover, each laid over it where it fits...
ENTRY_FIT = .8           # ...best, and where that covers at least this share of the smaller of the two
ENTRY_HOLLOW = 1.3       # a step whose parts fill less of their outline than 1/this takes the step inside it...
ENTRY_SPAN = .85         # ...where that outline is at least this share of the inner step's (it fades, not shrinks)

_entry_cache = {}     # learned animations per mark, side and place: later attempts only change the masks' dilation
# Processes learning a video's animations at a time; each holds the crops of up to ENTRY_LEARN entrances.
ENTRY_WORKERS = max(1, min(4, (os.cpu_count() or 2)//4))
ENTRY_WORKER_MEMORY = 4 << 30   # bytes of memory that must be free for each of them (one was seen to take 2.5 GB)


@dataclass
class Step:
    """One step of an animation in the window around a mark, at reduced resolution."""
    signal: np.ndarray
    weights: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.rect = cv2.boundingRect(self.mask)

    def score(self, window):
        """Best match of the step in a window, within two pixels of its place."""
        x, y, w, h = self.rect
        region = patch(window.astype(np.float32), (x-2, y-2, w+4, h+4))
        return float(response(region, self.signal[y:y+h, x:x+w], self.weights[y:y+h, x:x+w]).max())

    def follow(self, crop, keep, offset, radius):
        """The window offset within `radius` of `offset` at which the step matches an entrance's crop best. The
        offset moves only to a confident match that is clearly better than the one in place."""
        x, y, w, h = self.rect
        dx, dy = offset
        rect = (crop.shape[1]//5+dx+x-radius, dy+y-radius, w+2*radius, h+2*radius)
        region = patch(crop, rect).astype(np.float32)*patch(keep, rect)
        scores = response(region, self.signal[y:y+h, x:x+w], self.weights[y:y+h, x:x+w])
        _, best, _, (bx, by) = cv2.minMaxLoc(scores)
        if best < ENTRY_MATCH or best-float(scores[radius, radius]) <= ENTRY_TIE:
            return offset
        return dx+bx-radius, dy+by-radius


def entry_edges(video, layer, direction):
    """(edge frame, free frames beyond it) for each entrance (direction -1) or exit (+1) of a layer."""
    gap = max(1, round(ENTRY_GAP*float(video.fps)))
    spans = list(runs(layer.present))
    edges = []
    for index, (start, stop) in enumerate(spans):
        if direction < 0:
            edge, room = start, start-(spans[index-1][1] if index else 0)
        else:
            edge, room = stop-1, (spans[index+1][0] if index+1 < len(spans) else video.count)-stop
        if room >= gap:
            edges.append((edge, room))
    return edges


def entry_groups(video, layer, layers, edges, direction):
    """The edges grouped by what the mark comes out of (or goes into): the other marks it overlaps just beyond the
    edge, as lists of indices. A mark that appears out of nothing plays one animation wherever it appears, while
    one that unfolds out of another badge plays that badge's change. Edges where the mark is mostly outside the
    picture are left out: it slid out of the picture rather than animated."""
    gap = max(1, round(ENTRY_GAP*float(video.fps)))
    groups = {}
    for n, (edge, room) in enumerate(edges):
        if visible_share(layer.motif, layer.positions[edge], video) < .5:
            continue
        x, y = (int(v) for v in layer.positions[edge])
        rect = (x, y, layer.motif.width, layer.motif.height)
        near = set()
        for j in range(1, min(room, gap)+1):
            frame = edge+direction*j
            for q, other in enumerate(layers):
                if other is not layer and 0 <= frame < video.count and other.present[frame]:
                    ox, oy = (int(v) for v in other.positions[frame])
                    if overlap_share(rect, (ox, oy, other.motif.width, other.motif.height)) > .25:
                        near.add(q)
        groups.setdefault(tuple(sorted(near)), []).append(n)
    return list(groups.values())


def entry_size(layer):
    """The motif's size at reduced resolution."""
    s = ENTRY_SCALE
    return -(-layer.motif.width//s), -(-layer.motif.height//s)


def entry_region(layer, edge):
    """The region read around a mark's place at an edge, at reduced resolution: twice the motif's width beside it on
    either side and its height above and below, so an animation can be followed while it travels."""
    s = ENTRY_SCALE
    w, h = entry_size(layer)
    x, y = (int(v)//s for v in layer.positions[edge])
    return x-2*w, y-h, 5*w, 3*h


def window(crop, offset):
    """The window a step is learned in, from a crop of the region: the motif's place and once its size around it,
    moved by the offset the animation was followed to."""
    w = crop.shape[1]//5
    dx, dy = offset
    return patch(crop, (w+dx, dy, 3*w, crop.shape[0]))


def entry_crops(video, layer, edges, direction, depth, spread, cuda, inner=0):
    """High-pass crops of the region per (edge number, j): j frames away from the edge, outwards from the showing
    (up to `spread` beyond the reach); j <= 0 lies inside it (down to max(spread, inner) frames). The frames are
    read in whole runs."""
    inner = max(spread, inner)
    s = ENTRY_SCALE
    small = (video.width//s, video.height//s)
    wanted = {}
    for number, (edge, room) in enumerate(edges):
        rect = entry_region(layer, edge)
        for j in range(-inner, min(depth+spread, room)+1):
            frame = edge+direction*j
            if 0 <= frame < video.count:
                wanted.setdefault(frame, []).append((number, j, rect))
    needed = np.zeros(video.count, bool)
    needed[list(wanted)] = True
    crops = {}
    for start, stop in runs(needed):
        for i, raw in read_frames(video, start, stop, cuda):
            signal = highpass(cv2.resize(video.planes(raw)[0], small, interpolation=cv2.INTER_AREA), HIGHPASS_SIGMA/s)
            for number, j, rect in wanted[i]:
                crops[number, j] = patch(signal, rect).astype(np.float16)
    return crops


def entry_colours(video, layer, edges, direction, wanted, cuda):
    """YUV crops of the region at reduced resolution for the wanted (edge number, j) pairs."""
    s = ENTRY_SCALE
    small = (video.width//s, video.height//s)
    frames = {}
    for number, j in wanted:
        edge = edges[number][0]
        frame = edge+direction*j
        if 0 <= frame < video.count:
            frames.setdefault(frame, []).append((number, j, entry_region(layer, edge)))
    if not frames:
        return {}
    needed = np.zeros(video.count, bool)
    needed[list(frames)] = True
    colours = {}
    for start, stop in runs(needed):
        for i, raw in read_frames(video, start, stop, cuda):
            if i not in frames:
                continue
            yuv = [cv2.resize(plane, small, interpolation=cv2.INTER_AREA) for plane in video.planes(raw)]
            for number, j, rect in frames[i]:
                colours[number, j] = np.dstack([patch(plane, rect) for plane in yuv])
    return colours


def colour_window(crop, offset):
    """window() of a YUV crop."""
    return np.dstack([window(crop[..., c], offset) for c in range(crop.shape[2])]).astype(np.float32)


def widen_steps(video, layer, edges, chosen, direction, steps, shifts, whens, paths, keep, depth, spread, cuda):
    """Steps beyond the edge widened to the pixels whose colour most entrances share there, where it differs from
    the scene before the animation: a flying badge's disc keeps its colour while the sign of its edges turns with
    the scene behind it, and its inside turns, so the high-pass consensus finds only parts of it. Only parts next
    to the step's own mask and within its outline join, and holes they close are filled."""
    beyond = sorted(k for k in steps if k >= 1)
    if not beyond:
        return steps
    frames = {}
    for k in beyond:
        for r in range(len(chosen)):
            j = frame_for(whens[r], shifts[r], k)
            if j is not None and j >= 1:
                frames[r, k] = j
    scene_j = {r: min(depth+spread, edges[chosen[r]][1]) for r in range(len(chosen))}
    wanted = {(r, j) for (r, _), j in frames.items()} | {(r, j) for r, j in scene_j.items()}
    colours = entry_colours(video, layer, [edges[n] for n in chosen], direction, wanted, cuda)
    widened = dict(steps)
    for k in beyond:
        own = [r for r in range(len(chosen)) if (r, k) in frames and (r, frames[r, k]) in colours
               and (r, scene_j[r]) in colours]
        if len(own) < ENTRY_MEMBERS:
            continue
        offsets = {r: offset_at(paths[r], frames[r, k]) for r in own}
        stack = np.stack([colour_window(colours[r, frames[r, k]], offsets[r]) for r in own])
        scene = np.stack([colour_window(colours[r, scene_j[r]], offsets[r]) for r in own])
        median = np.median(stack, axis=0)
        agree = (np.abs(stack-median).max(axis=-1) <= ENTRY_COLOUR).mean(axis=0) >= ENTRY_AGREE
        changed = (np.abs(stack-scene).max(axis=-1) > ENTRY_COLOUR).mean(axis=0) >= ENTRY_CHANGED
        open_scene = np.mean([window(keep, offsets[r]) for r in own], axis=0) >= .5
        found = cv2.morphologyEx((agree & changed & open_scene).astype(np.uint8), cv2.MORPH_OPEN, ellipse(3))
        mask = steps[k].mask
        if not found.any() or not mask.any():
            continue
        points = cv2.findNonZero(mask)
        outline = np.zeros_like(mask)
        cv2.fillConvexPoly(outline, cv2.convexHull(points), 1)
        outline = cv2.dilate(outline, ellipse(9)) > 0
        touching = cv2.dilate(mask, ellipse(5)) > 0
        count, labels = cv2.connectedComponents(found, connectivity=8)
        parts = [n for n in np.unique(labels[touching & (found > 0)]) if n]
        if not parts:
            continue
        grown = fill_holes(np.maximum(mask, (np.isin(labels, parts) & outline).astype(np.uint8)))
        widened[k] = Step(steps[k].signal, steps[k].weights, grown)
    return widened


def complete_steps(steps, gap):
    """Steps beyond the edge filled to the convex outline of their parts. A badge that spins as it flies blurs into
    colours that differ from entrance to entrance, so neither the high-pass nor the colour consensus finds more than
    scattered arcs of it, and a translucent plate shows only its frame; what is missed shows in the frame and bleeds
    into the repaint of the frames around it. Parts within `gap` pixels of each other (both reaching out by it) form a
    cluster, and each cluster takes its outline where that is at most ENTRY_HULL_SCATTER times its area, or for a
    cluster of one part between ENTRY_HULL_SOLID and ENTRY_HULL times: a lone stroke such as a ribbon's edge keeps its
    shape, as do clusters far apart and a part that already fills most of its outline (a banner's bays are its own).
    Specks far smaller than a cluster's largest part (a sparkle beside a banner) stay as they are and shape no outline.
    Where no outline is taken, a part still has notches, gaps and holes the consensus missed (a spinning ring's
    blurred side, the hole round its play sign): bays it wraps round are filled and narrow notches closed, while
    shallow bays stay open. A step that is only pieces of a fading mark takes the shape of the step inside it, and a
    step gains what the steps on both sides of it cover. Any piece of the mark left beside the repaint shows, and the
    repaint carries it along the motion into the frames around. The step's signal and weights, and so its matching,
    stay as they were."""
    completed = dict(steps)
    reach = ellipse(2*gap+1)
    notch = ellipse(ENTRY_NOTCH*gap)
    for k, step in steps.items():
        if k < 1 or not step.mask.any():
            continue
        mask = (step.mask > 0).astype(np.uint8)
        count, clusters = cv2.connectedComponents(cv2.dilate(mask, reach), connectivity=8)
        grown = mask.copy()
        for c in range(1, count):
            cluster = mask & (clusters == c).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(cluster, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            if not len(areas):
                continue
            main = 1+np.flatnonzero(areas >= ENTRY_HULL_PART*areas.max())
            points = cv2.findNonZero(np.isin(labels, main).astype(np.uint8))
            outline = np.zeros_like(mask)
            cv2.fillConvexPoly(outline, cv2.convexHull(points), 1)
            ratio = np.count_nonzero(outline)/len(points)
            if ratio <= ENTRY_HULL_SCATTER if len(main) > 1 else ENTRY_HULL_SOLID <= ratio <= ENTRY_HULL:
                grown |= outline
            else:
                grown |= enclosed_bays(np.isin(labels, main).astype(np.uint8), outline, ENTRY_MOUTH)
        grown = close_holes(cv2.morphologyEx(grown, cv2.MORPH_CLOSE, notch), ENTRY_HOLE)
        if np.count_nonzero(grown) > np.count_nonzero(mask):
            completed[k] = Step(step.signal, step.weights, grown.astype(step.mask.dtype))
    # A mark fading in (or out) differs plainly from the scene only in its pale parts: the entrances agree on pieces
    # of it that leave its saturated arcs out, too far apart for an outline to reach them. A step whose pieces fill
    # little of their outline takes the step inside it (already completed), laid over where it fits best, where that
    # covers most of its pieces: the mark keeps its shape while it fades. A lettered banner's own mask fills little
    # of its outline too, so the pieces must span about as much as the step inside: one that shrinks (a banner
    # rolling up) keeps its own shape. Specks far smaller than its largest piece shape no outline here either.
    for k in sorted(steps):
        if k < 1:
            continue
        inner = completed.get(k-1)
        raw = (steps[k].mask > 0).astype(np.uint8)
        if not raw.any() or inner is None or inner.mask.shape != raw.shape or not inner.mask.any():
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        pieces = np.isin(labels, 1+np.flatnonzero(areas >= ENTRY_HULL_PART*areas.max())).astype(np.uint8)
        outline = np.zeros_like(raw)
        cv2.fillConvexPoly(outline, cv2.convexHull(cv2.findNonZero(pieces)), 1)
        if np.count_nonzero(outline) < ENTRY_HOLLOW*np.count_nonzero(pieces):
            continue
        around = np.zeros_like(raw)
        cv2.fillConvexPoly(around, cv2.convexHull(cv2.findNonZero((inner.mask > 0).astype(np.uint8))), 1)
        if np.count_nonzero(outline) < ENTRY_SPAN*np.count_nonzero(around):
            continue
        moved, fit = laid_over(raw > 0, inner.mask > 0, 2*gap)
        if fit >= ENTRY_FIT:
            step = completed[k]
            completed[k] = Step(step.signal, step.weights, np.maximum(step.mask > 0, moved).astype(step.mask.dtype))
    # A few steps where the entrances disagree (a spin's blur) lose a piece of the mark that steps on both sides of
    # them cover: it is the mark's there too. The mark moves across the canvas from step to step, so each step
    # around is first laid over the step where it fits best; one that fits nowhere well (another shape) is passed by.
    shown = {k: completed[k].mask > 0 for k in completed}
    for k in sorted(completed):
        if k < 1 or not shown[k].any():
            continue
        sides = []
        for near in (range(k-1, k-ENTRY_AROUND-1, -1), range(k+1, k+ENTRY_AROUND+1)):
            cover = np.zeros_like(shown[k])
            for j in near:
                if j in shown and shown[j].shape == shown[k].shape and shown[j].any():
                    moved, fit = laid_over(shown[k], shown[j], gap*(1+abs(k-j)))
                    if fit >= ENTRY_FIT:
                        cover |= moved
            sides.append(cover)
        between = sides[0] & sides[1] & ~shown[k]
        if between.any():
            step = completed[k]
            completed[k] = Step(step.signal, step.weights, (shown[k] | between).astype(step.mask.dtype))
    return completed


def entry_path(video, layer, edge, direction, spread):
    """The mark's offsets from its place at the edge, at reduced resolution, in the frames inside its showing."""
    s = ENTRY_SCALE
    path, offset = {}, (0, 0)
    for j in range(0, -spread-1, -1):
        frame = edge+direction*j
        if 0 <= frame < video.count and layer.present[frame]:
            dx, dy = (layer.positions[frame].astype(np.float64)-layer.positions[edge])/s
            offset = int(round(dx)), int(round(dy))
        path[j] = offset
    return path


def offset_at(path, j):
    """An entrance's offset j frames from its edge; beyond where it was followed, the nearest one known."""
    return path[j] if j in path else path[min(path, key=lambda known: abs(known-j))]


def latest(steps, k, patience):
    """The last step before k, unless the animation has not been seen for longer than `patience`."""
    before = [q for q in steps if k-patience <= q < k]
    return steps[max(before)] if before else None


def advance(crops, keep, r, j, step, path, radius):
    """Follows entrance r to j frames from its edge with a step of the animation, searching around where it would be
    if it kept moving as in the frames before: a folding mark matches the step before it only loosely while it
    flies, and keeps its course until it matches again. Without a step, it keeps its course."""
    previous = offset_at(path, j-1)
    before = offset_at(path, j-4)
    guess = (previous[0]+int(round((previous[0]-before[0])/3)), previous[1]+int(round((previous[1]-before[1])/3)))
    path[j] = step.follow(crops[r, j], keep, guess, radius) if step is not None and (r, j) in crops else guess


def follow_entrance(crops, keep, r, steps, shift, path, spread, depth, patience, radius):
    """Entrance r followed beyond its edge with steps already learned, each frame with its own step; with no steps,
    it keeps the course it had at the edge."""
    path = {j: offset for j, offset in path.items() if j <= 0}
    for k in range(1-spread, depth+1):
        if k+shift >= 1:
            advance(crops, keep, r, k+shift, steps.get(k) or latest(steps, k, patience), path, radius)
    return path


def cut(crops, keep, r, j, path):
    """Entrance r's window j frames from its edge, where it was followed to, with the scene around the mark left
    out."""
    offset = offset_at(path, j)
    return window(crops[r, j], offset).astype(np.float32)*window(keep, offset)


def entry_step(stack, reach):
    """The step that aligned windows share, or None. Specks far from the step's main part are left out."""
    if len(stack) < 2:
        return None
    signal, weights, mask = consensus_fields(np.stack(stack).astype(np.float32))
    mask = fill_holes(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1+int(np.argmax(areas))
        near = cv2.dilate((labels == largest).astype(np.uint8), ellipse(2*reach+1)) > 0
        keep = [n for n in range(1, count)
                if n == largest or areas[n-1] >= .25*areas.max() or near[labels == n].any()]
        mask = np.isin(labels, keep).astype(np.uint8)
    weights = weights*mask
    if mask.sum() < ENTRY_AREA or np.count_nonzero(weights) < 16:
        return None
    return Step(signal, weights.astype(np.float32), mask)


def frame_for(when, shift, k):
    """The frame j of an entrance that shows step k: by its timing {step: frame} where that was found (None for a
    step its frames skip); beyond it, at the offset from the step the timing ends with; without one, by its shift."""
    if not when:
        return k+shift
    first, last = min(when), max(when)
    if first <= k <= last:
        return when.get(k)
    end = first if k < first else last
    return k+when[end]-end


def aligned(crops, keep, members, shifts, paths, k, whens=None):
    """The members' windows at step k, and which members they belong to."""
    frames = {r: frame_for(whens[r] if whens else None, shifts[r], k) for r in members}
    own = [r for r in members if frames[r] is not None and (r, frames[r]) in crops]
    return own, [cut(crops, keep, r, frames[r], paths[r]) for r in own]


def entry_timing(score, low, high, shift, drift):
    """One entrance's timing: for each frame j from low to high, the step k it shows, no more than `drift` steps
    from j-shift. From one frame to the next the animation moves on one step, or (at a small cost) none, a repeated
    frame, or two, a dropped one. score(j, k) is how well frame j shows step k. Returns {j: k}."""
    width = 2*drift+1
    value, back = None, []
    for j in range(low, high+1):
        here = np.array([score(j, j-shift+d-drift) for d in range(width)], np.float64)
        if value is None:
            value = here
            continue
        # From d at the frame before (the step after it), from d-1 (a step skipped) or from d+1 (the same step).
        options = np.full((3, width), -np.inf)
        options[0] = value
        options[1, 1:] = value[:-1]-ENTRY_SKIP
        options[2, :-1] = value[1:]-ENTRY_SKIP
        back.append(options.argmax(axis=0))
        value = here+options.max(axis=0)
    if value is None:
        return {}
    d = int(value.argmax())
    timing = {high: high-shift+d-drift}
    for j, choice in zip(range(high, low, -1), reversed(back)):
        d += (0, -1, 1)[int(choice[d])]
        timing[j-1] = j-1-shift+d-drift
    return timing


def entry_when(score, timing, patience):
    """The frame that shows each step best, from an entrance's timing {j: k}: {k: j}, over the steps around the edge
    that the entrance matches, no more than `patience` steps in a row unmatched. Beyond them the timing is a guess:
    where the steps learned so far stop (a mark flying off, which the first round does not follow), frames that
    match nothing let it skip steps on its way to a match far off, and a skipped step is left out of the next round,
    which then cannot follow the mark into it."""
    when = {}
    for j, k in timing.items():
        if k not in when or score(j, k) > score(when[k], k):
            when[k] = j
    matched = sorted(k for k, j in when.items() if score(j, k) >= ENTRY_MATCH)
    if not matched:
        return {}
    first = last = min(range(len(matched)), key=lambda n: abs(matched[n]-.5))
    while first > 0 and matched[first]-matched[first-1] <= patience:
        first -= 1
    while last+1 < len(matched) and matched[last+1]-matched[last] <= patience:
        last += 1
    return {k: j for k, j in when.items() if matched[first] <= k <= matched[last]}


def timed_extent(score, timing, steps, room, patience):
    """Frames beyond the edge that the animation covers in one entrance, by its timing: up to the last frame that
    matched its step, as long as no more than `patience` frames in a row go unmatched; and the step each of those
    frames shows, {j: k} (a step that was not learned gives way to the nearest one learned, within `patience`)."""
    last, top = 0, max(steps, default=0)
    for j in range(1, room+1):
        k = timing.get(j)
        if k is None or k > top:
            break
        if k in steps and score(j, k) >= ENTRY_MATCH:
            last = j
            continue
        if j-last > patience:
            break
    shown = {}
    for j in range(1, last+1):
        k = timing.get(j)
        near = min(steps, key=lambda q: abs(q-k)) if steps and k is not None else None
        if near is not None and abs(near-k) <= patience:
            shown[j] = near
    return last, shown


def entry_keep(crops, members, shifts, depth):
    """0 on pixels that agree across the entrances even at the end of the reach: the scene around the mark (letterbox
    edges, another badge nearby), not its animation; 1 elsewhere."""
    found = np.zeros(next(iter(crops.values())).shape, np.uint8)
    for k in range(depth-ENTRY_TAIL+1, depth+1):
        stack = [crops[r, k+shifts[r]] for r in members if (r, k+shifts[r]) in crops]
        if len(stack) >= ENTRY_MEMBERS:
            found = np.maximum(found, consensus_fields(np.stack(stack).astype(np.float32))[2])
    return (cv2.dilate(found, ellipse(5)) == 0).astype(np.float32)


def best_shift(crops, keep, r, steps, spread, path):
    """The shift that aligns entrance r best with the steps, preferring none on a tie."""
    def mean_score(shift):
        scores = [step.score(cut(crops, keep, r, k+shift, path)) for k, step in steps.items() if (r, k+shift) in crops]
        return float(np.mean(scores)) if scores else -1.
    return max(range(-spread, spread+1), key=lambda shift: (mean_score(shift), -abs(shift)))


def learn_animation(video, layer, edges, direction, cuda):
    """Steps of the animation a mark shows at its entrances (or exits) at one place and, per edge, how it is aligned,
    how many frames beyond the edge the animation covers and where it was followed to there, and the part of the
    animation that plays on around the mark inside its showing:
    ({step: Step}, [(shift, extent, {j: offset}, {j: step})], ({step: Step}, [[(j, step, offset)]]))."""
    fps = float(video.fps)
    depth = round(ENTRY_REACH*fps)
    spread = max(2, round(ENTRY_SHIFT*fps))
    inner = max(spread, round(ENTRY_INSIDE*fps))
    patience = max(2, round(ENTRY_PATIENCE*fps))
    radius = max(3, round(ENTRY_SPEED*video.width/ENTRY_SCALE))
    reach = max(4, layer.motif.height//(2*ENTRY_SCALE))
    travels = layer.mark.kind == 'moving'
    chosen = sorted(set(np.linspace(0, len(edges)-1, min(len(edges), ENTRY_LEARN)).round().astype(int).tolist()))
    crops = entry_crops(video, layer, [edges[n] for n in chosen], direction, depth, spread, cuda, inner)
    members = range(len(chosen))
    inside = [entry_path(video, layer, edges[n][0], direction, inner) for n in chosen]
    paths = [dict(path) for path in inside]
    shifts = [0]*len(chosen)
    whens = [None]*len(chosen)
    drift = max(1, round(ENTRY_DRIFT*fps))
    keep, steps = None, {}

    def steps_score(steps, r, path, memo):
        def score(j, k):
            if (j, k) not in memo:
                memo[j, k] = steps[k].score(cut(crops, keep, r, j, path)) if k in steps and (r, j) in crops else 0.
            return memo[j, k]
        return score
    for attempt in range(3):
        if attempt:
            shifts = [best_shift(crops, keep, r, steps, spread, paths[r]) for r in members]
            # Each entrance's own timing against the steps learned so far: its repeated frames are not the others'.
            for r in members:
                score = steps_score(steps, r, paths[r], {})
                timing = entry_timing(score, 1-spread, min(depth+spread, edges[chosen[r]][1]), shifts[r], drift)
                whens[r] = entry_when(score, timing, patience)
        keep = entry_keep(crops, members, shifts, depth)
        steps = {}
        paths = [dict(path) for path in inside]
        for k in range(1-spread, depth+1):
            # The first round keeps each mark on the course it had at the edge, which aligns the entrances in time
            # without being misled by a path followed at the wrong phase. Later rounds follow a moving mark into
            # step k with the consensus of the step before; a badge animates at its place.
            step = latest(steps, k, patience) if attempt and travels else None
            for r in members:
                if k+shifts[r] >= 1:
                    advance(crops, keep, r, k+shifts[r], step, paths[r], radius)
            _, stack = aligned(crops, keep, members, shifts, paths, k, whens)
            if len(stack) >= ENTRY_MEMBERS and (found := entry_step(stack, reach)) is not None:
                steps[k] = found
    # Beyond the edge, a step counts only where most entrances match the consensus of the others. Each entrance's
    # frames around its place in time are scored too, to time it by.
    scores = [{} for _ in members]
    table = [{} for _ in members]
    for k, step in list(steps.items()):
        own, stack = aligned(crops, keep, members, shifts, paths, k, whens)
        for r, crop in zip(own, stack):
            if k <= 0:
                judge = step
            else:
                others = [other for q, other in zip(own, stack) if q != r]
                judge = entry_step(others, reach)
            scores[r][k] = judge.score(crop) if judge is not None else 0.
            if judge is not None:
                for j in range(k+shifts[r]-drift, k+shifts[r]+drift+1):
                    if (r, j) in crops:
                        table[r][j, k] = judge.score(cut(crops, keep, r, j, paths[r]))
        if k >= 1:
            passed = sum(scores[r][k] >= ENTRY_MATCH for r in own)
            if passed < 2 or 2*passed < len(own):
                del steps[k]
                for r in own:
                    del scores[r][k]
    if ENTRY_WIDEN:
        steps = widen_steps(video, layer, edges, chosen, direction, steps, shifts, whens, paths, keep, depth, spread,
                            cuda)
    steps = complete_steps(steps, max(2, round(ENTRY_HULL_GAP*video.height/ENTRY_SCALE)))
    # Steps where too few entrances matched (motion blur, the lettering half unfolded next to the edge) take the
    # union of the steps around them; step 0 is the finished mark.
    found = sorted(k for k in steps if k >= 0)
    for a, b in zip(found, found[1:]):
        if 1 < b-a <= patience:
            for k in range(a+1, b):
                steps[k] = Step(steps[a].signal, np.maximum(steps[a].weights, steps[b].weights),
                                np.maximum(steps[a].mask, steps[b].mask))
    placed = [None]*len(edges)
    for r, number in enumerate(chosen):
        score = lambda j, k, r=r: table[r].get((j, k), 0.) if k in steps else 0.
        room = edges[number][1]
        timing = entry_timing(score, 1-spread, min(depth+spread, room), shifts[r], drift)
        extent, shown = timed_extent(score, timing, steps, room, patience)
        placed[number] = shifts[r], extent, {j: offset_at(paths[r], j) for j in range(1, extent+1)}, shown
    lingering, linger_scores = linger_steps(layer, crops, keep, members, shifts, paths, inner, reach, patience)
    spans = [None]*len(edges)
    for r, number in enumerate(chosen):
        spans[number] = linger_span(linger_scores[r], lingering, shifts[r], paths[r], inner, patience)
    rest = [n for n in range(len(edges)) if placed[n] is None]
    scored = {k: step for k, step in steps.items() if k <= 0 or any(k in row for row in scores)}
    for batch in range(0, len(rest), ENTRY_LEARN):
        numbers = rest[batch:batch+ENTRY_LEARN]
        crops = entry_crops(video, layer, [edges[n] for n in numbers], direction, depth, spread, cuda, inner)
        keep = np.ones(next(iter(crops.values())).shape, np.float32)
        for r, number in enumerate(numbers):
            path = follow_entrance(crops, keep, r, {}, 0, entry_path(video, layer, edges[number][0], direction, inner),
                                   spread, depth, patience, radius)
            shift = best_shift(crops, keep, r, scored, spread, path)
            if travels:
                path = follow_entrance(crops, keep, r, scored, shift, path, spread, depth, patience, radius)
                shift = best_shift(crops, keep, r, scored, spread, path)
                path = follow_entrance(crops, keep, r, scored, shift, path, spread, depth, patience, radius)
            score = steps_score(scored, r, path, {})
            room = edges[number][1]
            timing = entry_timing(score, 1-spread, min(depth+spread, room), shift, drift)
            extent, shown = timed_extent(score, timing, scored, room, patience)
            placed[number] = shift, extent, {j: offset_at(path, j) for j in range(1, extent+1)}, shown
            own = {k: step.score(cut(crops, keep, r, k+shift, path))
                   for k, step in lingering.items() if (r, k+shift) in crops}
            spans[number] = linger_span(own, lingering, shift, path, inner, patience)
    return steps, placed, (lingering, spans)


def linger_steps(layer, crops, keep, members, shifts, paths, inner, reach, patience):
    """The animation around a mark inside its showing (a ring that sweeps on round a badge after it is shown, or
    starts before it leaves): steps of what the entrances share beyond the mark's own mask, each kept where most
    entrances match the consensus of the others there. Returns ({step: Step}, [{step: score}] per entrance)."""
    w, h = entry_size(layer)
    own_mask = np.zeros((3*h, 3*w), np.uint8)
    own_mask[h:2*h, w:2*w] = cv2.resize(layer.motif.mask, (w, h), interpolation=cv2.INTER_AREA) > 0
    near = cv2.dilate(own_mask, ellipse(5)) > 0
    steps, scores = {}, [{} for _ in members]
    for k in range(-max(shifts)-inner, 1-min(shifts)):
        own, stack = aligned(crops, keep, members, shifts, paths, k)
        inside = [n for n, r in enumerate(own) if k+shifts[r] <= 0]     # frames past the edge are the animation's
        own, stack = [own[n] for n in inside], [stack[n] for n in inside]
        if len(stack) < ENTRY_MEMBERS:
            continue
        found = entry_step(stack, reach)
        if found is None:
            continue
        beyond = beyond_mark(found.mask, near)
        if beyond is None:
            continue
        for r, crop in zip(own, stack):
            loo = entry_step([other for q, other in zip(own, stack) if q != r], reach)
            scores[r][k] = Step(loo.signal, loo.weights*beyond, beyond).score(crop) \
                if loo is not None and np.count_nonzero(loo.weights*beyond) >= 16 else 0.
        passed = sum(scores[r][k] >= ENTRY_MATCH for r in own)
        if passed < 2 or 2*passed < len(own):
            for r in own:
                del scores[r][k]
            continue
        steps[k] = Step(found.signal, found.weights*beyond, beyond)
    found = sorted(steps)
    for a, b in zip(found, found[1:]):
        if 1 < b-a <= patience:
            for k in range(a+1, b):
                steps[k] = Step(steps[a].signal, np.maximum(steps[a].weights, steps[b].weights),
                                np.maximum(steps[a].mask, steps[b].mask))
    return steps, scores


def beyond_mark(mask, near):
    """The part of a step's mask away from the mark's own (specks left out), or None where too little is."""
    beyond = (mask > 0) & ~near
    count, labels, stats, _ = cv2.connectedComponentsWithStats(beyond.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 12)+1
    beyond[np.isin(labels, small)] = False
    if beyond.sum() < ENTRY_AREA:
        return None
    return beyond.astype(np.uint8)


def linger_span(scores, steps, shift, path, inner, patience):
    """The frames inside the showing that one entrance's lingering animation covers, from the edge inwards while it
    matches (no more than `patience` frames in a row unmatched): [(j, step, offset)]."""
    last = 1
    for j in range(0, -inner-1, -1):
        if scores.get(j-shift, 0.) >= ENTRY_MATCH:
            last = j
        elif last-j > patience:
            break
    return [(j, j-shift, offset_at(path, j)) for j in range(last, 1) if j-shift in steps]


def animation_layers(video, layer, direction, steps, spans, edges, settings, side):
    """Layers that mask the animation, shown per edge at the frames of its span [(j, step, offset)]. Consecutive
    steps share a layer while the shared mask stays within ENTRY_SLACK of every step's own, so no frame is repainted
    much beyond what its step needs."""
    s = ENTRY_SCALE
    count = video.count
    used = sorted({k for span in spans for _, k, _ in span} & set(steps))
    pad = settings.get('dilation', 4)+max(settings.get('grown', {}).values(), default=0)+2   # room for any dilation
    full = {}
    for k in used:
        mask = steps[k].mask
        full[k] = cv2.copyMakeBorder(cv2.resize(mask, (mask.shape[1]*s, mask.shape[0]*s),
                                                interpolation=cv2.INTER_NEAREST), pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=0)
    groups = []     # [steps, shared mask, smallest step's area]
    for k in used:
        area = np.count_nonzero(full[k])
        if groups and k == groups[-1][0][-1]+1:
            union = np.maximum(groups[-1][1], full[k])
            smallest = min(groups[-1][2], area)
            if np.count_nonzero(union) <= ENTRY_SLACK*smallest:
                groups[-1][0].append(k)
                groups[-1][1:] = union, smallest
                continue
        groups.append([[k], full[k].copy(), area])
    width, _ = entry_size(layer)
    layers = []
    for group, union, _ in groups:
        label = f'{layer.mark.label} ({side} {group[0]}-{group[-1]})'
        # Steps that each leave a bay open on a different side (a ring spinning) together enclose it.
        union = close_holes(union, ENTRY_HOLE)
        dilation = mask_growth(settings, label)
        if dilation > 0:
            union = cv2.dilate(union, ellipse(2*dilation+1))
        x, y, w, h = cv2.boundingRect(union)
        mask = np.ascontiguousarray(union[y:y+h, x:x+w])
        largest = max(group, key=lambda k: int(steps[k].mask.sum()))
        size = (steps[largest].signal.shape[1]*s, steps[largest].signal.shape[0]*s)

        def full_size(field):
            plane = cv2.copyMakeBorder(cv2.resize(field, size, interpolation=cv2.INTER_LINEAR), pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=0)
            return np.ascontiguousarray(plane[y:y+h, x:x+w])
        weights = np.max([full_size(steps[k].weights) for k in group], axis=0)*(mask > 0)
        try:
            motif = Motif(full_size(steps[largest].signal).astype(np.float32),
                          np.clip(weights, 0, 1).astype(np.float32), mask.astype(np.uint8))
        except ValueError:
            continue
        positions = np.zeros((count, 2), np.int32)
        scores = np.zeros(count, np.float32)
        present = np.zeros(count, bool)
        for (edge, _), span in zip(edges, spans):
            rx, ry, _, _ = entry_region(layer, edge)
            for j, k, (dx, dy) in span:
                frame = edge+direction*j
                if k in group and 0 <= frame < count:
                    present[frame] = True
                    positions[frame] = s*(rx+width+dx)-pad+x, s*(ry+dy)-pad+y
                    scores[frame] = layer.positive
        if not present.any():
            continue
        mark = Mark('part', label, layer.mark.keys, layer.mark.line_height, [], [])
        layers.append(Layer(mark, motif, layer.origin, None, layer.positive, positions, scores,
                            np.full(count, -1., np.float32), present, positions.copy()))
    return layers


ALONE_SCALES = (.95, .975, 1., 1.025, 1.05)  # rescaling searched around the predicted size, frame to frame
ALONE_CHUNK = 12                # frames read at a time while following one
ALONE_COVERED = .8              # share of the followed mark another state's layer must cover to take over
ALONE_WEAK = .3                 # least score of a followed mark that matches less clearly (blurred, glowing)...
ALONE_MARGIN = .1               # ...when it also beats the motif's best match elsewhere in the picture by this much
ALONE_AREA = .2                 # share of the picture a followed mark's mask may cover
ALONE_TURNS = (0., -1., 1., -2., 2.)  # turns searched around the last angle, frame to frame, in degrees
ALONE_TILT = 15.                # largest angle a followed mark may turn to (a ribbon swinging in)
ALONE_TURNED = .03              # a turned match is taken only where it beats the upright one by this much
ALONE_GROUP_TURN = 1.           # degrees by which the frames sharing one part layer may differ
LOOK_MATCH = .8       # masked correlation (luma and colour) with the mark's look where its motif was last followed that
                      # keeps it followed where the motif no longer matches (a ribbon before its lettering fades in)...
LOOK_GONE = .3        # ...taken only where the correlation is then seen to fall by this much below the frames kept...
LOOK_PATIENCE = .5    # ...within this many seconds
LOOK_RADIUS = 24      # pixels the look is searched around the predicted place, frame to frame
LOOK_TURNS = (0., -1., 1.)  # turns searched around the last angle, frame to frame, in degrees


def turned(image, angle, expand=False):
    """An image rotated about its centre by angle degrees (counter-clockwise), in its own canvas or, expanded, in
    one holding all of it."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.)
    size = (w, h)
    if expand:
        c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
        size = (math.ceil(w*c+h*s), math.ceil(w*s+h*c))
        matrix[0, 2] += (size[0]-w)/2
        matrix[1, 2] += (size[1]-h)/2
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def follow_alone(video, layer, layers, edge, room, direction, cuda):
    """Follow an entrance or exit the mark plays too seldom to compare (flying into its corner once, at the
    opening, say): its motif moved and rescaled from frame to frame for as long as it matches, it stays in the
    picture and no other state of the badge covers it there.

    A mark shown larger than learned is matched in the picture shrunk to the motif's size, where its high-pass
    detail compares with the motif's; a smaller one with the motif shrunk. A mark swinging in (a ribbon tilted
    as it slides into place) is matched turned, frame by frame, where that matches clearly better than upright.
    Returns {frame: (x, y, scale, angle)}, the motif's top-left at that scale before turning it about its centre,
    from where the layer matches in place.
    """
    fps = float(video.fps)
    depth = min(round(ENTRY_REACH*fps), room)
    spread = max(2, round(ENTRY_SHIFT*fps))
    motif = layer.motif
    minimum = max(.4, .55*layer.positive)
    weak = max(ALONE_WEAK, .4*layer.positive)
    start = None    # a showing's first or last frames may be decided before the mark has settled in place
    for j in range(0, -spread-1, -1):
        frame = edge+direction*j
        if not 0 <= frame < video.count or not layer.present[frame]:
            break
        if layer.scores[frame] >= minimum:
            start = j
            break
    if start is None:
        return {}
    x, y = (float(v) for v in layer.positions[edge+direction*start])
    centre = (x+motif.width/2, y+motif.height/2)
    scale, growth, angle, velocity, followed, sized, planes = 1., 1., 0., (0., 0.), {}, {}, {}
    area = int(np.count_nonzero(motif.mask))

    def resized(factor, tilt=0.):
        key = (round(factor, 4), round(tilt, 3))
        if key not in sized:
            size = (max(8, round(motif.width*factor)), max(8, round(motif.height*factor)))
            signal = cv2.resize(motif.signal, size, interpolation=cv2.INTER_AREA)
            weights = np.clip(cv2.resize(motif.weights, size, interpolation=cv2.INTER_AREA), 0, 1)
            if tilt:
                signal, weights = turned(signal, tilt), turned(weights, tilt)
            sized[key] = (signal, weights)
        return sized[key]

    def match(plane, full, factor, tilt, guess, radius):
        """Best (score, top-left at full resolution) of the motif at this scale and angle within radius of guess (its
        centre)."""
        if factor <= 1:
            template, weights = resized(factor, tilt)
            h, w = template.shape
            left, top = round(guess[0]-w/2)-radius, round(guess[1]-h/2)-radius
            scores = response(patch(full, (left, top, w+2*radius, h+2*radius)), template, weights)
            _, value, _, (a, b) = cv2.minMaxLoc(scores)
            return value, (left+a, top+b)
        shrunk = cv2.resize(plane, (round(plane.shape[1]/factor), round(plane.shape[0]/factor)),
                            interpolation=cv2.INTER_AREA)
        reach = max(2, round(radius/factor))
        left, top = round(guess[0]/factor-motif.width/2)-reach, round(guess[1]/factor-motif.height/2)-reach
        template, weights = resized(1., tilt)
        scores = response(patch(highpass(shrunk, HIGHPASS_SIGMA), (left, top, motif.width+2*reach,
                                                                    motif.height+2*reach)), template, weights)
        _, value, _, (a, b) = cv2.minMaxLoc(scores)
        return value, (round((left+a)*factor), round((top+b)*factor))

    def elsewhere(plane, full, factor, tilt, place):
        """The motif's best match at this scale and angle away from place: what the picture alone reaches."""
        if factor <= 1:
            (template, weights), signal, (x, y) = resized(factor, tilt), full, place
        else:
            template, weights = resized(1., tilt)
            signal = highpass(cv2.resize(plane, (round(plane.shape[1]/factor), round(plane.shape[0]/factor)),
                                         interpolation=cv2.INTER_AREA), HIGHPASS_SIGMA)
            x, y = round(place[0]/factor), round(place[1]/factor)
        h, w = template.shape
        scores = response(cv2.copyMakeBorder(signal, h, h, w, w, cv2.BORDER_CONSTANT, value=0), template, weights)
        x, y = x+w, y+h
        scores[max(0, y-h//2):max(0, y+h//2+1), max(0, x-w//2):max(0, x+w//2+1)] = -1
        return float(scores.max())

    def covered(frame, rect):
        for other in layers:
            if other is not layer and other.present[frame]:
                ox, oy = (int(v) for v in other.positions[frame])
                if overlap_share(rect, (ox, oy, other.motif.width, other.motif.height)) >= ALONE_COVERED and \
                        rect[2]*rect[3] <= other.motif.width*other.motif.height/ALONE_COVERED:
                    return True
        return False
    for j in range(start+1, depth+1):
        frame = edge+direction*j
        if not 0 <= frame < video.count:
            break
        if frame not in planes:
            first, last = sorted((frame, frame+direction*(ALONE_CHUNK-1)))
            first, last = max(0, first), min(video.count-1, last)
            planes = {i: video.planes(raw)[0].astype(np.float32) for i, raw in read_frames(video, first, last+1, cuda)}
        # Around the predicted place, as far as where the mark stopped or sped up to twice its pace.
        radius = max(10, round(.06*video.width)) if len(followed) < 2 else round(math.hypot(*velocity))+8
        guess = (centre[0]+velocity[0], centre[1]+velocity[1])
        # A mark flying in shrinks or grows steadily: the size, like the place, is predicted from the last step.
        best, full = None, highpass(planes[frame], HIGHPASS_SIGMA)
        for f in (scale*growth*s for s in ALONE_SCALES):
            for tilt in (angle+t for t in ALONE_TURNS):
                if abs(tilt) > ALONE_TILT:
                    continue
                value, place = match(planes[frame], full, f, tilt, guess, radius)
                # Turned further from upright only where that matches clearly better.
                if best is None or value > best[0]+(ALONE_TURNED if abs(tilt) > abs(best[2]) else 0.):
                    best = (value, f, tilt, place)
        value, factor, tilt, (bx, by) = best
        w, h = motif.width*factor, motif.height*factor
        inside = max(0, min(bx+w, video.width)-max(bx, 0))*max(0, min(by+h, video.height)-max(by, 0))
        if (value < weak or inside < .5*w*h or area*factor*factor > ALONE_AREA*video.width*video.height
                or covered(frame, (bx, by, round(w), round(h)))):
            break
        # A weaker match counts only where it stands out from the picture: where the mark has gone, the best
        # place nearby scores no better than places elsewhere.
        if value < minimum and value < elsewhere(planes[frame], full, factor, tilt, (bx, by))+ALONE_MARGIN:
            break
        moved = (bx+w/2, by+h/2)
        velocity = (moved[0]-centre[0], moved[1]-centre[1])
        growth = float(np.clip(factor/scale, .9, 1.1))
        centre, scale, angle = moved, factor, tilt
        followed[frame] = (int(bx), int(by), float(factor), float(tilt))
    if followed:
        followed.update(follow_look(video, layer, followed, edge, depth, direction, cuda, covered))
    return followed


def look_planes(video, raw):
    """A frame's luma and colour as one three-channel picture, colour weighted up to count next to luma."""
    y, u, v = video.planes(raw)
    return np.dstack([y.astype(np.float32), 2*full_chroma(u, y.shape).astype(np.float32),
                      2*full_chroma(v, y.shape).astype(np.float32)])


def look_cut(image, rect):
    """The rectangle of an image of any channel count, zero beyond its edges."""
    x, y, w, h = rect
    result = np.zeros((h, w)+image.shape[2:], image.dtype)
    view, section = clipped_plane(image, rect)
    result[section] = view
    return result


def follow_look(video, layer, followed, edge, depth, direction, cuda, covered):
    """Frames beyond the last one a lone animation's motif was followed to where the mark's look there (its pixels
    under the mask, colour included) still matches nearby: a ribbon shown before its lettering fades in, or left after
    the lettering faded out, which the motif (high-pass detail, mostly the lettering's) no longer finds.

    Taken only where the look is then seen to go, its correlation falling by LOOK_GONE within LOOK_PATIENCE: a look
    that stays the same is the background's. Frames as the look fades count while closer to it than to what is left.
    Returns {frame: (x, y, scale, angle)} as follow_alone does."""
    motif = layer.motif
    last = max(followed, key=lambda f: direction*(f-edge))
    x, y, scale, angle = followed[last]
    far = min(video.count-1, max(0, edge+direction*depth))
    if direction*(far-last) < 1:
        return {}
    size = (max(8, round(motif.width*scale)), max(8, round(motif.height*scale)))
    mask = (cv2.resize(motif.mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= .5).astype(np.float32)
    if abs(angle) >= .25:
        mask = (turned(mask, angle, expand=True) >= .5).astype(np.float32)
    h, w = mask.shape

    def centre_of(frame):
        fx, fy, fscale, _ = followed[frame]
        return fx+motif.width*fscale/2, fy+motif.height*fscale/2
    centre = centre_of(last)
    previous = last-direction
    velocity = (centre[0]-centre_of(previous)[0], centre[1]-centre_of(previous)[1]) if previous in followed else (0., 0.)
    looks = {}

    def look(frame):
        if frame not in looks:
            first, end = sorted((frame, frame+direction*(ALONE_CHUNK-1)))
            looks.clear()
            looks.update({i: look_planes(video, raw)
                          for i, raw in read_frames(video, max(0, first), min(video.count-1, end)+1, cuda)})
        return looks[frame]
    left, top = round(centre[0]-w/2), round(centre[1]-h/2)
    inside = look_cut(np.ones((video.height, video.width), np.float32), (left, top, w, h))
    template, valid = look_cut(look(last), (left, top, w, h)), mask*inside
    if valid.sum() < .5*mask.sum():
        return {}
    templates = {}

    def turned_template(turn):
        key = round(turn, 3)
        if key not in templates:
            image, weights = (turned(template, turn), turned(valid, turn) >= .5) if turn else (template, valid > 0)
            templates[key] = image, np.dstack([weights.astype(np.float32)]*3)
        return templates[key]
    kept, fading, gone, turn = [], [], None, 0.
    patience = max(2, round(LOOK_PATIENCE*float(video.fps)))
    for n in range(1, abs(far-last)+1):
        frame = last+direction*n
        since = n-(len(kept)+1)
        radius = LOOK_RADIUS+round(math.hypot(*velocity))*since
        guess = (centre[0]+velocity[0]*(since+1), centre[1]+velocity[1]*(since+1))
        bx, by = round(guess[0]-w/2)-radius, round(guess[1]-h/2)-radius
        region = look_cut(look(frame), (bx, by, w+2*radius, h+2*radius))
        best = None
        for tilt in (turn+t for t in LOOK_TURNS):
            if abs(angle+tilt) > ALONE_TILT:
                continue
            image, weights = turned_template(tilt)
            scores = np.nan_to_num(cv2.matchTemplate(region, image, cv2.TM_CCOEFF_NORMED, mask=weights),
                                   nan=-1., posinf=-1., neginf=-1.)
            _, value, _, (a, b) = cv2.minMaxLoc(scores)
            if best is None or value > best[0]+(ALONE_TURNED if abs(angle+tilt) > abs(angle+best[1]) else 0.):
                best = (value, tilt, bx+a, by+b)
        value, tilt, px, py = best
        place = (px+w/2, py+h/2)
        if not fading and value >= LOOK_MATCH and not covered(frame, (px, py, w, h)):
            kept.append((frame, value, tilt, place))
            velocity = (place[0]-centre[0], place[1]-centre[1])
            centre, turn = place, tilt
            continue
        if not kept:
            return {}
        if value <= min(v for _, v, _, _ in kept)-LOOK_GONE:
            gone = value
            break
        if len(fading) >= patience:
            return {}
        fading.append((frame, value, tilt, place))
    if gone is None:
        return {}
    halfway = (min(v for _, v, _, _ in kept)+gone)/2
    for item in fading:
        frame, value, _, (cx, cy) = item
        if value < halfway or covered(frame, (round(cx-w/2), round(cy-h/2), w, h)):
            break
        kept.append(item)
    found = {}
    for frame, _, tilt, (cx, cy) in kept:
        found[frame] = (int(round(cx-motif.width*scale/2)), int(round(cy-motif.height*scale/2)), float(scale),
                        float(angle+tilt))
    return found


def alone_layers(video, layer, followed, settings, side):
    """Part layers for a followed lone animation: frames beyond the layer's own placement, grouped while the
    motif's size stays within a few percent and its angle within a degree, each group masked with the layer's mask
    at its largest size, turned by the group's middle angle."""
    extra = [f for f, (x, y, scale, angle) in sorted(followed.items())
             if not (layer.present[f] and abs(scale-1) < .03 and abs(angle) < .5
                     and np.abs(layer.positions[f]-(x, y)).max() <= 3)]
    groups = []
    for f in extra:
        if groups and f == groups[-1][-1]+1 and abs(followed[f][2]/followed[groups[-1][0]][2]-1) <= .04 and \
                max(abs(followed[f][3]-followed[g][3]) for g in groups[-1]) <= ALONE_GROUP_TURN:
            groups[-1].append(f)
        else:
            groups.append([f])
    count = video.count
    base = mask_growth(settings, layer.mark.label)
    layers = []
    for group in groups:
        label = f'{layer.mark.label} ({side} {group[0]}-{group[-1]})'
        largest = max(followed[f][2] for f in group)
        size = (max(8, round(layer.motif.width*largest)), max(8, round(layer.motif.height*largest)))
        mask = (cv2.resize(layer.motif.mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= .5)
        mask = mask.astype(np.uint8)
        # The layer's mask carries its dilation, rescaled with it; add what this part's dilation lacks.
        grow_by = mask_growth(settings, label)-largest*base
        if grow_by >= .5:
            mask = cv2.dilate(mask, ellipse(2*round(grow_by)+1))
        signal = cv2.resize(layer.motif.signal, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        weights = np.clip(cv2.resize(layer.motif.weights, size, interpolation=cv2.INTER_AREA), 0, 1).astype(np.float32)
        angle = float(np.median([followed[f][3] for f in group]))
        if abs(angle) >= .25:
            # Turned in a canvas holding all of it, so the ends of a long ribbon are not cut off.
            mask = (turned(mask.astype(np.float32), angle, expand=True) >= .5).astype(np.uint8)
            signal, weights = turned(signal, angle, expand=True), turned(weights, angle, expand=True)
        try:
            motif = Motif(signal, weights, mask)
        except ValueError:
            continue
        positions = np.zeros((count, 2), np.int32)
        scores = np.zeros(count, np.float32)
        present = np.zeros(count, bool)
        for f in group:
            x, y, scale, _ = followed[f]
            # Centred where the motif was followed at its own size.
            positions[f] = (round(x+(layer.motif.width*scale-motif.width)/2),
                            round(y+(layer.motif.height*scale-motif.height)/2))
            scores[f] = layer.positive
            present[f] = True
        mark = Mark('part', label, layer.mark.keys, layer.mark.line_height, [], [])
        layers.append(Layer(mark, motif, layer.origin, None, layer.positive, positions, scores,
                            np.full(count, -1., np.float32), present, positions.copy()))
    return layers


def animation_key(video, layer, direction, chosen):
    """What one learned animation is kept under: the mark, the side and its edges with the mark's place at each."""
    return (str(video.path), id(layer.mark), layer.mark.label, direction,
            tuple((edge, room, *map(int, layer.positions[edge])) for edge, room in chosen))


def learn_job(job):
    """learn_animation in a worker process."""
    return learn_animation(*job)


def free_memory():
    """Bytes of memory free to be taken, or None where that cannot be told."""
    try:
        if os.name == 'nt':
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                            *((name, ctypes.c_uint64) for name in ('total', 'free', 'total_paged', 'free_paged',
                                                                   'total_virtual', 'free_virtual', 'extended'))]
            status = Status(length=ctypes.sizeof(Status))
            return status.free if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        with open('/proc/meminfo', encoding='ascii') as lines:
            for line in lines:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) << 10
    except (OSError, ValueError):
        pass
    return None


def learn_ahead(video, layers, cuda):
    """Learn the animations follow_animations is about to ask for, each in a process of its own, where there are
    several: each is learned from its own mark and frames alone, and learning takes one core for a minute or more."""
    jobs = {}
    for layer in layers:
        for direction in (-1, 1):
            edges = entry_edges(video, layer, direction)
            for group in entry_groups(video, layer, layers, edges, direction):
                chosen = [edges[n] for n in group]
                key = animation_key(video, layer, direction, chosen)
                if len(group) >= ENTRY_MEMBERS and key not in _entry_cache:
                    jobs[key] = (video, layer, chosen, direction, cuda)
    # No more processes than the free memory holds: short of it, the animations are learned one by one in this one.
    free = free_memory()
    workers = min(len(jobs), ENTRY_WORKERS, ENTRY_WORKERS if free is None else free//ENTRY_WORKER_MEMORY)
    if workers < 2:
        return
    # Started afresh rather than forked: a copy of this process would take along the state of its native threads.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        try:
            for key, learned in zip(jobs, pool.map(learn_job, jobs.values())):
                _entry_cache[key] = learned
        except BrokenProcessPool:
            pass    # a worker died: what it left is learned in this process, as it is without workers


def follow_animations(video, layers, cuda, settings, trace):
    """Layers for the animations the marks show when they enter or leave the picture."""
    parts = []
    learn_ahead(video, layers, cuda)
    for layer in layers:
        for direction in (-1, 1):
            edges = entry_edges(video, layer, direction)
            grouped = entry_groups(video, layer, layers, edges, direction)
            places = [group for group in grouped if len(group) >= ENTRY_MEMBERS]
            for n in sorted(n for group in grouped if len(group) < ENTRY_MEMBERS for n in group):
                edge, room = edges[n]
                key = (str(video.path), id(layer.mark), layer.mark.label, direction, 'alone', edge, room,
                       *map(int, layer.positions[edge]))
                if key not in _entry_cache:
                    _entry_cache[key] = follow_alone(video, layer, layers, edge, room, direction, cuda)
                followed = _entry_cache[key]
                if not followed:
                    continue
                side = f"{'entrance' if direction < 0 else 'exit'} alone"
                found = alone_layers(video, layer, followed, settings, side)
                parts.extend(found)
                trace('plan.animation_alone', label=layer.mark.label, side=side, edge=int(edge),
                      frames=len(followed), first=min(followed), last=max(followed),
                      scales=[round(min(s for _, _, s, _ in followed.values()), 3),
                              round(max(s for _, _, s, _ in followed.values()), 3)],
                      angles=[round(min(a for _, _, _, a in followed.values()), 1),
                              round(max(a for _, _, _, a in followed.values()), 1)], layers=len(found))
            for place, group in enumerate(places):
                chosen = [edges[n] for n in group]
                key = animation_key(video, layer, direction, chosen)
                if key not in _entry_cache:
                    _entry_cache[key] = learn_animation(video, layer, chosen, direction, cuda)
                steps, placed, (lingering, lingers) = _entry_cache[key]
                side = 'entrance' if direction < 0 else 'exit'
                if len(places) > 1:
                    side = f'{side} {place+1}'
                spans = [[(j, shown[j], path[j]) for j in range(1, extent+1) if j in shown]
                         for _, extent, path, shown in placed]
                found = animation_layers(video, layer, direction, steps, spans, chosen, settings, side)
                kept = animation_layers(video, layer, direction, lingering, lingers, chosen, settings, f'{side} inside')
                parts.extend(found+kept)
                travel = [ENTRY_SCALE*max((abs(dx)+abs(dy) for dx, dy in path.values()), default=0)
                          for _, _, path, _ in placed]
                trace('plan.animation', label=layer.mark.label, side=side,
                      place=[int(v) for v in layer.positions[chosen[0][0]]], edges=len(chosen),
                      steps=len([k for k in steps if k >= 1]), matched=sum(extent > 0 for _, extent, _, _ in placed),
                      frames=sum(extent for _, extent, _, _ in placed), shifts=[shift for shift, _, _, _ in placed],
                      travel=travel, layers=len(found), inside_steps=len(lingering),
                      inside_frames=sum(len(span) for span in lingers), inside_layers=len(kept))
    return parts


# ----------------------------------------------------------------------------
# Turned holds
#
# A mark may rest at one place for a while, then jump to the next and rest there set at another slant. Its motif is
# learned at the slant most samples showed and matches the other holds only weakly: the scan finds where the mark
# rests, but presence there rests on scores too low to trust, and the mask does not lie on the letters. Each place the
# scan rests at is searched again with the motif turned; where a turned motif matches clearly better, the hold is
# followed frame by frame to its ends and masked turned, and the layer's own motif gives way there.

TURN_RANGE = 60.      # degrees searched either way of the slant the motif was learned at
TURN_STEP = 3.        # degrees between the angles searched first; the best is then refined to a degree
TURN_RADIUS = 24      # pixels the turned motif may lie from where the scan rested
TURN_REST = .5        # seconds the scanned placement must rest to seed a hold
TURN_STILL = 2        # pixels the scanned placement may move from frame to frame while resting
TURN_LOOKS = 5        # frames of each seed searched
TURN_MATCH = .5       # mean score the turned motif must reach on them
TURN_GAIN = .1        # by which it must beat the motif as learned
TURN_LEAST = 2.       # degrees below which the layer's own motif serves the hold
TURN_FOLLOW = .55     # share of the seed's score that keeps a hold followed, frame by frame, towards its ends
TURN_FLOOR = .3       # score below which a hold is never followed
TURN_PATIENCE = .5    # seconds a hold may go unmatched while followed (someone passing in front)
TURN_DRIFT = 3        # pixels a hold may drift from frame to frame while followed

_turn_cache = {}      # holds found per layer and seeds: later attempts only change the masks' dilation


def turned_motif(motif, angle):
    """A motif turned about its centre in a canvas holding all of it."""
    if abs(angle) < .25:
        return motif
    mask = (turned(motif.mask.astype(np.float32), angle, expand=True) >= .5).astype(np.uint8)
    return Motif(turned(motif.signal, angle, expand=True), turned(motif.weights, angle, expand=True), mask)


def rest_seeds(video, layer):
    """Stretches where the scan rested a moving layer's motif, with its centre in each frame:
    [(start, stop, {frame: centre})]."""
    located = layer.located
    half = np.array([layer.motif.width/2, layer.motif.height/2])
    still = np.zeros(video.count, bool)
    still[1:] = np.abs(np.diff(located, axis=0)).max(1) <= TURN_STILL
    least = max(3, round(TURN_REST*float(video.fps)))
    return [(start-1, stop, {f: located[f]+half for f in range(start-1, stop)})
            for start, stop in runs(still) if stop-start+1 >= least]


def read_seeds(video, layer):
    """Frames the samples read the mark in, with its motif's centre there, and a few frames around each: a hold the
    scan did not rest on (the mark faded in late, or a passing badge drew the scan away) is still read.
    [(frame, frame+1, {frame: centre})]"""
    size = np.array([layer.motif.width, layer.motif.height])
    seeds = []
    for f, (x, y, _, _) in layer.mark.observations:
        centre = np.add((x, y), layer.origin)+size/2
        seeds.append((f, f+1, {g: centre for g in range(f-4, f+5, 2) if 0 <= g < video.count}))
    return seeds


def centred_score(motif, signal, centre, radius):
    """Best score of a motif whose centre lies within radius of centre, and the centre it was found at."""
    guess = (int(round(centre[0]-motif.width/2)), int(round(centre[1]-motif.height/2)))
    x, y, score = match_near(motif, signal, guess, radius)
    return score, (x+motif.width/2, y+motif.height/2)


def turn_looks(places):
    """The few frames of a seed ({frame: centre}) that are searched, spread over it."""
    frames = sorted(places)
    return sorted({frames[round(k*(len(frames)-1)/max(1, TURN_LOOKS-1))] for k in range(TURN_LOOKS)})


def search_turn(decoded, layer, places, motifs):
    """The angle the layer's motif matches best at around the given centres ({frame: centre}, a few looked at):
    (angle, score, score as learned, offset of the turned motif's centre from the given ones)."""
    picks = turn_looks(places)
    signals = [decoded.pick(f).signal(f) for f in picks]
    centres = [places[f] for f in picks]
    results = {}

    def mean_at(angle):
        if angle not in results:
            if angle not in motifs:
                motifs[angle] = turned_motif(layer.motif, angle)
            found = [centred_score(motifs[angle], s, c, TURN_RADIUS) for s, c in zip(signals, centres)]
            results[angle] = (float(np.mean([score for score, _ in found])),
                              np.median([np.subtract(at, c) for (_, at), c in zip(found, centres)], axis=0))
        return results[angle][0]

    best = max(np.arange(-TURN_RANGE, TURN_RANGE+.5, TURN_STEP), key=mean_at)
    best = max([best+d for d in (-2., -1., 0., 1., 2.)], key=mean_at)
    return float(best), results[best][0], mean_at(0.), results[best][1]


def follow_turn(video, decoded, motif, edge, centre, threshold, direction):
    """Frames beyond a hold's edge, in one direction, where its turned motif still matches near where it was last
    found: {frame: centre}."""
    patience = max(2, round(TURN_PATIENCE*float(video.fps)))
    found, misses, place = {}, 0, centre
    i = edge+direction
    while 0 <= i < video.count:
        score, at = centred_score(motif, decoded.signal(i, direction), place, TURN_DRIFT)
        if score >= threshold:
            found[i], place, misses = at, at, 0
        else:
            misses += 1
            if misses > patience:
                break
        i += direction
    return found


def find_holds(video, layer, cuda, taken, trace):
    """Turned holds of one moving layer: [(first, last, angle, score, as learned, {frame: centre})]. Seeds already
    followed by a hold of this or another layer (taken: [(first, last, centre)]) are skipped. Where the scan rested
    is tried first, then where the samples read the mark."""
    seeds = rest_seeds(video, layer)+read_seeds(video, layer)
    key = (str(video.path), id(layer.mark), layer.mark.label, tuple((start, stop) for start, stop, _ in seeds))
    if key in _turn_cache:
        return _turn_cache[key]
    decoded, motifs, holds = Decoded(video, cuda), {}, []
    reach = TURN_RADIUS+max(layer.motif.width, layer.motif.height)/2
    searched = 0

    def followed(start, stop, places, found):
        middle = (start+stop)//2
        return any(first <= middle <= last and math.dist(places[middle], centre) <= reach
                   for first, last, centre in found)
    room, ahead = decoded.room(), set()
    for number, (start, stop, places) in enumerate(seeds):
        if followed(start, stop, places, taken+[(h[0], h[1], h[5][(h[0]+h[1])//2]) for h in holds]):
            continue
        if not ahead.issuperset(turn_looks(places)):
            # The frames searched are decoded ahead together, for as many of the seeds to come as fit: a decoder
            # for each frame takes most of the search's time. A hold found on the way only spares some the search.
            ahead = set(turn_looks(places))
            for a, b, later in seeds[number+1:]:
                if followed(a, b, later, taken):
                    continue
                if len(ahead.union(turn_looks(later))) > room:
                    break
                ahead.update(turn_looks(later))
            decoded.preload(ahead)
        searched += 1
        angle, score, upright, offset = search_turn(decoded, layer, places, motifs)
        if score < TURN_MATCH or score < upright+TURN_GAIN or abs(angle) < TURN_LEAST:
            continue
        motif = motifs[angle]
        centres = {f: tuple(places[f]+offset) for f in range(start, stop)}
        threshold = max(TURN_FLOOR, TURN_FOLLOW*score)
        centres.update(follow_turn(video, decoded, motif, start, centres[start], threshold, -1))
        centres.update(follow_turn(video, decoded, motif, stop-1, centres[stop-1], threshold, 1))
        first, last = min(centres), max(centres)
        last_seen = centres[first]
        for f in range(first, last+1):
            last_seen = centres.setdefault(f, last_seen)
        holds.append((first, last, angle, score, upright, centres))
        trace('plan.turned_hold', label=layer.mark.label, seed=[start, stop], angle=angle, score=round(score, 3),
              upright=round(upright, 3), first=first, last=last,
              centre=[round(v) for v in centres[(first+last)//2]])
    trace('plan.turned_seeds', label=layer.mark.label, seeds=len(seeds), searched=searched, holds=len(holds))
    _turn_cache[key] = holds
    return holds


def hold_layer(video, layer, hold, settings):
    """A layer masking one turned hold with the layer's mask turned (which carries the layer's dilation), grown by
    what earlier attempts added for the hold itself."""
    first, last, angle, _, _, centres = hold
    label = f'{layer.mark.label} (turned {angle:+.0f} {first}-{last})'
    motif = turned_motif(layer.motif, angle)
    signal, weights, mask = motif.signal, motif.weights, motif.mask
    grow_by = settings.get('grown', {}).get(label, 0)
    if grow_by >= .5:
        pad = int(round(grow_by))
        signal, weights, mask = (cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                                 for a in (signal, weights, mask))
        mask = cv2.dilate(mask, ellipse(2*pad+1))
    motif = Motif(signal, weights, mask)
    count = video.count
    positions = np.zeros((count, 2), np.int32)
    scores = np.zeros(count, np.float32)
    present = np.zeros(count, bool)
    for f, (cx, cy) in centres.items():
        positions[f] = round(cx-motif.width/2), round(cy-motif.height/2)
        scores[f] = layer.positive
        present[f] = True
    mark = Mark('hold', label, layer.mark.keys, layer.mark.line_height, [], [])
    return Layer(mark, motif, layer.origin, None, layer.positive, positions, scores, np.full(count, -1., np.float32),
                 present, positions.copy())


def turned_holds(video, layers, cuda, settings, trace):
    """Layers for the holds of moving layers that rest set at another slant. The layer, and the layers of the same
    words learned at other slants, give way where its holds are."""
    found, taken = [], []
    # The same words: each line of either mark reads like a line of the other. One shared-looking line is not
    # enough: "黄果短剧 huangguo.com" passing by a turned "huangguo.me" is another mark, not this one at a slant.
    def covers(a, b):
        return all(any(similar(x, y) >= .6 for y in b) for x in a)
    for layer in layers:
        same = [other for other in layers if other is layer or
                covers(layer.mark.keys, other.mark.keys) and covers(other.mark.keys, layer.mark.keys)]
        for hold in find_holds(video, layer, cuda, taken, trace):
            first, last, centres = hold[0], hold[1], hold[5]
            taken.append((first, last, centres[(first+last)//2]))
            found.append(hold_layer(video, layer, hold, settings))
            for other in same:
                half = np.array([other.motif.width/2, other.motif.height/2])
                for f in range(first, last+1):
                    if other.present[f] and math.dist(other.positions[f]+half, centres[f]) <= max(half):
                        other.present[f] = False
    return found


# ----------------------------------------------------------------------------
# Logos set into the scene
#
# Some videos show the watermark's brand inside the picture as well: a sign on a door, a poster on a wall, a print on
# a shirt. It moves with the camera rather than with the screen and shows in one shot only, so the scan over the
# whole video cannot learn it. Each is found where the samples read the brand's lettering and no layer repaints it,
# followed frame by frame through its shot, and masked by the boxes of its lettering and emblem.

SCENE_SIMILAR = .75     # likeness to a mark's text from which lettering in the scene counts as the watermark's brand
SCENE_COVERED = .3      # share of a reading under a present layer's mask from which it counts as repaired already
SCENE_UNHIDE = 1.5      # reach, in its line heights, of the region around a sign an animation hid that is read again
SCENE_NEAR = .8         # gap between the parts of one logo, in line heights of its lettering
SCENE_MATCH = .5        # score against its last placement that keeps a followed logo in place
SCENE_ANCHOR = .35      # score against the logo as first read below which following has drifted off it
SCENE_LOCK = .6         # score against the logo as first read from which its place there is taken
SCENE_PART = .45        # score of one part at the logo's place that shows the part there
SCENE_KEEP = .4         # share of checked frames a part must show in, moving with the logo, to belong to it
SCENE_CHECKS = 10       # frames each part is checked in
SCENE_HIDDEN = .5       # seconds a followed logo may stay hidden (someone passing in front) and be picked up again
SCENE_LOOKS = 6         # frames of each logo read again for parts the samples missed
SCENE_CHUNK = 24        # frames decoded at a time while following
SCENE_ONE_SHOT = .25    # share of the samples below which a mark moving within one shot is taken for a scene logo
SCENE_LARGEST = .08     # share of the picture a logo's box may cover: larger lettering is an overlay card, not a sign
SCENE_EMBLEM_LOOKS = 16 # frames of a logo and of its other showings compared for an emblem beside its lettering
# A logo shows for a second or two, often over one dark scene: near-black pixels that stay put tell nothing then.
SCENE_EMBLEM_LEAST = 48
# Share of the picture around a logo that must change between its frames for it to count as laid over the picture:
# around a sign in the scene it changes only where something passes in front (about a quarter or a third), around
# a badge over changing scenes in two thirds or more.
SCENE_OVERLAY = .5
SCENE_ANIMATION = 2.    # seconds searched beyond a logo's first and last frames for how it enters and leaves
SCENE_AGREE = 10        # largest difference, in any channel, of one showing of an animation from all of them
SCENE_SHIFT = 2         # frames by which a showing's animation may run early or late against the first one's
SCENE_STEP_AREA = .05   # least area of an animation step's mask, in squared line heights
# Share of the bright picture away from a logo that two of its showings agree on, from which they show the same footage
# there (an opening teaser repeating a later scene): their agreeing then tells nothing about the animation.
SCENE_REPEAT = .5
SCENE_REREADS = 8       # times per logo it is read again past where following lost it inside its shot
SCENE_REREAD_TRIES = 3  # frames past the end of its run read for it, the nearest first
RESIDUAL_SEED_LIKE = .6 # likeness to the watermark's text of lettering read around a missed residual (as the check's)
RESIDUAL_AROUND = 4.    # residual heights around a missed residual read again on their own (small lettering)
RESIDUAL_TRACE = .3     # |correlation| with an unblended overlay's lettering from which a residual is a trace of it
SCENE_FAINT_LIKE = .3   # |correlation| with a faint overlay's picture from which a reading under it is its lettering
SCENE_EMBLEM_READ = .9  # recognition score from which a letter read on its own may be the watermark's emblem
SCENE_BADGE_REACH = 4.  # emblem heights to either side searched for the lettering of the badge it belongs to
SCENE_CARRY_GAP = .2   # seconds between the frames compared for whether the picture around a logo moves with it
SCENE_CARRY_LEAST = 3.  # pixels a logo must move between them for the comparison to tell anything
SCENE_CARRIED = .6      # share of its moves the picture around a logo makes along from which it is in the scene


def lettering(key, rect, line):
    """Letter height of a reading: a tall box of several letters is a vertical line, as high as it is wide."""
    if len(key) >= 2 and rect[3] > 1.5*rect[2]:
        return float(min(line, rect[2]))
    return float(line)


def too_large(rect, video):
    return rect[2]*rect[3] > SCENE_LARGEST*video.width*video.height


def scene_text(key, keys):
    return len(key) >= 3 and (bool(BRAND.search(key)) or
                              max((similar(key, k) for k in keys), default=0.) >= SCENE_SIMILAR)


def icon_like(key):
    """A logo's emblem read as a letter or two (the 'H' of a brand)."""
    return len(key) <= 2 and key.isascii() and key.isalpha()


def within_one_shot(mark, samples):
    """A mark seen moving within one shot of a video of three or more, in few samples: something the camera passes
    (a sign on a door), not an overlay moving over the whole video."""
    shots = {int(samples.shots[frame]) for frame, _ in mark.observations}
    return (mark.kind == 'moving' and len(shots) == 1 and len(samples.cuts) >= 2
            and len(mark.observations) < SCENE_ONE_SHOT*len(samples.frames))


def rect_gap(a, b):
    dx = max(0, max(a[0], b[0])-min(a[0]+a[2], b[0]+b[2]))
    dy = max(0, max(a[1], b[1])-min(a[1]+a[3], b[1]+b[3]))
    return math.hypot(dx, dy)


def under_layers(layers, frame, rect):
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return 0.
    return float(covered_by(layers, frame, (x, y, w, h)).mean())


def lettering_likeness(layer, signal, frame, rect):
    """|Correlation|, over the layer's weights, of the picture's detail in rect with the layer's own picture where it
    is placed in frame: a reading of the layer's lettering follows it, lettering of its own seen through a faint
    overlay (a badge on a costume behind it) does not. None where too little of the layer lies under rect."""
    if not layer.present[frame]:
        return None
    lx, ly = (int(v) for v in layer.positions[frame])
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(x, lx, 0), max(y, ly, 0)
    x1 = min(x+w, lx+layer.motif.width, signal.shape[1])
    y1 = min(y+h, ly+layer.motif.height, signal.shape[0])
    if x1-x0 < 8 or y1-y0 < 8:
        return None
    inside = (slice(y0-ly, y1-ly), slice(x0-lx, x1-lx))
    weights = layer.motif.weights[inside]*(layer.motif.mask[inside] > 0)
    total = float(weights.sum())
    if np.count_nonzero(weights) < 64 or total <= 0:
        return None
    own = layer.motif.signal[inside]
    picture = signal[y0:y1, x0:x1].astype(np.float32)
    own = own-float((own*weights).sum())/total
    picture = picture-float((picture*weights).sum())/total
    norm = math.sqrt(float((weights*own*own).sum())*float((weights*picture*picture).sum()))
    return abs(float((weights*own*picture).sum()))/norm if norm > 0 else None


def repaired_already(layers, frame, rect, signal):
    """Whether a reading is lettering a present layer repairs: under an opaque mark's mask, or under a faint overlay's
    and following its letters. A faint overlay shows the picture through it, and lettering of the scene's own there is
    not removed with it. signal() gives the frame's detail (high-pass), asked for only under a faint overlay."""
    if under_layers([l for l in layers if l.mark.kind != 'faint'], frame, rect) >= SCENE_COVERED:
        return True
    faint = [l for l in layers if l.mark.kind == 'faint' and l.present[frame]]
    if not faint or under_layers(faint, frame, rect) < SCENE_COVERED:
        return False
    picture = signal()
    likeness = [lettering_likeness(l, picture, frame, rect) for l in faint]
    # Where the likeness cannot be told, the reading is taken for the overlay's own.
    return max((v for v in likeness if v is not None), default=1.) >= SCENE_FAINT_LIKE


@dataclass(frozen=True)
class Part:
    key: str            # what OCR read
    rect: tuple         # where, in the coordinates of the frame the logo was first read in
    line: float
    frame: int          # frame it was read in


def attach_parts(parts, rows, line):
    """Grow a logo's parts with the rows within reach of them until none is added; the reach is measured in the
    logo's line height, as an emblem's box says nothing of its lettering."""
    parts, rows = list(parts), list(rows)
    while True:
        near = [row for row in rows if any(rect_gap(row.rect, part.rect) <= SCENE_NEAR*line for part in parts)]
        if not near:
            return parts
        parts += near
        rows = [row for row in rows if row not in near]


def enclosing_icon(row, others):
    """A letter or two read in a box around other lettering: the picture around a logo read as letters (the folds of a
    costume around a badge pinned to it), not an emblem beside the logo - the parts of a logo sit side by side."""
    w, h = row.rect[2:]
    return icon_like(row.key) and any(other is not row and overlap_share(row.rect, other.rect) >= .8 and
                                      w*h >= 2*other.rect[2]*other.rect[3] for other in others)


def new_parts(rows, parts):
    """Rows that are not a part already read, nor another reading of one another."""
    rows = [row for row in rows if not enclosing_icon(row, rows+parts)]
    kept = []
    for row in rows:
        if all(iou(row.rect, other.rect) < .4 for other in parts+kept):
            kept.append(row)
    return kept


def text_line(parts):
    """Line height of a logo's lettering: an emblem read as a letter has its own box height."""
    lines = [p.line for p in parts if not icon_like(p.key)]
    return float(np.median(lines or [p.line for p in parts]))


@dataclass(eq=False)
class Scene:
    frame: int              # frame the logo was first read in
    shot: tuple             # (first, stop) frames of its shot
    parts: list             # [Part]
    mirrored: bool = False  # printed back to front, read flipped
    track: dict = None      # frame -> (dx, dy, score): the logo's shift from where it was first read
    emblem: tuple = None    # (dx, dy, mask): pixels beside the lettering that belong to the badge, relative to rect
    animations: list = None # [(side, {step: mask}, {frame: (step, x, y)})]: how it enters and leaves
    overlay: bool = False   # laid over the picture, which changes around it, rather than set into the scene

    @property
    def line(self):
        return text_line(self.parts)

    @property
    def rect(self):
        return union_rect([p.rect for p in self.parts])

    @property
    def keys(self):
        return [p.key for p in self.parts]

    @property
    def label(self):
        return f'scene {" ".join(self.keys)} ({min(self.track)}-{max(self.track)})'


DECODED_AHEAD = 1 << 30         # bytes of frames held ahead at once (preload)
DECODED_AHEAD_SIGNALS = 16      # frames held ahead whose high-pass detail is kept


class Decoded:
    """Frames decoded a chunk at a time in the direction they are visited, or one at a time when picked; frames known
    to be picked later can be decoded ahead together."""

    def __init__(self, video, cuda):
        self.video, self.cuda, self.raws, self.signals, self.picked, self.ahead = video, cuda, {}, {}, {}, {}

    def room(self):
        """How many frames may be held ahead at once."""
        return max(8, DECODED_AHEAD//(self.video.width*self.video.height*3//2))

    def preload(self, frames):
        """Decode frames that will be picked, in place of those held ahead before: one decoder reads each run of
        them, where picking them one at a time starts a decoder for every frame."""
        self.ahead = {}
        self.ahead = dict(read_selected(self.video, frames, self.cuda))
        self.signals = {k: v for k, v in self.signals.items() if k in self.picked or k in self.raws}

    def raw(self, i, direction=1):
        if i in self.picked:
            return self.picked[i]
        if i in self.ahead:
            return self.ahead[i]
        if i not in self.raws:
            first, last = sorted((i, i+direction*(SCENE_CHUNK-1)))
            first, last = max(0, first), min(self.video.count-1, last)
            self.raws = dict(read_frames(self.video, first, last+1, self.cuda))
            self.signals = {k: v for k, v in self.signals.items() if k in self.picked}
        return self.raws[i]

    def pick(self, i):
        if i not in self.picked and i not in self.raws and i not in self.ahead:
            if len(self.picked) > 64:
                self.picked.clear()
                self.signals = {k: v for k, v in self.signals.items() if k in self.raws}
            self.picked[i] = get_frame(self.video, i, self.cuda)
        return self

    def signal(self, i, direction=1):
        raw = self.raw(i, direction)
        if i not in self.signals:
            if i in self.ahead:
                # Many frames may be held ahead; their detail is kept for the last few looked at only.
                for k in [k for k in self.signals if k in self.ahead][:-DECODED_AHEAD_SIGNALS]:
                    del self.signals[k]
            self.signals[i] = highpass(self.video.planes(raw)[0], HIGHPASS_SIGMA)
        return self.signals[i]

    def bgr(self, i):
        return yuv_to_rgb(self.video.planes(self.raw(i)), self.video.matrix)[..., ::-1].copy()


def detail_motif(signal, layers, frame, box):
    """The picture's detail in box as a motif, leaving out what lies outside the picture or under another layer's
    mask (a badge over the logo); a faint overlay shows the picture through it."""
    picture = patch(signal, box).astype(np.float32)
    weights = (~covered_by([l for l in layers if l.mark.kind != 'faint'], frame, box)).astype(np.float32)
    x, y, w, h = box
    outside = np.ones((h, w), bool)
    view, _ = clipped_plane(outside, (-x, -y, signal.shape[1], signal.shape[0]))
    view[:] = False
    weights[outside] = 0
    try:
        return Motif(picture, weights, np.ones(picture.shape, np.uint8))
    except ValueError:
        return None


def match_near(motif, signal, guess, radius):
    """Best (x, y, score) of a motif within radius of guess, scored on its part inside the picture."""
    x, y = guess
    inside = (min(motif.width, signal.shape[1]-x)-max(0, -x))*(min(motif.height, signal.shape[0]-y)-max(0, -y))
    if inside >= .97*motif.width*motif.height:
        return motif.local(signal, guess, radius, radius)
    return clipped_local(motif, signal, guess, radius, radius)


def parts_visible(scene, anchor, signal, shift, box):
    """Whether any part of a logo hidden at this shift still shows (someone passing in front of half of it)."""
    for part in scene.parts:
        x, y, w, h = part.rect
        if w < 8 or h < 8 or part.frame != scene.frame:
            continue
        sub = (x-box[0], y-box[1])
        piece = np.ascontiguousarray(anchor.signal[sub[1]:sub[1]+h, sub[0]:sub[0]+w])
        weights = np.ascontiguousarray(anchor.weights[sub[1]:sub[1]+h, sub[0]:sub[0]+w])
        if piece.shape != (h, w) or np.count_nonzero(weights) < 32:
            continue
        try:
            motif = Motif(piece, weights, np.ones(piece.shape, np.uint8))
        except ValueError:
            continue
        if match_near(motif, signal, (round(x+shift[0]), round(y+shift[1])), 3)[2] >= SCENE_PART:
            return True
    return False


def follow_scene(video, decoded, scene, layers, direction):
    """Follow a logo from the frame it was read in for as long as it shows: {frame: (dx, dy, score)}.

    Each frame is matched against the last (the view turns and zooms slowly) and checked against the logo as first
    read, so following cannot drift off it. Frames where it is hidden count where a part of it still shows. A logo
    in the scene ends at its shot's cut, where the picture around it changes; a badge sliding over the picture
    (lettering read in one sample only) carries on across the cut, so cuts do not bound the following.
    """
    first, stop = 0, video.count
    line = scene.line
    margin = max(8, round(.5*line))
    box = grow(scene.rect, margin, margin)
    anchor = detail_motif(decoded.signal(scene.frame, direction), layers, scene.frame, box)
    if anchor is None:
        return {}
    current, shift, velocity = anchor, (0., 0.), (0., 0.)
    patience = max(2, round(SCENE_HIDDEN*float(video.fps)))
    track, hidden = {}, []
    frame = scene.frame
    while first <= frame+direction < stop:
        frame += direction
        signal = decoded.signal(frame, direction)
        steps = len(hidden)+1
        guess = (round(box[0]+shift[0]+steps*velocity[0]), round(box[1]+shift[1]+steps*velocity[1]))
        radius = max(6, round(.4*line))+round(math.hypot(*velocity)*steps)
        x, y, score = match_near(current, signal, guess, radius)
        ax, ay, anchored = match_near(anchor, signal, (x, y), 2)
        if score >= SCENE_MATCH and anchored >= SCENE_ANCHOR:
            if anchored >= SCENE_LOCK:
                # Matched only frame to frame, a logo creeping a fraction of a pixel a frame (a badge on someone
                # breathing) never moves at all and slides out of its box; where the logo as first read still
                # matches well, the place it matches at is the logo's.
                x, y = ax, ay
            moved = (x-box[0], y-box[1])
            for k, f in enumerate(hidden, 1):
                # Hidden in between: in place, wherever a part of it still shows.
                between = tuple(shift[a]+(moved[a]-shift[a])*k/steps for a in (0, 1))
                if parts_visible(scene, anchor, decoded.signal(f, direction), between, box):
                    track[f] = (*between, 0.)
            velocity = ((moved[0]-shift[0])/steps, (moved[1]-shift[1])/steps)
            shift, hidden = moved, []
            track[frame] = (*shift, score)
            current = detail_motif(signal, layers, frame, (x, y, box[2], box[3])) or current
        else:
            hidden.append(frame)
            if len(hidden) > patience:
                break
    last = frame-direction*len(hidden)
    for f in hidden:
        guess = tuple(shift[a]+velocity[a]*abs(f-last) for a in (0, 1))
        if not parts_visible(scene, anchor, decoded.signal(f, direction), guess, box):
            break
        track[f] = (*guess, 0.)
    return track


def shot_bounds(cuts, count, frame):
    bounds = [0]+[int(c) for c in cuts]+[count]
    k = int(np.searchsorted(bounds, frame, side='right'))-1
    return bounds[k], bounds[k+1]


def mirrored_rows(ocr, picture, rect, line, frame):
    """Readings of a horizontally flipped region around an emblem: a logo printed back to front."""
    x0, y0, w, h = grow(rect, round(6*line), round(1.5*line))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(picture.shape[1], x0+w), min(picture.shape[0], y0+h)
    crop = np.ascontiguousarray(picture[y0:y1, x0:x1][:, ::-1])
    return [Part(key, (x0+(x1-x0)-x-w, y0+y, w, h), lettering(key, (x, y, w, h), height), frame)
            for key, text, (x, y, w, h), score, height in ocr.read(crop)]


def emblem_badge(ocr, picture, emblem, frame, layers, signal):
    """The badge around an emblem read on its own: what the text detector finds next to it, however poorly it is
    read (a small badge's lettering shows as text yet reads as nothing), in a region a few emblem heights wide
    around it. A box running out of that region is a line of the scene's own (a caption), not the badge's."""
    line = emblem.rect[3]
    x0, y0, w, h = grow(emblem.rect, round(SCENE_BADGE_REACH*line), round(line))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(picture.shape[1], x0+w), min(picture.shape[0], y0+h)
    rows = []
    width, height_ = x1-x0, y1-y0
    for key, text, (x, y, rw, rh), score, height in ocr.read(np.ascontiguousarray(picture[y0:y1, x0:x1]), 0.):
        cut = (x <= 1 and x0 > 0) or (y <= 1 and y0 > 0) or \
              (x+rw >= width-1 and x1 < picture.shape[1]) or (y+rh >= height_-1 and y1 < picture.shape[0])
        rect = (x0+x, y0+y, rw, rh)
        if not cut and not repaired_already(layers, frame, rect, signal):
            rows.append(Part(key, rect, lettering(key, rect, height), frame))
    return attach_parts([emblem], new_parts(rows, [emblem]), line)


def hidden_signs(video, samples, layers, keys, ocr, decoded, frame, rows):
    """Lettering of the scene's own (a poster on a wall) that only a mark's animation covers in a sample (a banner
    flying across it), read again in the nearest frame of its shot where no animation step covering it shows. Video
    inpainting brings back what a passing mark covers from the frames around, so a sign under it stays, while the
    animation's own lettering leaves with it and reads as nothing there. Returns [(frame, parts)]."""
    lasting = [l for l in layers if l.mark.kind != 'part']
    signal = lambda: samples.signals[frame]
    first, stop = shot_bounds(samples.cuts, video.count, frame)
    found = []
    for r in rows:
        rect = tuple(int(v) for v in r.rect)
        # A reading among the parts found already is read; one near a region read again that held no sign is not:
        # the animation's own lettering beside a sign reads as nothing there.
        if not scene_text(r.key, keys) or any(rect_gap(rect, p.rect) == 0 for _, parts in found for p in parts) or \
                not repaired_already(layers, frame, rect, signal) or repaired_already(lasting, frame, rect, signal):
            continue
        hiding = [l for l in layers
                  if l.mark.kind == 'part' and l.present[frame] and under_layers([l], frame, rect) > 0]
        clear = [g for g in range(first, stop) if not any(l.present[g] for l in hiding)]
        if not clear:
            continue
        at = min(clear, key=lambda g: (abs(g-frame), g))
        reach = max(4, round(SCENE_UNHIDE*lettering(r.key, rect, r.thickness)))
        gx, gy, gw, gh = grow(rect, reach, reach)
        x0, y0 = max(0, gx), max(0, gy)
        x1, y1 = min(video.width, gx+gw), min(video.height, gy+gh)
        crop = np.ascontiguousarray(decoded.pick(at).bgr(at)[y0:y1, x0:x1])
        there = lambda: decoded.pick(at).signal(at)
        parts = []
        for key, text, (x, y, w, h), score, height in ocr.read(crop):
            box = (x0+x, y0+y, w, h)
            if (scene_text(key, keys) or icon_like(key)) and not repaired_already(layers, at, box, there):
                parts.append(Part(key, box, lettering(key, box, height), at))
        parts = new_parts(parts, [])
        brand = [p for p in parts if not icon_like(p.key)]
        if brand:
            found.append((at, attach_parts([brand[0]], [p for p in parts if p is not brand[0]], brand[0].line)))
    return found


def scene_seeds(video, samples, layers, keys, ocr, decoded, trace):
    """Logos in the samples: watermark lettering no present layer repaints, with the emblem and lines next to it."""
    by_frame = {}
    for r in samples.readings:
        by_frame.setdefault(r.frame, []).append(r)
    # The letter the watermark's own emblem reads as: the same letter read sharply on its own in the scene is taken
    # for the emblem of a badge whose lettering is too small to read.
    emblems = {k for k in keys if icon_like(k)}
    scenes = []
    for frame, rows in sorted(by_frame.items()):
        signal = lambda: samples.signals[frame]
        unrepaired = [r for r in rows if not repaired_already(layers, frame, r.rect, signal)]
        free = [Part(r.key, tuple(int(v) for v in r.rect), lettering(r.key, r.rect, r.thickness), frame)
                for r in unrepaired]
        sure = {id(p) for p, r in zip(free, unrepaired) if r.score >= SCENE_EMBLEM_READ}
        brand = [p for p in free if scene_text(p.key, keys)]
        icons = [p for p in free if icon_like(p.key)]
        found = []
        while brand:
            parts = attach_parts([brand[0]], brand[1:]+icons, brand[0].line)
            brand = [p for p in brand if p not in parts]
            icons = [p for p in icons if p not in parts]
            found.append((parts, False, False))
        if icons:
            picture = decoded.pick(frame).bgr(frame)
            for icon in icons:
                flipped = [p for p in mirrored_rows(ocr, picture, icon.rect, icon.line, frame)
                           if scene_text(p.key, keys) and not repaired_already(layers, frame, p.rect, signal)]
                if flipped:
                    found.append((attach_parts([icon], flipped, text_line(flipped)), True, False))
                elif icon.key in emblems and id(icon) in sure:
                    found.append((emblem_badge(ocr, picture, icon, frame, layers, signal), False, True))
        for parts, mirrored, emblem in found:
            rect = union_rect([p.rect for p in parts])
            trace('plan.scene_seed', frame=frame, rect=list(rect), keys=[p.key for p in parts], mirrored=mirrored,
                  emblem=emblem, too_large=too_large(rect, video))
            if not too_large(rect, video):
                scenes.append(Scene(frame, shot_bounds(samples.cuts, video.count, frame), parts, mirrored))
        for at, parts in hidden_signs(video, samples, layers, keys, ocr, decoded, frame, rows):
            rect = union_rect([p.rect for p in parts])
            trace('plan.scene_seed', frame=at, rect=list(rect), keys=[p.key for p in parts], mirrored=False,
                  emblem=False, too_large=too_large(rect, video), hidden_in=frame)
            if not too_large(rect, video):
                scenes.append(Scene(at, shot_bounds(samples.cuts, video.count, at), parts))
    return scenes


def clearest_frames(scene, count):
    """Up to count frames spread over a followed logo, each the best matched of its stretch (the least hidden)."""
    frames = sorted(f for f, (_, _, score) in scene.track.items() if score >= SCENE_MATCH)
    if not frames:
        return []
    return [int(max(stretch, key=lambda f: scene.track[f][2]))
            for stretch in np.array_split(np.asarray(frames), min(count, len(frames))) if len(stretch)]


def look_again(video, decoded, scene, layers, keys, ocr):
    """Read the logo again in a few followed frames, once as it is and once with its known parts painted over (an
    emblem next to lettering is often read only on its own), and add the parts found next to it."""
    for frame in clearest_frames(scene, SCENE_LOOKS):
        dx, dy, _ = scene.track[frame]
        line = scene.line
        x0, y0, w, h = grow(scene.rect, round(2.5*line), round(2.5*line))
        x0, y0 = max(0, int(x0+dx)), max(0, int(y0+dy))
        x1, y1 = min(video.width, x0+w), min(video.height, y0+h)
        if x1-x0 < 16 or y1-y0 < 16:
            continue
        crop = np.ascontiguousarray(decoded.pick(frame).bgr(frame)[y0:y1, x0:x1])
        known = np.zeros(crop.shape[:2], np.uint8)
        pad = max(2, round(.2*line))
        for part in scene.parts:
            left, top = int(part.rect[0]+dx-x0), int(part.rect[1]+dy-y0)
            cv2.rectangle(known, (left-pad, top-pad), (left+part.rect[2]+pad, top+part.rect[3]+pad), 1, cv2.FILLED)
        painted = cv2.inpaint(crop, known, 5, cv2.INPAINT_TELEA)
        rows = []
        for picture in (crop, painted):
            seen = np.ascontiguousarray(picture[:, ::-1]) if scene.mirrored else picture
            for key, text, (x, y, rw, rh), score, height in ocr.read(seen):
                if scene.mirrored:
                    x = seen.shape[1]-x-rw
                rect = (x0+x, y0+y, rw, rh)
                if (scene_text(key, keys) or icon_like(key)) and \
                        not repaired_already(layers, frame, rect, lambda: decoded.pick(frame).signal(frame)):
                    rows.append(Part(key, (int(rect[0]-dx), int(rect[1]-dy), rw, rh), lettering(key, rect, height),
                                     frame))
        scene.parts = attach_parts(scene.parts, new_parts(rows, scene.parts), line)


def moving_parts(decoded, scene, layers):
    """The parts that show where the logo is in most followed frames, and each one's share of frames shown: a badge
    animation or a caption passing over the logo while it was read does not belong to it."""
    frames = sorted(f for f, (_, _, score) in scene.track.items() if score >= SCENE_MATCH)
    checks = sorted({frames[round(k*(len(frames)-1)/max(1, SCENE_CHECKS-1))]
                     for k in range(min(SCENE_CHECKS, len(frames)))})
    kept, shares = [], []
    for part in scene.parts:
        dx, dy, _ = scene.track[part.frame]
        x, y, w, h = part.rect
        piece = detail_motif(decoded.pick(part.frame).signal(part.frame), layers, part.frame,
                             (round(x+dx), round(y+dy), w, h)) if min(w, h) >= 8 and checks else None
        if piece is None:
            kept.append(part)
            shares.append(None)
            continue
        shown = 0
        for f in checks:
            fx, fy, _ = scene.track[f]
            shown += match_near(piece, decoded.pick(f).signal(f), (round(x+fx), round(y+fy)), 3)[2] >= SCENE_PART
        shares.append(round(shown/len(checks), 2))
        if shown >= SCENE_KEEP*len(checks):
            kept.append(part)
    return kept or scene.parts, shares


def fill_hidden(video, decoded, scene, layers):
    """Frames inside a followed logo's run where it was lost, taken in wherever one of its final parts shows at the
    place between its neighbours (an emblem found by the second look shows past a ribbon over the lettering).
    Returns the frames taken in."""
    patience = max(2, round(SCENE_HIDDEN*float(video.fps)))
    followed = sorted(scene.track)
    pieces = []
    for part in scene.parts:
        dx, dy, _ = scene.track[part.frame]
        x, y, w, h = part.rect
        if min(w, h) >= 8:
            piece = detail_motif(decoded.pick(part.frame).signal(part.frame), layers, part.frame,
                                 (round(x+dx), round(y+dy), w, h))
            if piece is not None:
                pieces.append((piece, x, y))
    filled = []
    for before, after in zip(followed, followed[1:]):
        if not 1 < after-before <= patience+1:
            continue
        for f in range(before+1, after):
            t = (f-before)/(after-before)
            shift = [scene.track[before][a]+(scene.track[after][a]-scene.track[before][a])*t for a in (0, 1)]
            signal = decoded.pick(f).signal(f)
            if any(match_near(piece, signal, (round(x+shift[0]), round(y+shift[1])), 3)[2] >= SCENE_PART
                   for piece, x, y in pieces):
                scene.track[f] = (*shift, 0.)
                filled.append(f)
    return filled


def scene_logos(video, samples, layers, keys, ocr, cuda, trace):
    """Find and follow the logos set into the scene; their layers are built from them per attempt (scene_layer)."""
    decoded = Decoded(video, cuda)
    return follow_logos(video, decoded, scene_seeds(video, samples, layers, keys, ocr, decoded, trace), layers, keys,
                        ocr, trace)


def residual_scenes(video, findings, layers, unblended, keys, ocr, cuts, cuda, trace):
    """Logos set into the scene that the samples did not read (a badge pinned to a costume in shots between them):
    where the check found the watermark's lettering left outside every repaired region and a learned mark's picture
    matches the source there. The source frame is read again around each such residual and what is read there is
    followed like a logo found in the samples. A residual over an unblended overlay is a trace of it, not a logo."""
    decoded = Decoded(video, cuda)
    by_frame = {}
    for f in findings:
        if f['cause'] != 'missed':
            continue
        frame, rect = f['frame'], tuple(int(v) for v in f['rect'])
        likeness = unblend_likeness(unblended, frame, rect, video.planes(decoded.pick(frame).raw(frame))[0])
        if likeness is not None and likeness >= RESIDUAL_TRACE:
            trace('plan.residual_trace', frame=frame, rect=list(rect), likeness=round(likeness, 3))
            continue
        by_frame.setdefault(frame, []).append((rect, text_key(f['text'])))
    emblems = {k for k in keys if icon_like(k)}
    seeds = []
    for frame, found in sorted(by_frame.items()):
        rects = [rect for rect, _ in found]
        picture = decoded.pick(frame).bgr(frame)
        readings = list(ocr.read(picture))
        # Small lettering (a badge on a costume) that a reading of the whole frame misses reads in a crop around it.
        for x, y, w, h in rects:
            m = round(RESIDUAL_AROUND*h)
            x0, y0 = max(0, x-m), max(0, y-m)
            crop = np.ascontiguousarray(picture[y0:min(picture.shape[0], y+h+m), x0:min(picture.shape[1], x+w+m)])
            readings += [(key, text, (x0+rx, y0+ry, rw, rh), score, height)
                         for key, text, (rx, ry, rw, rh), score, height in ocr.read(crop)]
        rows = []
        for key, text, rect, score, height in readings:
            rect = tuple(int(v) for v in rect)
            if not (icon_like(key) or max(similar(key, k) for k in keys) >= RESIDUAL_SEED_LIKE):
                continue
            if any(rect_gap(rect, r) <= .5*r[3] for r in rects) and \
                    not repaired_already(layers, frame, rect, lambda: decoded.pick(frame).signal(frame)):
                rows.append(Part(key, rect, lettering(key, rect, height), frame))
        rows = new_parts(rows, [])
        # Where the source reads nothing there (OCR of small lettering comes and goes with the noise), the check's
        # own reading stands for it: it was read outside every repaired region, where the result is the source.
        for rect, key in found:
            if key and not any(rect_gap(p.rect, rect) <= .5*rect[3] for p in rows) and \
                    (icon_like(key) or max(similar(key, k) for k in keys) >= RESIDUAL_SEED_LIKE) and \
                    not repaired_already(layers, frame, rect, lambda: decoded.pick(frame).signal(frame)):
                rows.append(Part(key, rect, lettering(key, rect, rect[3]), frame))
        # Lettering first, so that an emblem read as a letter joins the lettering beside it.
        rows.sort(key=lambda p: icon_like(p.key))
        while rows:
            parts = attach_parts([rows[0]], rows[1:], rows[0].line)
            rows = [p for p in rows if p not in parts]
            if all(icon_like(p.key) for p in parts) and parts[0].key in emblems:
                # The watermark's emblem with lettering too small to read: the detector's boxes beside it.
                parts = emblem_badge(ocr, decoded.pick(frame).bgr(frame), parts[0], frame, layers,
                                     lambda: decoded.pick(frame).signal(frame))
            rect = union_rect([p.rect for p in parts])
            trace('plan.residual_seed', frame=frame, rect=list(rect), keys=[p.key for p in parts],
                  too_large=too_large(rect, video))
            if not too_large(rect, video):
                seeds.append(Scene(frame, shot_bounds(cuts, video.count, frame), parts))
    return follow_logos(video, decoded, seeds, layers, keys, ocr, trace) if seeds else []


def unblend_likeness(unblended, frame, rect, luma):
    """How closely the picture's detail in rect follows the lettering of an overlay unblended there (|correlation| of
    the high-pass pictures over the lettering): what is left of the overlay follows its letters, either way as it was
    removed too little or too much; a badge of its own does not. None where no unblended overlay's lettering is."""
    best = None
    fx, fy, fw, fh = rect
    for u in unblended:
        if not u.present[frame] or u.zone is None:
            continue
        x, y, w, h = u.rect
        x0, y0, x1, y1 = max(fx, x), max(fy, y), min(fx+fw, x+w), min(fy+fh, y+h)
        if x1-x0 < 8 or y1-y0 < 8:
            continue
        inside = (slice(y0-y, y1-y), slice(x0-x, x1-x))
        zone = u.zone[inside]
        if np.count_nonzero(zone) < 64:
            continue
        alpha = highpass(1.-u.variants[0][0][0].astype(np.float32), 1.5)[inside][zone]
        picture = highpass(luma[y:y+h, x:x+w].astype(np.float32), 1.5)[inside][zone]
        alpha, picture = alpha-alpha.mean(), picture-picture.mean()
        norm = math.sqrt(float(np.square(alpha).sum())*float(np.square(picture).sum()))
        if norm > 0:
            likeness = abs(float((alpha*picture).sum()))/norm
            best = likeness if best is None else max(best, likeness)
    return best


def scene_covers(scene, frame, rect):
    """Whether a followed logo shows next to rect in frame."""
    if frame not in scene.track:
        return False
    dx, dy, _ = scene.track[frame]
    x, y, w, h = scene.rect
    return rect_gap(rect, (round(x+dx), round(y+dy), w, h)) <= .5*scene.line


def read_on(video, decoded, scene, direction, scenes, layers, keys, ocr):
    """The logo read again just past the end of its run inside its shot, near where it was last: a logo seeded there,
    or None where its lettering does not read there or another followed logo shows there already. A sign in the scene
    shows for its whole shot, but following loses one that turns or bends far from how it was first read (a banner in
    the wind); its lettering still reads, and where it reads it is."""
    lettering_keys = [p.key for p in scene.parts if not icon_like(p.key)]
    if not lettering_keys:
        return None     # an emblem alone reads as a letter: too little to tell it from another
    end = (min if direction < 0 else max)(scene.track)
    first, stop = scene.shot
    dx, dy, _ = scene.track[end]
    x, y, w, h = scene.rect
    line = scene.line
    reach = round(max(line, .25*max(w, h)))
    rect = (round(x+dx), round(y+dy), w, h)
    for k in range(1, SCENE_REREAD_TRIES+1):
        frame = end+direction*k
        if not first <= frame < stop or any(s is not scene and scene_covers(s, frame, rect) for s in scenes):
            return None
        gx, gy, gw, gh = grow(rect, reach, reach)
        x0, y0, x1, y1 = max(0, gx), max(0, gy), min(video.width, gx+gw), min(video.height, gy+gh)
        if x1-x0 < 16 or y1-y0 < 16:
            return None
        crop = np.ascontiguousarray(decoded.pick(frame).bgr(frame)[y0:y1, x0:x1])
        seen = np.ascontiguousarray(crop[:, ::-1]) if scene.mirrored else crop
        rows = []
        for key, text, (rx, ry, rw, rh), score, height in ocr.read(seen):
            if scene.mirrored:
                rx = seen.shape[1]-rx-rw
            found = (x0+rx, y0+ry, rw, rh)
            if (scene_text(key, keys) or icon_like(key)) and \
                    not repaired_already(layers, frame, found, lambda: decoded.pick(frame).signal(frame)):
                rows.append(Part(key, found, lettering(key, found, height), frame))
        brand = [p for p in rows if not icon_like(p.key) and .5 <= p.line/line <= 2.
                 and max(similar(p.key, q) for q in lettering_keys) >= RESIDUAL_SEED_LIKE]
        if brand:
            return Scene(frame, scene.shot, attach_parts([brand[0]], [p for p in rows if p is not brand[0]],
                                                         brand[0].line), scene.mirrored)
    return None


def follow_logos(video, decoded, seeds, layers, keys, ocr, trace):
    """Follow seeded logos through their shots, a seed shown where a logo already followed joining it; settle their
    parts, emblems and the animations they enter and leave with."""
    scenes = []
    for scene in seeds:
        known = None
        for other in scenes:
            if scene.frame in other.track:
                dx, dy, _ = other.track[scene.frame]
                x, y, w, h = other.rect
                if overlap_share(scene.rect, (round(x+dx), round(y+dy), w, h)) >= .5:
                    known = other
                    break
        if known is not None:
            # Read again in a later sample: what this reading adds joins the logo already followed.
            dx, dy, _ = known.track[scene.frame]
            rows = [Part(p.key, (int(p.rect[0]-dx), int(p.rect[1]-dy), p.rect[2], p.rect[3]), p.line, p.frame)
                    for p in scene.parts]
            known.parts = attach_parts(known.parts, new_parts(rows, known.parts), known.line)
            trace('plan.scene_joined', frame=scene.frame, to=known.frame)
            continue
        scene.track = {scene.frame: (0., 0., 1.)}
        for direction in (-1, 1):
            scene.track.update(follow_scene(video, decoded, scene, layers, direction))
        scenes.append(scene)
    # Where following lost a logo before its shot's cut, it is read again past there and followed on from where it
    # reads, as a logo of its own (masked by the boxes of its lettering as read there), until the shot is covered or
    # it no longer reads. Frames the picture shows it in unmasked would also bring it back into the repaired ones
    # next to them, as the repair carries what neighbouring frames show into the hole.
    pending = [(s, d, 0) for s in scenes for d in (-1, 1)]
    while pending:
        scene, direction, depth = pending.pop(0)
        if depth >= SCENE_REREADS:
            continue
        again = read_on(video, decoded, scene, direction, scenes, layers, keys, ocr)
        if again is None:
            continue
        end = (min if direction < 0 else max)(scene.track)
        # The frames tried in between, where it did not read, are taken in where it was read.
        again.track = {f: (0., 0., 0.) for f in range(end+direction, again.frame, direction)}
        again.track[again.frame] = (0., 0., 1.)
        again.track.update(follow_scene(video, decoded, again, layers, direction))
        trace('plan.scene_reread', frame=again.frame, after=scene.label, rect=list(again.rect), keys=again.keys,
              frames=len(again.track))
        scenes.append(again)
        pending.append((again, direction, depth+1))
    kept = []
    for scene in scenes:
        # Parts that do not move with the logo are dropped before looking for more next to them, and what the
        # second look adds is checked the same way.
        scene.parts, _ = moving_parts(decoded, scene, layers)
        look_again(video, decoded, scene, layers, keys, ocr)
        read = [p.key for p in scene.parts]
        scene.parts, shares = moving_parts(decoded, scene, layers)
        filled = fill_hidden(video, decoded, scene, layers)
        trace('plan.scene_logo', label=scene.label, rect=list(scene.rect), frames=len(scene.track),
              shot=list(scene.shot), mirrored=scene.mirrored, parts=list(zip(read, shares)), filled=filled,
              too_large=too_large(scene.rect, video))
        if not too_large(scene.rect, video):
            kept.append(scene)
    for scene in kept:
        scene.emblem, changing = scene_emblem(video, decoded, scene, kept)
        carried = carried_by_picture(video, decoded, scene, layers) if changing is not None and \
            changing >= SCENE_OVERLAY else None
        # Two shots of one set show a sign at one place before different pictures as well; only a badge stays on
        # through a cut.
        crossing = crosses_cut(scene)
        scene.overlay = changing is not None and changing >= SCENE_OVERLAY and not carried and crossing
        scene.animations = []
        trace('plan.scene_emblem', label=scene.label, changing=None if changing is None else round(changing, 3),
              carried=carried, crossing=crossing, overlay=scene.overlay,
              offset=None if scene.emblem is None else list(scene.emblem[:2]),
              pixels=0 if scene.emblem is None else int(scene.emblem[2].sum()))
    grouped = set()
    for scene in kept:
        if id(scene) in grouped:
            continue
        group = [scene]+[s for s in kept if s is not scene and id(s) not in grouped and same_logo(scene, s)]
        grouped.update(id(s) for s in group)
        # Only a badge laid over the picture enters and leaves the same way each time; a sign in the scene comes
        # and goes with the shots around it.
        if len(group) < 2 or not any(s.overlay for s in group):
            continue
        for direction in (-1, 1):
            found = scene_animation(video, decoded, group, layers, direction)
            if found is None:
                continue
            steps, placed = found
            side = 'entrance' if direction < 0 else 'exit'
            for member, frames in zip(group, placed):
                if frames:
                    member.animations.append((side, steps, frames))
            trace('plan.scene_animation', label=scene.label, side=side, showings=len(group),
                  steps=[min(steps), max(steps)], area=max(int(m.sum()) for m in steps.values()),
                  frames=[[min(p), max(p)] for p in placed if p])
    return kept


def crosses_cut(scene):
    """Whether a followed logo shows on both sides of a cut: a badge laid over the picture stays through it, a sign
    in the scene leaves with its shot."""
    first, stop = scene.shot
    return any(not first <= f < stop for f, (_, _, score) in scene.track.items() if score >= SCENE_MATCH)


def carried_by_picture(video, decoded, scene, layers):
    """Whether the picture around a logo moves along with it: a sign in a shot the camera travels through, which the
    picture changing around it would otherwise pass for a badge laid over the picture. Over pairs of frames a moment
    apart in which the logo moves, the picture in a ring around it (the logo itself left out) is found again where
    the logo went, or where it was; None where it never moves far enough, or the ring is never found again."""
    line = scene.line
    frames = sorted(f for f, (_, _, score) in scene.track.items() if score >= SCENE_MATCH)
    gap = max(2, round(SCENE_CARRY_GAP*float(video.fps)))
    pairs = []
    for a in frames:
        later = [b for b in frames if a+gap <= b <= a+2*gap]
        if later:
            pairs.append((a, later[0]))
    if not pairs:
        return None
    pairs = list(dict.fromkeys(pairs[round(k*(len(pairs)-1)/max(1, SCENE_CHECKS-1))]
                               for k in range(min(SCENE_CHECKS, len(pairs)))))
    x, y, w, h = scene.rect
    reach = max(8, round(line))
    along = against = 0
    for a, b in pairs:
        (ax, ay, _), (bx, by, _) = scene.track[a], scene.track[b]
        move = (bx-ax, by-ay)
        distance = math.hypot(*move)
        if distance < SCENE_CARRY_LEAST:
            continue
        placed = (round(x+ax), round(y+ay), w, h)
        box = grow(placed, reach, reach)
        signal = decoded.pick(a).signal(a)
        picture = patch(signal, box).astype(np.float32)
        weights = (~covered_by([l for l in layers if l.mark.kind != 'faint'], a, box)).astype(np.float32)
        outside = np.ones(picture.shape, bool)
        view, _ = clipped_plane(outside, (-box[0], -box[1], signal.shape[1], signal.shape[0]))
        view[:] = False
        weights[outside] = 0
        weights[reach:reach+h, reach:reach+w] = 0
        try:
            ring = Motif(picture, weights, np.ones(picture.shape, np.uint8))
        except ValueError:
            continue
        radius = round(distance)+4
        fx, fy, score = match_near(ring, decoded.pick(b).signal(b), (box[0], box[1]), radius)
        if score < SCENE_MATCH:
            continue
        found = (fx-box[0], fy-box[1])
        if math.hypot(found[0]-move[0], found[1]-move[1]) <= max(2., .3*distance):
            along += 1
        elif math.hypot(*found) <= max(2., .3*distance):
            against += 1
    if along+against == 0:
        return None
    return along/(along+against) >= SCENE_CARRIED


def same_logo(a, b):
    """Two followed logos showing one badge: like lettering in boxes of like size."""
    (_, _, aw, ah), (_, _, bw, bh) = a.rect, b.rect
    return (abs(aw-bw) <= .1*max(aw, bw) and abs(ah-bh) <= .1*max(ah, bh)
            and all(max(similar(k, o) for o in b.keys) >= .6 for k in a.keys if not icon_like(k)))


def scene_emblem(video, decoded, scene, scenes):
    """Pixels beside a logo's lettering that keep one colour wherever it shows while the picture behind changes: the
    emblem of a badge passing over the picture, compared over its own frames and those of its other showings. What
    surrounds a sign in the scene moves with it and keeps its colour as well: the picture around it changes too
    little and nothing is added. Returns ((dx, dy, mask) relative to the logo's rect or None, share of the picture
    around the logo that changes between its frames or None where they are too alike to tell)."""
    line = scene.line
    x, y, w, h = scene.rect
    reach_x, reach_y = round(2.2*line), round(1.2*line)
    shown = []      # (frame, left, top): the lettering's box wholly in the picture
    for other in [scene]+[s for s in scenes if s is not scene and same_logo(scene, s)]:
        for f, (dx, dy, score) in sorted(other.track.items()):
            left, top = round(other.rect[0]+dx), round(other.rect[1]+dy)
            if score >= SCENE_MATCH and left >= 0 and top >= 0 and left+w <= video.width and top+h <= video.height:
                shown.append((f, left, top))
    if len(shown) < 4:
        return None, None
    picks = list(dict.fromkeys(shown[round(k*(len(shown)-1)/(SCENE_EMBLEM_LOOKS-1))]
                               for k in range(SCENE_EMBLEM_LOOKS)))
    # The search around the lettering, cut to what is in the picture in every frame compared.
    lo_x = max(-reach_x, max(-left for _, left, _ in picks))
    lo_y = max(-reach_y, max(-top for _, _, top in picks))
    hi_x = min(w+reach_x, min(video.width-left for _, left, _ in picks))
    hi_y = min(h+reach_y, min(video.height-top for _, _, top in picks))
    seed = np.zeros((hi_y-lo_y, hi_x-lo_x), np.uint8)
    seed[-lo_y:h-lo_y, -lo_x:w-lo_x] = 1
    open_sides = (lo_x == -reach_x, lo_y == -reach_y, hi_x == w+reach_x, hi_y == h+reach_y)
    pictures = (decoded.pick(f).bgr(f)[top+lo_y:top+hi_y, left+lo_x:left+hi_x] for f, left, top in picks)
    mask, _, changing = steady_pixels(pictures, seed, line, open_sides, SCENE_EMBLEM_LEAST, SCENE_OVERLAY)
    if mask is None:
        return None, changing
    mask[seed > 0] = 0
    return ((int(lo_x), int(lo_y), mask) if mask.any() else None), changing


def shared_pixels(stack):
    """Pixels most of the stacked pictures show in one colour."""
    stack = np.stack(stack).astype(np.float32)
    inliers = np.abs(stack-np.median(stack, axis=0)).max(axis=-1) <= SCENE_AGREE
    return inliers.sum(axis=0) >= max(2, math.ceil(.6*len(stack))), np.median(stack, axis=0)


def scene_animation(video, decoded, group, layers, direction):
    """How a logo shown more than once enters (direction -1) or leaves (+1). An app badge pops up and flies off the
    same way every time while the scene behind it does not repeat: beyond the ends of its showings, the pixels
    around it that the showings agree on, where the pictures behind them differ once it has gone, are its
    animation. Returns ({step: mask}, [{frame: (step, x, y)} per showing]) with the masks' places in the frames,
    or None."""
    fps = float(video.fps)
    depth = round(SCENE_ANIMATION*fps)
    patience = max(2, round(ENTRY_PATIENCE*fps))
    line = group[0].line
    _, _, w, h = group[0].rect
    reach_x, reach_y = round(2.2*line), round(1.2*line)
    edges = []      # (edge frame, centre) of each showing
    for scene in group:
        shown = [f for f, (_, _, score) in scene.track.items() if score >= SCENE_MATCH]
        edge = min(shown) if direction < 0 else max(shown)
        dx, dy, _ = scene.track[edge]
        x, y, sw, sh = scene.rect
        edges.append((edge, round(x+dx+sw/2), round(y+dy+sh/2)))
    # The search around the logo, cut to what is in the picture at every showing.
    lo_x = max(-(w//2)-reach_x, max(-cx for _, cx, _ in edges))
    lo_y = max(-(h//2)-reach_y, max(-cy for _, _, cy in edges))
    hi_x = min(w-w//2+reach_x, min(video.width-cx for _, cx, _ in edges))
    hi_y = min(h-h//2+reach_y, min(video.height-cy for _, _, cy in edges))
    if hi_x-lo_x < w//2 or hi_y-lo_y < h//2:
        return None
    crops = []      # per showing: {frames beyond its edge: (crop, pixels under a watermark's layer)}
    for edge, cx, cy in edges:
        own = {}
        region = (cx+lo_x, cy+lo_y, hi_x-lo_x, hi_y-lo_y)
        for j in range(-SCENE_SHIFT, depth+1):
            f = edge+direction*j
            if 0 <= f < video.count:
                decoded.raw(f, direction)
                own[j] = (decoded.bgr(f)[cy+lo_y:cy+hi_y, cx+lo_x:cx+hi_x].astype(np.int16),
                          covered_by(layers, f, region))
        crops.append(own)
    # Only showings seen out to the full reach tell the animation from the picture behind it: two showings of the
    # same footage (an opening teaser repeating a later scene) agree on that picture too, and only where it is seen
    # once the animation has gone does it show as the scene's own. A showing too near the video's ends is left out.
    learned = [r for r, own in enumerate(crops) if depth in own]
    if len(learned) < 2:
        return None
    every = edges
    edges, crops = [edges[r] for r in learned], [crops[r] for r in learned]
    # Where the pictures behind agree in a colour bright enough to pass for the animation, agreeing tells nothing.
    # A watermark there (the badge the animation turns into) hides the picture behind and tells nothing either.
    clean = [own[depth] for own in crops if depth in own]
    differ = np.ones((hi_y-lo_y, hi_x-lo_x), bool)
    if len(clean) >= 2:
        shared, median = shared_pixels([picture for picture, _ in clean])
        hidden = np.any([covered for _, covered in clean], axis=0)
        differ = ~(shared & (median.max(axis=-1) >= SCENE_EMBLEM_LEAST) & ~hidden)

    def agreement(stack):
        # What a watermark's layer masks in every showing is left to it.
        shared, median = shared_pixels([picture for picture, _ in stack])
        hidden = np.all([covered for _, covered in stack], axis=0)
        return shared & (median.max(axis=-1) >= SCENE_EMBLEM_LEAST) & differ & ~hidden

    def total(r, shift):
        return sum(int(agreement([crops[0][k], crops[r][k+shift]]).sum())
                   for k in range(1, 1+2*patience) if k in crops[0] and k+shift in crops[r])
    shifts = [0]+[max(range(-SCENE_SHIFT, SCENE_SHIFT+1), key=lambda s: (total(r, s), -abs(s)))
                  for r in range(1, len(crops))]
    # The animation grows out of the logo and moves on a little from frame to frame: what agrees away from where it
    # was a frame before is the scene behind.
    motion = 2*max(3, round(ENTRY_SPEED*video.width), round(.3*line))+1
    near = np.zeros((hi_y-lo_y, hi_x-lo_x), bool)
    stamp(near, np.ones((h, w), bool), -(w//2)-lo_x, -(h//2)-lo_y)
    if group[0].emblem is not None:
        ex, ey, emblem = group[0].emblem
        stamp(near, emblem > 0, -(w//2)-lo_x+ex, -(h//2)-lo_y+ey)
    near = cv2.dilate(near.astype(np.uint8), ellipse(motion)) > 0
    # Away from where the animation can reach, showings of different footage differ: of those that agree there, one
    # stands for them all at a step.
    outer = ~near

    def repeats(a, b):
        (first, hidden), (second, covered) = a, b
        seen = outer & ~hidden & ~covered & (np.maximum(first.max(axis=-1), second.max(axis=-1)) >= SCENE_EMBLEM_LEAST)
        if seen.sum() < line*line:
            return False
        return (np.abs(first-second).max(axis=-1) <= SCENE_AGREE)[seen].mean() >= SCENE_REPEAT

    def distinct(stack):
        kept = []
        for item in stack:
            if not any(repeats(item, other) for other in kept):
                kept.append(item)
        return kept

    steps, last = {}, 0
    for k in range(1, depth-SCENE_SHIFT+1):
        stack = [own[k+shift] for own, shift in zip(crops, shifts) if k+shift in own]
        if len(stack) < 2:
            break
        stack = distinct(stack)
        if len(stack) < 2:
            if k-last > patience:
                break
            continue
        mask = cv2.morphologyEx(agreement(stack).astype(np.uint8), cv2.MORPH_OPEN, ellipse(3))
        mask = fill_holes(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(.15*line)))
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        mask = np.isin(labels, np.unique(labels[near & (mask > 0)])).astype(np.uint8)
        if mask.sum() >= SCENE_STEP_AREA*line*line:
            steps[k] = mask
            last = k
            near = cv2.dilate(mask, ellipse(motion)) > 0
        elif k-last > patience:
            break
    if not steps:
        return None
    found = sorted(steps)
    for a, b in zip(found, found[1:]):
        for k in range(a+1, b):
            steps[k] = np.maximum(steps[a], steps[b])
    placed = [{} for _ in every]
    for r, (edge, cx, cy), shift in zip(learned, edges, shifts):
        frames = {edge+direction*(k+shift): (k, cx+lo_x, cy+lo_y) for k in steps}
        placed[r] = {f: place for f, place in frames.items() if 0 <= f < video.count}
    return steps, placed


def scene_layer(video, scene, settings):
    """A logo's layer: the boxes of its parts, closed and grown, at the followed places. It carries no picture of
    the logo, so the residual check never looks for it elsewhere."""
    line = scene.line
    label = scene.label
    growth = mask_growth(settings, label)
    pad = round(.25*line)
    margin = growth+pad+2
    box = grow(scene.rect, margin, margin)
    if scene.emblem is not None:
        ex, ey, emblem = scene.emblem
        sx, sy, sw, sh = cv2.boundingRect(emblem)
        placed = (scene.rect[0]+ex+sx, scene.rect[1]+ey+sy, sw, sh)
        box = union_rect([box, grow(placed, growth+2, growth+2)])
    mask = np.zeros((box[3], box[2]), np.uint8)
    for part in scene.parts:
        x, y, w, h = part.rect
        cv2.rectangle(mask, (x-box[0]-pad, y-box[1]-pad), (x-box[0]+w+pad-1, y-box[1]+h+pad-1), 1, cv2.FILLED)
    if scene.emblem is not None:
        left, top = placed[0]-box[0], placed[1]-box[1]
        mask[top:top+sh, left:left+sw] |= emblem[sy:sy+sh, sx:sx+sw]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(.5*line))
    if growth > 0:
        mask = cv2.dilate(mask, ellipse(2*growth+1))
    count = video.count
    positions = np.zeros((count, 2), np.int32)
    scores = np.zeros(count, np.float32)
    present = np.zeros(count, bool)
    for f, (dx, dy, score) in scene.track.items():
        positions[f] = (round(box[0]+dx), round(box[1]+dy))
        scores[f] = score
        present[f] = True
    motif = Motif(np.zeros(mask.shape, np.float32), np.ones(mask.shape, np.float32), mask)
    rect = scene.rect
    mark = Mark('scene', label, scene.keys, line, [(scene.frame, rect)],
                [(p.rect[0]-rect[0], p.rect[1]-rect[1], p.rect[2], p.rect[3]) for p in scene.parts])
    return Layer(mark, motif, (box[0]-rect[0], box[1]-rect[1]), None, 1., positions, scores,
                 np.full(count, -1., np.float32), present, positions.copy())


def scene_layers(video, scene, settings):
    """A logo's layer and those of the animations it enters and leaves with. Consecutive steps share a layer while
    the shared mask stays within ENTRY_SLACK of every step's own."""
    layers = [scene_layer(video, scene, settings)]
    count = video.count
    for side, steps, placed in scene.animations or []:
        groups = []     # [steps, shared mask, smallest step's area]
        for k in sorted(steps):
            area = int(steps[k].sum())
            if groups and k == groups[-1][0][-1]+1:
                union = np.maximum(groups[-1][1], steps[k])
                smallest = min(groups[-1][2], area)
                if np.count_nonzero(union) <= ENTRY_SLACK*smallest:
                    groups[-1][0].append(k)
                    groups[-1][1:] = union, smallest
                    continue
            groups.append([[k], steps[k].copy(), area])
        for group, union, _ in groups:
            label = f'{scene.label} ({side} {group[0]}-{group[-1]})'
            growth = mask_growth(settings, label)
            pad = growth+2
            union = cv2.copyMakeBorder(union, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            if growth > 0:
                union = cv2.dilate(union, ellipse(2*growth+1))
            x, y, w, h = cv2.boundingRect(union)
            mask = np.ascontiguousarray(union[y:y+h, x:x+w])
            positions = np.zeros((count, 2), np.int32)
            scores = np.zeros(count, np.float32)
            present = np.zeros(count, bool)
            for f, (k, rx, ry) in placed.items():
                if k in group:
                    positions[f] = (rx-pad+x, ry-pad+y)
                    scores[f] = 1.
                    present[f] = True
            if not present.any():
                continue
            motif = Motif(np.zeros(mask.shape, np.float32), np.ones(mask.shape, np.float32), mask)
            mark = Mark('scene', label, scene.keys, scene.line, [(scene.frame, scene.rect)], [])
            layers.append(Layer(mark, motif, (0, 0), None, 1., positions, scores, np.full(count, -1., np.float32),
                                present, positions.copy()))
    return layers


REPAIR_AREA_LIMIT = .16     # the renderer's default: it refuses frames whose repaired area exceeds this share
# Marks that OCR read and the scan confirmed may together cover more: a frame with two opaque corner badges, a
# caption box and a moving mark can need a fifth of the picture repainted. Each mark is repaired in its own region,
# so the total does not lower the quality; the ceiling only guards against a runaway mask.
MARK_AREA_LIMIT = .30


def area_exceeded(video, layers, share=REPAIR_AREA_LIMIT):
    """Frames whose union of masks is larger than the renderer accepts (same rule as verify_mask_area)."""
    limit = share*video.width*video.height
    upper = np.zeros(video.count, np.int64)
    for layer in layers:
        upper[layer.present] += int(np.count_nonzero(layer.motif.mask))
    over = []
    canvas = np.zeros((video.height, video.width), np.uint8)
    for i in np.flatnonzero(upper > limit):
        canvas.fill(0)
        for layer in layers:
            if layer.present[i]:
                x, y = (int(v) for v in layer.positions[i])
                view, section = clipped_plane(canvas, (x, y, layer.motif.width, layer.motif.height))
                np.maximum(view, layer.motif.mask[section], out=view)
        if np.count_nonzero(canvas) > limit:
            over.append(int(i))
    return over


def fit_area_budget(video, layers, trace):
    """Give up faint overlays, largest first, until every frame's repair fits the renderer's default limit.

    Only faint overlays give way: repainting a large translucent overlay does more harm than leaving it. The other
    marks are kept whatever they cover together; the recipe raises the renderer's limit to MARK_AREA_LIMIT for them.
    Past that ceiling the renderer refuses the whole attempt, so logos followed in the scene give way there, the
    largest of those in the frames over it first: the residual check then reports what they leave.
    """
    dropped = []
    while over := area_exceeded(video, layers):
        faint = [layer for layer in layers if layer.mark.kind == 'faint']
        if not faint:
            trace('plan.area_over_limit', frames=len(over), layers=[layer.mark.label for layer in layers])
            break
        largest = max(faint, key=lambda layer: np.count_nonzero(layer.motif.mask))
        layers.remove(largest)
        dropped.append(largest.mark.label)
        trace('plan.faint_dropped', label=largest.mark.label, frames_over_limit=len(over),
              mask_pixels=int(np.count_nonzero(largest.motif.mask)))
    while over := area_exceeded(video, layers, MARK_AREA_LIMIT):
        scenes = [layer for layer in layers if layer.mark.kind == 'scene' and layer.present[over].any()]
        if not scenes:
            trace('plan.mark_area_over_limit', frames=len(over), first=over[0])
            break
        largest = max(scenes, key=lambda layer: np.count_nonzero(layer.motif.mask))
        layers.remove(largest)
        trace('plan.scene_dropped', label=largest.mark.label, frames_over_limit=len(over), first=over[0],
              mask_pixels=int(np.count_nonzero(largest.motif.mask)))
    return layers, dropped


# ----------------------------------------------------------------------------
# Burned-in subtitles under a moving watermark
#
# Repainting a moving watermark also repaints any subtitle it passes over, erasing or garbling the letters. Where
# the watermark is translucent the subtitle still shows through it, so once the watermark's opacity and colour
# are known, inverse blending recovers the letters. The opacity is fitted per motif pixel against the repaired
# video (where the watermark moves, the repair mostly reproduces the true background), and only subtitle-like
# pixels are taken back: the inverse blend leaves faint traces along the watermark's edges that the repair does not.

CAPTION_WHITE = 176     # luma of a subtitle's letters
CAPTION_DARK = 80       # luma of the outline next to them
CAPTION_FIT_FRAMES = 300
CAPTION_MIN_FIT = 60
CAPTION_GAP = .035      # letters of one subtitle line lie at most about this far apart, as a share of picture height
CAPTION_SEEN = 40       # visible letter pixels that make a line real
CAPTION_LETTER = 120    # recovered pixels that make a line real on their own (a subtitle wholly under the watermark)
CAPTION_STEEP = .1      # opacity change within a few pixels that marks a watermark's stroke or edge
# A subtitle stays put for a second or more while the watermark moves on: the letters it covers in one frame show
# clear of it in another frame with the same subtitle, and are taken from there as they are.
CAPTION_REACH = 2.5     # seconds before and after a frame searched for its subtitle shown clear of the watermarks
CAPTION_SAME = .85      # share of two frames' visible letters (within a pixel) that must match for the same subtitle
# Letters never seen clear are inverse blended in the frame where the watermark over them is thinnest.
CAPTION_THIN = .35      # most opacity of the watermark over a pixel blended so
CAPTION_THINNER = .1    # by which it must be thinner there than over the frame being restored
# Under a translucent moving watermark the repair guesses at what lies beneath, and over a moving subject guesses
# wrong (a wristwatch wiped away, the edge of a garment pushed out of place), while the inverse blend shows it. The
# blend is taken back only where the two differ by more than its own traces along the watermark's strokes: where
# they agree the repair stays, and the watermark's outline does not show again. Experimental (--structure): each
# frame is decided alone, so an object can come back in one frame and not the next where it passes under the
# watermark's denser strokes, and a poorer fit (a small, heavily compressed mark, one whose domain changes between
# showings) leaves its strokes' shape in what is taken back.
STRUCTURE_ALPHA = .35   # median opacity of a moving watermark up to which the picture under it is taken back
STRUCTURE_OPAQUE = .5   # opacity over a pixel up to which its inverse blend is trusted
STRUCTURE_AGREE = 12    # difference of the repair from the inverse blend up to which the repair is kept
STRUCTURE_LOST = 30     # difference from which the inverse blend is taken whole (between the two, mixed)
STRUCTURE_LEAST = 30    # least pixels of a region where the repair lost the picture
# Median difference over such a region from which it is taken back: a fainter one is mostly shading the repair
# smoothed over, and taking back only its darkest part (the rest lying under the watermark's strokes) shows as a blot.
STRUCTURE_STRONG = 40
STRUCTURE_MOST = .35    # share of a watermark's trusted pixels lost in one frame beyond which the blend is suspect


@dataclass
class Opacity:
    layer: Layer
    alpha: np.ndarray       # per motif pixel
    colour: list            # opacity times colour per plane: luma at motif size, chroma at half size
    small_alpha: np.ndarray
    unreliable: np.ndarray  # never taken for letters: its own light strokes, which no inverse blend quite clears


def full_chroma(plane, shape):
    return cv2.resize(plane, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def glyph_like(luma, u, v):
    """White or yellow, bright pixels with a dark outline close by: how burned-in subtitles look, a word or two often
    picked out in yellow (u, v at luma size)."""
    dark_near = cv2.erode(luma, np.ones((9, 9), np.uint8)) < CAPTION_DARK
    du, dv = u.astype(np.int16)-128, v.astype(np.int16)-128
    neutral = (np.abs(du) < 18) & (np.abs(dv) < 18)
    # Yellow, pale to deep: well short of blue, a little towards red, never as far as orange.
    yellow = (du <= -18) & (dv > -12) & (dv < 50) & (-du > 2*dv)
    return (luma > CAPTION_WHITE) & (neutral | yellow) & dark_near


def stamp(canvas, picture, x, y):
    """OR a motif-sized boolean picture into a canvas at (x, y), clipped."""
    view, section = clipped_plane(canvas, (int(x), int(y), picture.shape[1], picture.shape[0]))
    view |= picture[section]


def covered_by(layers, i, rect, skip=None):
    """Pixels of rect (x, y, w, h) under any layer's mask in frame i, except skip."""
    x, y, w, h = rect
    covered = np.zeros((h, w), bool)
    for layer in layers:
        if layer is not skip and layer.present[i]:
            stamp(covered, layer.motif.mask > 0, layer.positions[i][0]-x, layer.positions[i][1]-y)
    return covered


def letter_rows(planes, layers, i):
    """Subtitle-like pixels per picture row in frame i, away from every watermark."""
    shape = planes[0].shape
    letters = glyph_like(planes[0], full_chroma(planes[1], shape), full_chroma(planes[2], shape))
    return (letters & ~covered_by(layers, i, (0, 0, shape[1], shape[0]))).sum(1)


def caption_band(rows, count, height):
    """Rows (top, bottom) where subtitle letters keep appearing, from letter_rows summed over count frames."""
    smooth = np.convolve(rows, np.ones(9)/9, mode='same')
    if not count or smooth.max() < 5*count:
        return None
    hot = smooth >= .25*smooth.max()
    # Join lines of a two-line subtitle, then keep the band with the most letters.
    hot = fill_gaps(hot, round(.04*height))
    best = max(runs(hot), key=lambda run: smooth[run[0]:run[1]].sum())
    return max(0, best[0]-12), min(height, best[1]+12)


def fit_opacity(layer, sources, repairs):
    """Opacity and colour of a moving watermark from motif-sized YUV patches of source and repair in frames where
    it is wholly in view and clear of other layers. Model per pixel and plane, with a per-frame strength k:
    source = repair + k*(colour - alpha*repair)."""
    mask = layer.motif.mask > 0
    S = np.stack([s[0] for s in sources]).astype(np.float32)
    R = np.stack([r[0] for r in repairs]).astype(np.float32)
    D = S-R
    k = np.ones(len(S), np.float32)
    weight = np.ones_like(S)
    for _ in range(6):
        # Weighted least squares per pixel of D = k*colour - k*alpha*R, then k per frame, then Tukey weights.
        wk = weight*(k*k)[:, None, None]
        s11, s12, s22 = wk.sum(0), -(wk*R).sum(0), (wk*R*R).sum(0)
        wkd = weight*k[:, None, None]*D
        t1, t2 = wkd.sum(0), -(wkd*R).sum(0)
        det = s11*s22-s12*s12
        det = np.where(np.abs(det) < 1e-6, 1e-6, det)
        colour = np.where(mask, (t1*s22-t2*s12)/det, 0).astype(np.float32)
        alpha = np.where(mask, np.clip((s11*t2-s12*t1)/det, 0, .95), 0).astype(np.float32)
        model = colour-alpha*R
        k = np.clip((weight*D*model).sum((1, 2))/np.maximum((weight*model*model).sum((1, 2)), 1e-6), 0, 2)
        k = (k/max(float(np.median(k)), 1e-3)).astype(np.float32)
        residual = D-k[:, None, None]*model
        spread = 1.4826*np.median(np.abs(residual), axis=0)+2
        u = residual/(4.685*spread)
        weight = np.where(np.abs(u) < 1, (1-u*u)**2, 0).astype(np.float32)
    del S, R, D, weight, residual, u, model
    h2, w2 = layer.motif.height//2, layer.motif.width//2
    small_alpha = cv2.resize(alpha, (w2, h2), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(layer.motif.mask, (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
    usable = k > .5
    colours = [colour]
    for p in (1, 2):
        Sc = np.stack([s[p] for s in sources]).astype(np.float32)[usable]
        Rc = np.stack([r[p] for r in repairs]).astype(np.float32)[usable]
        estimate = np.median((Sc-Rc)/k[usable, None, None]+small_alpha*Rc, axis=0)
        colours.append(np.where(small_mask, estimate, 0).astype(np.float32))
    # Not to be taken for letters: the watermark's own light strokes, which an inverse blend never quite clears. Only
    # where its opacity changes steeply (thin strokes, the edges of a shape): through the flat inside of a translucent
    # shape (a badge's disc) the inverse blend shows what lies beneath well.
    lightness = colour/np.maximum(alpha, 1e-3)
    steep = cv2.dilate(alpha, np.ones((5, 5), np.uint8))-cv2.erode(alpha, np.ones((5, 5), np.uint8)) > CAPTION_STEEP
    own_white = cv2.dilate(((alpha > .2) & (lightness > 150) & steep & mask).astype(np.uint8), ellipse(5)) > 0
    return Opacity(layer, alpha, colours, small_alpha, own_white), {
        'frames': int(len(sources)), 'alpha_median': round(float(np.median(alpha[mask])), 3),
        'strength_p5_p95': [round(float(v), 3) for v in np.percentile(k, [5, 95])]}


def restore_frame(fit, layers, band, i, source, repair):
    """Subtitle letters under fit's watermark in frame i, taken back from the inverse blend of the source.
    Returns (x0, y0, [Y, U, V] uint8 crops to paste over the repair, restored pixels) or None."""
    layer = fit.layer
    if not layer.present[i]:
        return None
    x, y = (int(v) for v in layer.positions[i])
    w, h = layer.motif.width, layer.motif.height
    H, W = source[0].shape
    if y+h <= band[0] or y >= band[1]:
        return None
    x0, y0 = max(0, x)//2*2, max(0, y, band[0])//2*2
    x1, y1 = min(W, -(-(x+w)//2)*2), min(H, -(-min(y+h, band[1])//2)*2)
    if x1-x0 < 16 or y1-y0 < 16:
        return None
    shape = (y1-y0, x1-x0)
    place = lambda picture, dx, dy, dtype=np.float32: _placed(picture, shape, dx, dy, dtype)
    mask = place(layer.motif.mask > 0, x-x0, y-y0, bool)
    alpha, colour = place(fit.alpha, x-x0, y-y0), place(fit.colour[0], x-x0, y-y0)
    unreliable = place(fit.unreliable, x-x0, y-y0, bool)
    others = covered_by(layers, i, (x0, y0, shape[1], shape[0]), skip=layer)
    s0 = source[0][y0:y1, x0:x1].astype(np.float32)
    r0 = repair[0][y0:y1, x0:x1].astype(np.float32)
    chroma = [p[y0//2:y1//2, x0//2:x1//2] for p in source[1:]]
    letters = glyph_like(source[0][y0:y1, x0:x1], *(full_chroma(c, shape) for c in chroma))
    # The frame's strength from pixels that are neither letters nor another watermark.
    d, model = s0-r0, colour-alpha*r0
    good = mask & (np.abs(d) < 40) & ~(cv2.dilate(letters.astype(np.uint8), ellipse(7)) > 0) & ~others
    if good.sum() < 500:
        return None
    k = float(np.clip((d*model)[good].sum()/max(float((model*model)[good].sum()), 1e-6), .5, 1.3))
    unblended = [np.clip((s0-k*colour)/np.maximum(1-k*alpha, .05), 0, 255)]
    cx, cy = x//2-x0//2, y//2-y0//2
    small = (shape[0]//2, shape[1]//2)
    small_alpha = _placed(fit.small_alpha, small, cx, cy, np.float32)
    for p, c in ((1, chroma[0]), (2, chroma[1])):
        small_colour = _placed(fit.colour[p], small, cx, cy, np.float32)
        unblended.append(np.clip((c.astype(np.float32)-k*small_colour)/np.maximum(1-k*small_alpha, .05), 0, 255))
    u8 = [np.rint(p).astype(np.uint8) for p in unblended]
    region = mask & ~others & (k*alpha <= .7)
    seeds = (glyph_like(u8[0], full_chroma(u8[1], shape), full_chroma(u8[2], shape)) & region & ~unreliable
             & (np.abs(unblended[0]-r0) > 30))
    seeds = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds)
    small_parts = [c for c in range(1, count) if stats[c, cv2.CC_STAT_AREA] < 25]
    if small_parts:
        seeds[np.isin(labels, small_parts)] = 0
    if not seeds.any():
        return None
    seeds &= in_caption_line(seeds, x0, y0, layers, band, i, source).astype(np.uint8)
    if not seeds.any():
        return None
    # Each letter with its outline.
    restored = (cv2.dilate(seeds, ellipse(7)) > 0) & region
    blend = cv2.GaussianBlur(restored.astype(np.float32), (0, 0), .8)*(mask & ~others)
    small_blend = cv2.resize(blend, (small[1], small[0]), interpolation=cv2.INTER_AREA)
    planes = [r0*(1-blend)+unblended[0]*blend]
    for p in (1, 2):
        base = repair[p][y0//2:y1//2, x0//2:x1//2].astype(np.float32)
        planes.append(base*(1-small_blend)+unblended[p]*small_blend)
    return x0, y0, [np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in planes], int(restored.sum())


def subpixel_fit(alpha, colour, s0, k, edges):
    """The watermark moves by fractions of a pixel: of its opacity and colour shifted by up to half a pixel, those
    whose inverse blend leaves the least relief along its strokes (edges). Returns (alpha, colour, (dx, dy))."""
    shape = (alpha.shape[1], alpha.shape[0])

    def relief(dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(alpha, M, shape, flags=cv2.INTER_LINEAR)
        c = cv2.warpAffine(colour, M, shape, flags=cv2.INTER_LINEAR)
        u = (s0-k*c)/np.maximum(1-k*a, .05)
        g = np.abs(cv2.Sobel(u, cv2.CV_32F, 1, 0))+np.abs(cv2.Sobel(u, cv2.CV_32F, 0, 1))
        return float(g[edges].sum()), a, c

    best = (relief(0., 0.)[0], 0., 0.)
    for step in (.5, .25):
        cx, cy = best[1], best[2]
        for dy in (-step, 0., step):
            for dx in (-step, 0., step):
                if (dx or dy) and abs(cx+dx) <= .5 and abs(cy+dy) <= .5:
                    score = relief(cx+dx, cy+dy)[0]
                    if score < best[0]:
                        best = (score, cx+dx, cy+dy)
    if not (best[1] or best[2]):
        return alpha, colour, (0., 0.)
    _, a, c = relief(best[1], best[2])
    return a, c, (best[1], best[2])


def restore_structure(fit, layers, i, source, repair):
    """The picture under fit's translucent watermark in frame i, from the inverse blend of the source where the repair
    lost it. Returns (x0, y0, [Y, U, V] uint8 crops to paste over the repair, restored pixels) or None."""
    layer = fit.layer
    if not layer.present[i]:
        return None
    x, y = (int(v) for v in layer.positions[i])
    w, h = layer.motif.width, layer.motif.height
    H, W = source[0].shape
    x0, y0 = max(0, x)//2*2, max(0, y)//2*2
    x1, y1 = min(W, -(-(x+w)//2)*2), min(H, -(-(y+h)//2)*2)
    if x1-x0 < 16 or y1-y0 < 16:
        return None
    shape = (y1-y0, x1-x0)
    place = lambda picture, dtype=np.float32: _placed(picture, shape, x-x0, y-y0, dtype)
    mask = place(layer.motif.mask > 0, bool)
    alpha, colour = place(fit.alpha), place(fit.colour[0])
    unreliable = place(fit.unreliable, bool)
    usable = mask & ~covered_by(layers, i, (x0, y0, shape[1], shape[0]), skip=layer)
    s0 = source[0][y0:y1, x0:x1].astype(np.float32)
    r0 = repair[0][y0:y1, x0:x1].astype(np.float32)
    d, model = s0-r0, colour-alpha*r0
    good = usable & (np.abs(d) < 40)
    if good.sum() < 500:
        return None
    k = float(np.clip((d*model)[good].sum()/max(float((model*model)[good].sum()), 1e-6), .5, 1.3))
    box = np.ones((5, 5), np.uint8)
    steep = cv2.dilate(alpha, box)-cv2.erode(alpha, box) > CAPTION_STEEP
    alpha, colour, (dx, dy) = subpixel_fit(alpha, colour, s0, k, steep & usable)
    steep = cv2.dilate(alpha, box)-cv2.erode(alpha, box) > CAPTION_STEEP
    unblended = np.clip((s0-k*colour)/np.maximum(1-k*alpha, .05), 0, 255)
    trusted = usable & (k*alpha <= STRUCTURE_OPAQUE) & ~unreliable
    # Where the repair lost the picture: away from the watermark's strokes, whose blend leaves traces of its own.
    lost = (trusted & ~steep & (np.abs(unblended-r0) > .8*STRUCTURE_LOST)).astype(np.uint8)
    lost = cv2.morphologyEx(lost, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lost)
    small = [c for c in range(1, count) if stats[c, cv2.CC_STAT_AREA] < STRUCTURE_LEAST]
    if small:
        lost[np.isin(labels, small)] = 0
    if not lost.any() or lost.sum() > STRUCTURE_MOST*max(1, int((trusted & ~steep).sum())):
        return None
    area = cv2.morphologyEx(lost, cv2.MORPH_CLOSE, ellipse(9))
    area = ((cv2.dilate(area, ellipse(5)) > 0) & usable).astype(np.uint8)
    count, labels = cv2.connectedComponents(area)
    lost_labels = labels[lost > 0]
    gap = np.abs(unblended-r0)[lost > 0]
    strong = [c for c in range(1, count) if (gap[lost_labels == c].size and
                                             np.median(gap[lost_labels == c]) >= STRUCTURE_STRONG)]
    area = np.isin(labels, strong)
    if not area.any():
        return None
    fill = (area & ~trusted).astype(np.uint8)
    luma = np.rint(unblended).astype(np.uint8)
    if fill.any():
        luma = cv2.inpaint(luma, fill, 3, cv2.INPAINT_TELEA)
    # Taken in as far as the two differ: where they nearly agree, the repair is kept.
    differ = cv2.GaussianBlur(np.abs(luma.astype(np.float32)-r0), (0, 0), 1.)
    weight = np.clip((differ-STRUCTURE_AGREE)/(STRUCTURE_LOST-STRUCTURE_AGREE), 0, 1)
    blend = cv2.GaussianBlur(area.astype(np.float32), (0, 0), 1.5)*usable
    blend = cv2.GaussianBlur(blend*weight, (0, 0), 1.)*usable
    planes = [r0*(1-blend)+luma.astype(np.float32)*blend]
    small_shape = (shape[0]//2, shape[1]//2)
    cx, cy = x//2-x0//2, y//2-y0//2
    M = np.float32([[1, 0, dx/2], [0, 1, dy/2]])
    shift = lambda picture: cv2.warpAffine(picture, M, (small_shape[1], small_shape[0]), flags=cv2.INTER_LINEAR)
    small_alpha = shift(_placed(fit.small_alpha, small_shape, cx, cy, np.float32))
    small_fill = cv2.resize(fill, (small_shape[1], small_shape[0]), interpolation=cv2.INTER_NEAREST)
    small_fill = cv2.dilate(small_fill, np.ones((2, 2), np.uint8))
    small_blend = cv2.resize(blend, (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA)
    for p in (1, 2):
        c = source[p][y0//2:y1//2, x0//2:x1//2].astype(np.float32)
        small_colour = shift(_placed(fit.colour[p], small_shape, cx, cy, np.float32))
        chroma = np.rint(np.clip((c-k*small_colour)/np.maximum(1-k*small_alpha, .05), 0, 255)).astype(np.uint8)
        if small_fill.any():
            chroma = cv2.inpaint(chroma, small_fill, 2, cv2.INPAINT_TELEA)
        base = repair[p][y0//2:y1//2, x0//2:x1//2].astype(np.float32)
        planes.append(base*(1-small_blend)+chroma.astype(np.float32)*small_blend)
    return x0, y0, [np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in planes], int((blend > .5).sum())


def in_caption_line(seeds, x0, y0, layers, band, i, source):
    """Which seed pixels (a crop at x0, y0) belong to a line of subtitle letters: one that also shows letters outside
    every watermark in this frame, or that holds a whole letter's worth of seeds by itself. A lone bright speck in
    the picture (a highlight on metal, say) belongs to neither."""
    H, W = source[0].shape
    b0, b1 = band[0]//2*2, min(H, -(-band[1]//2)*2)
    luma = source[0][b0:b1]
    seen = glyph_like(luma, full_chroma(source[1][b0//2:b1//2], luma.shape),
                      full_chroma(source[2][b0//2:b1//2], luma.shape))
    seen &= ~covered_by(layers, i, (0, b0, W, b1-b0))
    found = np.zeros_like(seen)
    found[y0-b0:y0-b0+seeds.shape[0], x0:x0+seeds.shape[1]] = seeds > 0
    # Letters of one line are at most about one letter apart (a space between phrases).
    reach = max(8, round(CAPTION_GAP*H))
    lines = cv2.dilate((seen | found).astype(np.uint8), np.ones((9, 2*reach+1), np.uint8))
    count, labels = cv2.connectedComponents(lines)
    backed = ((np.bincount(labels[seen], minlength=count) >= CAPTION_SEEN)
              | (np.bincount(labels[found], minlength=count) >= CAPTION_LETTER))
    backed[0] = False
    return backed[labels[y0-b0:y0-b0+seeds.shape[0], x0:x0+seeds.shape[1]]]


def _placed(picture, shape, x, y, dtype):
    canvas = np.zeros(shape, dtype)
    view, section = clipped_plane(canvas, (int(x), int(y), picture.shape[1], picture.shape[0]))
    view[:] = picture[section]
    return canvas


@dataclass
class CaptionFrame:
    """One frame's subtitle band as CaptionWindow keeps it (chroma at half size)."""
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    letters: np.ndarray     # subtitle-like pixels
    near: np.ndarray        # within a pixel of them
    covered: np.ndarray     # under any layer, dilated a little: a pixel counts as clear only well away from an edge
    need: np.ndarray        # under a layer that moves and none that stays
    # Through a fitted watermark: its opacity over each pixel (1 where another layer leaves it unknown), the inverse
    # blend, its subtitle-like pixels, and the watermark's own steep strokes, where the blend is not to be trusted.
    # None where no fitted watermark crosses the band.
    opacity: np.ndarray = None
    small_opacity: np.ndarray = None
    unblended: list = None
    unblended_letters: np.ndarray = None
    unsure: np.ndarray = None
    small_unsure: np.ndarray = None


def pooled(picture, reduce):
    """A picture of even size halved, each output pixel the reduction (np.max) of its 2x2 block."""
    h, w = picture.shape
    return reduce(picture.reshape(h//2, 2, w//2, 2), axis=(1, 3))


class CaptionWindow:
    """The source's subtitle band in the frames around the one being restored, read ahead by its own decoder.

    Letters of the subtitle under a moving watermark in one frame are taken from other frames showing the same
    subtitle: as they are where the watermark has moved off them, else inverse blended where the watermark over
    them is thinnest (between its strokes, through a translucent shape)."""

    def __init__(self, video, layers, band, cuda, fits=()):
        self.video, self.layers, self.fits = video, layers, list(fits)
        self.top, self.bottom = band[0]//2*2, min(video.height, -(-band[1]//2)*2)
        self.reach = max(1, round(CAPTION_REACH*float(video.fps)))
        self.frames = read_frames(video, cuda=cuda)
        self.cache, self.last = {}, -1
        self.margin = ellipse(5)
        fitted = {id(fit.layer) for fit in self.fits}
        self.unfitted = [layer for layer in layers if id(layer) not in fitted]
        self.small_masks = {id(fit): cv2.resize(fit.layer.motif.mask, (fit.layer.motif.width//2,
                                                                       fit.layer.motif.height//2),
                                                interpolation=cv2.INTER_NEAREST) > 0 for fit in self.fits}

    def through(self, j, y, u, v, rect):
        """Frame j's band seen through its fitted watermarks: (opacity, small opacity, [Y, U, V] inverse blend, its
        subtitle-like pixels, unsure, small unsure), or None where no fitted watermark crosses the band. The blend
        takes each watermark at its usual strength: kept to where it is thin, that errs little."""
        crossing = [fit for fit in self.fits if fit.layer.present[j] and
                    fit.layer.positions[j][1] < self.bottom and fit.layer.positions[j][1]+fit.layer.motif.height > self.top]
        if not crossing:
            return None
        opacity = np.zeros(y.shape, np.float32)
        unsure = np.zeros(y.shape, bool)
        planes = [y.astype(np.float32), u.astype(np.float32), v.astype(np.float32)]
        small_opacity = np.zeros(u.shape, np.float32)
        for fit in crossing:
            x, top = (int(c) for c in fit.layer.positions[j])
            dy = top-self.top
            mask = _placed(fit.layer.motif.mask > 0, y.shape, x, dy, bool)
            alpha = _placed(fit.alpha, y.shape, x, dy, np.float32)*mask
            colour = _placed(fit.colour[0], y.shape, x, dy, np.float32)
            planes[0] = np.where(mask, (planes[0]-colour)/np.maximum(1-alpha, .05), planes[0])
            opacity = np.maximum(opacity, alpha)
            unsure |= _placed(fit.unreliable, y.shape, x, dy, bool) & mask
            cx, cy = x//2, top//2-self.top//2
            small_mask = _placed(self.small_masks[id(fit)], u.shape, cx, cy, bool)
            small_alpha = _placed(fit.small_alpha, u.shape, cx, cy, np.float32)*small_mask
            for p in (1, 2):
                small_colour = _placed(fit.colour[p], u.shape, cx, cy, np.float32)
                planes[p] = np.where(small_mask, (planes[p]-small_colour)/np.maximum(1-small_alpha, .05), planes[p])
            small_opacity = np.maximum(small_opacity, small_alpha)
        # Under a layer with no fit, nothing is known.
        others = covered_by(self.unfitted, j, rect)
        opacity[others] = 1.
        small_opacity = np.maximum(small_opacity, pooled(opacity, np.max))
        unblended = [np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in planes]
        letters = glyph_like(unblended[0], full_chroma(unblended[1], y.shape), full_chroma(unblended[2], y.shape))
        return opacity, small_opacity, unblended, letters, unsure, pooled(unsure, np.max)

    def advance(self, i):
        rect = (0, self.top, self.video.width, self.bottom-self.top)
        while self.last < min(self.video.count-1, i+self.reach):
            try:
                j, raw = next(self.frames)
            except StopIteration:
                break
            planes = self.video.planes(raw)
            y = planes[0][self.top:self.bottom].copy()
            u = planes[1][self.top//2:self.bottom//2].copy()
            v = planes[2][self.top//2:self.bottom//2].copy()
            letters = glyph_like(y, full_chroma(u, y.shape), full_chroma(v, y.shape))
            near = cv2.dilate(letters.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            fixed = covered_by([layer for layer in self.layers if layer.base is not None], j, rect)
            moving = covered_by([layer for layer in self.layers if layer.base is None], j, rect)
            covered = cv2.dilate((fixed | moving).astype(np.uint8), self.margin) > 0
            # Pixels also under a layer that never moves never show clear: nothing to look for there.
            frame = CaptionFrame(y, u, v, letters, near, covered, moving & ~fixed)
            seen = self.through(j, y, u, v, rect)
            if seen is not None:
                (frame.opacity, frame.small_opacity, frame.unblended, frame.unblended_letters, frame.unsure,
                 frame.small_unsure) = seen
            self.cache[j] = frame
            self.last = j
        for j in [j for j in self.cache if j < i-self.reach]:
            del self.cache[j]

    def same_subtitle(self, i, j):
        """Whether frames i and j show the same subtitle, judged on the letters both show clear of every layer (each
        letter pixel of one within a pixel of a letter pixel of the other)."""
        a_frame, b_frame = self.cache[i], self.cache[j]
        common = ~a_frame.covered & ~b_frame.covered
        a, b = a_frame.letters & common, b_frame.letters & common
        total = int(a.sum())+int(b.sum())
        if total < 2*CAPTION_SEEN:
            return False
        return int((a & b_frame.near).sum())+int((b & a_frame.near).sum()) >= CAPTION_SAME*total

    def recover(self, i):
        """Letters of frame i's subtitle under a moving layer, from frames with the same subtitle. Returns (settled,
        letters, small_letters, y, u, v) in band coordinates (chroma at half size) or None: settled pixels were seen
        clear, or through a thinner watermark than frame i's own, and letters among them hold subtitle letters (with
        their outline) taken from there."""
        own = self.cache[i]
        need = own.need
        if not need.any():
            return None
        settled = np.zeros_like(need)
        letters = np.zeros_like(need)
        small_letters = np.zeros(own.u.shape, bool)
        y, u, v = np.zeros_like(own.y), np.zeros_like(own.u), np.zeros_like(own.v)
        open_sides = {-1: True, 1: True}
        same = []
        for distance in range(1, self.reach+1):
            for side in (1, -1):
                j = i+side*distance
                if not open_sides[side] or j not in self.cache:
                    continue
                if not self.same_subtitle(i, j):
                    # The subtitle changed (or cannot be told): farther frames on this side show another one.
                    open_sides[side] = False
                    continue
                same.append(j)
                other = self.cache[j]
                take = need & ~settled & ~other.covered
                if not take.any():
                    continue
                letter = take & (cv2.dilate(other.letters.astype(np.uint8), self.margin) > 0)
                y[letter] = other.y[letter]
                small = cv2.resize(letter.astype(np.uint8), (u.shape[1], u.shape[0]), interpolation=cv2.INTER_AREA) > 0
                small &= ~small_letters
                u[small], v[small] = other.u[small], other.v[small]
                small_letters |= small
                letters |= letter
                settled |= take
            if not any(open_sides.values()):
                break
        self.thinnest(own, same, need & ~settled, settled, letters, small_letters, y, u, v)
        return (settled, letters, small_letters, y, u, v) if settled.any() else None

    def thinnest(self, own, same, remaining, settled, letters, small_letters, y, u, v):
        """Pixels never seen clear, from the frame of the same subtitle where the watermark over them is thinnest
        (by CAPTION_THINNER below frame i's own, which restore_frame blended already), inverse blended there. Luma and
        chroma choose apart, each at its own size. Updates the arrays given in place."""
        if not remaining.any():
            return
        if own.opacity is None:
            best = np.zeros(remaining.shape, np.float32)
            small_best = np.zeros(u.shape, np.float32)
        else:
            best = np.where(own.unsure, 1., own.opacity).astype(np.float32)
            small_best = np.where(own.small_unsure, 1., own.small_opacity).astype(np.float32)
        best[~remaining] = 0.
        small_remaining = pooled(remaining, np.min) & ~small_letters
        small_best[~small_remaining] = 0.
        chosen = np.zeros_like(remaining)
        lettered = np.zeros_like(remaining)
        small_chosen = np.zeros_like(small_remaining)
        small_lettered = np.zeros_like(small_remaining)
        for j in same:
            other = self.cache[j]
            if other.opacity is None:
                continue
            better = (remaining & ~other.unsure & (other.opacity <= CAPTION_THIN)
                      & (other.opacity < best-CAPTION_THINNER))
            if better.any():
                best[better] = other.opacity[better]
                y[better] = other.unblended[0][better]
                near = cv2.dilate(other.unblended_letters.astype(np.uint8), self.margin) > 0
                lettered[better] = near[better]
                chosen |= better
            small_better = (small_remaining & ~other.small_unsure & (other.small_opacity <= CAPTION_THIN)
                            & (other.small_opacity < small_best-CAPTION_THINNER))
            if small_better.any():
                small_best[small_better] = other.small_opacity[small_better]
                u[small_better] = other.unblended[1][small_better]
                v[small_better] = other.unblended[2][small_better]
                near = pooled(cv2.dilate(other.unblended_letters.astype(np.uint8), self.margin) > 0, np.max)
                small_lettered[small_better] = near[small_better]
                small_chosen |= small_better
        settled |= chosen
        letters |= chosen & lettered
        small_letters |= small_chosen & small_lettered


def paste_recovered(planes, before, top, taken):
    """Over the repaired subtitle band of one frame: settled pixels go back to the plain repair (letters an inverse
    blend drew there are not the subtitle's), then the letters taken from clear frames go in, feathered inward."""
    settled, letters, small_letters, y, u, v = taken
    weight = cv2.GaussianBlur(letters.astype(np.float32), (0, 0), .8)*letters
    small_weight = cv2.resize(weight, (u.shape[1], u.shape[0]), interpolation=cv2.INTER_AREA)*small_letters
    small_settled = cv2.resize(settled.astype(np.uint8), (u.shape[1], u.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    for plane, old, value, keep, w, rows in ((planes[0], before[0], y, settled, weight, top),
                                             (planes[1], before[1], u, small_settled, small_weight, top//2),
                                             (planes[2], before[2], v, small_settled, small_weight, top//2)):
        band = plane[rows:rows+value.shape[0]]
        base = np.where(keep, old, band).astype(np.float32)
        band[:] = np.clip(np.rint(base*(1-w)+value.astype(np.float32)*w), 0, 255).astype(np.uint8)


def restore_captions(video, attempt, layers, cuda, cq, trace, captions=True, structure=False):
    """Put back subtitle letters that repainting a moving watermark erased (captions), and the picture under a
    translucent one where the repair lost it (structure, experimental: decided frame by frame, so what it takes back
    can come and go between frames). Writes attempt/restored.mp4 and returns a summary, or returns None when there
    is nothing to restore."""
    moving = [layer for layer in layers
              if layer.base is None and layer.mark.kind not in ('part', 'scene', 'hold') and layer.present.any()]
    if not moving:
        return None
    repaired = Video.open(attempt/'result.mp4')
    if (repaired.width, repaired.height, repaired.count) != (video.width, video.height, video.count):
        raise ValueError('The repaired video does not match its source.')
    step = max(1, video.count//CAPTION_FIT_FRAMES)
    rows, sampled = np.zeros(video.height), 0
    patches = {id(layer): ([], []) for layer in moving}
    for (i, raw), (_, fixed) in zip(read_frames(video, cuda=cuda, step=step),
                                    read_frames(repaired, cuda=cuda, step=step)):
        planes, fixed_planes = video.planes(raw), repaired.planes(fixed)
        rows += letter_rows(planes, layers, i)
        sampled += 1
        for layer in moving:
            x, y = (int(v) for v in layer.positions[i])
            w, h = layer.motif.width, layer.motif.height
            if (not layer.present[i] or x < 0 or y < 0 or x+w > video.width or y+h > video.height
                    or (covered_by(layers, i, (x, y, w, h), skip=layer) & (layer.motif.mask > 0)).any()):
                continue
            for target, picture in zip(patches[id(layer)], (planes, fixed_planes)):
                target.append([picture[0][y:y+h, x:x+w].copy(),
                               picture[1][y//2:y//2+h//2, x//2:x//2+w//2].copy(),
                               picture[2][y//2:y//2+h//2, x//2:x//2+w//2].copy()])
    band = caption_band(rows, sampled, video.height) if captions else None
    if captions and band is None:
        trace('captions.none', reason='no subtitle band')
    if band is None and not structure:
        return None
    fits, summary = [], {'band': None if band is None else list(band), 'layers': []}
    for layer in moving:
        sources, repairs = patches.pop(id(layer))
        if len(sources) < CAPTION_MIN_FIT:
            trace('captions.layer_skipped', label=layer.mark.label, reason='too few clear frames', frames=len(sources))
            continue
        fit, details = fit_opacity(layer, sources, repairs)
        del sources, repairs
        mask = layer.motif.mask > 0
        if float(np.mean(fit.alpha[mask] <= .7)) < .5:
            trace('captions.layer_skipped', label=layer.mark.label, reason='mostly opaque', **details)
            continue
        trace('captions.fit', label=layer.mark.label, **details)
        fits.append(fit)
        summary['layers'].append({'label': layer.mark.label, **details})
    clear = [fit for fit in fits
             if structure and float(np.median(fit.alpha[fit.layer.motif.mask > 0])) <= STRUCTURE_ALPHA]
    if band is None and not clear:
        return None
    # One streaming pass: restore and encode frame by frame, so memory does not grow with the video's length (the
    # window holds only the subtitle band of a few seconds around the frame).
    output = attempt/'restored.mp4'
    frames = pixels = taken_frames = taken_pixels = structure_frames = structure_pixels = 0
    window = CaptionWindow(video, layers, band, cuda, fits) if band is not None else None

    def paste(crops, x0, y0):
        h, w = crops[0].shape
        fixed_planes[0][y0:y0+h, x0:x0+w] = crops[0]
        fixed_planes[1][y0//2:y0//2+h//2, x0//2:x0//2+w//2] = crops[1]
        fixed_planes[2][y0//2:y0//2+h//2, x0//2:x0//2+w//2] = crops[2]

    with encode(video, output, 0, video.count, cq) as stream:
        for (i, raw), (_, fixed) in zip(read_frames(video, cuda=cuda), read_frames(repaired, cuda=cuda)):
            planes, fixed_planes = video.planes(raw), repaired.planes(fixed)
            # The picture first: subtitle letters are then restored over it.
            rebuilt = [got for fit in clear if (got := restore_structure(fit, layers, i, planes, fixed_planes))]
            for x0, y0, crops, count in rebuilt:
                paste(crops, x0, y0)
                structure_pixels += count
            structure_frames += bool(rebuilt)
            found, taken = [], None
            if window is not None:
                top, bottom = window.top, window.bottom
                window.advance(i)
                before = [fixed_planes[0][top:bottom].copy(), fixed_planes[1][top//2:bottom//2].copy(),
                          fixed_planes[2][top//2:bottom//2].copy()]
                found = [got for fit in fits if (got := restore_frame(fit, layers, band, i, planes, fixed_planes))]
                for x0, y0, crops, count in found:
                    paste(crops, x0, y0)
                    pixels += count
                taken = window.recover(i) if i in window.cache else None
                if taken is not None:
                    paste_recovered(fixed_planes, before, top, taken)
                    taken_pixels += int(taken[1].sum())
                    taken_frames += bool(taken[1].any())
            frames += bool(rebuilt) or bool(found) or (taken is not None and bool(taken[1].any()))
            stream.write(fixed)
    summary.update(frames=frames, pixels=int(pixels), taken_frames=taken_frames, taken_pixels=taken_pixels,
                   structure_frames=structure_frames, structure_pixels=int(structure_pixels))
    trace('captions.found', band=summary['band'], frames=frames, pixels=int(pixels), taken_frames=taken_frames,
          taken_pixels=taken_pixels, structure_frames=structure_frames, structure_pixels=int(structure_pixels))
    if not frames:
        output.unlink(missing_ok=True)
        return None
    from watermark import verify
    verify(video.path, output, attempt/'restored-verification.json')
    summary['result'] = str(output)
    return summary


def render_cq(settings):
    options = settings.get('final_render', ['--cq', '16'])
    return int(options[options.index('--cq')+1]) if '--cq' in options else 16


def run_attempt(video, directory, number, recipe, cuda, trace):
    attempt = directory/f'attempt-{number:02d}'
    plan = build_plan(video.path, attempt, recipe, not cuda)
    attempt.mkdir()
    log = {'source': str(video.path), 'recipe': str(recipe),
           'inputs': fingerprints(input_files(video.path, recipe, attempt, plan)), 'stages': plan}
    with lock_job(attempt):
        execute_plan(attempt, log, 0, trace)
    return attempt


def publish(directory, attempt):
    source = attempt/'restored.mp4' if (attempt/'restored.mp4').is_file() else attempt/'result.mp4'
    target = directory/'result.mp4'
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def run_job(path, directory, settings, cuda=True):
    directory.mkdir(parents=True)
    trace = job_trace(directory/'events.jsonl')
    report = {'source': str(path), 'status': 'running', 'settings': settings, 'attempts': []}
    persist = lambda: write_object(directory/'auto.json', report)
    persist()
    started = perf_counter()
    ocr = OCRWorker()       # started at its first request
    try:
        video = normalized_source(path, directory, trace)
        report['video'] = {'path': str(video.path), 'size': [video.width, video.height], 'fps': str(video.fps),
                           'frames': video.count}
        samples, marks = plan_marks(video, cuda, ocr, settings, trace)
        if not marks:
            report.update(status='no_watermark', reason='No recurring watermark text was found.')
            return report
        unremoved = []      # faint overlays neither unblended nor small enough to repaint
        unblend_tried = False
        unblended = []      # [Unblend] inverted overlays
        remnants = []       # [(Unblend, its faint layer)] for overlays with parts the inverse blend left (repaired)
        traces = []         # unblended overlays that may have left traces
        identity = digest(video.path)
        dilation = settings['dilation']
        threshold_scale = settings['threshold_scale']
        keys = [k for m in marks for k in m.keys]
        scanned = {}        # mark -> its layer as scanned; retries only change masks and thresholds
        scenes = None       # logos set into the scene, followed once (scene_logos)
        grown = {}          # layer label -> pixels added to its mask where text was left inside it
        finished = []       # (residuals, attempt number, directory) per rendered attempt
        for number in range(1, settings['max_attempts']+1):
            current = {**settings, 'dilation': dilation, 'threshold_scale': threshold_scale, 'grown': dict(grown)}
            layers = learn_layers(samples, marks, current, trace)
            if not layers:
                report.update(status='needs_review', reason='No watermark could be learned from the samples.')
                return report
            if not all([reuse_scan(layer, scanned.get(id(layer.mark))) for layer in layers]):
                scan_layers(video, layers, cuda, trace)
            else:
                trace('plan.scan_reused', layers=len(layers))
            scanned.update((id(layer.mark), layer) for layer in layers)
            layers = decide_presence(video, layers, cuda, current, trace)
            if not unblend_tried:
                # Fitted before the scene is searched: the frames a faint overlay shows in are judged by its fitted
                # letters, and a reading under it there is its lettering.
                fitted = fit_unblends(video, samples, layers, cuda, trace)
            if scenes is None:
                # Logos set into the scene are found once, in the source and with every watermark layer known
                # (lettering an overlay about to be unblended repaints is no logo), and followed through their shots.
                scenes = scene_logos(video, samples, layers, keys+samples.scene_keys, ocr, cuda, trace)
            if not unblend_tried:
                # Translucent overlays are inverted once, before anything is repainted; later attempts work on the
                # unblended video.
                unblend_tried = True
                video, unblends = unblend_overlays(video, directory, fitted, cuda, trace)
                if unblends:
                    unblended = unblends
                    done = {u.label for u in unblends}
                    # The parts an inverse blend leaves are repaired too, with the faint layer's picture of them.
                    remnants = [(u, layer) for u in unblends if u.poor is not None
                                for layer in layers if layer.mark.label == u.label]
                    for u, _ in remnants:
                        trace('plan.unblend_remnant', label=u.label, pixels=int(u.poor.sum()),
                              removed=u.check['removed'], removed_rest=u.check.get('removed_rest'),
                              parts=[p for p in u.check.get('parts', []) if p['removed'] < UNBLEND_CLEAN])
                    marks = [m for m in marks if m.label not in done]
                    layers = [layer for layer in layers if layer.mark.label not in done]
                    identity = digest(video.path)
                    traces = [u.label for u in unblends if u.check['removed'] < UNBLEND_CLEAN]
                    report['unblended'] = [{'label': u.label, **u.check} for u in unblends]
                    report['video']['unblended'] = str(video.path)
                    persist()
                    if not layers and not scenes and not remnants:
                        target = directory/'result.mp4'
                        shutil.copyfile(video.path, target)
                        report.update(status='needs_review' if traces else 'complete', result=str(target))
                        if traces:
                            report['reason'] = 'Faint overlays were reduced by unblending; traces may remain: ' + \
                                               ', '.join(traces)
                        return report
            layers = layers+[layer for layer in (remnant_layer(video, u, faint, current) for u, faint in remnants)
                             if layer is not None]
            layers = layers+[layer for scene in scenes for layer in scene_layers(video, scene, current)]
            layers, dropped = fit_area_budget(video, layers, trace)
            # A remnant given up leaves its overlay unblended as it was, judged by all its letters.
            kept = {layer.mark.label for layer in layers}
            left = {remnant_label(u.label) for u in unblended}
            unremoved = sorted(set(unremoved) | (set(dropped)-left))
            traces = [u.label for u in unblended if unblend_removal(u, remnant_label(u.label) in kept) < UNBLEND_CLEAN]
            report['faint_unremoved'] = unremoved
            if not layers:
                report.update(status='needs_review', reason='Only faint overlays too large to repaint were found.')
                return report
            plan_dir = directory/f'plan-{number:02d}'
            recipe = write_recipe(video, plan_dir, layers, samples.cuts, identity, current)
            preview_layers(samples, layers, plan_dir)
            record = {'number': number, 'recipe': str(recipe), 'dilation': dilation, 'grown': dict(grown),
                      'threshold_scale': threshold_scale,
                      'layers': [{'kind': l.mark.kind, 'label': l.mark.label, 'present': int(l.present.sum()),
                                  'mask_pixels': int(l.motif.mask.sum()), 'size': [l.motif.width, l.motif.height]}
                                 for l in layers if l.mark.kind != 'part'],
                      'animation_layers': sum(l.mark.kind == 'part' for l in layers),
                      'animation_frames': int(sum(l.present.sum() for l in layers if l.mark.kind == 'part'))}
            report['attempts'].append(record)
            persist()
            if settings.get('plan_only'):
                report.update(status='planned', reason='Stopped after planning.')
                return report
            try:
                attempt = run_attempt(video, directory, number, recipe, cuda, trace)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                # A retry that cannot be rendered (its wider masks refused, say) leaves the attempts before it.
                if not finished:
                    raise
                record['error'] = f'{type(error).__name__}: {error}'
                trace('attempt.failed', error=str(error))
                break
            record['directory'] = str(attempt)
            if (attempt/'watermark-audit.json').is_file():
                audit = read_object(attempt/'watermark-audit.json')
                record['audit'] = {k: audit.get(k) for k in ('discovery_passed', 'discovery_residual_p95',
                                                             'discovery_residual_max', 'outside_mask_luma_mae')}
            final = attempt/'result.mp4'
            if settings.get('captions', True) or settings.get('structure', False):
                captions = restore_captions(video, attempt, layers, cuda, render_cq(settings), trace,
                                            settings.get('captions', True), settings.get('structure', False))
                if captions:
                    record['captions'] = captions
                    final = Path(captions['result'])
                persist()
            candidate = Video.open(final)
            findings, checked, confirmed = residual_check(video, candidate, keys, layers, cuda, ocr,
                                                          settings['check_samples'], trace)
            record['ocr'] = {'checked': checked, 'source_frames_with_marks': confirmed, 'residuals': findings}
            persist()
            ours = [f for f in findings if f['cause'] != 'foreign']
            if not ours and (unremoved or traces):
                reasons = []
                if unremoved:
                    reasons.append('Faint overlays remain that could neither be unblended nor repainted: '
                                   + ', '.join(unremoved))
                if traces:
                    reasons.append('Faint overlays were reduced by unblending; traces may remain: ' + ', '.join(traces))
                report.update(status='needs_review', result=str(publish(directory, attempt)), reason=' '.join(reasons))
                return report
            if not ours:
                report.update(status='complete', result=str(publish(directory, attempt)))
                if findings:
                    report['notes'] = ['Text resembling the watermark remains outside every learned watermark; '
                                       'it does not match the learned appearance and is treated as scene content.']
                return report
            finished.append((len(ours), number, attempt))
            # Retry: widen the masks that text was left inside, and only those: widening every mask can push the
            # total over the area budget and lose a faint overlay that was removed. Text missed outside every mask
            # is a logo set into the scene that the samples did not read, followed from where it was found, or else
            # frames a mark was missed in, tracked more sensitively.
            for label in sorted({layers[f['layer']].mark.label for f in ours
                                 if f['cause'] == 'inside_mask' and f['layer'] is not None}):
                grown[label] = grown.get(label, 0)+4
            missed = [f for f in ours if f['cause'] == 'missed']
            found = residual_scenes(video, missed, layers, unblended, keys+samples.scene_keys, ocr, samples.cuts,
                                    cuda, trace) if missed else []
            scenes += found
            if any(not any(scene_covers(s, f['frame'], f['rect']) for s in found) for f in missed):
                threshold_scale *= .75
            trace('attempt.retry', residuals=len(ours), grown=grown, threshold_scale=threshold_scale,
                  scenes_added=[s.label for s in found])
        # A retry can do worse than the attempt before it; keep the attempt that left the least text, the earlier
        # one on a tie as it repainted less.
        _, best, attempt = min(finished)
        report.update(status='needs_review', reason='Watermark text remained after all attempts.',
                      result=str(publish(directory, attempt)), published_attempt=best)
        return report
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        trace('job.error', error=str(error))
        print(f'{path.name}: {error}', file=sys.stderr, flush=True)
        return report
    finally:
        ocr.close()
        report['seconds'] = round(perf_counter()-started, 1)
        persist()
        trace('job.finished', status=report['status'])


FINISHED = ('complete', 'no_watermark', 'needs_review', 'failed')


def batch_summary(sources, job):
    jobs = []
    for path in sources:
        report_path = job/path.stem/'auto.json'
        report = read_object(report_path) if report_path.is_file() else {}
        status = report.get('status', 'pending')
        jobs.append({'source': str(path), 'status': status, 'result': report.get('result'),
                     'reason': report.get('reason') or report.get('error'), 'seconds': report.get('seconds')})
    counts = {s: sum(j['status'] == s for j in jobs) for s in sorted({j['status'] for j in jobs})}
    summary = {'total': len(sources), 'counts': counts, 'jobs': jobs}
    # Several workers may finish at once: each writes its own temporary file, then replaces atomically.
    temporary = job/f'.batch.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(job/'batch.json')
    return summary


BATCH_RUNS = 3      # processes a batch video is given: one that dies unfinished is run again in a fresh one


def job_arguments(settings, cuda):
    """The command-line options that give a single-video run these settings."""
    options = ['--samples', str(settings['samples']), '--check-samples', str(settings['check_samples']),
               '--max-attempts', str(settings['max_attempts']), '--dilation', str(settings['dilation'])]
    switches = (('--cpu-decode', not cuda), ('--plan-only', settings['plan_only']), ('--audit', settings['audit']),
                ('--no-captions', not settings['captions']), ('--structure', settings['structure']))
    return options+[flag for flag, on in switches if on]


def run_isolated(path, directory, settings, cuda):
    """One video of a batch in a process of its own. The native libraries' state (the OCR engine's threads and
    buffers) does not carry from one video to the next, and a process that dies (an interpreter brought down by a
    native library, the memory killer) takes only its own video with it: that video runs again in a fresh process,
    up to BATCH_RUNS times, then is recorded as failed and the batch goes on."""
    report_path = directory/'auto.json'
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), str(path), str(directory),
               *job_arguments(settings, cuda)]
    for run in range(1, BATCH_RUNS+1):
        if directory.exists():
            events = directory/'events.jsonl'
            if run > 1 and events.is_file():
                # The run that died leaves its events beside the batch: they tell the step it died in.
                kept = directory.parent/'.crashes'/f'{directory.name}-run{run-1}.jsonl'
                kept.parent.mkdir(exist_ok=True)
                shutil.copyfile(events, kept)
            shutil.rmtree(directory)    # an interrupted (or, when retrying, failed) earlier run of this video
        code = subprocess.run(command).returncode
        report = read_object(report_path) if report_path.is_file() else {}
        if report.get('status') in FINISHED+('planned',):
            return report
        print(f'{path.name}: the job process exited with code {code} unfinished (run {run} of {BATCH_RUNS})',
              file=sys.stderr, flush=True)
    report = {**report, 'source': str(path), 'status': 'failed',
              'error': f'The job process exited with code {code} unfinished in {BATCH_RUNS} runs.'}
    directory.mkdir(parents=True, exist_ok=True)
    write_object(report_path, report)
    return report


def run_batch(sources, job, settings, cuda, retry_failed):
    """Work through a directory of videos. Any number of workers (one per GPU, say) may share one job directory:
    each video is claimed with a lock, finished videos are skipped, and interrupted ones start over."""
    job.mkdir(parents=True, exist_ok=True)
    batch_summary(sources, job)     # marks the directory as a batch at once, so a restart may resume it
    claims = job/'.claims'
    started = time.time()
    for path in sources:
        claim = claims/path.stem
        claim.mkdir(parents=True, exist_ok=True)
        try:
            lock = lock_job(claim)
        except ValueError:
            continue    # another worker has it
        with lock:
            directory = job/path.stem
            report_path = directory/'auto.json'
            status = read_object(report_path).get('status') if report_path.is_file() else None
            # Retry only failures from before this worker started, not one another worker has just retried.
            retry = retry_failed and status == 'failed' and report_path.stat().st_mtime < started
            if status in FINISHED and not retry:
                continue
            report = run_isolated(path, directory, settings, cuda)
        batch_summary(sources, job)
        print(f'{path.name}: {report["status"]}', flush=True)
    summary = batch_summary(sources, job)
    print(json.dumps(summary['counts'], ensure_ascii=False), flush=True)
    return 0 if all(j['status'] in ('complete', 'no_watermark') for j in summary['jobs']) else 2


def main():
    if sys.argv[1:2] == ['--ocr-worker']:
        return serve_ocr(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='An MP4 file or a directory of MP4 files.')
    parser.add_argument('job', type=Path, help='New job directory (for a source directory: new, or one to resume).')
    parser.add_argument('--samples', type=int, default=0, help='Frames sampled for OCR planning (0 = automatic).')
    parser.add_argument('--check-samples', type=int, default=40, help='Frames OCR-checked on the restored video.')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=4, help='Mask dilation in pixels.')
    parser.add_argument('--cpu-decode', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Stop after writing the first plan and previews.')
    parser.add_argument('--retry-failed', action='store_true', help='Batch: run failed videos again.')
    parser.add_argument('--audit', action='store_true', help='Also run the pixel audit on every attempt (slower).')
    parser.add_argument('--no-captions', action='store_true',
                        help='Keep subtitles as repainted where a moving watermark passed over them '
                             '(by default their letters are recovered from under a translucent watermark).')
    parser.add_argument('--structure', action='store_true',
                        help='Experimental: also take back, from the inverse blend, the picture under a translucent '
                             'moving watermark where the repair lost it. Decided frame by frame, so what it takes '
                             'back can come and go between frames.')
    args = parser.parse_args()
    source, job = args.source.resolve(strict=True), args.job.resolve()
    settings = {'samples': args.samples, 'check_samples': args.check_samples, 'max_attempts': args.max_attempts,
                'dilation': args.dilation, 'threshold_scale': 1., 'plan_only': args.plan_only, 'audit': args.audit,
                'captions': not args.no_captions, 'structure': args.structure}
    cuda = not args.cpu_decode
    if source.is_file():
        if job.exists():
            raise ValueError('Choose a new job directory.')
        report = run_job(source, job, settings, cuda)
        print(json.dumps({k: report.get(k) for k in ('status', 'result', 'reason', 'error', 'seconds')},
                         ensure_ascii=False, indent=2))
        return 0 if report['status'] in ('complete', 'no_watermark', 'planned') else 2
    sources = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
    if not sources:
        raise ValueError('The source directory has no MP4 files.')
    if len({p.stem for p in sources}) != len(sources):
        raise ValueError('Two source videos share a name; each needs its own job folder.')
    if job.exists() and not (job/'batch.json').is_file() and any(job.iterdir()):
        raise ValueError('The job directory is neither new, empty nor an earlier batch.')
    return run_batch(sources, job, settings, cuda, args.retry_failed)


if __name__ == '__main__':
    # A crash in native code (a decoder, the OCR runtime) ends the process without a Python traceback; this prints
    # where each thread was, so a batch's log tells which step it died in.
    faulthandler.enable()
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
