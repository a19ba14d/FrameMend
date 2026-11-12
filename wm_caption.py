"""Recover stationary subtitle strokes from unoccluded pixels in nearby frames."""

import cv2
import numpy as np

from wm_matte import logo_rect
from wm_media import plane_rects


CAPTION_RADIUS = 27


class CaptionReferences:
    def __init__(self, video, matte, band, edges, context=False):
        if not 0 <= band < video.height or band % 2:
            raise ValueError('Caption band must start on an even row inside the frame.')
        self.video, self.matte, self.band, self.edges = video, matte, band, edges
        self.signatures = {}
        self.shape = cv2.dilate(edges.reference_mask, np.ones((5, 5), np.uint8))
        self.context = context

    def signature(self, index, frame):
        if index in self.signatures:
            return self.signatures[index]
        y = self.video.planes(frame)[0][self.band:]
        dark = (y < 68).astype(np.uint8)
        bright = (y > 188) & (cv2.dilate(dark, np.ones((13, 13), np.uint8)) > 0)
        # Compare subtitle shapes only where the model trusts the inverse matte.
        band_rect = (0, self.band, self.video.width, self.video.height-self.band)
        safe = self.edges.mask(index, band_rect, self.shape) == 0
        signature = bright, safe
        self.signatures[index] = signature
        return signature

    def clean_old(self, before):
        for i in tuple(self.signatures):
            if i < before:
                del self.signatures[i]

    def caption_rect(self, index):
        rect = logo_rect(self.matte.profile, self.matte.track, index)
        if rect is None or rect[1] + rect[3] <= self.band:
            return None
        x, y, w, h = rect
        top = max(self.band, y)
        return x, top, w, y+h-top

    @staticmethod
    def candidate_frames(index, shot_start, shot_stop):
        offsets = (-27, -20, -14, -9, -5, -2, 2, 5, 9, 14, 20, 27)
        return tuple(dict.fromkeys(min(max(shot_start, index+d), shot_stop-1) for d in offsets))

    def collect(self, index, neighbors, rect, indices, planes=False):
        x, top, w, h = rect
        target_bright, target_safe = self.signature(index, neighbors[index])
        donors, validity = [], []
        for j in indices:
            if j == index or j not in neighbors:
                continue
            other = neighbors[j]
            bright, safe = self.signature(j, other)
            common = safe & target_safe
            intersection = np.count_nonzero(bright & target_bright & common)
            union = np.count_nonzero((bright | target_bright) & common)
            if intersection < 360 or intersection < 0.76 * union:
                continue
            parts = plane_rects(self.video.planes(other), rect)
            donors.append(parts if planes else parts[0])
            validity.append(safe[top-self.band:top+h-self.band, x:x+w])
        return donors, validity

    def apply(self, index, frame, neighbors, shot_start, shot_stop):
        rect = self.caption_rect(index)
        if rect is None:
            return
        indices = self.candidate_frames(index, shot_start, shot_stop)
        donors, validity = self.collect(index, neighbors, rect, indices, planes=self.context)
        if len(donors) < 2:
            return
        output = plane_rects(self.video.planes(frame), rect)
        current = output[0]
        shape = self.edges.mask(index, rect, self.edges.repair_shapes[0]).astype(bool)
        luma = [parts[0] for parts in donors] if self.context else donors
        model = self.consensus(luma, validity, details=True)
        self.restore_luma(current, luma, validity, shape, model)
        if self.context:
            reference = plane_rects(self.video.planes(neighbors[index]), rect)[0]
            self.restore_context(current, reference, luma, validity, shape, model)
            self.restore_chroma(output, donors, validity, shape, model)

    @staticmethod
    def consensus(donors, validity, details=False):
        values = np.stack(donors).astype(np.int16)
        valid = np.stack(validity)
        # Pick the largest group of references agreeing within a few code values.
        count = np.sum((np.abs(values[:, None] - values[None, :]) <= 3) & valid[None, :], axis=1)
        count *= valid
        best = count.argmax(axis=0)[None]
        estimate = np.take_along_axis(values, best, axis=0)[0]
        votes = np.take_along_axis(count, best, axis=0)[0]
        consensus = (votes >= 2) & (votes * 2 >= valid.sum(axis=0))
        cluster = valid & (np.abs(values - estimate) <= 3)
        estimate = np.rint(np.sum(values * cluster, axis=0) / np.maximum(votes, 1)).astype(np.int16)
        bright = (estimate > 188) & consensus
        dark = (estimate < 68) & consensus
        kernel = np.ones((9, 9), np.uint8)
        outline = (bright & (cv2.dilate(dark.astype(np.uint8), kernel) > 0)) | (
            dark & (cv2.dilate(bright.astype(np.uint8), kernel) > 0))
        return (estimate, outline, votes) if details else (estimate, outline)

    @classmethod
    def restore_luma(cls, current, donors, validity, shape, model=None):
        estimate, outline = cls.consensus(donors, validity) if model is None else model[:2]
        delta = estimate - current
        use = shape & outline & (np.abs(delta) <= 42)
        current[use] = estimate[use]

    @classmethod
    def restore_context(cls, current, reference, donors, validity, shape, model):
        if len(donors) < 3:
            return
        stroke_estimate, strokes, stroke_votes = model
        nearby = cv2.dilate(strokes.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        candidate = shape & nearby & ~strokes
        if not candidate.any():
            return
        # Match actual background samples at the same screen coordinates.
        # Exclude the stationary high-contrast lettering from this match.
        stroke_border = cv2.dilate(strokes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        near_stroke = candidate & stroke_border & (stroke_votes >= 3)
        near_stroke &= (stroke_estimate > 68) & (stroke_estimate < 188) & (np.abs(stroke_estimate-current) <= 25)
        current[near_stroke] = stroke_estimate[near_stroke]
        known = ~shape & ~stroke_border & (reference > 66) & (reference < 191)
        source = reference.astype(np.float32)
        reliable = []
        for donor, valid in zip(donors, validity, strict=True):
            safe = (known & valid).astype(np.float32)
            mass = cv2.boxFilter(safe, -1, (23, 23))
            error = cv2.boxFilter(np.abs(donor-source)*safe, -1, (23, 23)) / np.maximum(mass, 0.06)
            reliable.append(valid & (mass > 0.08) & (error < 5))
        estimate, _, votes = cls.consensus(donors, reliable, details=True)
        delta = estimate - current
        use = candidate & (votes >= 3) & (np.abs(delta) <= 21)
        current[use] = estimate[use]

    @staticmethod
    def chroma_reference(donors, validity, stroke):
        width, height = donors[0][1].shape[::-1]
        def area(mask):
            return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        target = area(stroke) >= 0.75
        if not target.any():
            return target, None, None
        valid = np.stack([area(v)[target] >= 0.99 for v in validity])
        values = np.stack([np.stack((u[target], v[target]), axis=1) for _, u, v in donors]).astype(np.int16)
        valid &= (np.abs(values-128) <= 10).all(axis=2)
        count = ((np.abs(values[:, None]-values[None, :]) <= 3).all(axis=3) & valid[None]).sum(axis=1) * valid
        best = count.argmax(axis=0)
        columns = np.arange(values.shape[1])
        picked = values[best, columns]
        good = count[best, columns] >= 3
        return target, picked, good

    @classmethod
    def restore_chroma(cls, output, donors, validity, shape, model):
        estimate, strokes, votes = model
        stroke = shape & strokes & (votes >= 3) & (np.abs(estimate-output[0]) <= 3)
        target, picked, good = cls.chroma_reference(donors, validity, stroke)
        if picked is None:
            return
        for ch, plane in enumerate(output[1:]):
            previous = plane[target]
            keep = good & (np.abs(previous-picked[:, ch]) <= 15)
            plane[target] = np.where(keep, picked[:, ch], previous)


def caption_plan(source, base, profile_path, workdir, band):
    from pathlib import Path

    from wm_media import Video
    from wm_splice import video_packets
    from wm_track import load_profile, load_track

    video, base = Video.open(source), Video.open(base)
    if (video.width, video.height, video.count, video.fps) != (base.width, base.height, base.count, base.fps):
        raise ValueError('The base must match the source frame grid.')
    if not 0 <= band < video.height or band % 2:
        raise ValueError('Caption band must start on an even row inside the frame.')
    profile = load_profile(video, profile_path)
    track = load_track(video, profile_path, Path(workdir) / 'track.npz')
    if not profile.get('translucent'):
        raise ValueError('Caption planning requires a translucent watermark track.')
    positions = track['positions'].astype(np.int32)
    h = profile['translucent']['reference_rect'][3]
    overlap = (positions[:, 0] >= 0) & (positions[:, 1] + h > band)
    stream, packets = video_packets(base.path)
    pts = sorted(int(p['pts']) for p in packets)
    if len(pts) != video.count or stream.get('codec_name') != 'h264' or stream.get('time_base') != '1/90000':
        raise ValueError('Caption planning expects an aligned H.264 MP4 with a 90 kHz timeline.')
    keyframes = [i for i, packet in enumerate(packets) if 'K' in packet.get('flags', '')
                 and int(packet['pts']) == pts[i]]
    if not keyframes or keyframes[0] != 0:
        raise ValueError('The base must start on a keyframe.')
    ranges = []
    boundaries = keyframes + [base.count]
    for lo, hi in zip(boundaries, boundaries[1:]):
        if overlap[lo:hi].any():
            if ranges and ranges[-1][1] == lo:
                ranges[-1][1] = hi
            else:
                ranges.append([lo, hi])
    return dict(caption_band=band, tracked_overlapping_frames=int(overlap.sum()),
                keyframe_intervals=ranges, interval_frames=sum(b-a for a, b in ranges))
