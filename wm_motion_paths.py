"""Recover paired, straight screen-space watermark paths from source samples."""

import itertools

import cv2
import numpy as np

from wm_media import get_frame, planes, read_frames
from wm_track import crop, highpass, template_peak


def rough_tracks(video, rectangles, luma, cuda):
    size = (video.width//4//2*2, video.height//4//2*2)
    scale = np.asarray((video.width/size[0], video.height/size[1]))
    templates = [highpass(cv2.resize(crop(luma, rect), tuple(np.rint(rect[2:]/scale).astype(int)),
                                    interpolation=cv2.INTER_AREA), 1.35) for rect in rectangles]
    time, images, scores = [], [], []
    for i, frame in read_frames(video, cuda=cuda, size=size, step=4):
        y = planes(frame, *size)[0]
        normalized = highpass(y, 1.35)
        found = [template_peak(normalized, template) for template in templates]
        time.append(i)
        images.append(y.copy())
        scores.append([[x*scale[0], y*scale[1], strength] for x, y, strength in found])
    return np.asarray(time), np.asarray(scores).transpose(1, 0, 2), np.stack(images)


def find_segments(time, scores, mirror):
    agree = ((scores[:, :, 2] > .34).all(0) &
             (np.linalg.norm(scores[0, :, :2]+scores[1, :, :2]-mirror, axis=1) < 10))
    centers = (scores[0, :, :2] - scores[1, :, :2] + mirror)/2
    dt = np.diff(time[agree])
    velocity = np.abs(np.diff(centers[agree], axis=0)/dt[:, None])
    stable = (dt <= 12) & (velocity[:, 0] > 2) & (velocity[:, 0] < 9) & (velocity[:, 1] > 1) & (velocity[:, 1] < 6)
    if stable.sum() < 24:
        raise ValueError('The two overlay tracks do not agree on a continuous mirrored path.')
    speed = np.median(velocity[stable], axis=0)
    valid = (scores[:, :, 2] > .36) | agree[None]
    frame = np.tile(time, 2)[valid.ravel()]
    position = np.concatenate((scores[0, :, :2], mirror-scores[1, :, :2]))[valid.ravel()]
    segments = []
    for direction in itertools.product((-1, 1), repeat=2):
        step = speed*direction
        offsets = position-frame[:, None]*step
        distance = np.abs(offsets[:, None]-offsets[None, :]).max(axis=2)
        close = distance <= 7
        active = np.ones(len(frame), dtype=bool)
        while active.any():
            votes = np.sum(close & active[None], axis=1)
            best = int(np.argmax(np.where(active, votes, 0)))
            if votes[best] < 24:
                break
            group = close[best] & active
            active[group] = False
            matched = np.unique(frame[group])
            if len(matched) < 18:
                continue
            center = int(matched[len(matched)//2])
            slope, intercept = np.polyfit(frame[group]-center, position[group], 1)
            if (np.linalg.norm(slope-step) < .3 and matched[-1]-matched[0] <= 260
                    and np.max(np.diff(matched), initial=0) <= 40):
                segments.append((center, matched, slope, intercept))
    if not segments:
        raise ValueError('No repeatable dual paths could be fitted.')
    return sorted(segments, key=lambda row: row[0])


def refine_segment(video, segment, rects, templates, mirror, cuda):
    center, frames, step, start = segment
    observations = []
    for frame in frames[np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int)]:
        y = video.planes(get_frame(video, int(frame), cuda))[0]
        predicted = start+(frame-center)*step
        for j, position in enumerate((predicted, mirror-predicted)):
            x, t = np.rint(position).astype(int)
            w, h = rects[j][2:]
            if not 0 <= x <= video.width-w or not 0 <= t <= video.height-h:
                continue
            left, top = max(0, x-10), max(0, t-10)
            roi = y[top:t+h+10, left:x+w+10]
            dx, dy, score = template_peak(highpass(roi, 4), templates[j])
            located = np.asarray((left+dx, top+dy))
            if score >= .39 and np.linalg.norm(located-position) < 7:
                observations.append((frame-center, *(located if j == 0 else mirror-located)))
    samples = np.asarray(observations)
    if len(samples) < 6 or len(np.unique(samples[:, 0])) < 5:
        raise ValueError(f'Too few native-resolution path anchors around frame {center}.')
    slope, anchor = np.polyfit(samples[:, 0], samples[:, 1:], 1)
    residual = np.linalg.norm(samples[:, 1:]-(anchor+samples[:, :1]*slope), axis=1)
    good = residual < 2.6
    if good.sum() < 6 or np.linalg.norm(step-slope) > .24:
        raise ValueError(f'Native path calibration did not agree around frame {center}.')
    slope, anchor = np.polyfit(samples[good, 0], samples[good, 1:], 1)
    print(f'Motion around {center}: {int(good.sum())} verified observations; velocity {slope.round(4).tolist()}.', flush=True)
    return center, frames, slope, anchor


def paired_paths(video, rects, luma, cuda):
    time, scores, samples = rough_tracks(video, rects, luma, cuda)
    mirror = np.asarray(rects[0][:2])+np.asarray(rects[1][:2])
    templates = [highpass(crop(luma, rect), 4) for rect in rects]
    # A still title can conceal a path that is already moving behind it.
    delta = np.abs(samples.astype(np.int16)-samples[0]).mean(axis=(1, 2))
    first_change = np.flatnonzero(delta > 7)
    gate = int(time[first_change[0]]) if len(first_change) and first_change[0] >= 12 else 0
    paths = [[], []]
    for segment in find_segments(time, scores, mirror):
        center, anchors, velocity, origin = refine_segment(video, segment, rects, templates, mirror, cuda)
        low = max(gate, int(anchors[0])-230)
        high = min(video.count, int(anchors[-1])+231)
        frames = np.arange(low, high)
        predicted = origin + (frames-center)[:, None]*velocity
        for j, (location, step) in enumerate(((predicted, velocity), (mirror-predicted, -velocity))):
            w, h = rects[j][2:]
            inside = ((location[:, 0]+w > 0) & (location[:, 0] < video.width) &
                      (location[:, 1]+h > 0) & (location[:, 1] < video.height))
            if not inside.any():
                continue
            rows = np.flatnonzero(inside)
            start, stop = int(frames[rows[0]]), int(frames[rows[-1]])+1
            if paths[j] and start < paths[j][-1]['frames'][1]:
                raise ValueError('Two projected paths compete in the same frames.')
            paths[j].append({'frames': [start, stop], 'position': location[rows[0]].round(5).tolist(),
                             'velocity': step.round(7).tolist()})
    print(f'Validated paths per moving mark: {[len(path) for path in paths]}; opening title gate {gate}.', flush=True)
    return paths, gate


class PairedMotion:
    """Use the fitted pair as a search window and resolve its pose on every frame."""

    radius = 56

    def __init__(self, video, settings, cuda):
        self.video = video
        self.settings = settings
        self.rects = [item['reference_rect'] for item in settings]
        self.mirror = np.asarray(self.rects[0][:2])+np.asarray(self.rects[1][:2])
        self.prediction = np.full((video.count, 2), np.nan, np.float32)
        self.measured = np.full((video.count, 2), np.nan, np.float32)
        self.strength = np.zeros(video.count, np.float32)
        self.ranges = []
        primary = settings[0]
        floor = primary.get('motion_floor', 0)
        for segment in primary['motion_path']:
            lo, hi = segment['frames']
            start, stop = max(floor, lo-12), min(video.count, hi+12)
            if self.ranges and start < self.ranges[-1][1]:
                raise ValueError('Expanded motion search windows overlap.')
            points = np.asarray(segment['position'])+(np.arange(start, stop)-lo)[:, None]*np.asarray(segment['velocity'])
            self.prediction[start:stop] = points
            self.ranges.append((start, stop))
        self.templates = []
        for item, rect in zip(settings, self.rects, strict=True):
            plate = video.planes(get_frame(video, item['reference_frame'], cuda))[0]
            self.templates.append(self.feature(crop(plate, rect)))

    @staticmethod
    def feature(values):
        log = -np.log(np.clip((235-values.astype(np.float32))/219, .045, 1.1))
        return highpass(log, 2.6).clip(-.40, .40)

    def match(self, luma, position, template):
        x, y = np.rint(position).astype(int)
        height, width = template.shape
        radius = self.radius
        left, top = max(0, x-radius), max(0, y-radius)
        right, bottom = min(self.video.width, x+width+radius), min(self.video.height, y+height+radius)
        if min(right-left, bottom-top) < 20:
            return None
        detail = self.feature(luma[top:bottom, left:right])
        inside = np.clip((x, y), (0, 0), (self.video.width-width, self.video.height-height))
        inward = np.rint(np.clip(inside-(x, y), -radius*.8, radius*.8)).astype(int)
        best = None
        for dx, dy in ((0, 0), tuple(inward//2), tuple(inward)):
            u, v = x+dx, y+dy
            a, b = max(0, 4-u), max(0, 4-v)
            c, d = min(width, self.video.width-u-4), min(height, self.video.height-v-4)
            if min(c-a, d-b) < 18:
                continue
            shape = template[b:d, a:c]
            energy = np.count_nonzero(np.abs(shape) > .035)
            if energy < 130:
                continue
            xx, yy, score = template_peak(detail, shape)
            found = np.asarray((left+xx-a, top+yy-b))
            if score < .30 or np.max(np.abs(found-position)) > radius:
                continue
            merit = score * min(1., np.sqrt(energy/450))
            if best is None or merit > best[0]:
                best = merit, found
        return best

    def observe(self, index, luma):
        prior = self.prediction[index]
        if np.isnan(prior[0]):
            return
        samples = []
        for j, (position, template) in enumerate(zip((prior, self.mirror-prior), self.templates, strict=True)):
            found = self.match(luma, position, template)
            if found is not None:
                score, point = found
                samples.append((score, point if j == 0 else self.mirror-point))
        samples.sort(key=lambda sample: -sample[0])
        if not samples:
            return
        score, point = samples[0]
        if len(samples) == 2 and np.linalg.norm(point-samples[1][1]) < 3:
            point = (point+samples[1][1])/2
        elif score < .41:
            return
        self.measured[index], self.strength[index] = point, score

    def coordinates(self):
        from wm_track import motion_padding

        corrected = np.full_like(self.prediction, np.nan)
        for lo, hi in self.ranges:
            frames = np.arange(lo, hi)
            observed = frames[np.isfinite(self.measured[frames, 0])]
            if len(observed) < 18:
                raise ValueError(f'Too few source matches for the paired route at {lo}:{hi}.')
            delta = self.measured[observed]-self.prediction[observed]
            # Reject a transient peak that disagrees with both adjacent source matches.
            median = np.stack([np.median(delta[max(0, j-3):j+4], axis=0) for j in range(len(delta))])
            good = np.linalg.norm(delta-median, axis=1) < 9
            observed, delta = observed[good], delta[good]
            if len(observed) < 18 or np.max(np.diff(observed), initial=0) > 85:
                raise ValueError(f'Insufficient continuous evidence for the paired route at {lo}:{hi}.')
            change = np.column_stack([np.interp(frames, observed, col) for col in delta.T])
            for where, samples in ((frames < observed[0], slice(0, 8)), (frames > observed[-1], slice(-8, None))):
                slope = np.polyfit(observed[samples], delta[samples], 1)[0].clip(-1.9, 1.9)
                anchor = observed[0] if samples.start == 0 else observed[-1]
                change[where] += (frames[where]-anchor)[:, None]*slope
            corrected[frames] = self.prediction[frames]+change.clip(-72, 72)
            print(f'Paired route {lo}:{hi}, {len(observed)} verified native positions, '
                  f'median correction {np.median(np.linalg.norm(delta, axis=1)):.2f} px.', flush=True)
        outputs = []
        for layer, points, rect in zip(self.settings, (corrected, self.mirror-corrected), self.rects, strict=True):
            w, h = rect[2:]
            valid = ((points[:, 0]+w > 0) & (points[:, 0] < self.video.width) &
                     (points[:, 1]+h > 0) & (points[:, 1] < self.video.height))
            coords = np.zeros((self.video.count, 3), np.float32)
            coords[:, :2] = -1
            coords[valid, :2] = points[valid]+motion_padding(layer)
            coords[valid, 2] = self.strength[valid]
            outputs.append(coords)
        return outputs
