import json
import os
from pathlib import Path

import cv2
import numpy as np

from wm_media import digest, get_frame, read_frames


def check_rect(rect, width, height):
    if (len(rect) != 4 or any(type(v) is not int or v < 0 or v % 2 for v in rect)
            or min(rect[2:]) <= 0 or rect[0] + rect[2] > width or rect[1] + rect[3] > height):
        raise ValueError(f'Expected an in-frame even [x,y,w,h] rectangle, got {rect}.')
    return tuple(rect)


def crop(plane, rect):
    x, y, w, h = rect
    return plane[y:y+h, x:x+w]


def highpass(gray, sigma):
    gray = gray.astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def frame_variants(count, alternate_ranges):
    variants = np.zeros(count, np.int16)
    for variant, ranges in enumerate(alternate_ranges, 1):
        if not ranges:
            raise ValueError('Each alternate template needs frame ranges.')
        for lo, hi in ranges:
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= count or variants[lo:hi].any():
                raise ValueError('Template frame ranges must be disjoint and within the source.')
            variants[lo:hi] = variant
    return variants


def tracking_window(video, outer, requested):
    x, y, w, h = outer
    a, b, c, d = check_rect(requested, video.width, video.height)
    if not (x <= a and y <= b and a+c <= x+w and b+d <= y+h):
        raise ValueError('The tracking rectangle must be inside the full overlay plate.')
    return a, b, c, d


def corner_activity(settings, scores):
    active = scores >= settings.get('minimum_score', 0.7)
    active[..., settings.get('always_active', [])] = True
    return active


def save_npz(path, *, replace=False, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.part')
    try:
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        if replace:
            temporary.replace(path)
        elif path.exists():
            raise FileExistsError(path)
        else:
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def template_peak(roi, template, mask_scores=None):
    if any(a < b for a, b in zip(roi.shape, template.shape, strict=True)):
        return -1, -1, 0.0
    corr = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    if mask_scores:
        mask_scores(corr)
    _, score, _, (x, y) = cv2.minMaxLoc(corr)

    def subpixel(before, after):
        denominator = before + after - 2 * score
        return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5)) if denominator < -1e-6 else 0.0

    dx = subpixel(corr[y, x-1], corr[y, x+1]) if 0 < x < corr.shape[1]-1 else 0.0
    dy = subpixel(corr[y-1, x], corr[y+1, x]) if 0 < y < corr.shape[0]-1 else 0.0
    return x+dx, y+dy, score


class Tracker:
    def __init__(self, video, profile, cuda):
        self.video = video
        self.logo = profile.get('translucent')
        self.corner = profile.get('corners')
        anchors = {}
        self.variant_index = 0
        self.variant_by_frame = None
        self.output_offsets = None

        def get_anchor(index):
            if index not in anchors:
                anchors[index] = video.planes(get_frame(video, index, cuda))
            return anchors[index]

        self.chroma = None
        if self.logo:
            full = check_rect(self.logo['reference_rect'], video.width, video.height)
            separate_tracking = 'tracking_rect' in self.logo
            self.logo_rect = tracking_window(video, full, self.logo.get('tracking_rect', full))
            self.padding = tuple(self.logo.get('edge_pad', (0, 0)))
            if len(self.padding) != 2 or any(type(n) is not int or n < 0 or n > 96 or n % 2 for n in self.padding):
                raise ValueError('edge_pad needs two even values between zero and 96.')
            if any(self.padding) and (separate_tracking or 'search_rect' in self.logo or 'chroma_match_rect' in self.logo):
                raise ValueError('Padded tracking requires the full frame and a luma reference.')
            px, py = self.padding
            bounds = (video.width+2*px, video.height+2*py)
            self.logo_search = check_rect(self.logo.get('search_rect', (0, 0, *bounds)), *bounds)
            x, y, w, h = self.logo_rect
            sx, sy, sw, sh = self.logo_search
            if not (sx <= x+px and sy <= y+py and x+px+w <= sx+sw and y+py+h <= sy+sh):
                raise ValueError('The translucent reference must be inside its search rectangle.')
            self.scale = int(self.logo.get('downsample', 4))
            match = self.logo.get('match_rect')
            fine = check_rect(match, w, h) if match is not None else (8, 8, w-16, h-16)
            self.fine_offset = fine[:2]
            self.coarse_offset = self.fine_offset if match is not None else (0, 0)
            chroma_rect = self.logo.get('chroma_match_rect')
            if chroma_rect:
                chroma_rect = check_rect(chroma_rect, w, h)
                cx, cy = chroma_rect[:2]
                self.chroma_offset = cx//2, cy//2

            def templates(index, rect):
                luma, u, v = get_anchor(index)
                reference = crop(luma, rect)
                coarse = highpass(self.shrink(crop(reference, fine) if match is not None else reference), 1.5)
                chroma = None
                if chroma_rect:
                    cx, cy, cw, ch = chroma_rect
                    region = ((rect[0]+cx)//2, (rect[1]+cy)//2, cw//2, ch//2)
                    chroma = highpass(crop(v, region).astype(np.float32)-crop(u, region), 2.5)
                return coarse, crop(highpass(reference, 5), fine).copy(), highpass(reference, 3), chroma

            primary = templates(self.logo['reference_frame'], self.logo_rect)
            self.coarse, self.fine, self.visible_template, self.chroma = primary
            displacement = (full[0]-x, full[1]-y)
            displacements = [displacement]
            alternatives = self.logo.get('white_plate', {}).get('variants', [])
            if any('plate' in item for item in alternatives):
                self.variant_by_frame = frame_variants(video.count, [item['ranges'] for item in alternatives])
                self.variant_templates = [primary]
                for item in alternatives:
                    plate = item.get('plate')
                    if plate:
                        full_plate = check_rect(plate['rect'], video.width, video.height)
                        rect = tracking_window(video, full_plate, plate.get('tracking_rect', full_plate))
                        if rect[2:] != (w, h):
                            raise ValueError('Alternate tracking plates must share the primary template size.')
                        self.variant_templates.append(templates(plate['frame'], rect))
                        displacements.append((full_plate[0]-rect[0], full_plate[1]-rect[1]))
                    else:
                        self.variant_templates.append(primary)
                        displacements.append(displacement)
            if separate_tracking:
                offsets = np.asarray(displacements)+motion_padding(self.logo)
                self.output_offsets = (offsets[self.variant_by_frame] if self.variant_by_frame is not None
                                       else np.broadcast_to(offsets[0], (video.count, 2)))
            self.previous = (x+px, y+py)
        if self.corner:
            self.corner_search = [check_rect(r, video.width, video.height) for r in self.corner['search_rects']]
            references = self.corner.get('references')
            if references is None:
                references = [{'frame': self.corner['reference_frame'], 'rect': self.corner['template_rect']}] * len(self.corner_search)
            if not isinstance(references, list) or len(references) != len(self.corner_search):
                raise ValueError('Supply one reference per corner search rectangle.')
            self.corner_templates = []
            self.corner_color = []
            for reference, search in zip(references, self.corner_search, strict=True):
                rect = check_rect(reference['rect'], video.width, video.height)
                if any(a > b for a, b in zip(rect[2:], search[2:], strict=True)):
                    raise ValueError('A corner reference is larger than its search rectangle.')
                y, u, v = get_anchor(reference['frame'])
                self.corner_templates.append(highpass(crop(y, rect), 3))
                if self.corner.get('chroma_template'):
                    half = tuple(value//2 for value in rect)
                    self.corner_color.append(highpass(crop(v, half).astype(np.float32)-crop(u, half), 1.7))

    def shrink(self, plane):
        h, w = plane.shape
        return cv2.resize(plane, (w // self.scale, h // self.scale), interpolation=cv2.INTER_AREA)

    def padded_y(self, luma):
        px, py = self.padding
        return cv2.copyMakeBorder(luma, py, py, px, px, cv2.BORDER_REPLICATE) if px or py else luma

    def select_frame(self, index):
        if self.variant_by_frame is not None:
            variant = self.variant_by_frame[index]
            if variant != self.variant_index:
                self.coarse, self.fine, self.visible_template, self.chroma = self.variant_templates[variant]
                self.variant_index = variant

    def translate_coordinates(self, coords, *, to_tracking=False):
        if self.output_offsets is not None:
            valid = coords[:, 0] >= 0
            direction = -1 if to_tracking else 1
            coords[valid, :2] += self.output_offsets[valid]*direction

    def mask_opaque_scores(self, scores, columns, rows, blocked):
        h, w = self.fine.shape
        for left, top, right, bottom in blocked:
            width = np.minimum(columns+w, right)-np.maximum(columns, left)
            height = np.minimum(rows+h, bottom)-np.maximum(rows, top)
            scores[np.ix_(height >= .85*h, width >= .90*w)] = -1

    def _fine_match(self, y, guess, radius, blocked=()):
        _, _, w, h = self.logo_rect
        sx, sy, sw, sh = self.logo_search
        px, py = guess
        x0, y0 = max(sx, px-radius), max(sy, py-radius)
        x1 = min(sx+sw, px+w+radius)
        y1 = min(sy+sh, py+h+radius)
        roi = highpass(y[y0:y1, x0:x1], 5)
        mask_scores = None
        if len(blocked):
            def mask_scores(scores):
                rows, columns = scores.shape
                self.mask_opaque_scores(scores, x0+np.arange(columns), y0+np.arange(rows), blocked)
        x, y, score = template_peak(roi, self.fine, mask_scores)
        return x0 + x - self.fine_offset[0], y0 + y - self.fine_offset[1], score

    def _chroma_match(self, u, v):
        sx, sy, sw, sh = (n//2 for n in self.logo_search)
        px, py = (round(n/2) for n in self.previous)
        cx, cy = self.chroma_offset
        ch, cw = self.chroma.shape
        x0, y0 = max(sx, px+cx-10), max(sy, py+cy-10)
        x1, y1 = min(sx+sw, px+cx+cw+10), min(sy+sh, py+cy+ch+10)
        box = np.s_[y0:y1, x0:x1]
        x, y, score = template_peak(highpass(v[box].astype(np.float32)-u[box], 2.5), self.chroma)
        return 2*(x0+x-cx), 2*(y0+y-cy), score

    def locate_logo(self, planes, index=None, blocked=()):
        if not self.logo:
            return -1, -1, 0.0
        if index is not None:
            self.select_frame(index)
        y, u, v = planes
        y = self.padded_y(y)
        threshold = self.logo.get('minimum_score', 0.32)
        x, row, score = self._fine_match(y, self.previous, radius=10, blocked=blocked)
        sx, sy, sw, sh = self.logo_search
        if self.chroma is not None and score < threshold:
            other = self._chroma_match(u, v)
            if other[2] >= self.logo.get('chroma_minimum_score', 0.7):
                x, row, score = other
        if score < self.logo.get('local_score', 0.68):
            small = highpass(self.shrink(crop(y, self.logo_search)), 1.5)
            corr = cv2.matchTemplate(small, self.coarse, cv2.TM_CCOEFF_NORMED)
            if len(blocked):
                rows, columns = corr.shape
                offset_x, offset_y = np.asarray(self.fine_offset)-self.coarse_offset
                self.mask_opaque_scores(corr, sx+np.arange(columns)*sw//small.shape[1]+offset_x,
                                        sy+np.arange(rows)*sh//small.shape[0]+offset_y, blocked)
            _, _, _, (xx, yy) = cv2.minMaxLoc(corr)
            guess = (sx+xx*sw//small.shape[1]-self.coarse_offset[0],
                     sy+yy*sh//small.shape[0]-self.coarse_offset[1])
            other = self._fine_match(y, guess, radius=12, blocked=blocked)
            if other[2] > score:
                x, row, score = other
        w, h = self.logo_rect[2:]
        if score < threshold or not (sx <= x <= sx+sw-w and sy <= row <= sy+sh-h):
            return -1, -1, score
        self.previous = round(x), round(row)
        return x, row, score

    def locate_corners(self, y, u=None, v=None):
        if not self.corner:
            return []
        scores = [float(cv2.matchTemplate(highpass(crop(y, rect), 3), template, cv2.TM_CCOEFF_NORMED).max())
                  for rect, template in zip(self.corner_search, self.corner_templates, strict=True)]
        if self.corner_color and u is not None and v is not None:
            for i, (rect, template) in enumerate(zip(self.corner_search, self.corner_color, strict=True)):
                half = tuple(value//2 for value in rect)
                values = highpass(crop(v, half).astype(np.float32)-crop(u, half), 1.7)
                scores[i] = max(scores[i], float(cv2.matchTemplate(values, template, cv2.TM_CCOEFF_NORMED).max()))
        return scores


def bridge_matches(coords, cuts, limit):
    repaired = np.zeros(len(coords), bool)
    missing = np.flatnonzero(coords[:, 0] < 0)
    for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
        if not len(run) or len(run) > limit or run[0] == 0 or run[-1] == len(coords)-1:
            continue
        left, right = run[0]-1, run[-1]+1
        step = (coords[right, :2] - coords[left, :2]) / (right-left)
        if np.any((cuts > left) & (cuts <= right)) or np.linalg.norm(step) > 6:
            continue
        coords[run, :2] = coords[left, :2] + (run-left)[:, None] * step
        repaired[run] = True
    return repaired


def visible_tail(luma, tracker, guess, bottom):
    px, py = tracker.padding
    x, y = np.rint(guess).astype(int) - (px, py)
    h, w = tracker.visible_template.shape
    x0, x1 = max(0, x-9), min(luma.shape[1], x+w+9)
    y0, y1 = bottom+3, min(luma.shape[0], y+h+24)
    best = (-1., -1., 0.)
    if x1-x0 < w-16 or y1-y0 < 14:
        return best
    detail = highpass(luma[y0:y1, x0:x1], 3)
    for row in range(y-24, y+25):
        first = max(0, y0+5-row)
        stop = min(h, y1-row)
        template = tracker.visible_template[first:stop, 8:-8]
        if len(template) < 12 or template.std() < 2.5:
            continue
        start = row+first-y0
        dx, _, score = template_peak(detail[start:start+len(template)], template)
        left = dx+x0-8
        if score > best[2] and abs(left-x) <= 8:
            best = (left+px, row+py, score)
    return best


def recover_continuous_logo(video, tracker, coords, settings, cuda, corner, corner_scores):
    """Recheck weak detections near a motion path bounded by reliable observations."""
    limit = settings['max_gap']
    step_limit = float(settings.get('max_step', 6))
    overlap = settings.get('corner_overlap', False)
    if type(limit) is not int or not 2 <= limit <= (180 if overlap else 90) or not 1 <= step_limit <= 12:
        raise ValueError('Continuous logo recovery needs a bounded gap and per-frame displacement.')
    positions = coords[:, :2]
    missing = np.flatnonzero(positions[:, 0] < 0)
    recovered = np.zeros(video.count, dtype=bool)
    for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
        if not len(run) or len(run) > limit or run[0] < 5 or run[-1] >= video.count-5:
            continue
        left, right = run[0]-1, run[-1]+1
        a, b = positions[[left, right]]
        before, after = positions[left-4:left+1], positions[right:right+5]
        if np.any(before < 0) or np.any(after < 0):
            continue
        dt = right-left
        v0, v1 = (before[-1]-before[0])/4, (after[-1]-after[0])/4
        if max(np.linalg.norm(v) for v in ((b-a)/dt, v0, v1)) > step_limit:
            continue
        t = ((run-left)/dt)[:, None]
        path = ((2*t**3-3*t**2+1)*a + (t**3-2*t**2+t)*dt*v0
                + (-2*t**3+3*t**2)*b + (t**3-t**2)*dt*v1)
        sx, sy, sw, sh = tracker.logo_search
        w, h = tracker.logo_rect[2:]
        if overlap and len(run) > 90 and max(a[1], b[1]) < video.height*.16:
            path[:, 1] = np.maximum(sy, path[:, 1])
        if not ((sx <= path[:, 0]).all() and (path[:, 0] <= sx+sw-w).all()
                and (sy <= path[:, 1]).all() and (path[:, 1] <= sy+sh-h).all()):
            continue
        blocked = np.zeros(len(run), dtype=int)
        if overlap and corner:
            actual = path-np.asarray(tracker.padding)
            active = corner_activity(corner, corner_scores[run])
            for j, polygon in enumerate(corner['masks']):
                xx, yy = np.asarray(polygon).T
                inside = (actual[:, 0]+w*.65 > xx.min()) & (actual[:, 0]+w*.35 < xx.max()) & (actual[:, 1]+h*.3 < yy.max())
                blocked[inside & active[:, j]] = max(1, int(yy.max())-6)
        if len(run) > 90 and np.count_nonzero(blocked) < len(run)*.9:
            continue
        accepted = np.zeros(len(run), bool)
        for i, frame in read_frames(video, int(run[0]), int(run[-1])+1, cuda):
            tracker.select_frame(i)
            j = i-int(run[0])
            guess = path[j]
            luma = video.planes(frame)[0]
            if blocked[j]:
                x, y, score = visible_tail(luma, tracker, guess, blocked[j])
                threshold, radius = .34, 25
            else:
                x, y, score = tracker._fine_match(tracker.padded_y(luma), tuple(np.rint(guess).astype(int)), 6)
                threshold, radius = .18, 4
            if (score >= threshold and sx <= x <= sx+sw-w and sy <= y <= sy+sh-h
                    and np.linalg.norm(np.array([x, y])-guess) <= radius):
                positions[i] = x, y
                coords[i, 2] = score
                accepted[j] = True
            else:
                positions[i] = guess
            recovered[i] = True
        if overlap and accepted.sum() >= 4 and np.count_nonzero(blocked) >= len(run)*.9:
            delta = positions[run[accepted]]-path[accepted]
            for axis in range(2):
                positions[run[~accepted], axis] += np.interp(run[~accepted], run[accepted], delta[:, axis]).clip(-8, 8)
            positions[run, 0] = positions[run, 0].clip(sx, sx+sw-w)
            positions[run, 1] = positions[run, 1].clip(sy, sy+sh-h)
        print(f'Continuous overlay frames {left+1}:{right}, {int(accepted.sum())} local matches, '
              f'{len(run)-int(accepted.sum())} path estimates.', flush=True)
    return recovered


def motion_padding(settings):
    if settings.get('motion_path'):
        return tuple(length+2 for length in settings['reference_rect'][2:])
    if inner := settings.get('tracking_rect'):
        outer = settings['reference_rect']
        offsets = [(inner[0]-outer[0], inner[1]-outer[1])]
        for item in settings.get('white_plate', {}).get('variants', []):
            if (plate := item.get('plate')) and (rect := plate.get('tracking_rect')):
                offsets.append((rect[0]-plate['rect'][0], rect[1]-plate['rect'][1]))
        return tuple(max(axis) for axis in zip(*offsets))
    return tuple(settings.get('edge_pad', (0, 0)))


def motion_coordinates(video, settings):
    w, h = settings['reference_rect'][2:]
    offset = np.asarray(motion_padding(settings))
    coords = np.zeros((video.count, 3), np.float32)
    coords[:, :2] = -1
    previous = 0
    for segment in settings['motion_path']:
        lo, hi = segment['frames']
        if not previous <= lo < hi <= video.count:
            raise ValueError('Moving path ranges must be sorted, disjoint and inside the source.')
        previous = hi
        start = np.asarray(segment['position'], np.float32)
        speed = np.asarray(segment['velocity'], np.float32)
        if start.shape != (2,) or speed.shape != (2,) or not np.isfinite([start, speed]).all():
            raise ValueError('A moving path needs finite two-axis positions and velocities.')
        points = start + np.arange(hi-lo)[:, None]*speed
        visible = ((points[:, 0]+w > 0) & (points[:, 0] < video.width) &
                   (points[:, 1]+h > 0) & (points[:, 1] < video.height))
        selected = lo + np.flatnonzero(visible)
        coords[selected, :2] = points[visible] + offset
        coords[selected, 2] = .40
    return coords


def extra_tracks(profile, track):
    for i, layer in enumerate(profile.get('extra_white', []), 1):
        prefix = f'extra{i}_'
        yield {'translucent': layer}, {**track, **{name: track[prefix+name] for name in ('positions', 'precise', 'scores')}}, prefix


def quantized_positions(coords):
    return np.where(coords[:, :2] < 0, -1, np.floor((coords[:, :2] + 1) / 2) * 2).astype(np.int16)


def verify_colored_anchors(settings, coords):
    primary = settings.get('white_plate', {})
    if not primary.get('opacity_reference'):
        return
    plates = [primary, *(item['plate'] for item in primary.get('variants', []) if 'plate' in item)]
    padding = np.asarray(motion_padding(settings))
    for plate in plates:
        frame = plate['frame']
        point = coords[frame, :2]
        if point[0] < 0 or np.linalg.norm(point-padding-plate['rect'][:2]) > 3:
            raise ValueError(f'Moving mark tracking disagrees with the calibrated dark reference at frame {frame}.')
    print(f'Tracking matches all {len(plates)} calibrated colored references.', flush=True)


def recover_corner_track(video, profile, track, cuda=True):
    if not (profile.get('translucent') and profile.get('corners')):
        raise ValueError('Corner overlap recovery requires one moving mark and corner masks.')
    tracker = Tracker(video, profile, cuda)
    coords = np.column_stack((track['precise'], track['scores'])).copy()
    tracker.translate_coordinates(coords, to_tracking=True)
    settings = {'max_gap': 180, 'max_step': 6, 'corner_overlap': True}
    recovered = recover_continuous_logo(video, tracker, coords, settings, cuda, profile['corners'], track['corners'])
    tracker.translate_coordinates(coords)
    combined = track.get('motion_recovered', np.zeros(video.count, bool)) | recovered
    print(f'Recovered {int(recovered.sum())} moving positions in corner-overlap pass.', flush=True)
    return {**track, 'positions': quantized_positions(coords), 'precise': coords[:, :2],
            'scores': coords[:, 2], 'motion_recovered': combined}


def scan(video, profile, profile_path, output, cuda=True):
    if Path(output).exists():
        raise ValueError('Choose a new track file.')
    video.check_timestamps()
    if profile.get('discovery'):
        from wm_discovery import import_discovery
        import_discovery(video, profile, profile_path, output, cuda)
        return
    logo_settings = profile.get('translucent') or {}
    paired = None
    extras = profile.get('extra_white', [])
    if logo_settings.get('verify_paired_motion'):
        from wm_motion_paths import PairedMotion
        if len(extras) != 1:
            raise ValueError('Paired motion verification requires two layers.')
        paired = PairedMotion(video, [logo_settings, *extras], cuda)
    planned = bool(logo_settings.get('motion_path'))
    tracker = Tracker(video, {**profile, 'translucent': None} if planned else profile, cuda)
    opaque_bounds = []
    if tracker.output_offsets is not None and tracker.corner:
        for polygon in tracker.corner['masks']:
            xs, ys = zip(*polygon)
            opaque_bounds.append((min(xs), min(ys), max(xs)+1, max(ys)+1))
        opaque_bounds = np.asarray(opaque_bounds)
    coords, corner_scores, scene_scores = [], [], []
    previous = None
    for i, frame in read_frames(video, cuda=cuda):
        planes = video.planes(frame)
        y = planes[0]
        corners = tracker.locate_corners(*planes)
        if paired:
            paired.observe(i, y)
        elif not planned:
            blocked = opaque_bounds[corner_activity(tracker.corner, np.asarray(corners))] if len(opaque_bounds) else ()
            coords.append(tracker.locate_logo(planes, i, blocked))
        corner_scores.append(corners)
        # Ignore the top corner logos and the usual subtitle bands when looking for cuts.
        middle = y[video.height//6:video.height*5//6, video.width//16:video.width*15//16]
        thumb = cv2.resize(middle, (48, 28), interpolation=cv2.INTER_AREA).astype(np.int16)
        score = 0 if previous is None else np.abs(thumb - previous).mean()
        scene_scores.append(score)
        previous = thumb
        if (i+1) % 600 == 0:
            print(f'Tracked {i+1}/{video.count}', flush=True)
    if paired:
        coords, *extra_coordinates = paired.coordinates()
    else:
        coords = motion_coordinates(video, logo_settings) if planned else np.asarray(coords, dtype=np.float32)
        extra_coordinates = [motion_coordinates(video, layer) for layer in extras]
    corners = np.asarray(corner_scores, dtype=np.float32)
    cuts = np.flatnonzero(np.asarray(scene_scores) > profile.get('cut_score', 24))
    extra = {}
    if limit := logo_settings.get('bridge_gaps'):
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError('bridge_gaps must be a short frame count from 1 to 8.')
        extra['interpolated'] = bridge_matches(coords, cuts, limit)
        print(f'Bridged {int(extra["interpolated"].sum())} low-confidence frames within shots.', flush=True)
    if motion := logo_settings.get('continuous_motion'):
        extra['motion_recovered'] = recover_continuous_logo(video, tracker, coords, motion, cuda,
                                                          profile.get('corners'), corners)
        print(f'Recovered {int(extra["motion_recovered"].sum())} frames using continuous overlay motion.', flush=True)
    tracker.translate_coordinates(coords)
    verify_colored_anchors(logo_settings, coords)
    for i, companion in enumerate(extra_coordinates, 1):
        extra.update({f'extra{i}_positions': quantized_positions(companion), f'extra{i}_precise': companion[:, :2],
                      f'extra{i}_scores': companion[:, 2]})
        print(f'White layer {i+1}: {int(np.count_nonzero(companion[:, 0] >= 0))} visible frames from verified paths.', flush=True)
    positions = quantized_positions(coords)
    save_npz(output, positions=positions, precise=coords[:, :2], scores=coords[:, 2],
             corners=corners, cuts=cuts, scene_scores=np.asarray(scene_scores, dtype=np.float32),
             source_sha256=digest(video.path), profile_sha256=digest(profile_path), **extra)
    found = np.count_nonzero(coords[:, 0] >= 0)
    print(f'Tracked {video.count} frames. URL matches {found}. Cuts {len(cuts)}. Saved {output}', flush=True)


def load_profile(video, path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        profile = json.load(stream)
    if profile.get('size') != [video.width, video.height]:
        raise ValueError('Profile coordinates do not match the source resolution.')
    if corners := profile.get('corners'):
        always = corners.get('always_active', [])
        count = len(corners['search_rects'])
        if not isinstance(always, list) or any(type(i) is not int or not 0 <= i < count for i in always):
            raise ValueError('Corner always_active must list zero-based region indices.')
    return profile


def load_track(video, profile_path, path):
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    if (str(data['source_sha256']) != digest(video.path) or str(data['profile_sha256']) != digest(profile_path)
            or len(data['positions']) != video.count):
        raise ValueError('This track belongs to another source or profile. Run scan again.')
    return data
