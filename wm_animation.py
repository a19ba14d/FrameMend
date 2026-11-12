"""Restore a repeated animated wordmark using its matching occurrence."""

import json
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from wm_inpaint import ProPainter, composite, feather
from wm_matte import Matte
from wm_media import Video, digest, plane_rects, read_frames, yuv_to_rgb
from wm_opening import brand_colors, fill_shapes, rect_mask, wipe_mask
from wm_repair import dilate
from wm_track import check_rect, load_track, save_npz


def dark_rays(first, second, before_first, before_second):
    a, b, clear_a, clear_b = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.int16)
                              for image in (first, second, before_first, before_second))
    local_a, local_b = (cv2.GaussianBlur(image, (0, 0), 2.6) for image in (a, b))
    changed = ((clear_a-a > 18) & (clear_b-b > 19)) | ((local_a-a > 18) & (local_b-b > 17))
    dark = (a < 76) & (b < 76) & (np.abs(a-b) <= 23) & changed
    return fill_shapes(dark.astype(np.uint8), 4)


def shared_animation_mask(first, second, before_first, before_second, area, radius, include_rays):
    similar = np.abs(first.astype(np.int16)-second).max(axis=2) <= 53
    common_colors = np.logical_or.reduce([a & b for a, b in zip(brand_colors(first), brand_colors(second), strict=True)])
    seeds = (similar & common_colors & (area > 0)).astype(np.uint8)
    seeds = fill_shapes(cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), 5)
    if include_rays:
        seeds |= dark_rays(first, second, before_first, before_second) & area
    return dilate(seeds, radius) & area


def read_rgb(video, start, stop, rect, cuda, matte=None, corners=None):
    frames = iter(read_frames(video, start, stop, cuda))
    images = []
    while batch := list(islice(frames, 16)):
        if matte:
            matte.restore(video, [frame for _, frame in batch], batch[0][0], spatial=False)
        for i, frame in batch:
            if corners:
                corners.restore(i, frame)
            images.append(yuv_to_rgb(plane_rects(video.planes(frame), rect), video.matrix))
    return np.stack(images)


def bounded_mask(video, bounds, region):
    left, top, width, height = check_rect(bounds, video.width, video.height)
    x, y, w, h = region
    if not (x <= left < left+width <= x+w and y <= top < top+height <= y+h):
        raise ValueError('Animation mask bounds must be inside the model region.')
    return rect_mask((left, top, width, height), region)


def animation_palette(rgb, include_green=False):
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    cyan = (g-r > 19) & (b-r > 10)
    yellow = (r-b > 27) & (g-b > 25) & (r > 140) & (g > 125)
    white = (np.minimum(np.minimum(r, g), b) > 175) & (np.maximum(np.maximum(r, g), b)-np.minimum(np.minimum(r, g), b) < 30)
    labels = cyan.astype(np.uint8) + 2*yellow.astype(np.uint8) + 3*white.astype(np.uint8)
    if include_green:
        green = (g-r > 19) & (g-b > 34) & (g > 110)
        labels[green] = 4
    return labels


def isolated_rays(mask, center=None):
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rays = np.zeros_like(mask)
    for contour in contours:
        box = cv2.minAreaRect(contour)
        short, long = sorted(box[1])
        if not (6 <= long <= 155 and short <= 14 and long >= 2.4*max(short, 1)):
            continue
        if center is not None:
            vertices = cv2.boxPoints(box)
            axes = np.diff(vertices, axis=0)
            line = axes[np.argmax(np.linalg.norm(axes, axis=1))]
            radial = np.asarray(box[0])-center
            agreement = abs(np.dot(line, radial)) / max(np.linalg.norm(line)*np.linalg.norm(radial), 1)
            if agreement < 0.84:
                continue
        cv2.drawContours(rays, [contour], -1, 1, -1)
    return rays


def build_animation_template(source, output, profile_path, workdir, config_path, cuda=True):
    video = Video.open(source)
    output = Path(output)
    if output.exists() or output.suffix.lower() != '.npz':
        raise ValueError('Choose a new .npz animation template path.')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    rect = check_rect(config['region'], video.width, video.height)
    x, y = rect[:2]
    area = bounded_mask(video, config['mask_region'], rect)
    starts = config['signature_starts']
    count = config.get('template_frames', 78)
    radius = config.get('dilation', 18)
    if (not 3 <= len(starts) <= 16 or len(set(starts)) != len(starts) or not 2 <= count <= 120
            or not all(0 <= i <= video.count-count for i in starts) or not 1 <= radius <= 32):
        raise ValueError('Choose three or more distinct matching occurrences and a valid template length.')
    phases = {}
    for first, last, *bounds in config.get('age_regions', []):
        if not 0 <= first < last <= count:
            raise ValueError('Animation age ranges must be inside the template.')
        support = area & bounded_mask(video, bounds, rect)
        phases.update({age: support for age in range(first, last)})
    track = load_track(video, profile_path, Path(workdir) / 'track.npz')
    signature = read_rgb(video, starts[0], starts[0]+count, rect, cuda)
    green = config.get('include_green', False)
    signature_palette = np.stack([animation_palette(image, green) for image in signature])
    first_gray = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in signature])
    colors = np.zeros(signature.shape[:3], np.uint8)
    dark = np.zeros_like(colors)
    for start in starts[1:]:
        for i, frame in read_frames(video, start, start+count, cuda):
            age = i-start
            rgb = yuv_to_rgb(plane_rects(video.planes(frame), rect), video.matrix)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
            if age == 0:
                clear = gray
                continue
            palette = animation_palette(rgb, green)
            difference = np.abs(signature[age].astype(np.int16)-rgb).max(axis=2)
            colors[age] += (palette != 0) & (palette == signature_palette[age]) & (difference <= 58)
            if age <= config.get('dark_until', 0):
                local = cv2.GaussianBlur(gray, (0, 0), 2.4)
                changed = (clear-gray > 12) | (local-gray > 10)
                dark[age] += (gray < 105) & (first_gray[age] < 105) & (np.abs(gray-first_gray[age]) <= 34) & changed
        print(f'Compared animation at frame {start}.', flush=True)
    masks = np.zeros_like(colors)
    center = np.asarray(config.get('ray_center', (video.width//2, video.height//2)))-np.asarray((x, y))
    for age in range(1, count):
        support = phases.get(age, area)
        seeds = ((colors[age] >= 2) & (support > 0)).astype(np.uint8)
        seeds = fill_shapes(cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), 3)
        dark_pixels = (dark[age] >= 2) & (support > 0)
        if age > 12:
            dark_pixels &= cv2.distanceTransform(1-seeds, cv2.DIST_L2, 3) <= 60
        seeds |= isolated_rays(dark_pixels.astype(np.uint8), center if age <= 12 else None)
        masks[age] = dilate(seeds, radius) & support
    if hold := config.get('hold_mask'):
        first, last = hold
        if not 1 <= first < last <= count:
            raise ValueError('The stationary animation mask range must be inside the template.')
        masks[first:last] = np.mean(masks[first:last], axis=0) >= .5
    if not masks[1:].any():
        raise ValueError('No common animated features were found in the supplied occurrences.')
    save_npz(output, region=np.asarray(rect), masks=masks, source_sha256=track['source_sha256'],
             profile_sha256=track['profile_sha256'], config_sha256=digest(config_path))


def prepare_animation_patch(source, base, output, profile_path, workdir, config_path, cuda=True, job=None, template=None):
    source = Video.open(source)
    from_source = base is None
    base = source if from_source else Video.open(base)
    if (source.width, source.height, source.count, source.fps, source.matrix) != (
            base.width, base.height, base.count, base.fps, base.matrix):
        raise ValueError('Animation repair requires an aligned base with the same color matrix.')
    output = Path(output)
    if output.exists() or output.suffix.lower() != '.npz':
        raise ValueError('Choose a new .npz animation patch path.')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    jobs = config.pop('jobs', None)
    if jobs is not None:
        if job not in jobs:
            raise ValueError(f'Choose an animation job from {", ".join(jobs)}.')
        config.update(jobs[job])
    elif job is not None:
        raise ValueError('This animation configuration has a single job.')
    rect = check_rect(config['region'], source.width, source.height)
    width, height = rect[2:]
    lo, hi = config['active']
    start, stop = config['context']
    carry_until = config.get('carry_until', hi)
    reference, repeat = config.get('reference_start'), config['repeat_start']
    clean = config.get('clean_frames', [])
    blank = config.get('blank_frames', [])
    radius, feather_width = config.get('dilation', 18), config.get('feather', 7)
    scale, dark_until = config.get('scale', 0.58), config.get('dark_until', 0)
    split = config.get('split_at_cuts', False)
    if not (0 <= start <= lo < hi <= stop <= source.count
            and hi <= carry_until <= min(hi+4, source.count)
            and 0 <= repeat < lo
            and (template is not None or (reference is not None and 0 <= reference < reference+(hi-repeat) <= repeat))
            and all(0 <= i < source.count and not start <= i < stop for i in clean)
            and all(lo <= i < hi for i in blank)
            and 1 <= feather_width <= radius <= 32 and 0.25 <= scale <= 1):
        raise ValueError('Invalid animation reference, patch timing, dilation or scale.')
    track = load_track(source, profile_path, Path(workdir) / 'track.npz')
    cuts = np.union1d(track['cuts'], np.asarray(config.get('extra_cuts', []), dtype=np.int64))
    cuts = [int(i) for i in cuts if start < i < stop]
    if cuts and not split:
        raise ValueError('Animation model context cannot cross a detected shot boundary.')
    if split and clean:
        raise ValueError('Use individual shot jobs when supplying extra clean frames.')
    edges = [start, *cuts, stop]
    if any(b-a+len(clean) > 120 for a, b in zip(edges, edges[1:])):
        raise ValueError('Keep each animation model window within 120 frames.')

    common = bounded_mask(source, config['mask_region'], rect)
    limits = {}
    for first, last, *bounds in config.get('phase_regions', []):
        if not lo <= first < last <= hi:
            raise ValueError('Animation phase bounds must be inside the active range.')
        restrict = common & bounded_mask(source, bounds, rect)
        limits.update({i: restrict for i in range(first, last)})
    matte = None
    if from_source:
        from wm_track import load_profile
        profile = load_profile(source, profile_path)
        if profile.get('translucent') or profile.get('stationary'):
            matte = Matte(profile, track, Path(workdir) / 'matte.npz')
    context = read_rgb(base, start, stop, rect, cuda, matte)
    masks = np.zeros(context.shape[:3], np.uint8)
    if template is None:
        signature = read_rgb(source, reference, reference+hi-repeat, rect, cuda)
        observed = read_rgb(source, repeat, hi, rect, cuda)
        for i in range(lo, hi):
            age = i-repeat
            masks[i-start] = shared_animation_mask(signature[age], observed[age], signature[0], observed[0],
                                                   limits.get(i, common), radius, age <= dark_until)
    else:
        with np.load(template, allow_pickle=False) as data:
            if (str(data['source_sha256']) != str(track['source_sha256'])
                    or str(data['profile_sha256']) != str(track['profile_sha256'])
                    or str(data['config_sha256']) != digest(config_path) or tuple(data['region']) != rect):
                raise ValueError('The animation template belongs to another source, profile or configuration.')
            stencil = data['masks']
        if stencil.ndim != 3 or stencil.shape[1:] != (height, width) or len(stencil) < hi-repeat:
            raise ValueError('The animation template does not cover this job.')
        for i in range(lo, hi):
            masks[i-start] = stencil[i-repeat] & limits.get(i, common)
    if not masks.any():
        raise ValueError('No shared animated pixels found at the configured timings.')
    active = slice(lo-start, hi-start)
    alpha = np.stack([np.rint(feather(mask, feather_width)*255).astype(np.uint8) for mask in masks[active]])
    model = ProPainter(scale=scale)
    predicted = np.zeros_like(context[active])
    if clean:
        plates = [read_rgb(base, i, i+1, rect, cuda, matte) for i in clean]
        images = np.concatenate((context, *plates))
        model_masks = np.concatenate((masks, np.zeros((len(clean), height, width), np.uint8)))
        predicted = model.predict(images, model_masks, references=range(len(context), len(images)))[active]
    else:
        for a, b in zip(edges, edges[1:]):
            left, right = max(lo, a), min(hi, b)
            window = slice(a-start, b-start)
            if left < right and masks[window].any():
                frames = model.predict(context[window], masks[window])
                predicted[left-lo:right-lo] = frames[left-a:right-a]
    for i in blank:
        predicted[i-lo] = 0
    if carry_until > hi:
        tail = read_rgb(source, hi, carry_until, rect, cuda)
        # The wordmark is already stationary when the opaque transition begins.
        # Replace exposed fragments while preserving the transition above them.
        extra = np.stack([alpha[-1] * (1-wipe_mask(image)) for image in tail])
        alpha = np.concatenate((alpha, extra))
        predicted = np.concatenate((predicted, np.repeat(predicted[-1:], carry_until-hi, axis=0)))
    save_npz(output, region=np.asarray(rect), frames=np.asarray([lo, carry_until]), prediction=predicted, alpha=alpha,
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'], config_sha256=digest(config_path))
    print(json.dumps(dict(job=job, active=[lo, carry_until], masked_frames=int(alpha.any(axis=(1, 2)).sum()), saved_patch=str(output)),
                     indent=2))


class AnimationPatch:
    def __init__(self, video, track, path):
        self.video = video
        with np.load(path, allow_pickle=False) as data:
            if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
                raise ValueError('Animation patch belongs to a different source or profile.')
            self.region = check_rect(data['region'].tolist(), video.width, video.height)
            self.lo, self.hi = data['frames'].tolist()
            self.prediction, self.alpha = data['prediction'], data['alpha']
        _, _, width, height = self.region
        shape = (self.hi-self.lo, height, width)
        if (not 0 <= self.lo < self.hi <= video.count or self.alpha.shape != shape
                or self.prediction.shape != shape+(3,) or self.alpha.dtype != np.uint8
                or self.prediction.dtype != np.uint8):
            raise ValueError('Invalid animation patch timing or dimensions.')

    def apply(self, index, frame):
        if self.lo <= index < self.hi:
            k = index-self.lo
            if self.alpha[k].any():
                planes = plane_rects(self.video.planes(frame), self.region)
                composite(planes, self.prediction[k], self.alpha[k].astype(np.float32)/255, self.video.matrix)
