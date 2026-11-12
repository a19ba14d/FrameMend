"""Prepare GPU patches for difficult overlaps with burned-in subtitles."""

import json
from pathlib import Path

import cv2
import numpy as np

from wm_caption_layers import composite_layers, estimate_layer
from wm_inpaint import ProPainter, feather
from wm_matte import Matte, capsule
from wm_media import Video, digest, get_frame, plane_rects, read_frames, rgb_to_yuv, yuv_to_rgb
from wm_temporal import TemporalEdges
from wm_track import check_rect, load_profile, load_track, save_npz


def subtitle_guard(luma, y, band, diameter=13, white_min=184, black_max=70, neighborhood=13):
    rows = np.arange(y, y+luma.shape[0])[:, None]
    dark_nearby = cv2.erode(luma, np.ones((neighborhood, neighborhood), np.uint8)) < black_max
    white = (luma > white_min) & dark_nearby & (rows >= band[0]) & (rows < band[1])
    return cv2.dilate(white.astype(np.uint8), np.ones((diameter, diameter), np.uint8)).astype(bool)


def suppress_bright_streaks(picture, prediction, target):
    previous = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY).astype(np.int16)
    repaired = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY).astype(np.int16)
    surrounding = cv2.morphologyEx(repaired, cv2.MORPH_OPEN, np.ones((21, 1), np.uint8))
    bright = ((repaired-previous > 10) & (repaired-surrounding > 10) & target).astype(np.uint8)
    connected = cv2.dilate(bright, np.ones((5, 3), np.uint8))
    streaks = cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones((1, 22), np.uint8))
    support = cv2.dilate(streaks, np.ones((5, 7), np.uint8)) & cv2.dilate(bright, np.ones((5, 5), np.uint8))
    return cv2.inpaint(prediction, support, 6, cv2.INPAINT_NS) if support.any() else prediction


def protected_composite(planes, prediction, support, guard, matrix, strength):
    alpha = feather(support.astype(np.uint8), 6) * strength
    alpha[guard] = 0
    width, height = planes[1].shape[::-1]
    color_alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    color_guard = cv2.resize(guard.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA) > 0
    color_alpha[color_guard] = 0
    checked = 0
    for plane, target, weight, protect in zip(planes, rgb_to_yuv(prediction, matrix),
                                             (alpha, color_alpha, color_alpha),
                                             (guard, color_guard, color_guard), strict=True):
        old = plane[protect].copy()
        active = weight > 0
        blend = weight[active]
        plane[active] = np.rint(plane[active]*(1-blend) + target[active]*blend).astype(np.uint8)
        if not np.array_equal(plane[protect], old):
            raise RuntimeError('The GPU patch changed protected subtitle samples.')
        checked += old.size
    return checked


def caption_layer_references(video, edges, region, config, context, cuda):
    start, stop = config['frames']
    if not context[0] <= start < stop <= context[1] or not config['references']:
        raise ValueError('Caption layer frames must be inside the context and have unoccluded references.')
    _, y, _, height = region
    references = []
    for reference in config['references']:
        i = reference['frame']
        top, bottom = reference['rows']
        if not start <= i < stop or not y <= top < bottom <= y+height:
            raise ValueError('Each caption reference must be inside the caption range and patch region.')
        parts = plane_rects(video.planes(get_frame(video, i, cuda)), region)
        white = subtitle_guard(parts[0], y, (top, bottom), diameter=1, white_min=191,
                               black_max=62, neighborhood=11).astype(np.uint8)
        outline = cv2.dilate(white, np.ones((9, 9), np.uint8)).astype(bool)
        contaminated = edges.mask(i, region, edges.reference_mask).astype(bool)
        if not outline.any() or np.any(outline & contaminated):
            raise ValueError(f'Caption reference frame {i} needs clearly visible, unoccluded letters.')
        padding = cv2.dilate(white, np.ones((19, 19), np.uint8)).astype(bool)
        references.append((i, parts, padding & ~contaminated))
    return (start, stop), references


def prepare_caption_patch(source, base, output, profile_path, workdir, config_path, cuda=True):
    original, video = Video.open(source), Video.open(base)
    if (original.width, original.height, original.count, original.fps) != (
            video.width, video.height, video.count, video.fps):
        raise ValueError('Caption inpainting requires an aligned base video.')
    output = Path(output)
    if output.exists() or output.suffix.lower() != '.npz':
        raise ValueError('Choose a new .npz caption patch path.')
    settings = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    region = check_rect(settings['region'], video.width, video.height)
    start, stop = settings['segment']
    lo, hi = settings['active']
    context_start, context_stop = settings['context']
    if not (0 <= start <= lo < hi <= stop <= video.count
            and 0 <= context_start <= lo < hi <= context_stop <= video.count
            and context_stop-context_start <= 120):
        raise ValueError('The active frames must be inside the segment and the model context of at most 120 frames.')
    x, y, width, height = region
    top, bottom = settings['write_rows']
    left, right = settings.get('write_columns', (x, x+width))
    layer_config = settings.get('caption_layer')
    cover_capsule = bool(settings.get('cover_capsule', False))
    mask_subtitles = settings.get('mask_subtitles_in_model', False)
    band = settings.get('subtitle_rows', [0, video.height])
    diameter = settings.get('guard_diameter', 13)
    mask_diameter = settings.get('mask_diameter', 7)
    top_feather = settings.get('top_feather', 1)
    fade = settings.get('fade_frames', 5)
    scale, strength = settings.get('scale', 0.68), settings.get('strength', 0.86)
    if (not (y <= top < bottom <= y+height and x <= left < right <= x+width
             and 0 <= band[0] < band[1] <= video.height
             and 0 < scale <= 1 and 0 < strength <= 1 and 7 <= diameter <= 31 and diameter % 2
             and 1 <= mask_diameter <= 31 and mask_diameter % 2 and 1 <= top_feather <= height
             and 1 <= fade <= 30)):
        raise ValueError('Invalid patch bounds, scale, strength or subtitle guard.')
    profile = load_profile(original, profile_path)
    track = load_track(original, profile_path, Path(workdir) / 'track.npz')
    matte = Matte(profile, track, Path(workdir) / 'matte.npz', clean_edges=True)
    edges = TemporalEdges(original, matte, track)
    cuts = edges.cuts[(edges.cuts > context_start) & (edges.cuts < context_stop)] - context_start
    if cuts.size and not settings.get('split_at_cuts', False):
        raise ValueError('The GPU context cannot cross a detected shot boundary.')
    if layer_config:
        layer_frames, references = caption_layer_references(original, edges, region, layer_config,
                                                             (context_start, context_stop), cuda)
        letters = np.logical_or.reduce([ref[2] for ref in references])
    shape = edges.reference_mask
    if cover_capsule:
        margins = profile['translucent'].get('capsule_margins')
        if not layer_config or not margins:
            raise ValueError('Full capsule recovery requires caption layers and a profile with capsule margins.')
        shape = np.maximum(shape, capsule(matte.width, matte.height, margins))
    shape = cv2.dilate(shape, np.ones((mask_diameter, mask_diameter), np.uint8))
    rows = np.arange(y, y+height)[:, None]
    columns = np.arange(x, x+width)[None, :]
    write = (rows >= top) & (rows < bottom) & (columns >= left) & (columns < right)
    rgb, masks, targets, guards = [], [], [], []
    for i, frame in read_frames(video, context_start, context_stop, cuda):
        planes = plane_rects(video.planes(frame), region)
        rgb.append(yuv_to_rgb(planes, video.matrix))
        footprint = edges.mask(i, region, shape).astype(bool)
        target = footprint & write & (lo <= i < hi)
        if layer_config:
            target_letters = letters if layer_frames[0] <= i < layer_frames[1] else False
            masks.append(footprint | target_letters)
        else:
            guard = subtitle_guard(planes[0], y, band, diameter)
            target &= ~guard
            masks.append(target | guard if mask_subtitles else target)
            guards.append(guard)
        targets.append(target)
    masks = np.stack(masks)
    targets = np.stack(targets)
    if not targets.any():
        raise ValueError('No writable watermark pixels found inside the requested bounds.')
    model = ProPainter(scale=scale)
    clips = zip(np.split(np.stack(rgb), cuts), np.split(masks, cuts), strict=True)
    predicted = np.concatenate([model.predict(pictures, support) for pictures, support in clips])
    active = slice(lo-context_start, hi-context_start)
    if settings.get('suppress_bright_streaks', False):
        for i in range(active.start, active.stop):
            predicted[i] = suppress_bright_streaks(rgb[i], predicted[i], targets[i])
    if layer_config:
        alpha = np.zeros((height, width), np.float32)
        fill = np.zeros_like(alpha)
        for i, parts, guard in references:
            background = rgb_to_yuv(predicted[i-context_start], video.matrix)
            opacity, luma = estimate_layer(parts, background, guard)
            use = opacity > alpha
            alpha[use], fill[use] = opacity[use], luma[use]
        extra = dict(layer_frames=np.asarray(layer_frames), layer_alpha=alpha, layer_fill=fill,
                     top_weight=np.clip((rows-top+1).astype(np.float32)/top_feather, 0, 1),
                     cover_capsule=cover_capsule, layer_guard=letters)
    else:
        extra = dict(guards=np.asarray(guards)[active])
    save_npz(output, region=np.asarray(region), frames=np.asarray([lo, hi]), band=np.asarray(band),
             diameter=diameter, strength=strength, fade=fade, prediction=predicted[active],
             masks=targets[active], **extra,
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'],
             config_sha256=digest(config_path))
    print(json.dumps(dict(segment=[start, stop], active=[lo, hi],
                          active_frames_with_mask=int(targets[active].any(axis=(1, 2)).sum()),
                          saved_patch=str(output)), indent=2))


class CaptionPatch:
    def __init__(self, video, track, path):
        self.video = video
        self.checked = 0
        with np.load(path, allow_pickle=False) as data:
            if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
                raise ValueError('Caption patch belongs to a different source or profile.')
            self.region = check_rect(data['region'].tolist(), video.width, video.height)
            self.lo, self.hi = data['frames'].tolist()
            self.band = data['band'].tolist()
            self.diameter, self.strength = int(data['diameter']), float(data['strength'])
            self.fade = int(data.get('fade', 5))
            self.rgb, self.masks = (data[key] for key in ('prediction', 'masks'))
            self.layer_frames = data['layer_frames'].tolist() if 'layer_frames' in data else None
            if self.layer_frames is None:
                self.guards = data['guards']
            else:
                self.layer = data['layer_alpha'], data['layer_fill']
                self.top_weight = data['top_weight']
                self.layer_guard = data.get('layer_guard')
                self.cover_capsule = bool(data.get('cover_capsule', False))
        _, _, width, height = self.region
        shape = (self.hi-self.lo, height, width)
        if (not 0 <= self.lo < self.hi <= video.count or self.rgb.shape != shape+(3,)
                or self.masks.shape != shape or not 0 < self.strength <= 1 or not 1 <= self.fade <= 30):
            raise ValueError('Invalid caption patch data.')
        if self.layer_frames is None and self.guards.shape != shape:
            raise ValueError('Invalid caption guard size.')
        if self.layer_frames is not None and (not all(p.shape == shape[1:] for p in self.layer)
                or self.top_weight.shape != (height, 1)
                or not 0 <= self.layer_frames[0] < self.layer_frames[1] <= video.count):
            raise ValueError('Invalid caption layer geometry or timing.')
        if self.layer_frames is not None and self.cover_capsule and (
                self.layer_guard is None or self.layer_guard.shape != shape[1:]):
            raise ValueError('Full capsule recovery needs the caption boundary guard.')

    def apply(self, index, frame):
        if not self.lo <= index < self.hi:
            return
        k = index-self.lo
        if not self.masks[k].any():
            return
        planes = plane_rects(self.video.planes(frame), self.region)
        fade = min(1, (index-self.lo+1)/self.fade, (self.hi-index)/self.fade)
        if self.layer_frames is not None:
            layer = self.layer if self.layer_frames[0] <= index < self.layer_frames[1] else None
            support = self.masks[k]
            ring = None
            if self.cover_capsule:
                ring = cv2.dilate(support.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~support
                if layer is not None:
                    ring &= ~self.layer_guard
                ring = ring.astype(np.float32)
            composite_layers(planes, self.rgb[k], support, self.video.matrix,
                             self.strength*fade, self.top_weight, layer, ring)
            return
        guard = self.guards[k] | subtitle_guard(planes[0], self.region[1], self.band, self.diameter)
        self.checked += protected_composite(planes, self.rgb[k], self.masks[k], guard,
                                            self.video.matrix, self.strength*fade)
