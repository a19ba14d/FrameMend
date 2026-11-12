"""Refine a tracked translucent watermark in a short, single-shot region."""

from collections import OrderedDict
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_edge_retouch import polish_texture
from wm_inpaint import ProPainter, feather, model_size
from wm_matte import Matte, exclusion, logo_rect
from wm_media import Video, plane_rects, read_frames, yuv_to_rgb
from wm_repair import dilate
from wm_temporal import CapsuleRim
from wm_track import check_rect, corner_activity, load_profile, load_track, save_npz


def crop_box(region, context):
    x, y, w, h = region
    cx, cy = context[:2]
    return np.s_[y-cy:y-cy+h, x-cx:x-cx+w]


def expand_region(rect, width, height, horizontal, vertical):
    x, y, w, h = rect
    left, top = max(0, x-horizontal), max(0, y-vertical)
    right, bottom = min(width, x+w+horizontal), min(height, y+h+vertical)
    return left, top, right-left, bottom-top


def contains(outer, inner):
    x, y, w, h = outer
    a, b, c, d = inner
    return x<=a and y<=b and a+c<=x+w and b+d<=y+h


def watermark_region(video, profile, track, first, last):
    marks = [rect for i in range(first, last) if (rect := logo_rect(profile, track, i)) is not None]
    if not marks:
        raise ValueError('The tracker found no translucent watermark in this interval.')
    left, top = min(r[0] for r in marks), min(r[1] for r in marks)
    right = max(x+w for x, _, w, _ in marks)
    bottom = max(y+h for _, y, _, h in marks)
    return expand_region((left, top, right-left, bottom-top), video.width, video.height, 8, 8)


def read_interval(value, fps, seconds):
    try:
        first, last = value.split(':')
        if seconds:
            start, end = Fraction(first), Fraction(last)
            if not 0<=start<end:
                raise ValueError
            return [math.floor(start*fps), math.ceil(end*fps)]
        return [int(first), int(last)]
    except (ValueError, ZeroDivisionError) as error:
        unit = 'decimal seconds' if seconds else 'whole frame numbers'
        raise ValueError(f'Use START:STOP with {unit}.') from error


def valid_interval(value, name):
    if not isinstance(value, list) or len(value)!=2 or any(type(i) is not int for i in value):
        raise ValueError(f'{name} must be [first, stop] with whole frame numbers.')
    first, last = value
    if not 0<=first<last:
        raise ValueError(f'{name} must have 0 <= first < stop.')
    return first, last


def anchor_groups(config, first, last):
    if 'groups' not in config:
        anchors = sorted(set(range(first, last, 6))|{last-1})
        return [dict(frames=[first, last], anchors=anchors, radius=13)]
    groups = config['groups']
    if not isinstance(groups, list) or not groups or any(not isinstance(g, dict) for g in groups):
        raise ValueError('groups must be a nonempty list of anchor groups.')
    expected = first
    for group in groups:
        lo, hi = valid_interval(group.get('frames'), 'Group frames')
        anchors = group.get('anchors')
        radius = group.get('radius')
        if (lo!=expected or hi>last or not isinstance(anchors, list) or not anchors
                or any(type(i) is not int or not lo<=i<hi for i in anchors)
                or anchors!=sorted(set(anchors)) or type(radius) is not int or not 1<=radius<=28):
            raise ValueError('Use consecutive groups with sorted, distinct anchors and a radius of 1–28 frames.')
        expected = hi
    if expected!=last:
        raise ValueError('Anchor groups must cover the selected interval.')
    return groups


def edge_strength(fields, sigma, gain):
    kernel = np.ones((3, 3), np.uint8)
    return np.maximum.reduce([
        cv2.morphologyEx(cv2.GaussianBlur(f, (0, 0), sigma), cv2.MORPH_GRADIENT, kernel)
        for pair in fields for f in (pair[0]*gain, pair[1]*128)])


def protected_mask(shape, context, rectangles):
    guard = np.zeros(shape, np.uint8)
    left, top = context[:2]
    for x, y, w, h in rectangles:
        cv2.rectangle(guard, (x-left, y-top), (x+w-left-1, y+h-top-1), 1, -1)
    return guard


def collect_observations(video, compositor, matte, context, region, lo, hi, protect, background, cuda):
    profile, track = compositor.profile, matte.track
    paired = bool(compositor.extra)
    models = [a.cpu().numpy() for a in matte.models]
    fields = [a[0] for a in models]
    margins = matte.profile['translucent'].get('capsule_margins')
    rim = CapsuleRim(matte.width, matte.height, margins) if margins else None
    opacity = np.clip(matte.opacity, 0, 1)
    tint = np.max(np.abs(models[0][:, 1]), axis=0)
    edges = (np.maximum.reduce([edge_strength(f, .65, 65) for f in zip(*models, strict=True)])>5.2).astype(np.uint8)
    edge_weight = feather(dilate(edges, 2), 3)
    confidence = (1-.26*np.clip((opacity-.04)/.44, 0, 1))*(1-.85*edge_weight)
    footprint = ((opacity>.057)|(tint>.047)).astype(np.uint8)
    support = feather(dilate(footprint, 4), 4)
    if paired:
        confidence = np.minimum(confidence, 1-.94*support)
    contrast = np.maximum.reduce([edge_strength(f, .62, 66) for f in zip(*models, strict=True)])
    texture_mask = cv2.GaussianBlur(dilate((contrast>3.6).astype(np.uint8), 4).astype(np.float32), (0, 0), 1.3)
    left, top, width, height = context
    x, y, w, h = region
    box = crop_box(region, context)
    fixed_guard = protected_mask((height, width), context, protect)
    corner_polys = [np.asarray(points, dtype=np.int32)-np.asarray([left, top], np.int32)
                    for points in profile.get('corners', {}).get('masks', [])]
    corner_active = corner_activity(profile.get('corners', {}), track['corners'])
    images, certainties, supports = [], [], []
    transforms, exposed = [], []
    batch = []

    def append_batch(items):
        compositor.restore(video, [frame for _, frame, _ in items], items[0][0], spatial=paired)
        for i, frame, rect in items:
            planes = video.planes(frame)
            if rim and rect:
                rim.restore(plane_rects(planes, rect), matte.shifts[i])
            images.append(yuv_to_rgb(plane_rects(planes, context), video.matrix))

    for i, frame in read_frames(video, lo, hi, cuda):
        rect = logo_rect(matte.profile, track, i)
        translate = None
        if rect:
            shift = matte.shifts[i]
            if rim:
                luma = plane_rects(video.planes(frame), rect)[0]
                shift = rim.align(luma, shift, fields[0])
                matte.shifts[i] = shift
            dx, dy = np.asarray(rect[:2])+shift-(left, top)
            translate = np.float32([[1, 0, dx], [0, 1, dy]])
            certain = cv2.warpAffine(confidence, translate, (width, height), borderValue=1)
            active = cv2.warpAffine(support, translate, (width, height))
        else:
            certain = np.ones((height, width), np.float32)
            active = np.zeros_like(certain)
        guard = fixed_guard.copy()
        for active_corner, points in zip(corner_active[i], corner_polys, strict=True):
            if active_corner:
                cv2.fillPoly(guard, [points], 1)
        # Guard the full 2×2 block when a protected pixel shares a chroma sample.
        shared = guard.reshape(height//2, 2, width//2, 2)
        shared[:] = shared.max(axis=(1, 3), keepdims=True)
        certain[guard>0] = 0
        active[guard>0] = 0
        if background:
            actual = yuv_to_rgb(plane_rects(video.planes(frame), region), video.matrix)
            clear = (exclusion(video, {**profile, 'translucent': matte.profile['translucent']},
                               track, i)[y:y+h, x:x+w]>0)&(guard[box]==0)
            exposed.append((actual, clear))
        transforms.append(translate)
        certainties.append(certain)
        supports.append(active[box].copy())
        batch.append((i, frame, rect))
        if len(batch)==12:
            append_batch(batch)
            batch.clear()
    if batch:
        append_batch(batch)
    return np.stack(images), np.stack(certainties), np.stack(supports), transforms, texture_mask, exposed


def unit_grid(height, width):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device='cuda'),
                          torch.linspace(-1, 1, width, device='cuda'), indexing='ij')
    return torch.stack((x, y), dim=-1)[None]


def sample(pictures, grid, padding='border'):
    return F.grid_sample(pictures, grid, align_corners=True, padding_mode=padding)


class SharedMotion:
    def __init__(self, images, scale):
        h, w = model_size(images.shape[2:], scale)
        small = F.interpolate(images/127.5-1, (h, w), mode='bilinear', align_corners=False, antialias=True)
        net = ProPainter(scale=scale).raft
        pairs = [net(small[None, j:min(j+7, len(small))], iters=10) for j in range(0, len(small)-1, 6)]
        norm = small.new_tensor([2/(w-1), 2/(h-1)])[None, :, None, None]
        self.forward, self.backward = [torch.cat([a[side][0] for a in pairs])*norm for side in range(2)]
        self.grid = unit_grid(h, w)
        self.paths = OrderedDict()

    def path(self, start, stop):
        if start==stop:
            return torch.zeros_like(self.grid.permute(0, 3, 1, 2))
        key = start, stop
        if key in self.paths:
            self.paths.move_to_end(key)
            return self.paths[key]
        step = 1 if stop>start else -1
        first = self.forward[start:start+1] if step>0 else self.backward[start-1:start]
        tail = self.path(start+step, stop)
        grid = self.grid+first.permute(0, 2, 3, 1)
        path = first+sample(tail, grid)
        self.paths[key] = path
        if len(self.paths)>160:
            self.paths.popitem(last=False)
        return path


def local_mean(x):
    h = F.avg_pool2d(x, (1, 17), 1, (0, 8), count_include_pad=False)
    return F.avg_pool2d(h, (17, 1), 1, (8, 0), count_include_pad=False)


def estimate(images, confidence, motion, grid, box, index, others, size):
    own = images[index, :, box[0], box[1]]
    certainty = confidence[index, :, box[0], box[1]]
    if not others:
        return own, torch.zeros_like(certainty)
    left = torch.cat([motion.path(index, j) for j in others])
    right = torch.cat([motion.path(j, index) for j in others])
    identity = grid.expand(len(others), -1, -1, -1)
    displacement = sample(left, identity)
    flow_grid = identity+displacement.permute(0, 2, 3, 1)
    backward = sample(right, flow_grid)
    pixels = own.new_tensor([(size[0]-1)/2, (size[1]-1)/2])[None, :, None, None]
    roundtrip = torch.linalg.vector_norm((displacement+backward)*pixels, dim=1, keepdim=True)
    warped = sample(torch.cat((images[others], confidence[others]), dim=1), flow_grid, 'zeros')
    aligned, quality = warped[:, :3], warped[:, 3:]
    known = (certainty>.65)*quality
    absdiff = (own[None]-aligned).abs().mean(dim=1, keepdim=True)
    residual = (local_mean(absdiff.clamp_max(34)*known)+.1)/(local_mean(known)+.10)
    decay = own.new_tensor([abs(j-index) for j in others])[:, None, None, None]
    weight = quality.square()*torch.exp(-decay/13)/(1+(roundtrip/1.85)**2)/(1+(residual/6.5)**2)
    # Wide disagreements may be real changes. Keep the target structure there.
    weight *= 1/(1+(absdiff/38)**4)
    total = weight.sum(dim=0)
    neighbor = (weight*aligned).sum(dim=0)/total.clamp_min(.0001)
    divergence = (weight*(aligned-neighbor[None]).abs().mean(dim=1, keepdim=True)).sum(dim=0)/total.clamp_min(.0001)
    blend = (total/.96).clamp(0, 1)*((1-certainty)*1.13).clamp(0, .97)/(1+(divergence/8)**2)
    return own*(1-blend)+neighbor*blend, blend


def finish_edges(image, certainty, opacity, weight):
    active = ((certainty<.29)&(opacity>.65)&(weight<.56)).astype(np.uint8)
    if not active.any():
        return image
    patched = cv2.inpaint(image, active, 2, cv2.INPAINT_TELEA).astype(np.float32)
    image = image.astype(np.float32)
    delta = np.max(abs(patched-image), axis=2)
    blend = active*np.clip((27-delta)/20, 0, .86)*np.clip((.74-weight)/.53, 0, 1)
    return np.rint(image*(1-blend[..., None])+patched*blend[..., None]).clip(0, 255).astype(np.uint8)


def restore_static(picture, current, references, box, alpha):
    image, known = current
    image, known = image[box], known[box]
    minimum = min(2200, max(64, image.shape[0]*image.shape[1]//8))
    best = None
    for i, (reference, valid) in references.items():
        reference, valid = reference[box], valid[box]
        eligible = known & valid
        if eligible.sum()<minimum:
            continue
        correction = np.median(image[eligible].astype(np.float32)-reference[eligible], axis=0)
        adjusted = reference.astype(np.float32)+correction
        error = np.abs(adjusted-image).mean(axis=-1)[eligible]
        if np.quantile(error, .90)>7 or np.abs(correction).max()>12:
            continue
        score = float(error.mean())
        if best is None or score<best[0]:
            best = score, i, adjusted, valid
    if best is None:
        return None
    score, i, adjusted, valid = best
    blend = feather(valid.astype(np.uint8), 7)*alpha[box]
    fade = min(17, blend.shape[1])
    blend[:, :fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    picture[box] = picture[box]*(1-blend[..., None])+adjusted*blend[..., None]
    return dict(reference=i, visible_rgb_error=score, pixels=int((blend>.55).sum()))


def static_background(config, video, region, first, last):
    background = config.get('static_background')
    if background is None:
        return None, []
    if not isinstance(background, dict):
        raise ValueError('static_background must provide a region and reference frames.')
    rect = check_rect(background.get('region', []), video.width, video.height)
    anchors = background.get('frames')
    if (not contains(region, rect) or not isinstance(anchors, list) or not 1<=len(anchors)<=8
            or any(type(i) is not int or not first<=i<last for i in anchors)
            or anchors!=sorted(set(anchors))):
        raise ValueError('Use a static region inside the patch and 1–8 sorted reference frames in the clip.')
    return crop_box(rect, region), anchors


@torch.inference_mode()
def prepare_local_patch(args):
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise ValueError('Local patch preparation requires CUDA PyTorch.')
    source = Video.open(args.input)
    if args.output.suffix.lower()!='.npz' or args.output.exists():
        raise ValueError('Choose a new .npz patch path.')
    if args.report and (args.report.exists() or args.report.resolve()==args.output.resolve()):
        raise ValueError('Choose a new report path separate from the patch.')
    if args.config:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if not isinstance(config, dict):
            raise ValueError('The local patch configuration must be a JSON object.')
        if jobs := config.pop('jobs', None):
            if args.job not in jobs:
                raise ValueError(f'Choose a local patch job from {", ".join(jobs)}.')
            config.update(jobs[args.job])
        elif args.job is not None:
            raise ValueError('This local patch configuration has a single job.')
    else:
        if args.job is not None:
            raise ValueError('A local patch job requires --config.')
        in_seconds = args.seconds is not None
        interval = args.seconds if in_seconds else args.frames
        config = {'frames': read_interval(interval, source.fps, in_seconds)}
    lo, hi = valid_interval(config.get('frames'), 'Frames')
    if hi>source.count or hi-lo>150:
        raise ValueError(f'Choose at most 150 frames inside the source (0–{source.count-1}).')
    profile = load_profile(source, args.profile)
    if not profile.get('translucent'):
        raise ValueError('The source profile needs a tracked translucent watermark.')
    track = load_track(source, args.profile, args.workdir/'track.npz')
    cuts = track['cuts'][(lo<track['cuts'])&(track['cuts']<hi)]
    if len(cuts):
        raise ValueError(f'The interval crosses a shot cut at frame {cuts[0]}. Use one shot at a time.')
    layer = config.get('layer', 0)
    count = len(profile.get('extra_white', []))
    if type(layer) is not int or not 0 <= layer <= count:
        raise ValueError('Choose a moving watermark layer present in the profile.')
    fade_frames = config.get('fade_frames', 0)
    if type(fade_frames) is not int or not 0 <= fade_frames <= min(15, (hi-lo)//2):
        raise ValueError('Use at most 15 fade frames on each side of the repair.')
    rect = args.region or config.get('region')
    paired = bool(count)
    compositor = Matte(profile, track, args.workdir/'matte.npz', clean_edges=paired, narrow_edges=paired,
                       align_white=paired, white_flat=paired)
    matte = (compositor, *compositor.extra)[layer]
    region = (check_rect(rect, source.width, source.height) if rect is not None
              else watermark_region(source, matte.profile, matte.track, lo, hi))
    context = check_rect(config.get('context', expand_region(region, source.width, source.height, 96, 100)),
                         source.width, source.height)
    _, _, cw, ch = context
    if not contains(context, region) or cw*ch>850000:
        raise ValueError('The repair region must fit in a context of at most 850,000 pixels.')
    method = config.get('method', 'references')
    if method not in ('references', 'model'):
        raise ValueError('Choose references or model for the local repair.')
    max_scale = .90 if method == 'model' else .65
    scale = config.get('scale', .48)
    if type(scale) not in (int, float) or not .25<=scale<=max_scale:
        raise ValueError(f'Use a local repair scale between .25 and {max_scale}.')
    if method == 'model' and hi-lo>90:
        raise ValueError('Use at most 90 frames for one local model repair.')
    protected = config.get('protect', [])
    if not isinstance(protected, list):
        raise ValueError('protect must list the rectangles to preserve.')
    protect = [check_rect(r, source.width, source.height) for r in protected+args.protect]
    groups = anchor_groups(config, lo, hi)
    static_box, static_frames = static_background(config, source, region, lo, hi)
    original, quality, mask, transforms, texture, exposed = collect_observations(
        source, compositor, matte, context, region, lo, hi, protect, static_frames, not args.cpu_decode)
    if not np.any(mask>=.5/255):
        raise ValueError('The selected region has no tracked watermark pixels available for repair.')
    box = crop_box(region, context)
    prepared = time.perf_counter()
    generated = None
    if method == 'model':
        scale = min(scale, math.sqrt(170000/(cw*ch)))
        model_masks = np.zeros(original.shape[:3], np.uint8)
        model_masks[:, box[0], box[1]] = mask>.25
        generated = ProPainter(scale=scale).predict(original, model_masks)[:, box[0], box[1]].copy()
    else:
        pictures = torch.as_tensor(original, device='cuda').permute(0, 3, 1, 2).contiguous().float()
        certain = torch.as_tensor(quality, device='cuda')[:, None]
        grid = unit_grid(ch, cw)[:, box[0], box[1]].contiguous()
        motion = SharedMotion(pictures, scale) if hi-lo>1 else None
    aligned_at = time.perf_counter()
    x, y, w, h = region
    origin = np.asarray([x-context[0], y-context[1]], np.float32)
    references = {i: exposed[i-lo] for i in static_frames}
    prediction, alpha, checks = [], [], []
    for group in groups:
        first, stop = group['frames']
        for i in range(first, stop):
            j = i-lo
            active = mask[j]
            if generated is not None:
                guess = generated[j]
                active = np.minimum(active*1.1, 1)
                measurements = {'model_support_pixels': int((active>.5).sum())}
            else:
                nearby = [k for k in group['anchors'] if k!=i and abs(k-i)<=group['radius']]
                nearby = sorted(k-lo for k in sorted(nearby, key=lambda k: (abs(k-i), k))[:8])
                guess, applied = estimate(pictures, certain, motion, grid, box, j, nearby, (cw, ch))
                guess = guess.permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
                weight = applied[0].cpu().numpy()
                if paired:
                    # Leave the existing correction where motion has no reliable visible reference.
                    active = active*np.clip(weight/.48, 0, 1)
                else:
                    guess = finish_edges(guess, quality[j][box], active, weight)
                measurements = {'anchors': len(nearby),
                                'reference_mixed_pixels': int(((weight>.5)&(active>.5)).sum())}
            if fade_frames:
                active = active*min(1, j/fade_frames, (hi-i-1)/fade_frames)
            opacity = np.rint(active*255).astype(np.uint8)
            fixed = None
            polish = generated is None and not paired
            if transforms[j] is not None and opacity.any() and (polish or static_frames):
                transform = transforms[j].copy()
                transform[:, 2] -= origin
                alpha_fraction = opacity.astype(np.float32)/255
                detail = cv2.warpAffine(texture, transform, (w, h))*alpha_fraction
                polished = polish_texture(guess, detail) if polish else guess.astype(np.float32)
                if static_frames:
                    fixed = restore_static(polished, exposed[j], references, static_box, alpha_fraction)
                guess = np.rint(polished).clip(0, 255).astype(np.uint8)
            prediction.append(guess)
            alpha.append(opacity)
            checks.append(dict(frame=i, **measurements, changed_support_pixels=int((active>.5).sum()),
                               static_background=fixed))
    save_npz(args.output, frames=np.asarray([lo, hi]), region=np.asarray(region),
             prediction=np.stack(prediction), alpha=np.stack(alpha), source_sha256=track['source_sha256'],
             profile_sha256=track['profile_sha256'])
    report = dict(frames=[lo, hi], layer=layer, region=region, context=context, method=method, scale=round(scale, 3),
                  device=torch.cuda.get_device_name(),
                  protected_rectangles=protect, affected_frames=sum(bool(a.any()) for a in alpha),
                  neighbor_pairs=hi-lo-1, source_seconds=round(prepared-started, 3), model_seconds=(
                      round(aligned_at-prepared, 3) if method == 'model' else None),
                  alignment_seconds=round(aligned_at-prepared, 3), total_seconds=round(time.perf_counter()-started, 3),
                  measurements=checks)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print({k: v for k, v in report.items() if k!='measurements'}, flush=True)
