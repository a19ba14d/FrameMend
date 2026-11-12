"""Recover repeated full-frame wipes from the visible parts of nearby frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_caption_inpaint import subtitle_guard
from wm_inpaint import ProPainter, feather
from wm_matte import affine_between
from wm_media import Video, digest, plane_rects, read_frames, rgb_to_yuv, yuv_to_rgb
from wm_opening import fill_shapes
from wm_repair import dilate
from wm_track import check_rect, load_track, save_npz


def wipe_palette(image):
    r, g, b = np.moveaxis(image.astype(np.int16), -1, 0)
    cyan = (g-r > 57) & (b-r > 36) & (g > 145)
    lime = (r-b > 63) & (g-b > 77) & (r > 150) & (g > 175)
    return cyan | lime


def new_cache(path):
    path = Path(path)
    if path.exists() or path.suffix.lower() != '.npz':
        raise ValueError('Choose a new .npz cache path.')
    return path


def wipe_bounds(config, video):
    first, stop = config['active_ages']
    before, after = config['template_anchors']
    limits = np.asarray(config['columns'])
    if (not 0 <= before < first < stop <= after or limits.shape != (stop-first, 2)
            or np.any(limits[:, 0] < 0) or np.any(limits[:, 1] > video.width)
            or np.any(limits[:, 0] >= limits[:, 1]) or not first < config['exit_age'] < stop):
        raise ValueError('Invalid wipe timing, reference frames or phase limits.')
    return first, stop, limits


def build_wipe_template(source, output, profile, workdir, settings, cuda=True):
    video = Video.open(source)
    path = new_cache(output)
    config = json.loads(Path(settings).read_text(encoding='utf-8-sig'))
    first, stop, limits = wipe_bounds(config, video)
    pre, post = config['template_anchors']
    starts = [job['start'] for job in config['jobs'].values()]
    if len(set(starts)) != len(starts) or not 3 <= len(starts) <= 16 or any(
            i < 0 or i+post >= video.count for i in starts):
        raise ValueError('Supply at least three aligned wipe occurrences.')
    track = load_track(video, profile, Path(workdir) / 'track.npz')
    votes = np.zeros((stop-first, video.height, video.width), dtype=np.uint8)
    black = np.zeros_like(votes)
    stable = np.zeros(votes.shape[1:], dtype=np.uint8)
    for start in starts:
        frames = {i-start: frame for i, frame in read_frames(video, start+pre, start+post+1, cuda)}
        early, late = [video.planes(frames[k])[0].astype(np.int16) for k in (pre, post)]
        stable += wipe_palette(yuv_to_rgb(video.planes(frames[post]), video.matrix))
        for age in range(first, stop):
            parts = video.planes(frames[age])
            votes[age-first] += wipe_palette(yuv_to_rgb(parts, video.matrix))
            luma = parts[0].astype(np.int16)
            local = cv2.GaussianBlur(luma, (0, 0), 2.4)
            clear = early if age < config['exit_age'] else late
            black[age-first] += (luma < 72) & (local-luma >= 8) & (clear-luma > 13)

    x = np.arange(video.width)[None, :]
    consensus = (len(starts)+1)//2
    shapes, guards = [], []
    for age, color, dark, (left, right) in zip(range(first, stop), votes, black, limits, strict=True):
        bounds = (x >= left) & (x < right)
        seeds = (color >= consensus) & bounds
        if seeds.mean() > 0.975:
            mask = np.ones_like(color)
        else:
            seeds &= stable < consensus
            core = fill_shapes(seeds.astype(np.uint8), 0)
            shadow = (dark >= consensus) & bounds
            if age >= 78 and age <= config['exit_age']+5:
                core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            mask = dilate(core | shadow.astype(np.uint8), 7)
        spread = cv2.dilate(mask, np.ones((33, 25), np.uint8)).astype(bool)
        if age < config['exit_age']:
            edge = np.max(np.where(spread, x, -200), axis=1)[:, None]
            guard = x <= edge+115
        else:
            edge = np.min(np.where(spread, x, video.width+200), axis=1)[:, None]
            guard = x >= edge-115
        shapes.append(mask)
        guards.append((guard & bounds).astype(np.uint8) | mask)
        print(f'Wipe phase {age:2d}: {float(mask.mean()):.3%} supported.', flush=True)
    save_npz(path, ages=np.asarray([first, stop]), masks=np.stack(shapes), guards=np.stack(guards),
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'],
             config_sha256=digest(settings))


def picture_slice(rect):
    x, y, width, height = rect
    return np.s_[y:y+height, x:x+width]


def capture_cues(video, frames, rules, first_age, masks, continuous):
    cues = []
    for rule in rules:
        rect = check_rect(rule['rect'], video.width, video.height)
        _, y, _, height = rect
        box = picture_slice(rect)
        ranges = rule['ranges']
        backgrounds = {}
        for background in rule.get('background_frames', []):
            frame = frames[background]
            parts = video.planes(frame)
            if np.count_nonzero(subtitle_guard(parts[0][box], y, (y, y+height), 1, 188, 69)) > 500:
                raise ValueError(f'Background reference phase {background} still contains text in the cue region.')
            backgrounds[background] = (parts[0], yuv_to_rgb(parts, video.matrix))
        blank = next((b[0]-1 for a, b in zip(ranges, ranges[1:]) if a[1] < b[0]), None)
        usable = not backgrounds and continuous and blank is not None and 0 <= blank-first_age < len(masks)
        if usable or backgrounds:
            registration = np.full((video.height, video.width), 255, np.uint8)
            registration[box] = 0
        if usable:
            blank_parts = video.planes(frames[blank])
            blank_image = yuv_to_rgb(blank_parts, video.matrix)
            blank_mask = cv2.erode(1-masks[blank-first_age], np.ones((13, 13), np.uint8))*255
        for first, stop, ref in ranges:
            if not first < stop:
                raise ValueError('Text cue end must follow its start.')
            parts = plane_rects(video.planes(frames[ref]), rect)
            glyphs = subtitle_guard(parts[0], y, (y, y+height), 17, 188, 69).astype(np.uint8)
            if np.count_nonzero(glyphs) < 300:
                raise ValueError(f'Frame {ref} has no usable text in the configured region.')
            image = yuv_to_rgb(parts, video.matrix)
            removal = dilate(glyphs, 7)
            if backgrounds:
                luma, background_image = backgrounds[min(backgrounds, key=lambda a: abs(a-ref))]
                transform = affine_between(luma, video.planes(frames[ref])[0], registration, registration)
                clean = cv2.warpAffine(background_image, transform, (video.width, video.height),
                                       borderMode=cv2.BORDER_REFLECT_101)[box]
                if rule.get('blend_background'):
                    bx, by, bw, bh = cv2.boundingRect(removal)
                    clean = cv2.seamlessClone(clean.copy(), image.copy(), removal*255,
                                              (bx+bw//2, by+bh//2), cv2.NORMAL_CLONE)
            else:
                clean = cv2.inpaint(image, removal, 8, cv2.INPAINT_TELEA)
            if usable:
                transform = affine_between(blank_parts[0], video.planes(frames[ref])[0], blank_mask, registration)
                size = (video.width, video.height)
                known = cv2.warpAffine(blank_mask, transform, size, flags=cv2.INTER_NEAREST)[box] == 255
                observed = cv2.warpAffine(blank_image, transform, size, borderMode=cv2.BORDER_REFLECT_101)[box]
                support = feather(known.astype(np.uint8), 7)[..., None]
                clean = np.rint(clean*(1-support)+observed*support).astype(np.uint8)
            cues.append((first, stop, rect, image, clean, feather(glyphs, 4)[..., None],
                         feather(removal, 4)[..., None]))
    return cues


def apply_cues(image, age, cues, remove=False):
    for first, stop, rect, text, background, stamp, erase in cues:
        if first <= age < stop:
            region = image[picture_slice(rect)]
            replacement, alpha = (background, erase) if remove else (text, stamp)
            region[:] = np.rint(region*(1-alpha) + replacement*alpha).astype(np.uint8)


class MotionBetween:
    """Warp observed anchor pictures with bidirectional flow on the local GPU."""

    @torch.inference_mode()
    def __init__(self, first, last, model, confidence):
        h, w = first.shape[:2]
        self.frames = torch.from_numpy(np.stack((first, last))).to(model.device).permute(0, 3, 1, 2).float()
        scale = min(0.55, 960/w)
        ph, pw = (max(64, round(v*scale/8)*8) for v in (h, w))
        small = F.interpolate(self.frames/127.5-1, (ph, pw), mode='bilinear', align_corners=False,
                              antialias=True)[None]
        flows = model.raft(small, iters=12)
        norm = self.frames.new_tensor([2/(pw-1), 2/(ph-1)]).reshape(1, 2, 1, 1)
        self.flow = F.interpolate(torch.cat(flows, dim=0)[:, 0]*norm, (h, w), mode='bilinear',
                                  align_corners=True).clamp(-0.35, 0.35)
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=model.device),
                              torch.linspace(-1, 1, w, device=model.device), indexing='ij')
        self.grid = torch.stack((x, y), dim=-1)[None]
        forward = self.grid + self.flow.permute(0, 2, 3, 1)
        cycle = self.flow + self.sample(self.flow.flip(0), forward)
        magnitude = torch.linalg.vector_norm(cycle * self.frames.new_tensor([w, h])[None, :, None, None],
                                             dim=1, keepdim=True)
        observed = torch.from_numpy(confidence).to(model.device)[:, None]
        self.consistency = (1/(1+(magnitude/18)**2)).clamp_min(0.15) * observed

    @staticmethod
    def sample(image, grid):
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)

    @torch.inference_mode()
    def picture(self, fraction):
        fraction = float(np.clip(fraction, 0, 1))
        distance = self.frames.new_tensor([fraction, 1-fraction]).reshape(2, 1, 1, 1)
        pos = self.grid - (self.flow * distance).permute(0, 2, 3, 1)
        for _ in range(2):
            pos = self.grid - (self.sample(self.flow, pos)*distance).permute(0, 2, 3, 1)
        warped = self.sample(self.frames, pos)
        weight = self.sample(self.consistency, pos)*(1-distance)
        blend = weight[0] / weight.sum(0).clamp_min(1e-6)
        disagreement = ((torch.abs(warped[0]-warped[1]).amax(0, keepdim=True)-15)/32).clamp(0, 1)
        preferred = F.avg_pool2d((blend >= 0.5).float()[None], 11, stride=1, padding=5,
                                 count_include_pad=False)[0]
        blend = blend*(1-disagreement) + preferred*disagreement
        image = warped[0]*blend + warped[1]*(1-blend)
        return image.round().clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()


def round_particles(seeds):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(seeds.astype(np.uint8), connectivity=8)
    _, _, w, h, area = stats.T
    keep = (area >= 2) & (area <= 850) & (w <= 2.5*h) & (h <= 2.5*w)
    keep[0] = False
    return keep[labels]


def refined_mask(core, guard, actual_rgb, actual_y, predicted_y):
    if core.all():
        return core
    original = actual_y.astype(np.int16)
    local = cv2.GaussianBlur(original, (0, 0), 2.2)
    # A dark dot is supported by its sweep position, shape and negative contrast.
    shadow = (original < 74) & (local-original > 11) & (predicted_y.astype(np.int16)-original > 21)
    extra = guard.astype(bool) & (wipe_palette(actual_rgb) | round_particles(shadow))
    return core | dilate(extra.astype(np.uint8), 6)


def align_prediction(prediction, actual_y, video, good, guard, strict=False):
    prior = rgb_to_yuv(prediction, video.matrix)[0]
    visible = good*(1-guard)
    if np.count_nonzero(visible) < 70000:
        return prediction
    transform = affine_between(prior, actual_y, good, visible)
    if strict and (np.abs(transform[:, 2]).max() > 12 or not np.allclose(transform[:, :2], np.eye(2), atol=0.009)):
        return prediction
    size = (video.width, video.height)
    aligned = cv2.warpAffine(prior, transform, size, borderMode=cv2.BORDER_REFLECT_101)
    keep = visible > 0
    original = actual_y[keep].astype(np.int16)
    error = lambda y: float(np.minimum(np.abs(y[keep].astype(np.int16)-original), 35).mean())
    if error(aligned) + 0.15 >= error(prior):
        return prediction
    return cv2.warpAffine(prediction, transform, size, borderMode=cv2.BORDER_REFLECT_101)


def anchor_confidence(video, track, anchors, animation_dir, jobs):
    confidence = np.ones((2, video.height, video.width), np.float32)
    if animation_dir is None:
        return confidence
    for job in jobs:
        with np.load(Path(animation_dir) / f'{job}.npz', allow_pickle=False) as data:
            if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
                raise ValueError('An anchor confidence patch belongs to another source or profile.')
            lo, hi = data['frames']
            if not any(lo <= i < hi for i in anchors):
                continue
            rect = check_rect(data['region'].tolist(), video.width, video.height)
            alpha = data['alpha']
            if alpha.shape != (hi-lo, rect[3], rect[2]) or alpha.dtype != np.uint8:
                raise ValueError('An anchor confidence patch has invalid opacity data.')
            for j, i in enumerate(anchors):
                if lo <= i < hi:
                    area = confidence[j][picture_slice(rect)]
                    # Keep a coherent source across the edge of a previously synthesized object.
                    uncertain = dilate(alpha[i-lo], 30)
                    uncertain = cv2.GaussianBlur(uncertain, (0, 0), 11)
                    np.minimum(area, 1-0.96*uncertain/255, out=area)
    return confidence


def prepare_bleed(video, frames, options):
    rect = check_rect(options['rect'], video.width, video.height)
    box = picture_slice(rect)
    refs = {}
    for age in options['references']:
        image = yuv_to_rgb(video.planes(frames[age]), video.matrix)
        if np.count_nonzero(wipe_palette(image[box])) > 25:
            raise ValueError(f'Corner reference phase {age} still contains bright wipe colors.')
        refs[age] = image
    if not refs:
        raise ValueError('The corner color correction needs a clear same-shot reference.')
    return box, refs


def corner_bleed(video, age, actual, luma, good, options, prepared):
    mask = np.zeros(luma.shape, np.uint8)
    if options and options['ages'][0] <= age < options['ages'][1]:
        box, refs = prepared
        mask[box] = dilate(wipe_palette(actual[box]).astype(np.uint8), 24)
        if mask.any():
            reference = refs[min(refs, key=lambda k: abs(k-age))]
            return mask, align_prediction(reference, luma, video, good, mask)
    return mask, None


def prepare_wipe_patch(source, base, output, profile, workdir, settings, template, job, cuda=True,
                       animation_dir=None, anchor_bottom=None):
    source, video = Video.open(source), Video.open(base)
    if (source.width, source.height, source.count, source.fps, source.matrix) != (
            video.width, video.height, video.count, video.fps, video.matrix):
        raise ValueError('Wipe repair requires an aligned base with matching frame and color formats.')
    path = new_cache(output)
    config = json.loads(Path(settings).read_text(encoding='utf-8-sig'))
    first, stop, _ = wipe_bounds(config, video)
    if job not in config['jobs']:
        raise ValueError(f'Unknown wipe job {job}.')
    spec = config['jobs'][job]
    start = spec['start']
    anchors = spec['anchors']
    bottom = dict(anchor_bottom or [])
    if (len(bottom) != len(anchor_bottom or []) or any(
            age not in anchors or type(rows) is not int or not 1 <= rows <= min(8, video.height-1)
            for age, rows in bottom.items())):
        raise ValueError('Choose one to eight bottom rows for each distinct wipe reference.')
    cut = spec.get('cut')
    cues = spec.get('cues', [])
    anchor_repairs = spec.get('anchor_repairs', [])
    bleed = spec.get('corner_bleed')
    repair_start = min(first, bleed['ages'][0]) if bleed else first
    if (len(anchors) != 2 or not 0 <= anchors[0] < anchors[1] < video.count-start
            or not 0 <= start < video.count-stop or any(target not in anchors for target, _, _ in anchor_repairs)):
        raise ValueError('Invalid clean anchor positions.')
    if cut is not None and not first < cut < stop:
        raise ValueError('Cut must be inside the covered interval.')
    if bleed and not 0 <= repair_start <= bleed['ages'][0] < bleed['ages'][1] <= stop:
        raise ValueError('Corner correction must end within the wipe patch.')
    track = load_track(source, profile, Path(workdir) / 'track.npz')
    with np.load(template, allow_pickle=False) as data:
        if (any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256'))
                or str(data['config_sha256']) != digest(settings) or list(data['ages']) != [first, stop]):
            raise ValueError('Wipe template belongs to another source or configuration.')
        masks, guards = data['masks'], data['guards']
    if masks.shape != (stop-first, video.height, video.width) or guards.shape != masks.shape:
        raise ValueError('Wipe template dimensions do not match the source.')
    references = [*anchors, *(cue[2] for zone in cues for cue in zone['ranges']),
                  *(ref for zone in cues for ref in zone.get('background_frames', [])),
                  *(ref for _, ref, _ in anchor_repairs)]
    if bleed:
        references.extend(bleed['references'])
    if min(references) < 0 or start+max(references) >= video.count:
        raise ValueError('A wipe reference frame is out of bounds.')
    frames = {i-start: frame for i, frame in read_frames(video, start+min(repair_start, *references),
                                                        start+max(stop, max(references)+1), cuda)}
    layers = capture_cues(video, frames, cues, first, masks, cut is None)
    good = np.full((video.height, video.width), 255, dtype=np.uint8)
    for zone in cues:
        good[picture_slice(zone['rect'])] = 0
    prepared = prepare_bleed(video, frames, bleed) if bleed else None
    pictures = {a: yuv_to_rgb(video.planes(frames[a]), video.matrix) for a in anchors}
    for target, ref, rect in anchor_repairs:
        part = check_rect(rect, video.width, video.height)
        box = picture_slice(part)
        observed = yuv_to_rgb(plane_rects(video.planes(frames[ref]), part), video.matrix)
        width = spec.get('anchor_feather', 0)
        if width:
            support = np.zeros((video.height, video.width), np.uint8)
            support[box] = 1
            alpha = feather(support, width)[box][..., None]
            observed = np.rint(pictures[target][box]*(1-alpha)+observed*alpha).astype(np.uint8)
        pictures[target][box] = observed
    for age, image in pictures.items():
        mask, reference = corner_bleed(video, age, image, video.planes(frames[age])[0], good, bleed, prepared)
        if reference is not None:
            alpha = feather(mask, 9)[..., None]
            image[:] = np.rint(image*(1-alpha)+reference*alpha).astype(np.uint8)
        apply_cues(image, age, layers, remove=True)
    motion = None
    if cut is None:
        trust = anchor_confidence(video, track, [start+a for a in anchors], animation_dir, spec.get('animation', []))
        for a, confidence in zip(anchors, trust, strict=True):
            for lo, hi, rect, _, _, _, erase in layers:
                if lo <= a < hi:
                    confidence[picture_slice(rect)] *= 1-0.80*erase[..., 0]
        motion = MotionBetween(*(pictures[a] for a in anchors), ProPainter(), trust)
        with torch.inference_mode():
            for i, age in enumerate(anchors):
                if age in bottom:
                    rows = bottom[age]
                    motion.frames[i, :, -rows:] = motion.frames[i, :, -rows-1:-rows]
    else:
        for age, rows in bottom.items():
            pictures[age][-rows:] = pictures[age][-rows-1:-rows]
    repair_regions = [*config.get('corner_repairs', []), *spec.get('repair_regions', [])]
    predictions, alphas = [], []
    coverage = []
    for age in range(repair_start, first):
        parts = video.planes(frames[age])
        actual = yuv_to_rgb(parts, video.matrix)
        mask, reference = corner_bleed(video, age, actual, parts[0], good, bleed, prepared)
        estimate = np.zeros_like(actual)
        if reference is not None:
            region = dilate(mask, 2).astype(bool)
            estimate[region] = reference[region]
        predictions.append(estimate)
        alphas.append(np.rint(feather(mask, 9)*255).astype(np.uint8))
        coverage.append(round(float(mask.mean()), 6))
    for age, core, guard in zip(range(first, stop), masks, guards, strict=True):
        parts = video.planes(frames[age])
        actual = yuv_to_rgb(parts, video.matrix)
        if motion is not None:
            estimate = motion.picture((age-anchors[0])/(anchors[1]-anchors[0]))
        else:
            estimate = pictures[anchors[int(age >= cut)]].copy()
        estimate = align_prediction(estimate, parts[0], video, good, guard, strict=True)
        apply_cues(estimate, age, layers)
        mask = refined_mask(core, guard, actual, parts[0], rgb_to_yuv(estimate, video.matrix)[0])
        dense = first+3 <= age < stop-3
        if dense:
            # Individual replacements in the dense dot field make a grid of mismatched
            # edges when an object moves behind the wipe. Blend the swept band as a unit.
            mask |= guard
        if mask.mean() >= 0.995:
            mask[:] = 1
        for lo, hi, rect in repair_regions:
            if lo <= age < hi:
                mask[picture_slice(check_rect(rect, video.width, video.height))] = 1
        extra, reference = corner_bleed(video, age, actual, parts[0], good, bleed, prepared)
        if reference is not None:
            missing = (extra > 0) & (mask == 0)
            estimate[missing] = reference[missing]
            mask |= extra
        alpha = np.rint(feather(mask, 12 if dense else 5)*255).astype(np.uint8)
        predictions.append(estimate)
        alphas.append(alpha)
        coverage.append(round(float(mask.mean()), 6))
    save_npz(path, region=np.asarray([0, 0, video.width, video.height]), frames=np.asarray([start+repair_start, start+stop]),
             prediction=np.stack(predictions), alpha=np.stack(alphas), coverage=np.asarray(coverage),
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'],
             config_sha256=digest(settings), anchor_bottom=np.asarray(sorted(bottom.items()), dtype=int).reshape(-1, 2))
    print(json.dumps(dict(job=job, active=[start+repair_start, start+stop], anchor_frames=[start+a for a in anchors],
                          inferred_cut=None if cut is None else start+cut, average_repaired_fraction=round(float(np.mean(coverage)), 5),
                          saved_patch=str(path)), indent=2), flush=True)
