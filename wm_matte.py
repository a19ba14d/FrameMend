from pathlib import Path

import cv2
import numpy as np
import torch

from wm_media import clipped_plane, get_frame, plane_rects, read_frames
from wm_track import check_rect, corner_activity, extra_tracks, frame_variants, motion_padding, save_npz


def logo_rect(profile, track, index):
    x, y = map(int, track['positions'][index])
    if x < 0:
        return None
    settings = profile['translucent']
    dx, dy = motion_padding(settings)
    return (x-dx, y-dy, *settings['reference_rect'][2:])


def phase_offsets(profile, track):
    settings = profile['translucent']
    precise = track['precise']
    step = settings.get('phase_step', 0)
    if step:
        precise = np.rint((precise + settings.get('phase_bias', 0)) / step) * step
    return (precise - track['positions']).astype(np.float32)


def exclusion(video, profile, track, index):
    mask = np.full((video.height, video.width), 255, dtype=np.uint8)
    rect = logo_rect(profile, track, index)
    if rect:
        x, y, w, h = rect
        mask[max(0, y):max(0, y+h), max(0, x):max(0, x+w)] = 0
    corners = profile.get('corners')
    if corners:
        active = corner_activity(corners, track['corners'][index])
        for flag, points in zip(active, corners['masks'], strict=True):
            if flag:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 0)
    return cv2.erode(mask, np.ones((9, 9), np.uint8))


def affine_between(reference, target, reference_mask, target_mask):
    shape = (reference.shape[1]//2, reference.shape[0]//2)
    a, b, ma, mb = [cv2.resize(p, shape, interpolation=cv2.INTER_AREA)
                    for p in (reference, target, reference_mask, target_mask)]
    mask = (cv2.bitwise_and(ma, mb) == 255).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((19, 19), np.uint8))
    points = cv2.goodFeaturesToTrack(a, 400, 0.01, 9, mask=mask)
    identity = np.eye(2, 3, dtype=np.float32)
    if points is None or len(points) < 12:
        return identity
    moved, good, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(19, 19), maxLevel=2)
    good = good.ravel().astype(bool)
    if good.sum() < 12:
        return identity
    transform, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], ransacReprojThreshold=1.0)
    if transform is None or inliers.mean() < 0.43 or np.abs(transform[:, 2]).max() > 30:
        return identity
    transform[:, 2] *= 2
    return transform


def training_pair(video, profile, track, target_index, reference_index, target, reference, weight):
    rect = logo_rect(profile, track, target_index)
    x, y, width, height = rect
    source = video.planes(target)
    clean = video.planes(reference)
    good = exclusion(video, profile, track, reference_index)
    here = exclusion(video, profile, track, target_index)
    transform = affine_between(clean[0], source[0], good, here)
    full = cv2.warpAffine(clean[0], transform, (video.width, video.height))
    overlap = (good == 255) & (here == 255)
    if np.mean(np.abs(full[overlap].astype(np.float32) - source[0][overlap])) > np.mean(
            np.abs(clean[0][overlap].astype(np.float32) - source[0][overlap])):
        transform = np.eye(2, 3)

    mask = cv2.warpAffine(good, transform, (video.width, video.height))
    valid = mask[y:y+height, x:x+width] == 255
    # Exclude other foreground logos in the target, including an overlapping corner.
    corners = profile.get('corners')
    if corners:
        blocked = np.zeros((height, width), dtype=np.uint8)
        active = corner_activity(corners, track['corners'][target_index])
        for flag, polygon in zip(active, corners['masks'], strict=True):
            if flag:
                cv2.fillPoly(blocked, [np.asarray(polygon, np.int32) - (x, y)], 1)
        valid &= ~blocked.astype(bool)
    if not valid.any():
        return None
    parts = plane_rects(source, rect)
    predicted, weights = [], []
    for plane, part, s in zip(clean, parts, (1, 2, 2)):
        matrix = transform.copy()
        matrix[:, 2] = (matrix[:, 2] - (x, y)) / s
        patch = cv2.warpAffine(plane, matrix, (width//s, height//s)).astype(np.float32)
        sharp = np.abs(patch - cv2.GaussianBlur(patch, (0, 0), 2))
        confidence = weight / (1 + (sharp / 7) ** 2)
        keep = valid if s == 1 else cv2.resize(valid.astype(np.float32), (part.shape[1], part.shape[0]),
                                              interpolation=cv2.INTER_AREA) == 1
        predicted.append(patch)
        weights.append(confidence * keep)
    return tuple(p.copy() for p in parts), predicted, weights


def solve_affine(rows, column, device, prior=None):
    observed, clean, confidence = (torch.from_numpy(np.stack([r[j][column] for r in rows]))
                                   .to(device, dtype=torch.float32) for j in range(3))
    if prior is None:
        prior = torch.full_like(clean[0], 0.9)
        strength = 9.0
    else:
        strength = 65.0
    weights = confidence
    for _ in range(4):
        total = weights.sum(0).clamp_min(0.0001)
        mean_x = (weights * clean).sum(0) / total
        mean_y = (weights * observed).sum(0) / total
        delta = clean - mean_x
        var = (weights * delta.square()).sum(0)
        cov = (weights * delta * (observed - mean_y)).sum(0)
        gain = ((cov + prior * strength) / (var + strength)).clamp(0.18, 1.04)
        bias = mean_y - gain * mean_x
        residual = (observed - gain * clean - bias).abs()
        weights = confidence / (1 + (residual / (4.0 if column == 0 else 2.5)).square())
    known = confidence.sum(0) > 0.2
    gain = torch.where(known, gain, 1)
    bias = torch.where(known, bias, 0)
    return gain.cpu().numpy(), bias.cpu().numpy()


def capsule(width, height, margin):
    x0, y0, x1, y1 = margin
    a = np.zeros((height, width), dtype=np.uint8)
    radius = (height - y0 - y1) // 2
    left, right = x0 + radius, width - x1 - radius - 1
    if radius <= 0 or left > right:
        raise ValueError('Invalid capsule margins.')
    cv2.rectangle(a, (left, y0), (right, height-y1-1), 1, -1)
    for x in (left, right):
        cv2.circle(a, (x, y0+radius), radius, 1, -1)
    return a


def save_calibrations(output, models, variants, track, stationary=(), additional=()):
    arrays = {}
    if len(models) == 1:
        arrays = {f'fields{j}': values for j, values in enumerate(models[0])}
    elif models:
        arrays = {f'fields{j}': np.stack(values) for j, values in enumerate(zip(*models, strict=True))}
        arrays['variants'] = variants
    arrays.update({f'stationary{i}': alpha for i, alpha in enumerate(stationary)})
    for i, model in enumerate(additional, 1):
        arrays.update({f'extra{i}_fields{j}': values for j, values in enumerate(model)})
    save_npz(output, **arrays, source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'])


def calibrate(video, profile, track, output, cuda=True, extra_pairs=None, radius=16):
    if Path(output).exists():
        raise ValueError('Choose a new calibration file.')
    settings = profile.get('translucent', {})
    if settings.get('white_plate') or profile.get('stationary'):
        from wm_white_matte import calibrate_white
        if extra_pairs or (settings and not settings.get('white_plate')):
            raise ValueError('White plate calibration requires a white moving plate or only stationary layers.')
        calibrate_white(video, profile, track, output, cuda)
        return
    if not settings:
        raise ValueError('The profile has no translucent overlay.')
    if templates := settings.get('flat_templates'):
        from wm_flat_matte import calibrate_flat
        if extra_pairs or settings.get('calibration_variants'):
            raise ValueError('Choose either flat templates or same-shot calibration pairs.')
        calibrate_flat(video, profile, track, output, templates, cuda)
        return
    _, _, width, height = check_rect(settings['reference_rect'], video.width, video.height)
    positions = track['positions'].astype(np.int32)
    present = positions[:, 0] >= 0
    event = settings.get('calibration_events', 'jumps')
    if event == 'visibility':
        candidates = present[1:] != present[:-1]
    elif event == 'jumps':
        movement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        candidates = (movement > width * 0.22) & present[1:] & present[:-1]
    else:
        raise ValueError('Calibration events must be jumps or visibility.')
    changes = np.flatnonzero(candidates & (track['scene_scores'][1:] < 13)) + 1
    alternate_ranges = settings.get('calibration_variants', [])
    variants = frame_variants(video.count, alternate_ranges)
    rows = [[] for _ in range(len(alternate_ranges)+1)]
    phases = [[] for _ in rows]
    shift = phase_offsets(profile, track)

    def add(a, b, target, reference, weight):
        row = training_pair(video, profile, track, a, b, target, reference, weight)
        if row is not None:
            rows[variants[a]].append(row)
            phases[variants[a]].append(shift[a])

    for j in changes:
        frames = dict(read_frames(video, max(0, j-radius), min(video.count, j+radius), cuda))
        for a, target in frames.items():
            b = j if a < j else j-1
            gap = abs(a - b)
            if (present[a] and (event != 'visibility' or not present[b])
                    and track['scene_scores'][min(a, b)+1:max(a, b)+1].max() < 18):
                add(a, b, target, frames[b], 1 / (1 + gap * 0.18))
        print(f'Collected calibration samples near frame {j}. Total {sum(map(len, rows))}.', flush=True)
    for a, b, weight in extra_pairs or []:
        if a == b or logo_rect(profile, track, a) is None or weight <= 0:
            raise ValueError(f'Invalid calibration frame pair {(a, b, weight)}.')
        add(a, b, get_frame(video, a, cuda), get_frame(video, b, cuda), weight)
    if any(len(group) < 3 for group in rows):
        raise ValueError('Each template needs matching clean frame pairs. Supply --pairs with same-shot frame pairs.')
    fit_mattes(rows, phases, profile, track, output, width, height, variants)


def sampling_grid(shifts, width, height, scale):
    device = shifts.device
    x = torch.arange(width // scale, device=device) * scale + (scale - 1) / 2
    y = torch.arange(height // scale, device=device) * scale + (scale - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xx, yy), dim=-1)[None] - shifts[:, None, None, :]
    return grid * grid.new_tensor((2/(width-1), 2/(height-1))) - 1


def warp_fields(fields, grids):
    return torch.nn.functional.grid_sample(fields[None].expand(len(grids), -1, -1, -1), grids,
                                           align_corners=True, padding_mode='zeros')


def initial_fields(rows, phases, width, height, device):
    aligned = []
    for row, (dx, dy) in zip(rows, phases, strict=True):
        aligned.append([[cv2.warpAffine(p, np.float32([[1, 0, -dx/s], [0, 1, -dy/s]]),
                                       (width//s, height//s), borderMode=cv2.BORDER_REPLICATE)
                         for p, s in zip(parts, (1, 2, 2), strict=True)] for parts in row])
    gain, bias = solve_affine(aligned, 0, device)
    chroma = cv2.resize(gain, (width//2, height//2), interpolation=cv2.INTER_AREA)
    values = [(gain, bias)] + [solve_affine(aligned, j, device, torch.from_numpy(chroma).to(device)) for j in (1, 2)]
    return np.stack([np.stack((cv2.resize(g - 1, (width, height), interpolation=cv2.INTER_LINEAR),
                              cv2.resize(b / 128 + g - 1, (width, height), interpolation=cv2.INTER_LINEAR)))
                     for g, b in values])


def fit_matte(rows, phases, profile, track, output, width, height):
    fit_mattes([rows], [phases], profile, track, output, width, height, None)


def fit_mattes(rows, phases, profile, track, output, width, height, variants):
    margins = profile['translucent'].get('capsule_margins')
    support = capsule(width, height, margins) if margins else np.ones((height, width), np.uint8)
    models = [fit_fields(group, np.asarray(offsets), support) for group, offsets in zip(rows, phases, strict=True)]
    save_calibrations(output, models, variants, track)
    print(f'Calibrated using {sum(map(len, rows))} clean frame pairs. Saved {output}', flush=True)


def fit_fields(rows, phases, support, seed=None):
    if not torch.cuda.is_available():
        raise ValueError('CUDA PyTorch is required for matte fitting.')
    device = torch.device('cuda')
    height, width = support.shape
    support = torch.from_numpy(support).to(device).float()
    initial = initial_fields(rows, phases, width, height, device) if seed is None else seed
    shifts = torch.from_numpy(phases).to(device)
    rng = np.random.default_rng(1459)
    result = []
    for j, scale in enumerate((1, 2, 2)):
        observed, clean, weight = (torch.from_numpy(np.stack([r[k][j] for r in rows]))
                                   .to(device, dtype=torch.float32) for k in range(3))
        correction = (observed - clean) / 128
        background = (clean - 128) / 128
        grids = sampling_grid(shifts, width, height, scale)
        prior = torch.from_numpy(initial[j]).to(device)
        fields = torch.nn.Parameter(prior.clone())
        optimizer = torch.optim.Adam([fields], lr=0.009)
        batch_size = min(len(rows), 20)
        for i in range(340):
            selected = torch.as_tensor(rng.choice(len(rows), batch_size, replace=False), device=device)
            sample = warp_fields(fields * support, grids[selected])
            predicted = sample[:, 0] * background[selected] + sample[:, 1]
            error = torch.nn.functional.smooth_l1_loss(predicted, correction[selected], reduction='none', beta=0.025)
            loss = (error * weight[selected]).sum() / weight[selected].sum().clamp_min(1)
            if j:
                loss += 0.035 * (fields[0] - luma).square().mean()
            if seed is not None:
                loss += 0.1 * (fields - prior).square().mean()
            loss += 0.003 * (fields[:, 1:] - fields[:, :-1]).abs().mean()
            loss += 0.003 * (fields[:, :, 1:] - fields[:, :, :-1]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                fields[0].clamp_(-0.80, 0.04)
                fields[1].clamp_(-0.65, 0.65)
            if i in (160, 270):
                optimizer.param_groups[0]['lr'] *= 0.5
        values = fields.detach() * support
        if j == 0:
            luma = values[0].clone()
        result.append(values.cpu().numpy())
        print(f'Plane {j} calibrated on CUDA.', flush=True)
    return result


class Matte:
    def __init__(self, profile, track, path, clean_edges=False, narrow_edges=False, align_white=False, *,
                 prefix='', white_flat=False):
        self.profile, self.track = profile, track
        self.models = []
        self.fixed = None
        self.white_alignment = []
        self.white_alignment_changes = []
        self.white_flat = white_flat
        self.extra = [Matte(layer, points, path, clean_edges, narrow_edges, align_white, prefix=key, white_flat=white_flat)
                      for layer, points, key in extra_tracks(profile, track)]
        with np.load(path, allow_pickle=False) as saved:
            for key in ('source_sha256', 'profile_sha256'):
                if str(saved[key]) != str(track[key]):
                    raise ValueError('This calibration belongs to another source or profile.')
            if layers := profile.get('stationary'):
                from wm_white_matte import FixedWhiteMatte
                self.fixed = FixedWhiteMatte(layers, [saved[f'stationary{i}'] for i in range(len(layers))],
                                            len(track['positions']))
            if not profile.get('translucent'):
                return
            fields = [saved[f'{prefix}fields{j}'] for j in range(3)]
            variants_key = prefix+'variants'
            self.variants = saved[variants_key] if variants_key in saved else np.zeros(len(track['positions']), np.int16)
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise ValueError('CUDA PyTorch is required for inverse compositing.')
        self.shifts = phase_offsets(profile, track)
        self.width, self.height = profile['translucent']['reference_rect'][2:]
        fields = [p[None] if p.ndim == 3 else p for p in fields]
        if align_white:
            from wm_white_matte import WhiteEdgeAlignment
            self.white_alignment = [WhiteEdgeAlignment(-variant[0]) for variant in fields[0]]
        self.models = [torch.from_numpy(f).to(self.device) for f in fields]
        self.fields = [f[0] for f in self.models]
        self.edge_masks = []
        self.edge_shapes = []
        self.opacity = np.max(-fields[0][:, 0], axis=0)
        self.foreground_shape = (self.opacity > 0.30).astype(np.uint8)
        if clean_edges:
            # Codec ringing is concentrated where the foreground opacity or color changes sharply.
            size = 3 if narrow_edges else 5
            luma_limit = 7.0 if narrow_edges else 5.0
            chroma_limit = 5.0 if narrow_edges else 3.4
            luma_edges = np.maximum.reduce([edge_mask(f, luma_limit, size) for f in fields[0]])
            self.edge_shapes = [luma_edges] + [
                np.maximum(luma_edges, np.maximum.reduce([edge_mask(p, chroma_limit, size) for p in f]))
                for f in fields[1:]]
            self.edge_masks = [torch.from_numpy(m).float().to(self.device)[None] for m in self.edge_shapes]

    @torch.inference_mode()
    def restore(self, video, frames, start, *, spatial=True):
        if self.models:
            self._restore_moving(video, frames, start, spatial)
        if self.fixed:
            for i, frame in enumerate(frames, start):
                self.fixed.restore(i, video.planes(frame))
        for layer in self.extra:
            layer.restore(video, frames, start, spatial=spatial)

    def _restore_moving(self, video, frames, start, spatial):
        patches, targets, offsets, variants = [], [], [], []
        for i, frame in enumerate(frames, start):
            rect = logo_rect(self.profile, self.track, i)
            if rect:
                planes = video.planes(frame)
                shift = self.shifts[i]
                if self.white_alignment and self.track['scores'][i] < .72:
                    aligned = self.white_alignment[self.variants[i]].align(planes[0], rect, shift)
                    if aligned is not None:
                        self.white_alignment_changes.append((i, float(np.linalg.norm(aligned-shift))))
                        shift = aligned
                if 0 <= rect[0] <= video.width-self.width and 0 <= rect[1] <= video.height-self.height:
                    patch = plane_rects(planes, rect)
                    targets.append([(plane, np.s_[:, :]) for plane in patch])
                else:
                    patch, views = [], []
                    for plane, scale, fill in zip(planes, (1, 2, 2), (16, 128, 128), strict=True):
                        view, section = clipped_plane(plane, tuple(value//scale for value in rect))
                        padded = np.full((self.height//scale, self.width//scale), fill, np.uint8)
                        padded[section] = view
                        patch.append(padded)
                        views.append((view, section))
                    targets.append(views)
                patches.append(patch)
                offsets.append(shift)
                variants.append(self.variants[i])
        if not patches:
            return
        shifts = torch.from_numpy(np.asarray(offsets)).to(self.device)
        selection = torch.as_tensor(variants, dtype=torch.long, device=self.device) if len(self.models[0]) > 1 else None
        for j, column in enumerate(zip(*patches, strict=True)):
            source = torch.from_numpy(np.stack(column)).to(self.device).float()
            grid = sampling_grid(shifts, self.width, self.height, 1 if j == 0 else 2)
            if selection is None:
                corrected = warp_fields(self.fields[j], grid)
            else:
                corrected = torch.nn.functional.grid_sample(self.models[j][selection], grid, align_corners=True)
            d, k = corrected.unbind(dim=1)
            clean = ((source - 128 * (1 + k)) / (1 + d) + 128).round_().clamp_(0, 255).byte().cpu().numpy()
            if spatial and self.edge_masks:
                masks = (warp_fields(self.edge_masks[j], grid)[:, 0] > 0.20).byte().cpu().numpy()
                support = (d < -.023).byte().cpu().numpy() if self.white_flat else [None]*len(clean)
                for target, src, mask, body in zip(targets, clean, masks, support, strict=True):
                    dst, section = target[j]
                    result = repair_ringing(src, mask, j)
                    if body is not None:
                        result = repair_flat_white(result, body, section)
                    dst[:] = result[section]
            else:
                for target, src in zip(targets, clean, strict=True):
                    dst, section = target[j]
                    dst[:] = src[section]


def edge_mask(fields, threshold, size=5):
    k3 = np.ones((3, 3), np.uint8)
    strength = np.maximum.reduce([cv2.morphologyEx(p, cv2.MORPH_GRADIENT, k3)
                                  for p in (fields[0] * 68, fields[1] * 128)])
    edge = (strength > threshold).astype(np.uint8)
    return cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def repair_ringing(source, mask, channel):
    prediction = cv2.inpaint(source, mask, 3, cv2.INPAINT_NS)
    delta = prediction.astype(np.int16) - source
    # Keep sharp scene edges and captions when local interpolation disagrees strongly.
    limit = 22 if channel == 0 else 14
    strength = np.clip((limit * 2.5 - np.abs(delta)) / (limit * 1.5), 0.10, 1)
    return np.clip(np.rint(source + delta * strength), 0, 255).astype(np.uint8)


def repair_flat_white(source, support, section):
    mask = cv2.dilate(support, np.ones((3, 3), np.uint8))
    valid = np.zeros(source.shape, np.float32)
    valid[section] = 1
    clean = (1-mask)*valid
    values = source.astype(np.float32)
    mass = cv2.boxFilter(clean, -1, (15, 15))
    safe = np.maximum(.04, mass)
    average = cv2.boxFilter(clean*values, -1, (15, 15))/safe
    variance = cv2.boxFilter(clean*values**2, -1, (15, 15))/safe - average**2
    delta = average-values
    weight = mask * np.clip((35-variance)/25, 0, 1) * np.clip((mass-.10)*5, 0, 1)
    weight *= np.clip((19-np.abs(delta))/9, 0, 1)
    return np.clip(np.rint(values+weight*delta), 0, 255).astype(np.uint8)
