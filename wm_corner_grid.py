"""Suppress the stationary model grid inside reconstructed corners."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wm_inpaint import FEATURE_GRID_STRIDE, model_size
from wm_media import Video, encode, plane_rects, read_frames


HARMONICS = (
    (0, 1), (1, 0), (-1, 1), (1, 1),
    (0, 2), (2, 0), (-2, 1), (2, 1), (-1, 2), (1, 2), (-2, 2), (2, 2),
    (0, 3), (3, 0), (-3, 1), (3, 1), (-1, 3), (1, 3),
)


def lowpass(planes, rows, columns):
    half_x = columns//2
    half_y = rows//2
    horizontal = F.avg_pool2d(F.pad(planes, (half_x, half_x, 0, 0), mode='replicate'),
                              (1, columns), stride=1)
    return F.avg_pool2d(F.pad(horizontal, (0, 0, half_y, half_y), mode='replicate'),
                        (rows, 1), stride=1)


class GridPlane:
    def __init__(self, shape, periods, device):
        self.height, self.width = shape
        self.background = tuple(2*round(period/2)+1 for period in periods)
        rows, columns = self.tile = tuple(max(8, 2*round(2.5*period)) for period in periods)
        self.step = rows//2, columns//2
        pad_top, pad_left = self.step
        pad_bottom = pad_top + (-self.height % pad_top)
        pad_right = pad_left + (-self.width % pad_left)
        self.padding = (pad_left, pad_right, pad_top, pad_bottom)
        self.padded_shape = (self.height+pad_top+pad_bottom, self.width+pad_left+pad_right)
        self.window = torch.hann_window(rows, device=device)[:, None] * torch.hann_window(columns, device=device)
        self.window_mass = self.window.sum()
        window_tiles = self.tiles(torch.ones(1, 1, *shape, device=device))
        coverage = self.fold(torch.ones_like(window_tiles))
        self.coverage = coverage.clamp_min_(0.001)

        bins = [(5*y, 5*x) for y, x in HARMONICS if 5*abs(y)+2 < rows/2 and 5*x+2 < columns/2]
        y, x = np.asarray(bins, dtype=np.int64).reshape(-1, 2).T
        self.iy = torch.as_tensor(y % rows, device=device)
        self.ix = torch.as_tensor(x, device=device)
        self.vertical = self.ix == 0
        neighbor_y, neighbor_x = np.asarray([(-2, -2), (-2, 0), (-2, 2), (0, -2),
                                               (0, 2), (2, -2), (2, 0), (2, 2)]).T
        near_y, near_x = y[:, None]+neighbor_y, x[:, None]+neighbor_x
        near_y = np.where(near_x < 0, -near_y, near_y)
        self.near_y = torch.as_tensor(near_y % rows, device=device)
        self.near_x = torch.as_tensor(abs(near_x), device=device)

    def tiles(self, data):
        patches = F.unfold(F.pad(data, self.padding), self.tile, stride=self.step)
        return patches.transpose(1, 2).reshape(data.shape[0], -1, *self.tile)

    def fold(self, tiles):
        patches = (tiles*self.window).flatten(2).transpose(1, 2)
        padded = F.fold(patches, self.padded_shape, self.tile, stride=self.step)
        top, left = self.step
        return padded[:, :, top:top+self.height, left:left+self.width]

    def coefficients(self, samples, weight):
        support = self.tiles(weight)*self.window
        mass = support.sum(dim=(-2, -1))
        observed = (mass >= 0.21*self.window_mass).float()
        high = samples-lowpass(samples, *self.background)
        spectrum = torch.fft.rfft2(self.tiles(high)*support)
        exact = spectrum[..., self.iy, self.ix] / mass.clamp_min(1)[..., None]
        noise = spectrum[..., self.near_y, self.near_x].abs().median(dim=-1).values / mass.clamp_min(1)[..., None]
        return exact, noise, observed

    def phase_statistics(self, samples, weight, labels):
        exact, noise, observed = self.coefficients(samples, weight)
        amplitude = exact.abs()
        quality = ((amplitude-1.1*noise-.12)/(amplitude+.40)).clamp_(0, 1)
        unit = exact/(amplitude+.15)*quality*observed[..., None]
        total = (labels.to(exact.dtype) @ unit.flatten(1)).reshape(2, *exact.shape[1:])
        return total, labels @ observed

    def residual(self, samples, weight, temporal, floor, prior):
        if not self.iy.numel():
            return torch.zeros_like(samples)
        exact, noise, observed = self.coefficients(samples, weight)
        norm = (temporal @ observed).clamp_min(1)[..., None]
        measured = exact*observed[..., None]
        stable = (temporal.to(exact.dtype) @ measured.flatten(1)).reshape_as(exact)/norm
        absolute = (temporal @ measured.abs().flatten(1)).reshape_as(noise)/norm
        nearby = (temporal @ (noise*observed[..., None]).flatten(1)).reshape_as(noise)/norm

        amplitude = stable.abs()
        coherence = (amplitude / absolute.clamp_min(.001) - .46).div(.39).clamp_(0, 1)
        contrast = (amplitude - 1.15*nearby - floor).div(.43 + .65*nearby).clamp_(0, 1)
        phase = (exact*stable.conj()).real / (exact.abs()*amplitude).clamp_min(.001)
        same_phase = ((phase-.06)/.78).clamp_(0, 1)
        estimate = (.68*exact + .32*stable)*coherence*contrast*same_phase*observed[..., None]
        if prior is not None:
            certainty = ((prior.abs()-.05)/.13).clamp_(0, 1)
            alignment = (stable*prior.conj()).real/(amplitude*prior.abs()).clamp_min(.001)
            estimate *= .24 + .76*certainty*((alignment-.08)/.82).clamp_(0, 1)
        rows, columns = self.tile
        replacement = exact.new_zeros((*exact.shape[:-1], rows, columns//2+1))
        scaled = estimate*(self.tile[0]*self.tile[1])
        replacement[..., self.iy, self.ix] = scaled
        replacement[..., -self.iy[self.vertical], 0] = scaled[..., self.vertical].conj()
        artifact = torch.fft.irfft2(replacement, s=self.tile)
        return self.fold(artifact)/self.coverage


class CornerGrid:
    radius = 3

    def __init__(self, shape, context, scale=.625, strength=1.0):
        if not torch.cuda.is_available():
            raise ValueError('CUDA PyTorch is required for corner grid suppression.')
        if not .25 <= scale <= 1 or not 0 < strength <= 1.5:
            raise ValueError('Invalid grid model scale or strength.')
        self.device = torch.device('cuda')
        self.strength = strength
        processed = model_size(context, scale)
        period = tuple(FEATURE_GRID_STRIDE*a/b for a, b in zip(context, processed, strict=True))
        chroma_shape = tuple(p//2 for p in shape)
        self.luma = GridPlane(shape, period, self.device)
        self.chroma = GridPlane(chroma_shape, tuple(p/2 for p in period), self.device)
        self.totals = [None, None, None]
        self.priors = [None, None, None]

    @torch.inference_mode()
    def calibrate(self, parts, mask, generated):
        generated = torch.as_tensor(generated, device=self.device, dtype=torch.bool)
        labels = torch.stack((~generated, generated)).float()
        blend = torch.as_tensor(mask, device=self.device, dtype=torch.float32)[:, None]
        chroma_blend = F.avg_pool2d(blend, 2)
        for i, values in enumerate(parts):
            grid, weight = (self.luma, blend) if i == 0 else (self.chroma, chroma_blend)
            samples = torch.as_tensor(values, device=self.device, dtype=torch.float32)[:, None]
            stats = grid.phase_statistics(samples, weight, labels)
            if self.totals[i] is not None:
                stats = tuple(a+b for a, b in zip(self.totals[i], stats, strict=True))
            self.totals[i] = stats

    def finish_calibration(self):
        for i, (sums, counts) in enumerate(self.totals):
            mean = sums/counts.clamp_min(1)[..., None]
            coverage = (counts.min(dim=0).values/12).clamp_(0, 1)
            self.priors[i] = (mean[1]-mean[0])*coverage[..., None]
        self.totals = [None, None, None]

    @torch.inference_mode()
    def restore(self, parts, mask, groups):
        temporal = torch.as_tensor(groups, device=self.device)
        same_shot = temporal[:, None] == temporal[None]
        order = torch.arange(len(groups), device=self.device)
        temporal = (self.radius+1-(order[:, None]-order).abs()).clamp_min(0).float()*same_shot
        blend = torch.as_tensor(mask, device=self.device, dtype=torch.float32)[:, None]
        luma = torch.as_tensor(parts[0], device=self.device, dtype=torch.float32)[:, None]
        smooth = lowpass(luma, 7, 7)
        gx = smooth[..., 2:]-smooth[..., :-2]
        gy = smooth[:, :, 2:]-smooth[:, :, :-2]
        gx = F.pad(gx, (1, 1, 0, 0), mode='replicate')/2 if gx.numel() else torch.zeros_like(smooth)
        gy = F.pad(gy, (0, 0, 1, 1), mode='replicate')/2 if gy.numel() else torch.zeros_like(smooth)
        edges = torch.hypot(gx, gy)
        nearby_edges = F.max_pool2d(edges, 17, stride=1, padding=8)
        protect = (1-(torch.maximum(edges, .72*nearby_edges)-2.5)/8).clamp_(.06, 1)
        chroma_blend = -F.max_pool2d(-blend, 2)
        chroma_protect = F.avg_pool2d(protect, 2)
        result = []
        for i, values in enumerate(parts):
            grid, weight, guard = (self.luma, blend, protect) if i == 0 else (self.chroma, chroma_blend, chroma_protect)
            samples = luma if i == 0 else torch.as_tensor(values, device=self.device, dtype=torch.float32)[:, None]
            residual = grid.residual(samples, weight, temporal, .16 if i == 0 else .13, self.priors[i])
            cap = 16 if i == 0 else 9
            correction = (residual*self.strength).clamp_(-cap, cap)*weight*guard
            corrected = (samples-correction).round_().clamp_(0, 255).byte()
            result.append(corrected[:, 0].cpu().numpy())
        return tuple(result)


def corner_planes(video, items, rect):
    cropped = [plane_rects(video.planes(frame), rect) for _, frame in items]
    return tuple(np.stack(planes) for planes in zip(*cropped, strict=True))


def calibrate_grids(video, jobs, grids, cuda, start=0):
    sample_ids = set(np.linspace(start, video.count-1, min(video.count-start, 192), dtype=int))
    frames = ((i, frame) for i, frame in read_frames(video, start, cuda=cuda) if i in sample_ids)
    while batch := list(itertools.islice(frames, 12)):
        indices = [i for i, _ in batch]
        for job, grid in zip(jobs, grids, strict=True):
            planes = corner_planes(video, batch, job.patch_rect)
            weight = np.broadcast_to(job.alpha, (len(batch), *job.alpha.shape)).copy()
            grid.calibrate(planes, weight, job.active[indices])
    for grid in grids:
        grid.finish_calibration()
    print(f'Calibrated stationary corner grid on {len(sample_ids)} spread-out frames.', flush=True)


def protected_frames(video, track, first, patches):
    if not 0 <= first < video.count:
        raise ValueError('Opening protection must leave at least one frame available for grid cleanup.')
    protected = np.arange(video.count) < first
    for path in patches:
        with np.load(path, allow_pickle=False) as data:
            if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
                raise ValueError('Protection patch belongs to a different source or profile.')
            start, stop = data['frames'].tolist()
            alpha = data['alpha']
            if not 0 <= start < stop <= video.count or alpha.ndim != 3 or alpha.shape[0] != stop-start:
                raise ValueError('Invalid protection patch timing.')
            protected[start:stop] |= alpha.any(axis=(1, 2))
    return protected


def filtered_frames(video, jobs, grids, track, protected, start, stop, cuda, statistics):
    from wm_caption_inpaint import subtitle_guard

    radius, step = CornerGrid.radius, 12
    decode_start, decode_stop = max(0, start-radius), min(video.count, stop+radius)
    frames = iter(read_frames(video, decode_start, decode_stop, cuda))
    last = decode_start
    buffer = []
    numbers = np.arange(video.count)
    shots = np.searchsorted(track['cuts'], numbers, side='right')
    active = [job.active & ~protected for job in jobs]
    groups = [shots + np.cumsum(np.r_[0, valid[1:] != valid[:-1]]) for valid in active]

    for lo in range(start, stop, step):
        hi = min(lo+step, stop)
        need = min(decode_stop, hi+radius)
        buffer.extend(itertools.islice(frames, need-last))
        last = need
        ids = np.asarray([i for i, _ in buffer])
        center = np.flatnonzero((ids >= lo) & (ids < hi))
        if len(center) != hi-lo:
            raise RuntimeError(f'Grid filter stopped near frame {lo}.')
        chosen = np.any([valid[ids[center]] for valid in active], axis=0)
        rendered = [buffer[j][1].copy() for j in center]
        changed = np.zeros(len(center), dtype=bool)

        for job, grid, valid, scene in zip(jobs, grids, active, groups, strict=True):
            selected = np.flatnonzero(valid[ids[center]])
            if not selected.size:
                continue
            parts = corner_planes(video, buffer, job.patch_rect)
            y = job.patch_rect[1]
            weight = valid[ids, None, None] * job.alpha
            for luma, mask in zip(parts[0], weight, strict=True):
                if mask.any():
                    mask[subtitle_guard(luma, y, (y+100, y+len(luma)), 15)] = 0
            repaired = grid.restore(parts, weight, scene[ids])
            for k in selected:
                at = center[k]
                destination = plane_rects(video.planes(rendered[k]), job.patch_rect)
                for j, (plane, result) in enumerate(zip(destination, repaired, strict=True)):
                    difference = result[at] != plane
                    changed[k] |= difference.any()
                    statistics['changed_yuv_samples'][j] += int(difference.sum())
                    plane[:] = result[at]

        statistics['frames_filtered'] += int(chosen.sum())
        statistics['frames_changed'] += int(changed.sum())
        yield from zip(range(lo, hi), rendered, strict=True)
        buffer = [(i, frame) for i, frame in buffer if i >= hi-radius]
    if next(frames, None) is not None:
        raise RuntimeError('The grid decoder returned unexpected trailing frames.')


def degrid_video(args, start, stop):
    from wm_animation import AnimationPatch
    from wm_repair import CornerJob
    from wm_inpaint import corner_jobs
    from wm_retouch import Retouch
    from wm_track import load_profile, load_track

    source, base = Video.open(args.input), Video.open(args.base)
    if (source.width, source.height, source.fps, source.count, source.offset, source.matrix) != (
            base.width, base.height, base.fps, base.count, base.offset, base.matrix):
        raise ValueError('Grid cleanup requires an aligned base with matching color format.')
    if (args.output.suffix.lower() != '.mp4' or args.output.resolve() in (source.path, base.path)
            or args.output.exists()):
        raise ValueError('Choose a new MP4 output path.')
    if args.report and args.report.resolve() in (source.path, base.path, args.output.resolve()):
        raise ValueError('Choose a separate grid report path.')
    if args.dilation < 5:
        raise ValueError('Grid cleanup uses the corner repair dilation of at least five pixels.')
    profile = load_profile(source, args.profile)
    track = load_track(source, args.profile, Path(args.workdir)/'track.npz')
    jobs = [CornerJob(*job, args.dilation) for job in corner_jobs(source, profile, track)]
    if not jobs:
        raise ValueError('Grid cleanup needs corner regions in the watermark profile.')
    protected = protected_frames(source, track, args.protect_first, args.protect_patch)
    grids = [CornerGrid(job.alpha.shape, job.mask.shape, args.model_scale, args.strength) for job in jobs]
    patches = [AnimationPatch(base, track, path) for path in args.animation_patch]
    retouch = Retouch(base, args.retouch) if args.retouch else None
    cuda = not args.cpu_decode
    calibrate_grids(base, jobs, grids, cuda, start=args.protect_first)
    statistics = dict(frames=[start, stop], frames_protected=int(protected[start:stop].sum()),
                      frames_filtered=0, frames_changed=0, changed_yuv_samples=[0, 0, 0],
                      strength=args.strength, device=torch.cuda.get_device_name())
    began = time.monotonic()
    with encode(base, args.output, start, stop, args.cq) as stream:
        for i, frame in filtered_frames(base, jobs, grids, track, protected, start, stop, cuda, statistics):
            for patch in patches:
                patch.apply(i, frame)
            if retouch:
                retouch.apply(i, frame)
            stream.write(frame)
            if (i-start+1) % 600 == 0:
                elapsed = time.monotonic()-began
                print(f'Encoded {i-start+1}/{stop-start} frames ({(i-start+1)/elapsed:.1f} fps)', flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(statistics, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {args.output.resolve()}', flush=True)
    print(json.dumps(statistics, indent=2), flush=True)
