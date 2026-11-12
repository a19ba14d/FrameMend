"""Recover corner detail from the same shot when the corner mark changes sides."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_inpaint import corner_jobs, feather
from wm_matte import affine_between, exclusion
from wm_media import Video, digest, plane_rects, read_frames, yuv_to_rgb
from wm_repair import dilate
from wm_splice import video_packets
from wm_track import load_profile, load_track, save_npz


def phase_translation(target, donor, known, box):
    x, y, w, h = box
    if min(w, h) < 32:
        return None
    region = np.s_[y:y+h, x:x+w]
    good = known[region]
    a, b = target[region], donor[region]
    if good.mean() < .87 or min(a[good].std(), b[good].std()) < 7:
        return None
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    clean = [np.where(good, image, image[good].mean()).astype(np.float32) for image in (b, a)]
    (dx, dy), response = cv2.phaseCorrelate(*clean, window)
    if response >= .30 and max(abs(dx), abs(dy)) <= 46:
        return np.float32([[1, 0, dx], [0, 1, dy]])
    return None


def match_context(target, donor, target_valid, donor_valid, shell, samples, seeds=()):
    """Select the transform using only visible scene pixels around the repair."""
    candidates = [np.eye(2, 3, dtype=np.float32), affine_between(donor, target, donor_valid, target_valid)]
    candidates.extend(seeds)
    common = (target_valid > 0) & (donor_valid > 0)
    for box in samples:
        transform = phase_translation(target, donor, common, box)
        if transform is not None:
            candidates.append(transform)
    size = target.shape[::-1]
    observed = cv2.GaussianBlur(target.astype(np.float32), (0, 0), 1.1)
    reference = cv2.GaussianBlur(donor.astype(np.float32), (0, 0), 1.1)
    check = shell & (target_valid > 0)
    best = None
    for transform in candidates:
        aligned = cv2.warpAffine(reference, transform, size, flags=cv2.INTER_LINEAR)
        usable = cv2.warpAffine(donor_valid, transform, size, flags=cv2.INTER_NEAREST) > 0
        selected = usable & check
        if selected.sum() < max(2000, .35*shell.sum()):
            continue
        difference = observed[selected]-aligned[selected]
        bias = float(np.clip(np.median(difference), -10, 10))
        residual = np.abs(difference-bias)
        error = float(.85*np.minimum(residual, 34).mean() + .15*np.quantile(residual, .92) + abs(bias)*.13)
        score = error + .012*float(np.linalg.norm(transform[:, 2]))
        if best is None or score < best[0]:
            best = (score, transform, bias, usable)
    return best


def has_border_line(luma, valid, inside):
    # A one-pixel input edge can stretch into a stripe under an affine warp.
    good = (inside > 0) & (valid[:9] > 0).all(axis=0)
    if good.sum() < 64:
        return False
    difference = luma[0, good] - np.median(luma[4:9, good], axis=0)
    return max((difference < -24).mean(), (difference > 24).mean()) >= .68


class CornerReference:
    def __init__(self, video, job, side, height):
        self.rect, original, self.active = job
        self.side = side
        x, y, w, h = self.rect
        self.width, self.height = min(video.width, w+240), min(video.height, h+100)
        self.x = 0 if side == 0 else video.width-self.width
        self.context = (self.x, 0, self.width, self.height)
        self.polygon = np.zeros((self.height, self.width), np.uint8)
        self.polygon[y:y+h, x-self.x:x-self.x+w] = original
        self.support = dilate(self.polygon, 25)
        self.visible_border = self.support & (1-dilate(self.polygon, 6))
        self.shell = (dilate(self.polygon, 64) > 0) & (dilate(self.polygon, 18) == 0)
        yy, xx = np.where(self.polygon)
        below = int(yy.max())+22
        edge = int(xx.max())+29 if side == 0 else 8
        inner_width = self.width-edge-14 if side == 0 else int(xx.min())-43
        self.samples = ((32, below, self.width-72, self.height-below-4),
                        (edge, 16, inner_width, min(300, self.height-24)))
        self.device = torch.device('cuda')
        yy, xx = torch.meshgrid(torch.arange(height, device=self.device),
                                torch.arange(self.width, device=self.device), indexing='ij')
        self.grid = torch.stack((xx, yy), dim=-1).float()
        self.normalize = self.grid.new_tensor([2/(self.width-1), 2/(self.height-1)])

    def fit(self, target, donor, seeds=()):
        return match_context(target[0][0], donor[0][0], target[1], donor[1], self.shell, self.samples, seeds)

    def border_lines(self, frame):
        parts, valid = frame
        luma = parts[0]
        columns = slice(None, None, 1 if self.side == 0 else -1)
        return (has_border_line(luma, valid, self.polygon[0]),
                has_border_line(luma[:, columns].T, valid[:, columns].T, self.polygon[:, columns][:, 0]))

    def clean_border(self, frame):
        return not any(self.border_lines(frame))

    def holdout(self, frames, indices):
        errors = []
        donor = frames[indices[0]]
        for index in indices[2:10:3]:
            target = frames[index]
            trial = self.fit(target, donor)
            if trial is None:
                continue
            _, transform, bias, usable = trial
            selected = (self.polygon > 0) & usable & (target[1] > 0)
            if selected.sum() < self.polygon.sum()*.45:
                continue
            aligned = cv2.warpAffine(donor[0][0], transform, (self.width, self.height)).astype(np.float32)
            error = np.abs(target[0][0][selected]-aligned[selected]-bias).mean()
            errors.append(float(error))
        return errors

    @torch.inference_mode()
    def predict(self, rgb, transform, bias, border=(False, False)):
        inverse = torch.as_tensor(cv2.invertAffineTransform(transform), device=self.device, dtype=torch.float32)
        positions = self.grid @ inverse[:, :2].T + inverse[:, 2]
        # A dark one-pixel frame edge must not be stretched into the scene when
        # the target lies beyond the donor. Use the first unaffected scene row.
        if border[0]:
            positions[..., 1].clamp_min_(4)
        if border[1]:
            if self.side == 0:
                positions[..., 0].clamp_min_(4)
            else:
                positions[..., 0].clamp_max_(self.width-5)
        grid = positions*self.normalize-1
        image = torch.as_tensor(rgb, device=self.device).permute(2, 0, 1)[None].float()
        patch = F.grid_sample(image, grid[None], padding_mode='border', align_corners=True)
        return (patch[0].permute(1, 2, 0) + bias*(255/219)).round().clamp(0, 255).byte().cpu().numpy()


def reliable_runs(accepted):
    if not accepted:
        return []
    runs = np.split(np.asarray(sorted(accepted)), np.flatnonzero(np.diff(sorted(accepted)) > 1)+1)
    return [group.tolist() for group in runs if len(group) >= 4]


def keyframe_plan(base, patches):
    _, packets = video_packets(base.path)
    times = sorted(packet['pts'] for packet in packets)
    if len(times) != base.count:
        raise ValueError('The base must have one video packet per frame.')
    indices = [i for i, packet in enumerate(packets) if packet['pts'] == times[i] and 'K' in packet['flags']]
    indices.append(base.count)
    groups = []
    for file, start, stop in patches:
        first = int(indices[np.searchsorted(indices, start, side='right')-1])
        last = int(indices[np.searchsorted(indices, stop, side='left')])
        if first > start:
            raise ValueError('No independent keyframe was found before the corner patch.')
        if groups and first < groups[-1]['frames'][1]:
            groups[-1]['frames'][1] = max(last, groups[-1]['frames'][1])
            groups[-1]['patches'].append(file)
        else:
            groups.append(dict(frames=[first, last], patches=[file]))
    return groups


def prepare_corner_references(source, base, output, profile_path, workdir, config_path, animation_template=None, cuda=True):
    source, base = Video.open(source), Video.open(base)
    if (source.width, source.height, source.count, source.fps, source.offset, source.matrix) != (
            base.width, base.height, base.count, base.fps, base.offset, base.matrix):
        raise ValueError('Corner recovery needs a frame-aligned base with the same color format.')
    if not torch.cuda.is_available():
        raise ValueError('CUDA PyTorch is required to warp the corner references.')
    profile = load_profile(source, profile_path)
    track = load_track(source, profile_path, Path(workdir)/'track.npz')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    jobs = corner_jobs(source, profile, track)
    if len(jobs) != 2 or any(r[1] != 0 or r[0] != (0 if side == 0 else source.width-r[2])
                              for side, (r, _, _) in enumerate(jobs)):
        raise ValueError('Expected two opposing top corner regions.')
    height = min(source.height, max((int(np.where(poly)[0].max())+41)//2*2 for _, poly, _ in jobs))
    views = [CornerReference(source, job, side, height) for side, job in enumerate(jobs)]
    active = np.stack([view.active for view in views], axis=1)
    if not np.all(active.sum(axis=1) == 1):
        raise ValueError('Each source frame must have one identified corner mark.')
    switches = np.flatnonzero(np.any(active[1:] != active[:-1], axis=1))+1
    selected = config.get('switches', switches.tolist())
    if sorted(set(selected)) != selected or not np.all(np.isin(selected, switches)):
        raise ValueError('Each configured frame must be a distinct, detected side switch.')
    radii = [int(config.get('radius_by_switch', {}).get(str(i), config['radius'])) for i in selected]
    if not all(6 <= n <= 80 for n in radii):
        raise ValueError('Corner reference radius must be between 6 and 80 frames.')
    windows = config.get('frames_by_switch', {})
    if not isinstance(windows, dict) or set(windows) - {str(i) for i in selected}:
        raise ValueError('Each corner reference window needs a selected switch.')
    cut_frames = np.r_[0, track['cuts'], source.count]
    contexts = []
    for switch, radius in zip(selected, radii, strict=True):
        pos = np.searchsorted(cut_frames, switch, side='right')
        shot_start, shot_stop = map(int, cut_frames[pos-1:pos+1])
        if str(switch) not in windows:
            lo, hi = max(shot_start, switch-radius), min(shot_stop, switch+radius)
        else:
            window = windows[str(switch)]
            if (not isinstance(window, list) or len(window) != 2 or any(type(n) is not int for n in window)
                    or not shot_start <= window[0] <= switch-6 < switch+6 <= window[1] <= shot_stop
                    or window[1]-window[0] > 420):
                raise ValueError('A corner reference window needs at least six frames on each side of the switch '
                                 'and at most 420 frames inside one detected shot.')
            lo, hi = window
        contexts.append((switch, lo, hi, shot_start, shot_stop))
    origins = np.asarray(config.get('wordmark_starts', []), dtype=int)
    if animation_template:
        with np.load(animation_template, allow_pickle=False) as data:
            if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
                raise ValueError('The animation template belongs to another source or profile.')
            tx, ty, tw, th = data['region']
            templates = data['masks']

    def validity(i):
        mask = exclusion(source, profile, track, i)
        if animation_template:
            j = np.searchsorted(origins, i, side='right')-1
            if j >= 0 and i-origins[j] < len(templates):
                mask[ty:ty+th, tx:tx+tw] *= 1-templates[i-origins[j]]
        return mask

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    patches, reports = [], []
    for switch, lo, hi, shot_start, shot_stop in contexts:
        raw = dict(read_frames(source, lo, hi, cuda))
        safe = {i: validity(i) for i in raw}
        pictures = np.zeros((hi-lo, height, source.width, 3), np.uint8)
        alpha = np.zeros(pictures.shape[:3], np.uint8)
        report = dict(switch=switch, context=[lo, hi], sides=[])
        for side, view in enumerate(views):
            frames = {i: (plane_rects(source.planes(frame), view.context),
                          safe[i][:view.height, view.x:view.x+view.width]) for i, frame in raw.items()}
            exposed = sorted((i for i in frames if not view.active[i]), key=lambda i: abs(switch-.5-i))
            clean = [i for i in exposed if view.clean_border(frames[i])]
            # Exclude transient lines. A border shared across the shot may be scene detail.
            donors = clean if len(clean) >= max(4, (len(exposed)+1)//2) else exposed
            targets = [i for i in frames if view.active[i]]
            heldout = view.holdout(frames, donors) if len(donors) >= 4 else []
            details = dict(side=side, skipped_border_frames=sorted(set(exposed)-set(donors)),
                           heldout_luma_mae=heldout, frames=[])
            report['sides'].append(details)
            if not heldout or np.median(heldout) > 7.0:
                continue
            selected_models = {}
            for i in targets:
                options = []
                previous = selected_models.get(i-1)
                for j in donors[:7:2]:
                    # Recheck the preceding transform when the visible edge has
                    # too few points to produce a fresh tracking estimate.
                    seeds = (previous[2],) if str(switch) in windows and previous is not None and previous[1] == j else ()
                    match = view.fit(frames[i], frames[j], seeds)
                    if match is None:
                        continue
                    error, transform, bias, usable = match
                    coverage = float(usable[view.polygon > 0].mean())
                    if error <= 4.0 and coverage >= .87:
                        options.append((error+.025*abs(i-j)+3*(1-coverage), j, transform, bias, error, coverage))
                if options:
                    selected_models[i] = min(options, key=lambda item: item[0])
            cache = {}
            for run in reliable_runs(selected_models):
                for i in run:
                    _, j, transform, bias, error, coverage = selected_models[i]
                    if j not in cache:
                        cache[j] = yuv_to_rgb(frames[j][0], source.matrix), view.border_lines(frames[j])
                    donor_rgb, border = cache[j]
                    picture = view.predict(donor_rgb, transform, bias, border)
                    known = feather((view.visible_border & (frames[i][1] > 0))[:height].astype(np.uint8), 4)[..., None]
                    actual = yuv_to_rgb(frames[i][0], source.matrix)[:height]
                    picture = np.rint(picture*(1-known) + actual*known).astype(np.uint8)
                    # A narrow strip beyond a translated image uses the closest observed edge.
                    usable = cv2.warpAffine(frames[j][1], transform, (view.width, view.height),
                                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
                    start_weight = 1 if run[0] in (switch, shot_start) else min(1, (i-run[0]+1)/9)
                    end_weight = 1 if run[-1]+1 in (switch, shot_stop) else min(1, (run[-1]+1-i)/9)
                    trust = np.clip((4.2-error)/1.35, 0, 1)*min(start_weight, end_weight)
                    weight = feather((view.support & usable)[:height].astype(np.uint8), 18)*trust
                    area = np.s_[i-lo, :, view.x:view.x+view.width]
                    pictures[area] = picture
                    alpha[area] = np.rint(weight*255).astype(np.uint8)
                    details['frames'].append(dict(frame=i, reference=j, context_error=error, coverage=coverage,
                                                  mean_alpha=float(weight[view.polygon[:height] > 0].mean())))
        selected_frames = np.flatnonzero(alpha.any(axis=(1, 2)))
        if selected_frames.size:
            first, last = int(selected_frames[0]), int(selected_frames[-1])+1
            name = f'switch-{switch}.npz'
            save_npz(root/name, region=np.asarray([0, 0, source.width, height]), frames=np.asarray([lo+first, lo+last]),
                     prediction=pictures[first:last], alpha=alpha[first:last], source_sha256=track['source_sha256'],
                     profile_sha256=track['profile_sha256'], config_sha256=digest(config_path))
            patches.append((name, lo+first, lo+last))
        reports.append(report)
        print(f'Corner switch {switch}: {len(selected_frames)} repaired frames.', flush=True)
    if not patches:
        raise ValueError('No exposure passed the alignment and held-out checks.')
    segments = keyframe_plan(base, patches)
    (root/'plan.json').write_text(json.dumps(dict(segments=segments, switches=reports), indent=2)+'\n', encoding='utf-8')
