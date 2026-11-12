"""Recover moving translucent-mark pixels from validated same-shot observations."""

import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_inpaint import ProPainter, feather, model_size
from wm_matte import Matte, logo_rect
from wm_media import Video, digest, plane_rects, read_frames, yuv_to_rgb
from wm_repair import dilate
from wm_temporal import CapsuleRim
from wm_track import check_rect, corner_activity, load_profile, load_track, save_npz


def frame_window(values, count, name):
    if (not isinstance(values, list) or len(values)!=2 or any(type(i) is not int for i in values)
            or not 0<=values[0]<values[1]<=count):
        raise ValueError(f'{name} must be a zero-based [first, stop) frame range.')
    return tuple(values)


def local_box(rect, context):
    x, y, w, h = rect
    cx, cy = context[:2]
    return np.s_[y-cy:y-cy+h, x-cx:x-cx+w]


def local_average(value):
    rows = F.avg_pool2d(value, (1, 25), stride=1, padding=(0, 12), count_include_pad=False)
    return F.avg_pool2d(rows, (25, 1), stride=1, padding=(12, 0), count_include_pad=False)


class RevealMasks:
    def __init__(self, profile, track, matte, context, protect, donor_margin):
        self.profile, self.track, self.matte = profile, track, matte
        self.context = context
        fields = matte.fields[0].cpu().numpy()
        self.opacity = matte.opacity
        edges = np.maximum.reduce(matte.edge_shapes)
        severe = ((self.opacity>.32) | (edges>0)).astype(np.uint8)
        self.risk = dilate(severe, donor_margin)
        self.replace = feather(dilate(severe, 11), 6)
        self.protect = np.zeros((context[3], context[2]), np.uint8)
        x, y = context[:2]
        for px, py, pw, ph in protect:
            cv2.rectangle(self.protect, (px-x-2, py-y-2), (px+pw-x+1, py+ph-y+1), 1, -1)
        margins = profile['translucent'].get('capsule_margins')
        self.rim = CapsuleRim(matte.width, matte.height, margins) if margins else None
        self.fields = fields

    def at(self, video, i, source):
        x, y, w, h = self.context
        xy = self.track['positions'][i]
        shape = (h, w)
        risk = np.zeros(shape, np.uint8)
        opacity = np.zeros(shape, np.float32)
        support = opacity.copy()
        protect = self.protect.copy()
        rect = logo_rect(self.profile, self.track, i)
        if rect:
            shift = self.matte.shifts[i]
            if self.rim:
                luma = plane_rects(video.planes(source), rect)[0]
                shift = self.rim.align(luma, shift, self.fields)
                self.matte.shifts[i] = shift
            dx, dy = xy+shift-(x, y)
            move = np.float32([[1, 0, dx], [0, 1, dy]])
            risk = cv2.warpAffine(self.risk, move, (w, h), flags=cv2.INTER_NEAREST)
            support = cv2.warpAffine(self.replace, move, (w, h))
            opacity = cv2.warpAffine(self.opacity, move, (w, h))
        corners = self.profile.get('corners')
        if corners:
            active = corner_activity(corners, self.track['corners'][i])
            for flag, points in zip(active, corners['masks'], strict=True):
                if flag:
                    polygon = np.asarray(points, dtype=np.int32)-(x, y)
                    cv2.fillPoly(protect, [polygon.astype(np.int32)], 1)
        guard = cv2.dilate(protect, np.ones((5, 5), np.uint8))
        available = feather(1-(risk|guard), 3)
        certainty = (1-.42*np.clip((opacity-.04)/.17, 0, 1))*available
        return certainty, support*(1-guard)


class RevealedFrames:
    def __init__(self, source, base, context, first, last, masks, cuda):
        self.first = first
        references, current, certainty, support = [], [], [], []
        originals = iter(read_frames(source, first, last, cuda))
        cleaned = iter(read_frames(base, first, last, cuda))
        while batch := list(itertools.islice(zip(originals, cleaned, strict=True), 12)):
            for (i, src), (_, dst) in batch:
                good, replace = masks.at(source, i, src)
                certainty.append(good)
                support.append(replace)
                current.append(yuv_to_rgb(plane_rects(base.planes(dst), context), base.matrix))
            source_frames = [a[1] for a, _ in batch]
            masks.matte.restore(source, source_frames, batch[0][0][0], spatial=False)
            for src in source_frames:
                references.append(yuv_to_rgb(plane_rects(source.planes(src), context), source.matrix))
        to_gpu = lambda array: torch.as_tensor(np.stack(array), device='cuda')
        self.references = to_gpu(references).permute(0, 3, 1, 2).contiguous()
        self.current = to_gpu(current).permute(0, 3, 1, 2).contiguous()
        self.certainty = to_gpu(certainty)[:, None].half()
        self.support = to_gpu(support)[:, None].half()


class UrlReveal:
    def __init__(self, frames, region, context, scale, iterations, offsets, time_scale):
        self.frames = frames
        self.height, self.width = frames.current.shape[2:]
        self.h, self.w = model_size((self.height, self.width), scale)
        self.raft = ProPainter(scale=scale).raft
        self.iterations = iterations
        self.offsets = offsets
        self.time_scale = time_scale
        self.cache = {}
        self.working = torch.cat([F.interpolate(part.float()/127.5-1, (self.h, self.w), mode='bilinear',
                                 align_corners=False, antialias=True).half() for part in frames.current.split(12)])
        x, y, w, h = region
        self.box = local_box(region, context)
        cy, cx = torch.meshgrid(torch.arange(y, y+h, device='cuda')-context[1],
                                torch.arange(x, x+w, device='cuda')-context[0], indexing='ij')
        self.grid = torch.stack((cx*2/(self.width-1)-1, cy*2/(self.height-1)-1), dim=-1)[None]
        self.holdout = ((cx//8+cy//8*3)%7==0)[None, None]
        self.norm = self.grid.new_tensor([2/(self.w-1), 2/(self.h-1)]).reshape(1, 2, 1, 1)
        self.pixels = self.grid.new_tensor([(self.width-1)/2, (self.height-1)/2]).reshape(1, 2, 1, 1)
        self.luma = self.grid.new_tensor([.2126, .7152, .0722]).reshape(1, 3, 1, 1)

    @staticmethod
    def sample(pictures, grid):
        return F.grid_sample(pictures, grid, align_corners=True, padding_mode='zeros')

    def pair_flows(self, i, candidates):
        pairs = [(min(i, j), max(i, j)) for j in candidates]
        missing = [p for p in pairs if p not in self.cache]
        for start in range(0, len(missing), 4):
            keys = missing[start:start+4]
            images = self.working[torch.tensor(keys, device='cuda')].float()
            f, b = self.raft(images, iters=self.iterations)
            for key, a, c in zip(keys, (f[:, 0]*self.norm).half(), (b[:, 0]*self.norm).half(), strict=True):
                self.cache[key] = (a, c)
        direct, reverse = zip(*(self.cache[p] if i<j else self.cache[p][::-1]
                                for p, j in zip(pairs, candidates, strict=True)), strict=True)
        return torch.stack(direct).float(), torch.stack(reverse).float()

    @torch.inference_mode()
    def repair(self, i):
        frames = self.frames
        j = i-frames.first
        box = self.box
        old = frames.current[j, :, box[0], box[1]].float()
        support = frames.support[j, :, box[0], box[1]].float()
        if not support.any():
            return old, support[0], dict(frame=i, references=0, replaced=0)
        n = len(frames.current)
        nearby = sorted({min(n-1, max(0, j+d)) for d in self.offsets}-{j})
        direct, reverse = self.pair_flows(j, nearby)
        identity = self.grid.expand(len(nearby), -1, -1, -1)
        motion = self.sample(direct, identity)
        grid = identity+motion.permute(0, 2, 3, 1)
        cycle = torch.linalg.vector_norm((motion+self.sample(reverse, grid))*self.pixels, dim=1, keepdim=True)
        actual = torch.cat((frames.references[nearby].float(), frames.certainty[nearby].float()), dim=1)
        warped = self.sample(actual, grid)
        colors, donor = warped[:, :3], warped[:, 3:].clamp(0, 1)
        own = frames.certainty[j:j+1, :, box[0], box[1]].float()
        known = own*donor
        fit = known*(~self.holdout)
        count = local_average(fit)
        diff = old[None]-colors
        tone = (local_average(fit*diff.clamp(-25, 25))/(count+.10)).clamp(-9, 9)
        colors = (colors+tone).clamp(0, 255)
        delta = ((old[None]-colors)*self.luma).sum(dim=1, keepdim=True).abs()
        shared = (fit*delta.clamp_max(36)).sum(dim=(2, 3), keepdim=True)/fit.sum(dim=(2, 3), keepdim=True).clamp_min(1)
        error = (local_average(fit*delta.clamp_max(38))+.12*shared)/(count+.12)
        interval = self.grid.new_tensor([abs(j-k) for k in nearby]).reshape(-1, 1, 1, 1)
        weight = donor*(1+(cycle/2.7)**2).reciprocal()*(1+(error/6.3)**2).reciprocal()
        weight *= torch.exp(-interval/self.time_scale)*torch.clamp((17-shared)/10, 0, 1)

        luminance = (colors*self.luma).sum(dim=1, keepdim=True)
        ranked, order = luminance.sort(dim=0)
        sorted_weights = torch.gather(weight, 0, order)
        middle = (sorted_weights.cumsum(dim=0) < sorted_weights.sum(dim=0)*.5).sum(dim=0, keepdim=True)
        median = torch.gather(ranked, 0, middle.clamp_max(len(nearby)-1))
        agreement = (1+((luminance-median)/8)**2).reciprocal()
        agreed = weight*agreement
        total = agreed.sum(dim=0)
        predicted = (colors*agreed).sum(dim=0)/total.clamp_min(.0001)
        spread = (agreed*(luminance-median).abs()).sum(dim=0)/total.clamp_min(.0001)
        alpha = support*(total/.78).clamp(0, 1)*(1+(spread/6.8)**2).reciprocal()

        held = self.holdout*own*support
        hold_weight = held*(total>.65)
        hold_error = (((old-predicted)*self.luma[0]).sum(dim=0).abs()*hold_weight[0]).sum()
        mass = hold_weight.sum()
        risk = support>.6
        report = dict(frame=i, references=len(nearby), replaced=int((alpha>.65).sum()),
                      supported_share=round(float((alpha[risk]>.65).float().mean()), 5) if risk.any() else 0,
                      held_out_weight=round(float(mass), 1),
                      held_out_y_error=round(float(hold_error/mass), 3) if mass>=5 else None)
        cutoff = j-max(abs(d) for d in self.offsets)-1
        self.cache = {k: v for k, v in self.cache.items() if k[0]>=cutoff}
        return predicted, alpha[0], report


@torch.inference_mode()
def prepare_url_reveal(input_path, base_path, output, profile_path, workdir, config_path, report=None, cuda=True,
                       job=None):
    source, base = Video.open(input_path), Video.open(base_path)
    if (source.width, source.height, source.count, source.fps, source.offset, source.matrix) != (
            base.width, base.height, base.count, base.fps, base.offset, base.matrix):
        raise ValueError('URL repair requires frame-aligned video with matching dimensions and color format.')
    if not torch.cuda.is_available():
        raise ValueError('CUDA PyTorch is required for same-shot URL recovery.')
    output, workdir = Path(output), Path(workdir)
    if output.suffix.lower()!='.npz' or output.exists():
        raise ValueError('Choose a new .npz URL patch path.')
    if report is not None and Path(report).resolve() in (output.resolve(), source.path, base.path):
        raise ValueError('Choose a separate URL repair report path.')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    if job is not None:
        if job not in config.get('jobs', {}):
            raise ValueError(f'Unknown URL repair job {job!r}.')
        config = config['jobs'][job]
    elif 'jobs' in config:
        raise ValueError('Select a URL repair job with --job.')
    profile = load_profile(source, profile_path)
    if not profile.get('translucent'):
        raise ValueError('The watermark profile needs a tracked translucent mark.')
    track = load_track(source, profile_path, workdir/'track.npz')
    first, last = frame_window(config['frames'], source.count, 'frames')
    lo, hi = frame_window(config['references'], source.count, 'references')
    if not lo<=first<last<=hi or not 2<=hi-lo<=240 or np.any((track['cuts']>lo)&(track['cuts']<hi)):
        raise ValueError('References must include the target interval within one shot of at most 240 frames.')
    rect = lambda value: check_rect(value, source.width, source.height)
    context, region = rect(config['context']), rect(config['region'])
    cx, cy, cw, ch = context
    x, y, w, h = region
    if not (cx<=x<x+w<=cx+cw and cy<=y<y+h<=cy+ch and cw*ch<=850000):
        raise ValueError('The URL region must fit inside a context of at most 850,000 pixels.')
    offsets = config.get('offsets', [-39, -21, -12, -6, -3, 3, 6, 12, 21, 39])
    if (not isinstance(offsets, list) or not 4<=len(offsets)<=16
            or any(type(d) is not int or not 1<=abs(d)<=200 for d in offsets)
            or len(set(offsets))!=len(offsets)):
        raise ValueError('Give 4–16 distinct reference offsets between -200 and 200, excluding zero.')
    scale = float(config.get('model_scale', .48))
    iterations = config.get('raft_iterations', 9)
    if not .25<=scale<=.75 or type(iterations) is not int or not 4<=iterations<=18:
        raise ValueError('Invalid URL flow scale or iteration count.')
    donor_margin = config.get('donor_margin', 4)
    time_scale = float(config.get('time_scale', 47))
    if type(donor_margin) is not int or not 2<=donor_margin<=8 or not 12<=time_scale<=140:
        raise ValueError('The donor margin must be 2–8 pixels and time scale must be 12–140 frames.')
    fade = config.get('fade', [0, 0])
    if (not isinstance(fade, list) or len(fade)!=2 or any(type(v) is not int or v<0 for v in fade)
            or sum(fade)>=last-first):
        raise ValueError('The entry and exit fades must fit within the target interval.')
    protect = [rect(value) for value in config.get('protect', [])]
    matte = Matte(profile, track, workdir/'matte.npz', clean_edges=True)
    masks = RevealMasks(profile, track, matte, context, protect, donor_margin)
    frames = RevealedFrames(source, base, context, lo, hi, masks, cuda)
    restorer = UrlReveal(frames, region, context, scale, iterations, offsets, time_scale)
    result, alpha, measurements = [], [], []
    for i in range(first, last):
        guess, strength, evidence = restorer.repair(i)
        factor = min(1., (i-first+1)/max(fade[0], 1), (last-i)/max(fade[1], 1))
        result.append(guess.round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        alpha.append((strength*factor*255).round().clamp(0, 255).byte().cpu().numpy())
        measurements.append(evidence)
        if (i-first+1)%20==0:
            print(f'Aligned {i-first+1}/{last-first} frames on the GPU.', flush=True)
    save_npz(output, region=np.asarray(region), frames=np.asarray([first, last]),
             prediction=np.stack(result), alpha=np.stack(alpha), source_sha256=track['source_sha256'],
             profile_sha256=track['profile_sha256'], config_sha256=digest(config_path))
    if report:
        report = Path(report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(dict(job=job, frames=[first, last], region=region, device=torch.cuda.get_device_name(),
                                         measurements=measurements), indent=2)+'\n', encoding='utf-8')
    print(f'Saved {output.resolve()}', flush=True)
