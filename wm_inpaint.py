from argparse import Namespace
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_media import TOOLS, plane_rects, rgb_to_yuv, yuv_to_rgb
from wm_track import check_rect, corner_activity


# ProPainter folds quarter-resolution features with stride three.
FEATURE_GRID_STRIDE = 12


def model_size(shape, scale):
    return tuple(max(128, round(length*scale/8)*8) for length in shape)


def load_bidirectional_raft(path, device):
    from RAFT import RAFT
    from model.modules.flow_comp_raft import RAFT_bi

    class SingleDeviceRAFT(RAFT_bi):
        def __init__(self, model):
            torch.nn.Module.__init__(self)
            self.fix_raft = model

    # The checkpoint has DataParallel keys, but inference uses a single CUDA device.
    model = RAFT(Namespace(small=False, mixed_precision=False, alternate_corr=False))
    weights = torch.load(path, map_location='cpu', weights_only=True)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(weights, 'module.')
    model.load_state_dict(weights)
    return SingleDeviceRAFT(model.requires_grad_(False).to(device)).eval()


class ProPainter:
    def __init__(self, model_dir=TOOLS / 'ProPainter', scale=0.5, raft_iterations=6):
        if not torch.cuda.is_available():
            raise ValueError('CUDA PyTorch is required for ProPainter.')
        weights = model_dir / 'weights'
        for name in ('raft-things.pth', 'recurrent_flow_completion.pth', 'ProPainter.pth'):
            if not (weights / name).is_file():
                raise ValueError(f'Missing {weights / name}. Run setup.ps1 -ProPainter.')
        sys.path.insert(0, str(model_dir.resolve()))
        try:
            from model.propainter import InpaintGenerator
            from model.recurrent_flow_completion import RecurrentFlowCompleteNet
            self.device = torch.device('cuda', torch.cuda.current_device())
            self.raft = load_bidirectional_raft(weights / 'raft-things.pth', self.device)
        finally:
            sys.path.pop(0)
        self.scale = scale
        self.raft_iterations = raft_iterations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.flow = RecurrentFlowCompleteNet(weights / 'recurrent_flow_completion.pth').to(self.device).eval().half()
        self.model = InpaintGenerator(model_path=weights / 'ProPainter.pth').to(self.device).eval().half()
        print('CUDA inpainting on', self.device, flush=True)

    @torch.inference_mode()
    def predict(self, rgb, masks, references=()):
        count, height, width, _ = rgb.shape
        process_h, process_w = model_size((height, width), self.scale)
        frames = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
        frames = F.interpolate(frames, (process_h, process_w), mode='bilinear', align_corners=False, antialias=True)[None]
        m = torch.from_numpy(masks).to(self.device)[:, None].float()
        m = F.adaptive_max_pool2d(m, (process_h, process_w)).clamp_max_(1)[None].half()
        if count == 1:
            frames = frames.repeat(1, 2, 1, 1, 1)
            m = m.repeat(1, 2, 1, 1, 1)
        n = frames.shape[1]
        # RAFT is kept in float32. Small batches bound its correlation volume.
        batch = 8 if process_w * process_h < 140000 else 4
        pairs = [self.raft(frames[:, j:min(j + batch + 1, n)], iters=self.raft_iterations)
                 for j in range(0, n - 1, batch)]
        flows = tuple(torch.cat(axis, dim=1).half() for axis in zip(*pairs))
        frames = frames.half()
        completed, _ = self.flow.forward_bidirect_flow(flows, m)
        flows = self.flow.combine_flow(flows, completed, m)
        masked = frames * (1 - m)
        propagated, remaining = self.model.img_propagation(masked, flows, m, 'nearest')
        propagated = masked + propagated.reshape_as(frames) * m
        remaining = remaining.reshape_as(m)

        outputs = torch.zeros((n, 3, process_h, process_w), device=self.device, dtype=torch.float32)
        weights = torch.zeros(n, device=self.device, dtype=torch.float32)
        for mid in range(0, n, 4):
            lo, hi = max(0, mid - 4), min(n, mid + 5)
            if not masks[min(lo, count - 1):min(hi, count)].any():
                continue
            local = list(range(lo, hi))
            refs = [j for j in sorted(set(range(0, n, 10)).union(references)) if j < lo or j >= hi]
            selected = local + refs
            pred = self.model(propagated[:, selected], tuple(f[:, lo:hi-1] for f in flows),
                              m[:, selected], remaining[:, selected], len(local))
            pred = pred.reshape(-1, 3, process_h, process_w)[:len(local)]
            weight = 5 - (torch.arange(lo, hi, device=self.device) - mid).abs()
            outputs[lo:hi] += pred.float() * weight[:, None, None, None]
            weights[lo:hi] += weight
        outputs /= weights[:, None, None, None].clamp_min_(1)
        outputs = F.interpolate(outputs[:count], (height, width), mode='bicubic', align_corners=False)
        return ((outputs + 1) * 127.5).round_().clamp_(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()


def shifted_clip(clip, dy, dx):
    height, width = clip.shape[1:3]
    rows = np.clip(np.arange(height)-dy, 0, height-1)
    columns = np.clip(np.arange(width)-dx, 0, width-1)
    return clip[:, rows[:, None], columns[None, :]].copy()


def phase_detail(model, rgb, masks, direction):
    original = model.predict(rgb, masks)
    shape = rgb.shape[1:3]
    processed = model_size(shape, model.scale)
    periods = [FEATURE_GRID_STRIDE*a/b for a, b in zip(shape, processed, strict=True)]
    first = [max(1, round(period/3))*sign for period, sign in zip(periods, direction, strict=True)]
    second = [max(1, round(2*period/3))*sign for period, sign in zip(periods, direction, strict=True)]
    correction = np.zeros_like(original, dtype=np.float32)
    weight = np.ones(original.shape[:3], np.float32)
    for dy, dx in ((first[0], second[1]), (second[0], first[1])):
        variant = model.predict(shifted_clip(rgb, dy, dx), shifted_clip(masks, dy, dx))
        variant = shifted_clip(variant, -dy, -dx)
        for i, (alt, base) in enumerate(zip(variant, original, strict=True)):
            difference = alt.astype(np.float32)-base
            structure = cv2.GaussianBlur(difference, (0, 0), 7)
            agreement = np.clip(1-(np.mean(np.abs(structure), axis=-1)-4)/14, 0, 1)
            correction[i] += (difference-structure)*agreement[..., None]
            weight[i] += agreement
    return np.clip(np.rint(original+correction/weight[..., None]), 0, 255).astype(np.uint8)


def polygon_mask(rect, polygon):
    x, y, w, h = rect
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32) - (x, y)], 1)
    if not mask.any():
        raise ValueError(f'Empty mask in {rect}.')
    return mask


def feather(mask, width):
    return np.minimum(1, cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / width)


def composite(original, result, alpha, matrix, match_ring=None):
    refined = rgb_to_yuv(result, matrix)
    subsampled = cv2.resize(alpha, (alpha.shape[1]//2, alpha.shape[0]//2), interpolation=cv2.INTER_AREA)
    for plane, patch, blend in zip(original, refined, (alpha, subsampled, subsampled), strict=True):
        if match_ring is not None:
            patch = match_boundary(plane, patch, match_ring)
        active = blend > 0
        clean = plane[active] * (1 - blend[active]) + patch[active] * blend[active]
        plane[active] = np.clip(np.rint(clean), 0, 255).astype(np.uint8)


def match_boundary(source, patch, ring):
    # Extend low-frequency color corrections from a known clean ring into the prediction.
    size = (max(1, source.shape[1] // 4), max(1, source.shape[0] // 4))
    weights = cv2.resize(ring, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    delta = (source.astype(np.float32) - patch) * weights
    blur = lambda image: cv2.GaussianBlur(cv2.resize(image, size, interpolation=cv2.INTER_AREA), (0, 0), 5)
    correction = np.clip(blur(delta) / np.maximum(blur(weights), 0.045), -30, 30)
    return patch + cv2.resize(correction, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)


def corner_jobs(video, profile, track):
    corner = profile.get('corners')
    if not corner:
        return []
    jobs = []
    activity = corner_activity(corner, track['corners'])
    for j, (rect, polygon) in enumerate(zip(corner['regions'], corner['masks'], strict=True)):
        rect = check_rect(rect, video.width, video.height)
        mask = polygon_mask(rect, polygon)
        jobs.append((rect, mask, activity[:, j]))
    return jobs


def inpaint_roi(video, model, items, rect, masks, phase_direction=None):
    originals = [plane_rects(video.planes(frame), rect) for frame in items]
    rgb = np.stack([yuv_to_rgb(roi, video.matrix) for roi in originals])
    predicted = model.predict(rgb, masks) if phase_direction is None else phase_detail(model, rgb, masks, phase_direction)
    return originals, predicted
