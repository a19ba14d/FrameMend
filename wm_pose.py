"""Reconstruct a moving subject from annotated same-shot reference frames."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wm_animation import AnimationPatch
from wm_inpaint import ProPainter, feather
from wm_matte import Matte, capsule
from wm_media import Video, digest, get_frame, plane_rects, read_frames, rgb_to_yuv, yuv_to_rgb
from wm_repair import dilate
from wm_temporal import TemporalEdges
from wm_track import check_rect, load_profile, load_track, save_npz
from wm_wipe import MotionBetween, new_cache, picture_slice, refined_mask, wipe_palette


class PoseWarp:
    """Fit a smooth displacement through the supplied landmarks and sample on CUDA."""

    def __init__(self, video, region, fixed):
        if not torch.cuda.is_available():
            raise ValueError('CUDA PyTorch is required for pose repair.')
        self.device = torch.device('cuda')
        self.width, self.height = video.width, video.height
        self.scale = max(self.width, self.height) * 0.5
        self.fixed = self.tensor(fixed)
        x, y, self.w, self.h = region
        yy, xx = torch.meshgrid(torch.linspace(y, y+self.h-1, (self.h+3)//4, device=self.device),
                                torch.linspace(x, x+self.w-1, (self.w+3)//4, device=self.device),
                                indexing='ij')
        self.grid_shape = xx.shape
        self.grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2) / self.scale
        self.normalize = self.tensor([2*self.scale/(self.width-1), 2*self.scale/(self.height-1)])

    def tensor(self, values):
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def plate(self, image):
        return self.tensor(image).permute(2, 0, 1)[None]

    @staticmethod
    def radial(distance):
        return distance * distance.clamp_min(1e-7).log()

    @torch.inference_mode()
    def picture(self, points, reference, plate):
        target = torch.cat((self.tensor(points), self.fixed)) / self.scale
        donor = torch.cat((self.tensor(reference), self.fixed)) / self.scale
        count = len(target)
        linear = torch.cat((torch.ones_like(target[:, :1]), target), dim=1)
        system = target.new_zeros((count+3, count+3))
        system[:count, :count] = self.radial(torch.cdist(target, target).square())
        system[:count, :count].diagonal().add_(0.0006)
        system[:count, count:] = linear
        system[count:, :count] = linear.T
        weights = torch.linalg.solve(system, torch.cat((donor-target, target.new_zeros((3, 2)))))
        linear_grid = torch.cat((torch.ones_like(self.grid[:, :1]), self.grid), dim=1)
        shift = self.radial(torch.cdist(self.grid, target).square()) @ weights[:count] + linear_grid @ weights[count:]
        grid = ((self.grid+shift)*self.normalize-1).reshape(*self.grid_shape, 2)
        grid = F.interpolate(grid.permute(2, 0, 1)[None], (self.h, self.w), mode='bilinear', align_corners=True)
        image = F.grid_sample(plate, grid.permute(0, 2, 3, 1), align_corners=True, padding_mode='border')
        return image[0].permute(1, 2, 0).contiguous().cpu().numpy()


def mark_regions(mask, rects):
    for rect in rects:
        mask[picture_slice(rect)] = 1
    return mask


def opaque_region(patch, i, shape, threshold):
    result = np.zeros(shape, np.uint8)
    if patch.lo <= i < patch.hi:
        result[picture_slice(patch.region)] = patch.alpha[i-patch.lo] > threshold
    return result


@torch.inference_mode()
def transition_from_late(early, late, fraction, model, known):
    from wm_corner_grid import lowpass

    motion = MotionBetween(early, late, model, known)
    confidence = motion.consistency[:1]
    shift = lowpass(motion.flow[:1]*confidence, 37, 37)
    shift /= lowpass(confidence, 37, 37).clamp_min(.16)
    shift = shift.clamp(-.29, .29).permute(0, 2, 3, 1)
    anchor = motion.frames[1:]
    first = motion.sample(anchor, motion.grid+shift)
    moved = motion.sample(anchor, motion.grid+shift*(1-fraction))
    difference = motion.frames[:1]-first
    brightness = difference.mean(dim=1,keepdim=True)
    material = (1-(difference-brightness).abs().amax(dim=1,keepdim=True)/23).clamp(0, 1)
    tone = lowpass(brightness*material,23,23) / lowpass(material,23,23).clamp_min(.25)
    tone = tone.clamp(-14, 14)
    moved = (moved+tone*(1-fraction)).clamp(0, 255)
    # Ease into the single warped plate while it is still close to the earlier pose.
    entry = min(1, fraction/.10)
    result = motion.frames[:1]*(1-entry) + moved*entry
    return result[0].permute(1, 2, 0).cpu().numpy()


def prepare_pose_patch(source, base, output, profile_path, workdir, config_path, wordmark, wipe, template, cuda=True):
    source, base = Video.open(source), Video.open(base)
    if (source.width, source.height, source.fps, source.count, source.matrix) != (
            base.width, base.height, base.fps, base.count, base.matrix):
        raise ValueError('Pose repair requires an aligned base with matching video and color formats.')
    output = new_cache(output)
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    shape = (base.height, base.width)
    rect = lambda bounds: check_rect(bounds, base.width, base.height)
    region = rect(config['region'])
    box = picture_slice(region)
    lo, hi = config['frames']
    early, late = config['references']
    bottom_rows = config.get('reference_bottom_rows', [0, 0])
    if (not isinstance(bottom_rows, list) or len(bottom_rows) != 2
            or any(type(rows) is not int or not 0 <= rows <= min(8, base.height-1) for rows in bottom_rows)):
        raise ValueError('Supply the bottom-edge row count for each pose reference.')
    wipe_start, tail = config['wipe_start'], config['reveal_tail']
    switch = config['reference_switch']
    blend_start = config.get('reference_blend_start', switch)
    entries = config['landmarks']
    keys = [int(i) for i in entries]
    points = np.asarray(list(entries.values()), dtype=np.float32)
    fixed = np.asarray(config['fixed'], dtype=np.float32)
    if (not 0 <= lo <= early < wipe_start < switch < tail < hi <= late < source.count
            or not wipe_start <= blend_start <= switch
            or sorted(set(keys)) != keys or early not in keys or late not in keys
            or points.ndim != 3 or points.shape[2] != 2 or points.shape[1] < 3
            or fixed.ndim != 2 or fixed.shape[1] != 2 or len(fixed) < 3
            or not np.isfinite(points).all() or not np.isfinite(fixed).all()):
        raise ValueError('Invalid pose references, frame ranges or landmark geometry.')
    references = {i: points[keys.index(i)] for i in (early, late)}
    profile = load_profile(source, profile_path)
    workdir = Path(workdir)
    track = load_track(source, profile_path, workdir/'track.npz')
    wordmark = AnimationPatch(base, track, wordmark)
    wipe = AnimationPatch(base, track, wipe)
    with np.load(template, allow_pickle=False) as data:
        if any(str(data[key]) != str(track[key]) for key in ('source_sha256', 'profile_sha256')):
            raise ValueError('The wipe template belongs to another source or profile.')
        first_age, stop_age = data['ages']
        masks, guards = data['masks'], data['guards']
    origin = config['wipe_origin']
    start_reveal = min(int(i) for i in config['reveal'])
    if not origin+first_age <= start_reveal <= tail < hi <= origin+stop_age:
        raise ValueError('The configured reveal needs a corresponding wipe mask.')
    if masks.shape != (stop_age-first_age, *shape) or guards.shape != masks.shape:
        raise ValueError('Invalid wipe mask geometry.')
    reveals = {int(i): [rect(part) for part in regions] for i, regions in config['reveal'].items()}
    entries = {int(i): [rect(part) for part in regions] for i, regions in config['enter'].items()}
    early_rgb, late_rgb = [yuv_to_rgb(base.planes(get_frame(base, i, cuda)), base.matrix) for i in (early, late)]
    for picture, rows in zip((early_rgb, late_rgb), bottom_rows, strict=True):
        if rows:
            picture[-rows:] = picture[-rows-1:-rows]
    early_alpha = mark_regions(np.zeros(shape, np.uint8), [rect(config['early_region'])])
    mapper = PoseWarp(base, region, fixed)
    early_plate = mapper.plate(np.dstack((early_rgb, feather(early_alpha, 70))))
    late_plate = mapper.plate(late_rgb)
    flow_region = config.get('reference_flow_region')
    flow = None
    if flow_region:
        x, y, width, height = rect(flow_region)
        x -= region[0]
        y -= region[1]
        if (blend_start == switch or min(x, y) < 0 or min(width, height) < 32
                or x+width > region[2] or y+height > region[3]):
            raise ValueError('The reference flow needs a transition window and a region inside the pose patch.')
        flow_area = np.s_[y:y+height, x:x+width]
        known = np.ones((2, height, width), np.float32)
        flow_weight = feather(np.pad(np.ones((height-2, width-2), np.uint8), 1), 24)[..., None]
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            flow = ProPainter()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
    wordmark_region = mark_regions(np.zeros(shape, np.uint8), [rect(config['wordmark_region'])])
    wipe_region = mark_regions(np.zeros(shape, np.uint8), [rect(config['wipe_region'])])
    tail_extra = mark_regions(np.zeros(shape, np.uint8), [rect(config['tail_region'])])
    fade_samples = mark_regions(np.zeros(shape, np.uint8), [rect(config['fade_region'])])[box].astype(bool)
    matte = Matte(profile, track, workdir/'matte.npz', clean_edges=True)
    edges = TemporalEdges(source, matte, track)
    capsule_shape = capsule(matte.width, matte.height, profile['translucent']['capsule_margins'])
    url = dilate(np.maximum(capsule_shape, edges.reference_mask), 7)
    raw = dict(read_frames(source, start_reveal, hi, cuda))
    visible = {i: frame.copy() for i, frame in raw.items()}
    matte.restore(source, list(visible.values()), start_reveal, spatial=False)
    rgb_weights = np.float32([.2126, .7152, .0722] if base.matrix == 'bt709' else [.299, .587, .114])
    predictions, opacity = [], []
    for i, frame in read_frames(base, lo, hi, cuda):
        pose = np.float32([np.interp(i, keys, p) for p in points.reshape(len(keys), -1).T]).reshape(-1, 2)
        estimate = mapper.picture(pose, references[late], late_plate)
        if i < switch:
            donor = mapper.picture(pose, references[early], early_plate)
            weight = donor[..., 3:]
            early_estimate = estimate*(1-weight) + donor[..., :3]*weight
            if i >= blend_start:
                fraction = (i-blend_start)/(switch-blend_start)
                fraction = fraction*fraction*(3-2*fraction)
                interpolated = early_estimate*(1-fraction) + estimate*fraction
                if flow is not None and fraction > 0:
                    moved = transition_from_late(early_estimate[flow_area], estimate[flow_area], fraction, flow, known)
                    interpolated[flow_area] = interpolated[flow_area]*(1-flow_weight)+moved*flow_weight
                estimate = interpolated
            else:
                estimate = early_estimate
        luma = base.planes(frame)[0][box]
        predicted_luma = 16 + (estimate @ rgb_weights)*(219/255)
        clean = fade_samples & (predicted_luma > 60) & (predicted_luma < 215) & (luma > 24)
        minimum = .10 if i < config['fade_stop'] else .94
        maximum = 1.09 if i < config['fade_stop'] else 1.06
        gain = np.clip(np.median((luma[clean]-16)/(predicted_luma[clean]-16)), minimum, maximum) if clean.any() else 1
        estimate = np.rint(estimate*gain).clip(0, 255).astype(np.uint8)
        if i < wipe_start:
            mask = dilate(opaque_region(wordmark, i, shape, 23), 18) & wordmark_region
            mark_regions(mask, entries.get(i, ()))
        else:
            mask = wipe_region
        alpha = feather(mask[box], 32)
        if i in reveals:
            original = plane_rects(source.planes(raw[i]), region)
            rgb = yuv_to_rgb(original, source.matrix)
            clean_rgb = yuv_to_rgb(plane_rects(source.planes(visible[i]), region), source.matrix)
            k = i-origin-first_age
            unknown = refined_mask(masks[k][box], guards[k][box], rgb, original[0], rgb_to_yuv(estimate, base.matrix)[0])
            unknown |= dilate(wipe_palette(rgb).astype(np.uint8), 8) | edges.mask(i, region, url)
            solid = mark_regions(np.zeros(shape, np.uint8), reveals[i])[box]
            known = feather(solid & (1-unknown), 17)[..., None]
            estimate = np.rint(estimate*(1-known) + clean_rgb*known).astype(np.uint8)
        if i >= tail:
            support = dilate(opaque_region(wipe, i, shape, 0), 14) | tail_extra
            alpha *= feather(support[box], 12)
        predictions.append(estimate)
        opacity.append(np.rint(alpha*255).astype(np.uint8))
    save_npz(output, region=np.asarray(region), frames=np.asarray([lo, hi]), prediction=np.stack(predictions),
             alpha=np.stack(opacity), source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'],
             config_sha256=digest(config_path))
    print(json.dumps(dict(frames=[lo, hi], references=[early, late], reference_switch=switch,
                          gpu=torch.cuda.get_device_name(), saved_patch=str(output)), indent=2), flush=True)
