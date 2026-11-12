"""Extend a visible, straight wall or window edge through a corner mask."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wm_inpaint import corner_jobs, feather, match_boundary
from wm_matte import exclusion
from wm_media import Video, digest, plane_rects, read_frames, yuv_to_rgb
from wm_repair import dilate
from wm_track import check_rect, load_profile, load_track, save_npz


def edge_points(luma, columns, rows, sign):
    top, bottom = rows
    blurred = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 1.8)
    gradient = sign*cv2.Sobel(blurred, -1, 0, 1)/8
    centers = np.argmax(gradient[top:bottom, columns], axis=0)+top
    before, peak, after = (gradient[centers+step, columns] for step in (-1, 0, 1))
    offset = (.5*(before-after)/np.minimum(before+after-2*peak, -.001)).clip(-.5, .5)
    return centers+offset, peak


def fit_line(x, y, good):
    for _ in range(3):
        if good.sum() < 12:
            raise ValueError('Too few clearly visible trim samples for the line fit.')
        slope, offset = np.polyfit(x[good], y[good], 1)
        deviation = np.abs(y-(slope*x+offset))
        good &= deviation <= max(1.6, float(np.median(deviation[good])*3.2))
    return float(slope), float(offset)


@torch.inference_mode()
def extend_strip(image, slope, columns, output_x, output_width):
    height, width = image.shape[:2]
    picture = torch.as_tensor(image, device='cuda', dtype=torch.float32).permute(2, 0, 1)[None]
    yy = torch.arange(height, device=picture.device, dtype=torch.float32)[:, None]
    reference = float(np.mean(columns))
    x = torch.as_tensor(columns, device=picture.device, dtype=torch.float32)[None]
    y = yy+(x-reference)*slope
    grid = torch.stack((x.expand_as(y)*2/(width-1)-1, y*2/(height-1)-1), dim=-1)[None]
    aligned = F.grid_sample(picture, grid, align_corners=True, padding_mode='border')[0]
    profile = aligned.median(dim=-1).values[None, :, :, None]
    profile = F.avg_pool2d(F.pad(profile, (0, 0, 1, 1), mode='replicate'), (3, 1), stride=1)
    x = torch.arange(output_x, output_x+output_width, device=picture.device, dtype=torch.float32)[None]
    y = yy-(x-reference)*slope
    grid = torch.stack((torch.zeros_like(y), y*2/(height-1)-1), dim=-1)[None]
    return F.grid_sample(profile, grid, align_corners=True,
                         padding_mode='border')[0].permute(1, 2, 0).cpu().numpy()


def sample_edge(config, width, height):
    left, right = config['columns']
    top, bottom = config['rows']
    stride = config.get('stride', 1)
    sign = config['sign']
    if not (0 <= left < right <= width and 1 <= top < bottom < height
            and sign in (-1, 1) and type(stride) is int and stride > 0
            and (right-left)//stride >= 20):
        raise ValueError('Specify a visible sample range inside the trim context.')
    return np.arange(left, right, stride), (top, bottom), sign


def prepare_trim_patch(input_path, output, profile_path, workdir, config_path, cuda=True):
    source = Video.open(input_path)
    if not torch.cuda.is_available():
        raise ValueError('CUDA PyTorch is required for corner trim projection.')
    output = Path(output)
    if output.exists() or output.suffix.lower() != '.npz':
        raise ValueError('Choose a new .npz cache path.')
    profile = load_profile(source, profile_path)
    track = load_track(source, profile_path, Path(workdir)/'track.npz')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    context = check_rect(config['context'], source.width, source.height)
    region = check_rect(config['region'], source.width, source.height)
    cx, cy, cw, ch = context
    x, y, width, height = region
    jobs = corner_jobs(source, profile, track)
    side = config['side']
    if not 0 <= side < len(jobs):
        raise ValueError('Choose a corner from the watermark profile.')
    (px, py, pw, ph), corner, active = jobs[side]
    if not (cy == y == py == 0 and max(cx, px) <= x and x+width <= min(cx+cw, px+pw)
            and height <= ch and min(cw, ch) >= 32):
        raise ValueError('The output rectangle must fit inside the top corner and context.')
    polygon = np.zeros((ch, width), np.uint8)
    rows = min(ch, ph)
    polygon[:rows] = corner[:rows, x-px:x-px+width]
    opacity = np.rint(feather(dilate(polygon, 19), 14)[:height]*255).astype(np.uint8)
    ring = dilate(polygon, 33) & (1-dilate(polygon, 7))
    shots = config['shots']
    if not shots:
        raise ValueError('Supply at least one shot for corner trim projection.')
    first, last = shots[0]['frames'][0], shots[-1]['frames'][1]
    if not 0 <= first < last <= source.count or last-first > 420:
        raise ValueError('The corner trim patch must span at most 420 frames.')

    predictions, measurements = [], []
    expected = first
    for shot in shots:
        start, stop = shot['frames']
        if (expected != start or not first <= start < stop <= last or not active[start:stop].all()
                or np.any((track['cuts'] > start) & (track['cuts'] < stop))):
            raise ValueError('Trim windows must be consecutive, inside detected shots and on the active corner.')
        expected = stop
        sample = sample_edge(shot['edge'], cw, ch)
        columns = sample[0]
        donor_left, donor_right = shot['donor_columns']
        slope_min, slope_max = shot['slope_limits']
        minimum_luma = float(shot['edge'].get('minimum_luma', 106))
        parallel = shot.get('parallel_edge')
        if not (columns[0] <= donor_left < donor_right <= columns[-1]+1 and -.65 <= slope_min < slope_max <= .65
                and 0 <= minimum_luma <= 255):
            raise ValueError('Invalid donor strip, slope limits or sample luminance.')
        if parallel is not None:
            alternate = sample_edge(parallel, cw, ch)
            weight = float(parallel['weight'])
            if not 0 < weight <= 1:
                raise ValueError('Parallel edge weight must be between zero and one.')

        for index, frame in read_frames(source, start, stop, cuda):
            planes = plane_rects(source.planes(frame), context)
            luma = planes[0]
            picture = yuv_to_rgb(planes, source.matrix)
            safe = exclusion(source, profile, track, index)[:ch, cx:cx+cw]>0
            points, strength = edge_points(luma, *sample)
            light = luma[18:30, columns].mean(axis=0)
            good = (strength>1.7) & (light>minimum_luma) & safe[np.rint(points).astype(int), columns]
            held = (columns>=columns[0]+(columns[-1]-columns[0]+1)*2//3) & good
            if held.sum() < 6:
                raise ValueError(f'Too few visible holdout points at frame {index}.')
            trial_slope, intercept = fit_line(columns, points, good & ~held)
            error = float(np.abs(points[held]-(trial_slope*columns[held]+intercept)).mean())
            slope, _ = fit_line(columns, points, good.copy())
            if parallel is not None:
                rows, strength = edge_points(luma, *alternate)
                known = safe[np.rint(rows).astype(int), alternate[0]]
                slope2, _ = fit_line(alternate[0], rows, (strength>3.5)&known)
                slope = slope*(1-weight)+slope2*weight
            slope = float(np.clip(slope, slope_min, slope_max))
            donors = columns[good & (columns>=donor_left) & (columns<donor_right)]
            if len(donors)<12:
                raise ValueError(f'Too few visible donor columns at frame {index}.')
            guess = extend_strip(picture, slope, donors, x-cx, width)
            observed = picture[:, x-cx:x-cx+width]
            visible = safe[:, x-cx:x-cx+width]
            anchors = (visible & (ring>0)).astype(np.float32)
            repaired = np.stack([match_boundary(a, b, anchors) for a, b in zip(
                observed.transpose(2, 0, 1), guess.transpose(2, 0, 1), strict=True)], axis=-1)[:height]
            known = feather(visible[:height].astype(np.uint8), 6)[..., None]
            result = repaired*(1-known)+observed[:height]*known
            predictions.append(np.rint(result).clip(0, 255).astype(np.uint8))
            measurements.append(error)

    save_npz(output, region=np.asarray(region), frames=np.asarray([first, last]),
             prediction=np.stack(predictions), alpha=np.repeat(opacity[None], last-first, axis=0),
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'],
             config_sha256=digest(config_path), edge_holdout_mae=np.asarray(measurements))
    print(json.dumps(dict(frames=[first, last], heldout_edge_mae_median=float(np.median(measurements)),
                          saved_patch=str(output.resolve())), indent=2), flush=True)
