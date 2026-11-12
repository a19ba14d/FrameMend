"""Inpaint an opaque mark that is attached to the scene within a shot."""

import json
from pathlib import Path

import cv2
import numpy as np

from wm_animation import read_rgb
from wm_inpaint import ProPainter, feather
from wm_matte import Matte
from wm_media import Video, digest
from wm_repair import Corners, dilate, windows
from wm_track import check_rect, corner_activity, load_profile, load_track, recover_corner_track, save_npz


def masks_for_clip(context, settings, rect, start, track, corner):
    x, y, width, height = rect
    stop = start + len(context)
    keys = np.asarray(settings['bounds'], dtype=np.int32)
    if (keys.ndim != 2 or keys.shape[1] != 5 or not len(keys) or np.any(np.diff(keys[:, 0]) <= 0)
            or not start <= keys[0, 0] <= keys[-1, 0] < stop):
        raise ValueError('Region bounds must have increasing frame numbers within the patch.')
    bounds = np.column_stack([np.interp(np.arange(start, stop), keys[:, 0], column)
                              for column in keys[:, 1:].T]).round().astype(int)
    if (np.any(bounds[:, :2] < (x, y)) or np.any(bounds[:, 2:] > (x+width, y+height))
            or np.any(bounds[:, :2] > bounds[:, 2:])):
        raise ValueError('Every mask bound must be within its patch region.')
    masks = np.zeros(context.shape[:3], np.uint8)
    corner_active = corner_activity(corner, track['corners'][start:stop])
    corner_polygons = [np.asarray(polygon, np.int32) - (x, y) for polygon in corner.get('masks', [])]
    floor = settings.get('protect_luma_below')
    if floor is not None and (type(floor) is not int or not 1 <= floor <= 175):
        raise ValueError('The dark-detail protection level must be between 1 and 175.')
    for i, ((left, top, right, bottom), mask) in enumerate(zip(bounds, masks, strict=True)):
        mask[top-y:bottom-y, left-x:right-x] = 1
        if floor is not None:
            dark = (cv2.cvtColor(context[i], cv2.COLOR_RGB2GRAY) < floor).astype(np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if settings.get('protect_connected_foreground'):
                dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                _, labels = cv2.connectedComponents(dark)
                ids = np.unique(labels[-20:])
                dark = np.isin(labels, ids[ids != 0]).astype(np.uint8)
            mask[dilate(dark, 2) > 0] = 0
        if settings.get('cover_corner'):
            for active, polygon in zip(corner_active[i], corner_polygons, strict=True):
                if active:
                    extra = np.zeros_like(mask)
                    cv2.fillPoly(extra, [polygon], 1)
                    mask |= dilate(extra, 12)
    return masks


def fill_columns(image, mask):
    """Carry the observed wall's light and shadow down each masked column."""
    height, width = mask.shape
    active = mask.any(axis=0)
    start = mask.argmax(axis=0)
    stop = height - mask[::-1].argmax(axis=0)
    x = np.arange(width)
    distance = np.arange(7, 28)[:, None]
    upper = np.median(image[np.clip(start-distance, 0, height-1), x], axis=0)
    lower = np.median(image[np.clip(stop+distance, 0, height-1), x], axis=0)
    upper[start < 16] = lower[start < 16]
    lower[stop > height-16] = upper[stop > height-16]
    ratio = np.clip((np.arange(height)[:, None]-start) / np.maximum(stop-start, 1), 0, 1)
    fill = upper[None]*(1-ratio[..., None]) + lower[None]*ratio[..., None]
    image[:, active] = np.rint(fill[:, active]).astype(np.uint8)


def top_texture(image, mask):
    """Extend observed texture from below a top-edge mask along its measured direction."""
    active = mask.any(axis=0)
    if not active.any():
        return image
    if not np.all(mask[0, active]):
        raise ValueError('Texture extension needs a mask connected to the top edge.')
    h, w = mask.shape
    boundary = h - mask[::-1].argmax(axis=0)
    below = int(np.max(boundary[active]))+5
    if below+20 > h:
        raise ValueError('Texture extension needs clean rows below the mask.')
    luma = cv2.cvtColor(image[below:min(h, below+64), active], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    slope = float(np.clip(-np.mean(gx*gy)/max(1, np.mean(gx**2)), -2.5, 2.5))
    yy, xx = np.indices((h, w), dtype=np.float32)
    edge = boundary[None].astype(np.float32)+2
    my = edge + np.maximum(edge-yy, 0)*.27
    mx = xx+slope*(my-yy)
    filled = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    left, right = np.flatnonzero(active)[[0, -1]]
    for x, direction in ((left, -1), (right, 1)):
        samples = x+direction*np.arange(7, 27)
        if samples.min() < 0 or samples.max() >= w:
            continue
        inside = x-direction*np.arange(3, 13)
        offset = image[:, samples].mean(axis=1)-filled[:, inside].mean(axis=1)
        offset = cv2.GaussianBlur(offset[:, None], (0, 0), 5)[:, 0].clip(-44, 44)
        weight = np.maximum(0, 1-np.abs(np.arange(w)-x)/72)
        filled += offset[:, None]*weight[None, :, None]
    return np.clip(np.rint(filled), 0, 255).astype(np.uint8)


def fit_wall(image, samples, mask=None):
    """Estimate a smooth wall color from designated visible background strips."""
    height, width = image.shape[:2]
    coordinates, colors = [], []
    for x, y, w, h in samples:
        rows, cols = np.mgrid[y:y+h:4, x:x+w:4]
        patch = image[rows, cols].astype(np.float32)
        median = np.median(patch, axis=(0, 1))
        good = (np.abs(patch-median).max(axis=2) < 17) & (patch.mean(axis=2) > 80)
        if good.sum() >= 12:
            coordinates.append(np.column_stack((np.ones(good.sum()), cols[good]/width-.5, rows[good]/height-.5)))
            colors.append(patch[good])
    if not colors:
        raise ValueError('No clean wall samples remain in the configured strips.')
    design, observations = np.concatenate(coordinates), np.concatenate(colors)
    regularization = np.diag([0, .015, .09])*len(design)
    plane = np.linalg.solve(design.T @ design + regularization, design.T @ observations)
    plane[1:] = np.clip(plane[1:], -48, 48)
    xx, yy = np.arange(width)[None, :, None], np.arange(height)[:, None, None]
    surface = plane[0] + (xx/width-.5)*plane[1] + (yy/height-.5)*plane[2]
    if mask is not None and mask.any():
        bottom = np.flatnonzero(mask.any(axis=1))[-1]+1
        if 3 <= bottom <= height-24:
            boundary = np.median(image[bottom+3:bottom+21], axis=0)
            visible = (mask[bottom-3] > 0) & (boundary.mean(axis=1) > 75)
            columns = np.flatnonzero(visible)
            if len(columns) >= 16:
                residual = np.clip(boundary-surface[bottom+11], -54, 54)
                correction = np.column_stack([np.interp(xx.ravel(), columns, c[visible]) for c in residual.T])
                correction = cv2.GaussianBlur(correction[None], (0, 0), 11)[0]
                surface += correction[None] * (.8 + .2 * np.minimum(yy/bottom, 1))
    return np.clip(np.rint(surface), 0, 255).astype(np.uint8)


def prepare_region_patch(source, output, profile_path, workdir, config_path, cuda=True, job=None):
    video = Video.open(source)
    profile = load_profile(video, profile_path)
    track = load_track(video, profile_path, Path(workdir) / 'track.npz')
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    settings = config['jobs'][job] if job is not None else config
    rect = check_rect(settings['region'], video.width, video.height)
    lo, hi = settings['frames']
    scale = settings.get('scale', .55)
    if not 0 <= lo < hi <= min(video.count, lo+240) or not .25 <= scale <= 1:
        raise ValueError('Region patches need up to 240 in-range frames and a supported model scale.')
    if Path(output).exists():
        raise FileExistsError(output)
    root = Path(workdir)
    matte_path = root/'matte.npz'
    if settings.get('recover_corner_overlap'):
        track = recover_corner_track(video, profile, track, cuda)
    matte = (Matte(profile, track, matte_path)
             if profile.get('translucent') or profile.get('stationary') else None)
    corners = Corners.from_cache(video, profile, track, root/'corners', matte_path) if settings.get('use_corner_cache') else None
    context = read_rgb(video, lo, hi, rect, cuda, matte, corners)
    masks = masks_for_clip(context, settings, rect, lo, track, profile.get('corners', {}))
    if not masks.any():
        raise ValueError('The region patch has no visible mask.')
    method = settings.get('method', 'model')
    print(f'Region {job or "patch"}, {method} reconstruction, frames {lo}:{hi}, region {rect}.', flush=True)
    if method == 'column':
        prediction = context.copy()
        for image, mask in zip(prediction, masks, strict=True):
            fill_columns(image, mask)
    elif method == 'texture-edge':
        prediction = np.stack([top_texture(image, mask) for image, mask in zip(context, masks, strict=True)])
    elif method == 'plane':
        samples = [check_rect(rectangle, *rect[2:]) for rectangle in settings['wall_samples']]
        prediction = np.stack([fit_wall(image, samples, mask if settings.get('match_wall_boundary') else None)
                               for image, mask in zip(context, masks, strict=True)])
    elif method == 'reference':
        ref = settings['reference_frame']
        if (type(ref) is not int or not 0 <= ref < video.count or lo <= ref < hi
                or np.any((track['cuts'] > min(lo, ref)) & (track['cuts'] <= max(hi-1, ref)))):
            raise ValueError('A fixed region reference must be a clean frame outside the patch in the same shot.')
        plate = read_rgb(video, ref, ref+1, rect, cuda, matte)[0]
        prediction = np.repeat(plate[None], len(context), axis=0)
    elif method == 'model':
        cuts = track['cuts'][(track['cuts'] > lo) & (track['cuts'] < hi)] - lo
        model = ProPainter(scale=scale)
        prediction = np.zeros_like(context)
        weights = np.zeros(len(context), np.float32)
        for window in windows(len(context), cuts, 56, 7):
            if not masks[window.out_start:window.out_stop].any():
                continue
            inputs = slice(window.in_start, window.in_stop)
            result = model.predict(context[inputs], masks[inputs])
            for i in range(window.out_start, window.out_stop):
                weight = window.weight(i)
                mix = weight / (weight + weights[i])
                prediction[i] = np.rint(prediction[i]*(1-mix) + result[i-window.in_start]*mix).astype(np.uint8)
                weights[i] += weight
            print(f'Region {job or "patch"}, prepared frames {lo+window.out_start}:{lo+window.out_stop}', flush=True)
    else:
        raise ValueError('Choose model, column, texture-edge, plane or reference region reconstruction.')
    soft_edge = settings.get('feather', 7)
    if type(soft_edge) is not int or not 1 <= soft_edge <= 32:
        raise ValueError('Region feather must be between 1 and 32 pixels.')
    alpha = np.stack([np.rint(feather(mask, soft_edge)*255).astype(np.uint8) for mask in masks])
    save_npz(output, region=np.asarray(rect), frames=np.asarray([lo, hi]), prediction=prediction, alpha=alpha,
             source_sha256=track['source_sha256'], profile_sha256=track['profile_sha256'], config_sha256=digest(config_path))
    areas = np.count_nonzero(masks, axis=(1, 2))
    print(f'Region {job or "patch"}, {np.count_nonzero(areas)} masked frames, '
          f'maximum mask {areas.max()} pixels. Saved {output}', flush=True)
