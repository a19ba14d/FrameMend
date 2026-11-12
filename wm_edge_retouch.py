"""Repair residual marks while retaining nearby image structure."""

import cv2
import numpy as np


def polish_texture(rgb, strength):
    picture = rgb.astype(np.float32)
    guide = cv2.GaussianBlur(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), (0, 0), 1.85)

    def mean(value):
        return cv2.boxFilter(value, -1, (9, 9))

    average = mean(guide)
    spread = mean(guide*guide)-average**2
    color = mean(picture)
    covariance = mean(picture*guide[..., None])-color*average[..., None]
    slope = covariance/(spread[..., None]+2.8)
    offset = color-slope*average[..., None]
    guided = mean(slope)*guide[..., None]+mean(offset)
    dx = cv2.Sobel(guide, -1, 1, 0)/8
    dy = cv2.Sobel(guide, -1, 0, 1)/8
    detail = np.hypot(dx, dy)
    opacity = np.minimum(1, strength*(.48+.48/(1+(detail/3.5)**2)))
    return picture+(guided-picture).clip(-13, 13)*opacity[..., None]


def fit_rising_edge(plane, first, last):
    blurred = cv2.GaussianBlur(plane.astype(np.float32), (0, 0), 1.55)
    columns = np.arange(18, plane.shape[1]-20)
    rows, positions = [], []
    for y in range(max(first, 0), min(last, len(plane))):
        line = blurred[y]
        edge = ((line[columns-16] < 145) & (line[columns+15] > 180)
                & (line[columns-1] < 160) & (line[columns+1] >= 160))
        if edge.any():
            rows.append(y)
            positions.append(columns[edge][-1])
    if len(rows) < 20:
        return None
    rows, positions = np.asarray(rows), np.asarray(positions)
    inlier = np.ones(len(rows), dtype=bool)
    for _ in range(5):
        if inlier.sum() < 20:
            return None
        curve = np.polyfit(rows[inlier], positions[inlier], 2)
        error = np.abs(positions-np.polyval(curve, rows))
        inlier = error <= max(2, np.median(error)*3)
    slope = np.polyval(np.polyder(curve), (first, last))
    if np.median(error) > 3.5 or inlier.mean() < 0.60 or not np.all((0.45 < slope) & (slope < 3.8)):
        return None
    return curve


def repair_along_edge(destination, context, support, x, y, curve, rows, scale):
    height, width = destination.shape
    low, high = (y + r/scale for r in rows)
    marked = cv2.dilate(support.astype(np.float32), np.ones((9, 9), np.uint8))
    unknown = np.zeros_like(context, dtype=np.float32)
    unknown[y:y+height, x:x+width] = marked
    length, reach, span = len(context), 128//scale, 190//scale
    sample_y, sample_x = np.mgrid[:length, :span].astype(np.float32)
    sample_x += np.polyval(curve, sample_y*scale)/scale - reach
    aligned = cv2.remap(context, sample_x, sample_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    safe = cv2.remap(unknown, sample_x, sample_y, cv2.INTER_LINEAR) == 0
    safe &= (sample_x >= 1) & (sample_x < context.shape[1]-1)
    axis = np.arange(length)[:, None]
    above = np.maximum.accumulate(np.where(safe, axis, -1), axis=0)
    below = np.minimum.accumulate(np.where(safe, axis, length)[::-1], axis=0)[::-1]
    top, bottom = (np.take_along_axis(aligned, np.clip(index, 0, length-1), axis=0)
                   for index in (above, below))
    fraction = np.clip((axis-above)/np.maximum(below-above, 1), 0, 1)
    estimate = (top + (bottom-top)*fraction).astype(np.float32)
    smoothed = cv2.GaussianBlur(estimate, (0, 0), 3/scale, 0.75)
    distance = np.arange(span)[None] - reach
    smooth_weight = np.clip((np.abs(distance)-11/scale)/(19/scale), 0, 1).astype(np.float32)
    estimate += (smoothed-estimate)*smooth_weight
    agreement = np.clip((42-np.abs(bottom-top))/(28 if scale == 1 else 25), 0, 1)
    agreement *= (above >= 0) & (below < length)

    target_y, target_x = np.mgrid[y:y+height, x:x+width].astype(np.float32)
    target_x -= np.polyval(curve, target_y*scale)/scale - reach
    prediction = cv2.remap(estimate, target_x, target_y, cv2.INTER_LINEAR)
    confidence = cv2.remap(agreement, target_x, target_y, cv2.INTER_LINEAR)
    across = np.clip(np.minimum(target_x-8/scale, span-4/scale-target_x)/(18/scale), 0, 1)
    along = np.clip(np.minimum(target_y-low, high-target_y)/(5/scale), 0, 1)
    delta = prediction-destination
    trust = np.clip((49-np.abs(delta))/26, 0, 1)
    mask = cv2.GaussianBlur(marked, (5, 5), 1)
    destination[:] = np.rint(destination + delta*confidence*across*along*trust*mask).astype(np.uint8)
