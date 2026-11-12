"""Restore fixed subtitle coverage over a reconstructed YUV background."""

import cv2
import numpy as np

from wm_inpaint import feather, match_boundary
from wm_media import rgb_to_yuv


def estimate_layer(parts, background, guard):
    y, u, v = [p.astype(np.float32) for p in parts]
    bg_y, bg_u, bg_v = [p.astype(np.float32) for p in background]
    dark = (bg_y-y) / np.maximum(bg_y-16, 24)
    light = (y-bg_y) / np.maximum(235-bg_y, 24)
    opacity = np.maximum(dark, light)

    # White fill and black outline both attenuate the chroma of the background.
    bu, bv = bg_u-128, bg_v-128
    power = bu*bu + bv*bv
    attenuation = 1 - (bu*(u-128) + bv*(v-128)) / np.maximum(power, 1)
    color = np.clip(attenuation, 0, 1) * np.clip((power-120)/350, 0, 1)
    color = cv2.resize(color, y.shape[::-1], interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(np.maximum(opacity, color*np.clip((opacity+0.06)/0.20, 0, 1)), 0, 1)
    alpha *= np.clip((np.abs(y-bg_y)-4)/11, 0, 1) * guard
    alpha *= cv2.dilate((alpha > 0.70).astype(np.uint8), np.ones((5, 5), np.uint8))
    fill = np.clip(y - (1-alpha)*bg_y, alpha*16, alpha*235)
    return alpha, fill


def composite_layers(planes, prediction, support, matrix, strength, top_weight, layer=None, ring=None):
    alpha = feather(support.astype(np.uint8), 5) * top_weight
    targets = rgb_to_yuv(prediction, matrix)
    if ring is not None:
        targets = tuple(match_boundary(plane, target, ring) for plane, target in zip(planes, targets, strict=True))

    def subsample(value):
        return cv2.resize(value, planes[1].shape[::-1], interpolation=cv2.INTER_AREA)

    if layer is not None:
        opacity, fill = layer
        chroma = 1 - subsample(opacity)
        y, u, v = (p.astype(np.float32) for p in targets)
        targets = (y*(1-opacity)+fill, 128+(u-128)*chroma, 128+(v-128)*chroma)
        alpha = np.maximum(alpha, support * opacity * 0.97)
    alpha *= strength
    color_alpha = subsample(alpha)
    for plane, target, weight in zip(planes, targets, (alpha, color_alpha, color_alpha), strict=True):
        active = weight > 0
        blend = weight[active]
        plane[active] = np.clip(np.rint(plane[active]*(1-blend) + target[active]*blend), 0, 255).astype(np.uint8)
