"""Learn distinct overlay designs from local proposals and retain simultaneous copies."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
import json
from math import ceil, floor
import re

import cv2
import numpy as np
from PIL import Image

from wm_discovery import (AmbiguousPeak, MAX_LAYERS, MAX_TEMPLATES, MAX_TRACKS, Motif, absence_probes, agrees_in_halves,
                          aligned_consensus, chroma_signals, chroma_templates, color_patch, consensus_fields, patch,
                          peaks, response, strong_match)
from wm_gpu_search import pixel_search_frames, search_input, search_padding
from wm_grounding import classify_candidate, filled_mask_evidence, frame_evidence, jpeg, locate_source_overlays, preview_residue
from wm_media import clipped_plane, get_frame, read_frames
from wm_overlap import coherent_overlaps, exposed_match
from wm_track import highpass, save_npz


IDENTIFIER = re.compile(r'(?<![\w@])(?:@[\w.-]{3,}|(?:[a-z0-9-]+\.)+[a-z]{2,})(?![\w@])', re.I)
ACCOUNT_ID = re.compile(r'(?:(?<!\w)(?:account(?:\s+(?:id|number))?|(?:creator|channel)\s+id)|'
                        r'[\u3400-\u9fff]{2,10}[号號])\s*[:：#]?\s*([0-9]{7,15})(?![0-9])', re.I)
INDEPENDENT_CONTEXT = .65
MASKED_REVIEW_MINIMUM = .47


@dataclass(frozen=True)
class Observation:
    frame: int
    rect: tuple[int, int, int, int]


@dataclass
class Layer:
    motif: Motif
    floor: float
    matches: dict
    label: str = ''
    reviewed_regions: tuple = ()
    identifier_support: tuple[Motif, int, int] | None = None


class LayerNeedsEvidence(ValueError):
    def __init__(self, layer, frames, reason, hints=None, *, partial_mask=False, visible_frames=(), confirmed=(),
                 needs_classification=False, hidden_masks=None):
        super().__init__(reason)
        self.layer = layer
        self.frames = frames
        self.hints = hints or {}
        self.partial_mask = partial_mask
        self.visible_frames = visible_frames
        self.confirmed = confirmed
        self.needs_classification = needs_classification
        self.hidden_masks = hidden_masks or {}


def proposals_for(video, rows):
    proposals = []
    for i, boxes in sorted(rows.items()):
        if not 0 <= i < video.count:
            raise ValueError('A local proposal refers to a frame outside the source.')
        accepted = set()
        for row in boxes:
            left, top, right, bottom = row['rect']
            x, y = max(0, floor(left)), max(0, floor(top))
            w, h = min(video.width, ceil(right))-x, min(video.height, ceil(bottom))-y
            rect = (x, y, w, h)
            if rect not in accepted and min(w, h) >= 8 and w*h <= .18*video.width*video.height:
                proposals.append(Observation(i, rect))
                accepted.add(rect)
                if len(accepted) == 6:
                    break
    return proposals


def overlap(a, b):
    x, y, w, h = a
    u, v, width, height = b
    return max(0, min(x+w, u+width)-max(x, u))*max(0, min(y+h, v+height)-max(y, v))


def remainder_observations(video, rows, observations):
    reported = proposals_for(video, {i: [row for row in boxes if row.get('provider') == 'qwen-visible-remainder']
                                     for i, boxes in rows.items()})
    return [item for item in observations if any(
        overlap(item.rect, target.rect) >= .45*min(item.rect[2]*item.rect[3], target.rect[2]*target.rect[3])
        for target in reported)]


def visible_box(motif, x, y):
    left, top, width, height = motif.bounds
    return x+left, y+top, width, height


def trim_empty_motif_padding(video, layer):
    motif = layer.motif
    if motif.width <= video.width and motif.height <= video.height:
        return layer
    support = ((motif.weights != 0) | (motif.mask != 0)).astype(np.uint8)
    identifier = layer.identifier_support
    if identifier is not None:
        text, u, v = identifier
        support[v:v+text.height, u:u+text.width] = 1
    left, top, width, height = cv2.boundingRect(support)
    width, height = max(8, width), max(8, height)
    if width > video.width or height > video.height:
        pad_x, pad_y = search_padding(motif)
        if (motif.width > video.width+2*pad_x or motif.height > video.height+2*pad_y
                or any(not clipped_plane(motif.mask, (-x, -y, video.width, video.height))[0].any()
                       for x, y, _ in layer.matches.values())):
            raise ValueError('The clipped watermark does not fit the supported tracking canvas.')
        return layer
    left, top = min(left, motif.width-width), min(top, motif.height-height)
    region = np.s_[top:top+height, left:left+width]
    fitted = Motif(motif.signal[region].copy(), motif.weights[region].copy(), motif.mask[region].copy())
    # Moving the stored origins by the removed padding preserves every in-frame mask pixel.
    matches = {item: (x+left, y+top, score) for item, (x, y, score) in layer.matches.items()}
    if identifier is not None:
        identifier = text, u-left, v-top
    return replace(layer, motif=fitted, matches=matches, identifier_support=identifier)


def source_sized_tracks(video, track):
    height, width = track['mask'].shape
    fit_width, fit_height = min(width, video.width), min(height, video.height)
    if (width, height) == (fit_width, fit_height):
        return [track]
    present, positions = track['present'], track['positions']
    # A window of source size can contain every visible pixel at each verified origin.
    origins = np.clip(-positions.astype(np.int64), 0, (width-fit_width, height-fit_height))
    windows = np.unique(origins[present], axis=0)
    if not 1 <= len(windows) <= MAX_TRACKS:
        raise ValueError('The clipped watermark needs too many distinct repair windows.')
    output = []
    for left, top in windows:
        active = present & (origins == (left, top)).all(axis=1)
        region = np.s_[top:top+fit_height, left:left+fit_width]
        motif = Motif(*(np.ascontiguousarray(track[name][region]) for name in ('signal', 'weights', 'mask')))
        points = np.zeros_like(positions)
        points[active] = positions[active]+(left, top)
        if any(not clipped_plane(motif.mask, (-x, -y, video.width, video.height))[0].any()
               for x, y in np.unique(points[active].astype(np.int64), axis=0)):
            raise ValueError('A verified clipped watermark has no visible mask pixels.')
        output.append({**track, 'positions': points, 'present': active, 'scores': np.where(active, track['scores'], 0),
                       'signal': motif.signal, 'weights': motif.weights, 'mask': motif.mask})
    return output


def save_layer_evidence(path, layers, provenance):
    metadata, arrays = [], {}
    for index, layer in enumerate(layers):
        anchors = [[int(item.frame), *map(int, item.rect), int(x), int(y), float(score)]
                   for item, (x, y, score) in layer.matches.items()]
        metadata.append({'label': layer.label, 'floor': layer.floor, 'anchors': anchors,
                         'regions': [list(map(int, row)) for row in layer.reviewed_regions]})
        for field in ('signal', 'weights', 'mask'):
            arrays[f'{index}_{field}'] = getattr(layer.motif, field)
        if layer.identifier_support is not None:
            text, x, y = layer.identifier_support
            metadata[-1]['identifier_offset'] = [int(x), int(y)]
            for field in ('signal', 'weights', 'mask'):
                arrays[f'{index}_identifier_{field}'] = getattr(text, field)
    manifest = json.dumps({'provenance': provenance, 'layers': metadata}, ensure_ascii=False)
    save_npz(path, replace=True, manifest=manifest, **arrays)


def load_layer_evidence(path, video, identity, *, on_rejected=None):
    with np.load(path, allow_pickle=False) as saved:
        data = json.loads(str(saved['manifest']))
        provenance = data['provenance']
        if provenance['source_sha256'] != identity or not 1 <= len(data['layers']) <= MAX_TEMPLATES:
            raise ValueError('Confirmed motif evidence does not match this source.')
        def load_record(index, record):
            motif = Motif(*(saved[f'{index}_{name}'] for name in ('signal', 'weights', 'mask')))
            matches = {}
            for frame, left, top, width, height, x, y, score in record['anchors']:
                if (not all(type(n) is int for n in (frame, left, top, width, height, x, y))
                        or not 0 <= frame < video.count or not .0 <= score <= 1.01 or min(width, height) <= 0
                        or not overlap(visible_box(motif, x, y), (0, 0, video.width, video.height))):
                    raise ValueError('A saved motif anchor is invalid for this source.')
                matches[Observation(frame, (left, top, width, height))] = x, y, score
            label = record['label']
            if (len(diverse_frames(matches)) < 3 or not .0 < record['floor'] < 1
                    or not isinstance(label, str) or not label.strip()):
                raise ValueError('A saved motif lacks independent source anchors.')
            regions = tuple(tuple(row) for row in record['regions'])
            if any(len(row) != 5 or any(type(n) is not int for n in row) or not 0 <= row[0] < video.count
                   or not overlap(row[1:], (0, 0, video.width, video.height)) for row in regions):
                raise ValueError('A saved motif has an invalid reviewed region.')
            support = None
            if 'identifier_offset' in record:
                x, y = record['identifier_offset']
                text = Motif(*(saved[f'{index}_identifier_{field}'] for field in ('signal', 'weights', 'mask')))
                if (type(x) is not int or type(y) is not int or x < 0 or y < 0
                        or x+text.width > motif.width or y+text.height > motif.height):
                    raise ValueError('A saved identifier mask is outside its watermark.')
                if np.any(text.mask & (motif.mask[y:y+text.height, x:x+text.width] == 0)):
                    raise ValueError('A saved identifier mask is not covered by its watermark.')
                support = text, x, y
            return trim_empty_motif_padding(video, Layer(motif, record['floor'], matches, label, regions, support))

        layers = []
        for index, record in enumerate(data['layers']):
            try:
                layers.append(load_record(index, record))
            except (ValueError, KeyError, TypeError, cv2.error) as error:
                if on_rejected is None:
                    raise
                on_rejected(layer=index+1, reason=str(error))
    return layers, provenance


def near(motif, signal, box, *, aligned=None):
    x, y, w, h = box
    a, b, tw, th = motif.bounds
    if aligned is None:
        if not (.48 <= w/tw <= 2.10 and .46 <= h/th <= 2.16):
            return None
        rx = min(42, max(6, w//7+abs(w-tw)//2))
        ry = min(26, max(5, h//5+abs(h-th)//2))
        position = (x+(w-tw)//2-a, y+(h-th)//2-b)
    else:
        position, rx, ry = aligned, 2, 2
    u, v, score = motif.local(signal, position, rx, ry)
    target = visible_box(motif, u, v)
    if overlap(target, box) < .62*min(tw*th, w*h):
        return None
    return u, v, score


def anchor_matches(motif, signals, observations, minimum):
    matches = {}
    for item in observations:
        if (candidate := near(motif, signals[item.frame], item.rect)) is not None and candidate[2] >= minimum:
            matches[item] = candidate
    return matches


def seed(observation, signals):
    x, y, w, h = observation.rect
    dx, dy = max(3, w//12), max(3, h//9)
    rect = x-dx, y-dy, w+2*dx, h+2*dy
    image = patch(signals[observation.frame], rect)
    weights = np.zeros(image.shape, np.float32)
    weights[dy:dy+h, dx:dx+w] = np.minimum(np.abs(image[dy:dy+h, dx:dx+w])/12, 1)
    return Motif(image, weights, (weights > .25).astype(np.uint8))


def diverse_frames(matches):
    by_frame = {}
    for observation, found in matches.items():
        current = by_frame.get(observation.frame)
        if current is None or found[2] > current[1][2]:
            by_frame[observation.frame] = observation, found
    return [row for _, row in sorted(by_frame.items())]


def build_layer(signals, observations):
    guides = []
    for item in observations:
        try:
            motif = seed(item, signals)
        except ValueError:
            continue
        anchors = anchor_matches(motif, signals, observations, .40)
        matched = diverse_frames(anchors)
        if len(matched) >= 3:
            strength = float(np.median([found[2] for _, found in matched]))
            guides.append((len(matched), strength, item.frame, motif, anchors, matched))
    for _, _, seed_frame, guide, anchors, matched in sorted(guides, key=lambda row: row[:2], reverse=True)[:8]:
        held_out, (x, y, _) = next(row for row in reversed(matched) if row[0].frame != seed_frame)
        placements = [(o.frame, u, v) for o, (u, v, _) in matched if o.frame != held_out.frame]
        try:
            motif, (dx, dy) = aligned_consensus(guide.width, guide.height, placements, signals)
        except ValueError:
            continue
        found = near(motif, signals[held_out.frame], held_out.rect, aligned=(x+dx, y+dy))
        if found is None or found[2] < .55 or not agrees_in_halves(motif, signals[held_out.frame], *found[:2]):
            continue
        threshold = max(.51, min(.73, found[2]*.80))
        corroborated = anchor_matches(motif, signals, observations, threshold)
        validated = {o: p for o, p in corroborated.items() if agrees_in_halves(motif, signals[o.frame], *p[:2])}
        for item, (x, y, _) in anchors.items():
            if item in validated:
                continue
            signal = signals[item.frame]
            found = near(motif, signal, item.rect, aligned=(x+dx, y+dy))
            if found is not None and found[2] >= threshold and agrees_in_halves(motif, signal, *found[:2]):
                validated[item] = found
        if len(diverse_frames(validated)) >= 3:
            return Layer(motif, threshold, validated)
    return None


def recover_layer(video, rows, observations, signals, cuda, trace):
    """Search the sampled source frames when model boxes disagree on a repeated shape."""
    if len(rows) < 3 or not observations:
        return None
    anchors = [observations[i] for i in np.linspace(0, len(observations)-1, min(8, len(observations)), dtype=int)]
    seeded = {item.frame for item in anchors}
    others = sorted(rows.keys()-seeded)
    additional = min(12-len(seeded), len(others))
    frames = sorted(seeded.union(others[i] for i in np.linspace(0, len(others)-1, additional, dtype=int)))
    for i in frames:
        if i not in signals:
            signals[i] = highpass(video.planes(get_frame(video, i, cuda))[0], 2.2)
    spacing = max(1, round(float(video.fps)/2))
    best = None
    for anchor in anchors:
        try:
            guide = seed(anchor, signals)
        except ValueError:
            continue
        placements = [anchor]
        for frame in frames:
            if frame == anchor.frame:
                continue
            x, y, score, alternative = guide.locate(signals[frame], coarse=True)
            left, top, width, height = visible_box(guide, x, y)
            if (score >= .45 and score-alternative >= .10 and left >= 0 and top >= 0
                    and left+width <= video.width and top+height <= video.height):
                placements.append(Observation(frame, (left, top, width, height)))
        if len(placements) < 3 or (layer := build_layer(signals, placements)) is None or anchor not in layer.matches:
            continue
        separated = []
        for frame in sorted({item.frame for item in layer.matches}):
            if not separated or frame-separated[-1] >= spacing:
                separated.append(frame)
        if len(separated) < 3:
            continue
        _, correlation, _ = review_anchors(layer, signals)
        if correlation >= INDEPENDENT_CONTEXT:
            continue
        strength = (len(separated), float(np.median([value[2] for value in layer.matches.values()])), -correlation)
        if best is None or strength > best[0]:
            best = strength, anchor.frame, separated, layer
    if best is not None:
        strength, anchor, separated, layer = best
        trace('layers.pixel_seed_recovered', anchor=anchor, frames=separated, median_score=round(strength[1], 4),
              context_correlation=round(-strength[2], 4))
        return layer
    return None


def complete_layer(layer, signals, trace, *, wide=False):
    anchors = diverse_frames(layer.matches)
    if len(anchors) < 4:
        return None
    core = layer.motif
    _, _, width, height = core.bounds
    if wide:
        pad_x, pad_y = min(180, max(22, 2*width//3)), min(130, max(22, 3*height))
    else:
        pad_x, pad_y = min(140, max(16, width//3)), min(100, max(16, 2*height))
    size = core.width+2*pad_x, core.height+2*pad_y
    aligned = np.stack([patch(signals[item.frame], (x-pad_x, y-pad_y, *size))
                        for item, (x, y, _) in anchors])
    signal, weights, mask = consensus_fields(aligned[:-1])
    core_slice = np.s_[pad_y:pad_y+core.height, pad_x:pad_x+core.width]
    existing = np.zeros(mask.shape, np.uint8)
    existing[core_slice] = core.mask
    # The bounding box can already enclose an unmasked, disconnected part of the mark.
    protected = cv2.dilate(existing, np.ones((3, 3), np.uint8)).astype(bool)
    weights[protected] = 0
    mask[protected] = 0
    added = int(np.count_nonzero(weights))
    holdout = float(response(aligned[-1], signal, weights)[0, 0])
    trace('layers.extension_checked', label=layer.label, anchors=len(anchors), added_pixels=added, wide=wide,
          held_out_frame=anchors[-1][0].frame, held_out_score=round(holdout, 4))
    if added < 32 or holdout < .52:
        return None
    signal[existing > 0] = core.signal[core.mask > 0]
    weights[existing > 0] = core.weights[core.mask > 0]
    mask |= existing
    x, y, w, h = cv2.boundingRect(mask)
    if w*h > .18*signals[anchors[0][0].frame].size:
        return None
    x0, y0 = max(0, x-3), max(0, y-3)
    region = np.s_[y0:y+h+3, x0:x+w+3]
    motif = Motif(signal[region].copy(), weights[region].copy(), mask[region].copy())
    dx, dy = x0-pad_x, y0-pad_y
    matches = {}
    for item, (x, y, _) in layer.matches.items():
        a, b, score = motif.local(signals[item.frame], (x+dx, y+dy), 1, 1)
        if score >= layer.floor and agrees_in_halves(motif, signals[item.frame], a, b):
            matches[item] = (a, b, score)
    if len(diverse_frames(matches)) < max(3, ceil(.70*len(anchors))):
        return None
    return Layer(motif, layer.floor, matches, layer.label)


def proposed_extension(core, anchors, observations, px, py):
    """Keep local-only additions inside visual proposals that agree across source frames."""
    by_frame = defaultdict(list)
    for item in observations:
        by_frame[item.frame].append(item.rect)
    shape = core.height+2*py, core.width+2*px
    votes = np.zeros(shape, np.int32)
    for item, (x, y, _) in anchors:
        footprint = visible_box(core, x, y)
        supported = np.zeros(shape, np.uint8)
        for rect in by_frame[item.frame]:
            left, top, w, h = rect
            if overlap(rect, footprint) < .35*min(w*h, footprint[2]*footprint[3]):
                continue
            inside, _ = clipped_plane(supported, (left-x+px, top-y+py, w, h))
            inside[:] = 1
        votes += supported
    return votes >= ceil(2*len(anchors)/3)


def complete_sampled_layer(video, layer, signals, cuda, trace, *, observations=()):
    """Verify disconnected additions while keeping the tracking signature of the existing strokes."""
    core = layer.motif
    _, _, width, height = core.bounds
    px = min((video.width-core.width)//2, max(24, width//3))
    py = min((video.height-core.height)//2, max(18, height//2))
    original = diverse_frames(layer.matches)
    if min(px, py) < 4 or len(original) < 3:
        return None
    matches = dict(layer.matches)
    sampled = dict(signals)
    seen = {item.frame for item, _ in original}
    chroma_available = False

    def recruit(frames):
        nonlocal chroma_available
        for i, raw in frames:
            if i in seen:
                continue
            planes = video.planes(raw)
            detail = highpass(planes[0], 2.2)
            for x, y, score in locations(core, detail, limit=3):
                if score >= max(.67, layer.floor) and agrees_in_halves(core, detail, x, y):
                    matches[Observation(i, visible_box(core, x, y))] = x, y, score
                    sampled[i] = detail
                    seen.add(i)
                    chroma_available |= len(planes) == 3
                    break

    recruit(read_frames(video, cuda=cuda, step=max(1, ceil(video.count/48))))
    needed = max(9, len(original)+3)
    allowed = None
    outside = set()
    if len(seen) < needed:
        if not observations:
            return None
        radius = max(2, round(float(video.fps)/2))
        first, last = original[0][0].frame, original[-1][0].frame
        start, stop = max(0, first-radius), min(video.count, last+radius+1)
        if last-first > 3*radius:
            return None
        outside = {i for i in seen if not start <= i < stop}
        px = min(180, max(px, 2*width//3))
        allowed = proposed_extension(core, original, observations, px, py)
        if not allowed.any():
            return None
        recruit(read_frames(video, start, stop, cuda, step=max(1, ceil((stop-start)/32))))
        trace('layers.local_extension_window', label=layer.label, start=start, stop=stop,
              anchors=len(seen)-len(outside), independent_frames=sorted(outside), proposal_pixels=int(allowed.sum()))
    anchors = diverse_frames(matches)
    if len(anchors)-len(outside) < needed:
        return None
    size = core.width+2*px, core.height+2*py
    placements = [(x-px, y-py, *size) for _, (x, y, _) in anchors]
    aligned = np.stack([patch(sampled[item.frame], rect) for (item, _), rect in zip(anchors, placements, strict=True)])
    visible = None
    if any(x < 0 or y < 0 or x+w > video.width or y+h > video.height for x, y, w, h in placements):
        visible = np.zeros(aligned.shape, bool)
        for index, ((item, _), rect) in enumerate(zip(anchors, placements, strict=True)):
            _, section = clipped_plane(sampled[item.frame], rect)
            visible[index][section] = True
    remote = [i for i, (item, _) in enumerate(anchors) if item.frame in outside]
    local = [i for i, (item, _) in enumerate(anchors) if item.frame not in outside]
    selected = np.linspace(1, len(local)-1, min(5, len(local)//3), dtype=int)
    held = [local[i] for i in selected]
    train = np.ones(len(anchors), bool)
    train[held+remote] = False
    padding = ((py, py), (px, px))
    prior = np.pad(core.mask, padding)
    evidence = []

    def verified_extensions(samples, channel):
        signal, weights, mask = consensus_fields(samples[train], visible[train] if visible is not None else None)
        if allowed is not None:
            mask &= allowed
        _, groups, stats, _ = cv2.connectedComponentsWithStats(mask & (prior == 0))
        extra = np.zeros_like(mask)
        for component, (x, y, w, h, area) in enumerate(stats[1:], 1):
            region = np.s_[y:y+h, x:x+w]
            island = groups[region] == component
            support = weights[region]*island
            if area < 12 or np.count_nonzero(support) < 12:
                continue
            mass = support.sum()

            def held_out_scores(indices):
                scores = []
                for i in indices:
                    observable = support if visible is None else support*visible[i][region]
                    if observable.sum() >= .8*mass:
                        scores.append(float(response(samples[i][region], signal[region], observable)[0, 0]))
                return scores

            scores = held_out_scores(held)
            if sum(s >= .58 for s in scores) < 2 or sum(s >= .36 for s in scores) < ceil(.75*len(scores)):
                continue
            cross_checks = held_out_scores(remote)
            if any(score < .52 for score in cross_checks):
                continue
            extra[region] |= island
            evidence.append({'rect': [int(x-px), int(y-py), int(w), int(h)], 'channel': channel,
                             'scores': [round(s, 4) for s in scores],
                             'independent_scores': [round(s, 4) for s in cross_checks]})
        return extra

    extra = verified_extensions(aligned, 'luma')
    if extra.sum() < 32 and chroma_available:
        color = [color_patch(video, get_frame(video, item.frame, cuda), *rect)
                 for (item, _), rect in zip(anchors, placements, strict=True)]
        extra |= verified_extensions(np.stack(color), 'render-chroma')
    mask = prior|extra
    x, y, w, h = cv2.boundingRect(mask)
    trace('layers.sampled_extension_checked', label=layer.label, previous_anchors=len(original), anchors=len(anchors),
          held_out=[anchors[i][0].frame for i in held], components=evidence, added_pixels=int(extra.sum()))
    if extra.sum() < 32 or w*h > .18*video.width*video.height:
        return None
    left, top = min(x, px), min(y, py)
    right, bottom = max(x+w, px+core.width), max(y+h, py+core.height)
    if right-left > video.width or bottom-top > video.height:
        return None
    region = np.s_[top:bottom, left:right]
    motif = Motif(np.pad(core.signal, padding)[region].copy(), np.pad(core.weights, padding)[region].copy(),
                  mask[region].copy())
    signals.update(sampled)
    positions = {item: (x-px+left, y-py+top, score) for item, (x, y, score) in matches.items()}
    return Layer(motif, layer.floor, positions, layer.label, layer.reviewed_regions)


def fit_translucency(video, layer, cuda):
    """Estimate a blend against the background and test it on held-out frames."""
    motif = layer.motif
    matches = diverse_frames(layer.matches)
    if len(matches) < 6:
        return None
    anchors = [matches[i] for i in np.linspace(0, len(matches)-1, min(12, len(matches)), dtype=int)]
    pixels = np.stack([patch(video.planes(get_frame(video, item.frame, cuda))[0],
                             (x, y, motif.width, motif.height)) for item, (x, y, _) in anchors])
    background = np.stack([cv2.medianBlur(image, 9) for image in pixels]).astype(np.float32)
    pixels = pixels.astype(np.float32)
    held = [len(anchors)//3, len(anchors)-1]
    training = np.ones(len(anchors), bool)
    training[held] = False
    background_samples, marked_samples = background[training], pixels[training]
    background_mean, marked_mean = background_samples.mean(axis=0), marked_samples.mean(axis=0)
    variation = background_samples-background_mean
    slope = np.sum(variation*(marked_samples-marked_mean), axis=0)/np.maximum(1., np.sum(variation*variation, axis=0))
    offset = marked_mean-slope*background_mean
    opacity = 1-slope
    foreground = offset/np.maximum(.02, opacity)
    null_error = np.abs(pixels[held]-background[held]).mean(axis=0)
    error = np.abs(pixels[held]-(slope*background[held]+offset)).mean(axis=0)
    support = ((background_samples.std(axis=0) >= 10) & (opacity >= .20) & (opacity <= .93)
               & (foreground >= 0) & (foreground <= 255) & (null_error >= 3.5)
               & (error <= np.minimum(4., .58*null_error)))
    return slope, offset, support, [anchors[i][0].frame for i in held], len(anchors)


def complete_translucent_mask(video, layer, cuda, trace):
    fit = fit_translucency(video, layer, cuda)
    if fit is None:
        return None
    _, _, support, held_out, samples = fit
    motif = layer.motif
    a, b, w, h = motif.bounds
    within = np.zeros(motif.mask.shape, bool)
    within[b:b+h, a:a+w] = True
    _, components, stats, _ = cv2.connectedComponentsWithStats((support&within).astype(np.uint8))
    repeated = np.isin(components, np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 6)+1).astype(np.uint8)
    expanded = cv2.dilate(repeated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = motif.mask | (expanded&within)
    added = int(np.count_nonzero(mask & (motif.mask == 0)))
    trace('layers.translucent_mask_checked', samples=samples, added_pixels=added, held_out_frames=held_out)
    if added < 32:
        return None
    return Layer(Motif(motif.signal, motif.weights, mask), layer.floor, layer.matches, layer.label)


def covered_fragment(layer, prior, signals, *, video=None, cuda=False):
    fragment, full = layer.motif, prior.motif
    _, _, w, h = fragment.bounds
    if w > 1.15*full.bounds[2] or h > 1.15*full.bounds[3]:
        return None
    blend, learned_blend = None, False
    for x, y, score in locations(fragment, full.signal, limit=3):
        footprint = visible_box(fragment, x, y)
        covered = patch(full.mask, (x, y, fragment.width, fragment.height))
        if (abs(score) < .72 or overlap(footprint, full.bounds) < .88*w*h
                or np.any((fragment.mask != 0)&(covered == 0))
                or not agrees_in_halves(fragment, full.signal, x, y, -1 if score < 0 else 1)):
            continue
        mapped = {}
        for item, (u, v, _) in layer.matches.items():
            image = signals[item.frame]
            origin = u-x, v-y
            a, b, strength = full.local(image, origin, 2, 2, reversible=True)
            if (abs(strength) >= max(.58, prior.floor-.06)
                    and agrees_in_halves(full, image, a, b, -1 if strength < 0 else 1)):
                mapped[item] = (a, b, abs(strength))
                continue
            if regional := regional_match(full, image, origin):
                mapped[item] = regional
                continue
            if not learned_blend:
                blend = translucent_reference(video, prior, cuda) if video is not None else None
                learned_blend = True
            if blend is None:
                break
            luma = video.planes(get_frame(video, item.frame, cuda))[0]
            match = translucent_match(full, luma, origin, blend)
            if match is None or match < .42:
                break
            mapped[item] = (*origin, match)
        else:
            return mapped
    return None


def merge_nearby_templates(layer, prior, signals, video, cuda):
    """Share one track for the same graphic when both approved masks can be kept."""
    names = candidate_identifiers(layer.label)
    if len(names) != 1 or names != candidate_identifiers(prior.label):
        return None
    identifier = next(iter(names))
    if identifier.startswith('account:'):
        number = identifier.removeprefix('account:')
        if any(not re.search(rf'(?<![0-9]){number}(?![a-zA-Z0-9_])', item.label) for item in (layer, prior)):
            return None
    fragment, full = layer.motif, prior.motif
    if abs(np.count_nonzero(fragment.mask)-np.count_nonzero(full.mask)) > .03*np.count_nonzero(full.mask):
        return None
    for dx, dy, score in locations(fragment, full.signal, limit=3):
        if score < .95 or not agrees_in_halves(fragment, full.signal, dx, dy):
            continue
        left, top = min(0, dx), min(0, dy)
        width = max(full.width, dx+fragment.width)-left
        height = max(full.height, dy+fragment.height)-top
        if width > video.width or height > video.height:
            continue
        padding = ((-top, height-full.height+top), (-left, width-full.width+left))
        signal, weights, first = (np.pad(field, padding) for field in (full.signal, full.weights, full.mask))
        second = np.zeros_like(first)
        x, y = dx-left, dy-top
        second[y:y+fragment.height, x:x+fragment.width] = fragment.mask
        union = first | second
        if np.count_nonzero(first & second) < .97*np.count_nonzero(union):
            continue
        margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        if np.any(first & (cv2.dilate(second, margin) == 0)) or np.any(second & (cv2.dilate(first, margin) == 0)):
            continue
        support = None
        if prior.identifier_support is not None:
            text, u, v = prior.identifier_support
            support = text, u-left, v-top
        if layer.identifier_support is not None:
            text, u, v = layer.identifier_support
            alternate = text, u+x, v+y
            if support is not None and (support[1:] != alternate[1:]
                                        or not np.array_equal(support[0].mask, alternate[0].mask)):
                continue
            support = support or alternate
        motif = Motif(signal, weights, union)
        floor = max(layer.floor, prior.floor)
        matches = {}
        for original, offset in ((prior, (left, top)), (layer, (left-dx, top-dy))):
            for item, (u, v, _) in original.matches.items():
                position = u+offset[0], v+offset[1]
                if item in matches and matches[item][:2] != position:
                    break
                if item.frame not in signals:
                    signals[item.frame] = highpass(video.planes(get_frame(video, item.frame, cuda))[0], 2.2)
                image = signals[item.frame]
                _, _, strength = motif.local(image, position, 0, 0, reversible=True)
                if (abs(strength) < max(.63, floor-.06)
                        or not agrees_in_halves(motif, image, *position, -1 if strength < 0 else 1)):
                    break
                matches[item] = (*position, abs(strength))
            else:
                continue
            break
        else:
            reviewed = tuple(dict.fromkeys(prior.reviewed_regions+layer.reviewed_regions))
            return Layer(motif, floor, matches, prior.label, reviewed, support)
    return None


def same_motif(a, b):
    return (np.array_equal(a.mask, b.mask) and np.array_equal(a.signal, b.signal)
            and np.array_equal(a.weights, b.weights))


def merge_identical_templates(layer, prior):
    if layer.label.strip().casefold() != prior.label.strip().casefold() or not same_motif(layer.motif, prior.motif):
        return None
    first, second = prior.identifier_support, layer.identifier_support
    if first and second and (first[1:] != second[1:] or not same_motif(first[0], second[0])):
        return None
    matches = dict(prior.matches)
    for item, match in layer.matches.items():
        existing = matches.get(item)
        if existing and match[:2] != existing[:2]:
            return None
        if existing is None or match[2] > existing[2]:
            matches[item] = match
    return replace(prior, matches=matches, floor=max(prior.floor, layer.floor), identifier_support=first or second,
                   reviewed_regions=tuple(dict.fromkeys(prior.reviewed_regions+layer.reviewed_regions)))


def retain_confirmed_layer(confirmed, layer, signals, video, cuda, trace):
    for i, prior in enumerate(confirmed):
        if merged := merge_identical_templates(layer, prior):
            confirmed[i] = merged
            trace('layers.identical_templates_merged', label=merged.label, anchors=len(merged.matches),
                  reviewed_regions=len(merged.reviewed_regions))
            return True
        if merged := merge_nearby_templates(layer, prior, signals, video, cuda):
            confirmed[i] = merged
            trace('layers.nearby_templates_merged', label=merged.label, anchors=len(merged.matches),
                  reviewed_regions=len(merged.reviewed_regions), mask_pixels=int(merged.motif.mask.sum()))
            return True
    return False


def crop_for(video, motif, position, proposal):
    x, y, w, h = visible_box(motif, *position)
    left, top, width, height = proposal
    pad_x, pad_y = max(10, w//6), min(110, 2*h+8)
    right = min(video.width, max(x+w, left+width)+pad_x)
    bottom = min(video.height, max(y+h, top+height)+pad_y)
    left, top = max(0, min(left, x)-pad_x), max(0, min(top, y)-pad_y)
    return left, top, right-left, bottom-top


def review_anchors(layer, signals):
    """Prefer independent backgrounds without swapping in weak watermark matches."""
    matches = diverse_frames(layer.matches)
    middle = len(matches)//2
    groups = matches[:middle], matches[middle:]
    strongest = [max(group, key=lambda row: row[1][2]) for group in groups]
    width, height = layer.motif.width+24, layer.motif.height+24
    outside = np.ones((height, width), bool)
    outside[12:-12, 12:-12] = False

    def context(row):
        observation, (x, y, _) = row
        return patch(signals[observation.frame], (x-12, y-12, width, height))[outside]

    def correlation(a, b):
        return float(np.dot(a, b)/max(1., np.linalg.norm(a)*np.linalg.norm(b)))

    baseline = correlation(*(context(row) for row in strongest))
    if baseline < INDEPENDENT_CONTEXT:
        return strongest, baseline, baseline
    eligible = [[(row, context(row)) for row in group if row[1][2] >= max(layer.floor, best[1][2]-.10)]
                for group, best in zip(groups, strongest, strict=True)]
    selected, selected_correlation, strength = strongest, baseline, (0., 0.)
    for left, a in eligible[0]:
        for right, b in eligible[1]:
            left_score, right_score = left[1][2], right[1][2]
            pair_strength = min(left_score, right_score), max(left_score, right_score)
            if pair_strength <= strength:
                continue
            score = correlation(a, b)
            if score < INDEPENDENT_CONTEXT:
                selected, selected_correlation, strength = [left, right], score, pair_strength
    return selected, selected_correlation, baseline


def complete_chroma_mask(video, layer, signals, cuda, trace):
    anchors = diverse_frames(layer.matches)
    if len(anchors) < 4:
        return None
    held, correlation, _ = review_anchors(layer, signals)
    if correlation >= INDEPENDENT_CONTEXT:
        return None
    holdouts = {item.frame for item, _ in held}
    training = [row for row in anchors if row[0].frame not in holdouts]
    training = [training[i] for i in np.linspace(0, len(training)-1, min(10, len(training)), dtype=int)]
    core = layer.motif
    _, _, width, height = core.bounds
    px, py = min(150, max(22, width//2)), min(115, max(24, 2*height))
    w, h = core.width+2*px, core.height+2*py
    aligned = []
    for item, (x, y, _) in [*training, *held]:
        channels = chroma_signals(video.planes(get_frame(video, item.frame, cuda)))
        aligned.append([patch(channel, (x-px, y-py, w, h)) for channel in channels])
    aligned = np.asarray(aligned)
    original = np.zeros((h, w), np.uint8)
    original[py:py+core.height, px:px+core.width] = core.mask
    added = np.zeros_like(original)
    for channel in range(aligned.shape[1]):
        chroma = aligned[:, channel]
        detail, weights, mask = consensus_fields(chroma[:-2])
        _, components, stats, _ = cv2.connectedComponentsWithStats(mask & (original == 0))
        for i, (x, y, width, height, area) in enumerate(stats[1:], 1):
            if area < 16:
                continue
            region = np.s_[y:y+height, x:x+width]
            component = components[region] == i
            support = weights[region]*component
            if np.count_nonzero(support) >= 16 and all(
                    response(frame[region], detail[region], support)[0, 0] >= .65 for frame in chroma[-2:]):
                added[region] |= component
    count = int(np.count_nonzero(added))
    trace('layers.chroma_extension_checked', label=layer.label, added_pixels=count,
          training=[item.frame for item, _ in training], held_out=sorted(holdouts))
    if count < 32 or count > 1.3*np.count_nonzero(original):
        return None
    mask = original|added
    x, y, width, height = cv2.boundingRect(mask)
    if width*height > .18*video.width*video.height:
        return None
    x0, y0 = max(0, x-3), max(0, y-3)
    region = np.s_[y0:y+height+3, x0:x+width+3]
    signal, weights = [np.pad(field, ((py, py), (px, px)))[region].copy() for field in (core.signal, core.weights)]
    motif = Motif(signal, weights, mask[region].copy())
    dx, dy = x0-px, y0-py
    matches = {item: (x+dx, y+dy, score) for item, (x, y, score) in layer.matches.items()
               if agrees_in_halves(motif, signals[item.frame], x+dx, y+dy)}
    if len(diverse_frames(matches)) < max(3, ceil(.70*len(anchors))):
        return None
    return Layer(motif, layer.floor, matches, layer.label)


def layer_review(video, layer, signals, backend, cuda, trace):
    selected, correlation, baseline = review_anchors(layer, signals)
    trace('layers.review_pair', frames=[item.frame for item, _ in selected],
          context_correlation=round(correlation, 3), strongest_pair_correlation=round(baseline, 3))
    evidence, answers = [], []
    for observation, (x, y, score) in selected:
        rect = crop_for(video, layer.motif, (x, y), observation.rect)
        footprint = visible_box(layer.motif, x, y)
        image = frame_evidence(video, observation.frame, cuda, max_side=800, region=rect, outline=footprint,
                               mask_at=(layer.motif.mask, x, y))
        try:
            answer = classify_candidate(backend, image, 22)
        except (RuntimeError, ValueError) as error:
            answer = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.crop_checked', frame=observation.frame, rect=list(rect), footprint=list(footprint),
              score=score, answer=answer)
        evidence.append((observation.frame, rect, footprint, image))
        answers.append(answer)
    return evidence, answers, correlation


def candidate_identifiers(label):
    if not isinstance(label, str):
        return set()
    label = label.casefold()
    return {name.removeprefix('www.') for name in IDENTIFIER.findall(label)} | {
        f'account:{number}' for number in ACCOUNT_ID.findall(label)}


def identifier_kind(identifier):
    if identifier.startswith('@'):
        return 'handle'
    if identifier.startswith('account:'):
        return 'account'
    return 'domain'


def identifier_matches(motif, image, x, y):
    _, _, score = motif.local(image, (x, y), 0, 0)
    return abs(score) >= .55 and agrees_in_halves(motif, image, x, y, -1 if score < 0 else 1)


def unverified_identifier_frames(video, support, seen, cuda, trace):
    text, u, v = support
    pending = []
    for frame, raw in read_frames(video, cuda=cuda):
        if seen[frame]:
            image = highpass(video.planes(raw)[0], 2.2)
            if any(not identifier_matches(text, image, x+u, y+v) for x, y, _ in seen[frame]):
                pending.append(frame)
    trace('layers.identifier_track_checked', frames=sum(bool(row) for row in seen),
          unresolved=len(pending), sample=pending[:12])
    return pending


def registered_identifier_geometry(video, layer, sampled, backend, cuda, trace, read_pixels):
    """Locate readable text in two disjoint groups aligned to the verified watermark."""
    core = layer.motif
    px = min(75, max(20, core.bounds[2]//3))
    py = min(70, max(20, core.bounds[3]//2))
    w, h = core.width+2*px, core.height+2*py
    images, readings = [], []
    for offset in range(2):
        group = list(sampled.items())[offset::2]
        samples = np.zeros((len(group), h, w), np.float32)
        visible = np.zeros(samples.shape, bool)
        for i, (frame, (_, (x, y, _))) in enumerate(group):
            luma = video.planes(get_frame(video, frame, cuda))[0]
            view, section = clipped_plane(luma, (x-px, y-py, w, h))
            samples[i][section] = view
            visible[i][section] = True
        valid = visible.sum(axis=0) >= ceil(2*len(group)/3)
        consensus = np.full((h, w), 16., np.float32)
        consensus[valid] = np.nanmedian(np.where(visible[:, valid], samples[:, valid], np.nan), axis=0)
        image = np.clip((consensus-16.)*(255/219), 0, 255).astype(np.uint8)
        images.append(image)
        rows = defaultdict(list)
        for text, (a, b, c, d) in read_pixels(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)):
            if min(c-a, d-b) >= 8 and 0 <= a < c <= w and 0 <= b < d <= h:
                for name in candidate_identifiers(text):
                    rows[name].append((a, b, c-a, d-b))
        readings.append(rows)
    first, second = readings
    known = candidate_identifiers(layer.label) if layer.reviewed_regions else set()
    pairs = [(name, a, b) for name in first.keys() & second.keys()
             if not known or name in known for a in first[name] for b in second[name]
             if overlap(a, b) >= .66*min(a[2]*a[3], b[2]*b[3])]
    names = {name for name, *_ in pairs}
    if len(names) != 1:
        return None
    name, a, b = max(pairs, key=lambda row: overlap(*row[1:]))
    if any(not overlap(a, other) or not overlap(b, peer) for _, other, peer in pairs):
        return None
    visual, boxes = [], []
    for i, image in enumerate(images, 1):
        evidence = (f'Registered source group {i}', jpeg(Image.fromarray(image), 1000))
        labels = locate_source_overlays(backend, evidence, 22, read_labels=True)
        visual.append(set().union(*(candidate_identifiers(label) for label, _ in labels)))
        boxes.append([rect for label, rect in labels if candidate_identifiers(label) == {name}])
    trace('layers.registered_identifier_checked', frames=list(sampled), ocr=name,
          visual=[sorted(items) for items in visual])
    if visual != [{name}, {name}]:
        return None
    left, top = min(a[0], b[0])-px, min(a[1], b[1])-py
    right, bottom = max(a[0]+a[2], b[0]+b[2])-px, max(a[1]+a[3], b[1]+b[3])-py
    envelope = None
    if all(len(group) == 1 for group in boxes):
        native = [(floor(x*w/1000)-px, floor(y*h/1000)-py, ceil(u*w/1000)-px, ceil(v*h/1000)-py)
                  for [(x, y, u, v)] in boxes]
        regions = [(x, y, u-x, v-y) for x, y, u, v in native]
        common = overlap(*regions)
        union = sum(w*h for _, _, w, h in regions)-common
        text = left, top, right-left, bottom-top
        if common >= .65*union and all(overlap(text, region) >= .85*text[2]*text[3] for region in regions):
            envelope = (min(p[0] for p in native), min(p[1] for p in native),
                        max(p[2] for p in native), max(p[3] for p in native))
    return name, (left, top, right, bottom), envelope


def registered_identifier_proposals(video, layer, sampled, backend, cuda, trace, read_pixels):
    evidence = registered_identifier_geometry(video, layer, sampled, backend, cuda, trace, read_pixels)
    if evidence is None:
        return {}
    name, (left, top, right, bottom), _ = evidence
    return {name: {frame: [{'rect': [x+left, y+top, x+right, y+bottom]}]
                   for frame, (_, (x, y, _)) in sampled.items()
                   if 0 <= x+left < x+right <= video.width and 0 <= y+top < y+bottom <= video.height}}


def complete_identifier_layer(video, layer, signals, cuda, trace, read_text, identifiers, *, registered_backend=None):
    anchors = diverse_frames(layer.matches)
    if len(anchors) < 6:
        return None
    core = layer.motif
    verified = {}
    if registered_backend is None:
        reviews = review_anchors(layer, signals)[0]
        sampled = {item.frame: (item, position) for item, position in [
            *(anchors[i] for i in np.linspace(0, len(anchors)-1, min(6, len(anchors)-2), dtype=int)), *reviews]}
        proposals = defaultdict(lambda: defaultdict(list))
        for frame, (item, (x, y, _)) in sorted(sampled.items()):
            region = crop_for(video, core, (x, y), item.rect)
            for text, rect in read_text(video, frame, region, cuda):
                for identifier in candidate_identifiers(text).intersection(identifiers):
                    proposals[identifier][frame].append({'rect': rect})
        remaining = [item.frame for item, _ in anchors if item.frame not in sampled]
        held_out = [remaining[i] for i in np.linspace(0, len(remaining)-1, min(12, len(remaining)), dtype=int)]
    else:
        strong = [row for row in anchors if row[1][2] >= max(.73, layer.floor)]
        if len(strong) < 8:
            return None
        held_out = [strong[i][0].frame for i in np.linspace(1, len(strong)-1, min(6, len(strong)//4), dtype=int)]
        training = [row for row in strong if row[0].frame not in held_out]
        sampled = {item.frame: (item, position) for item, position in
                   [training[i] for i in np.linspace(0, len(training)-1, min(16, len(training)), dtype=int)]}
        verified = {item.frame: position for item, position in strong}
        proposals = registered_identifier_proposals(video, layer, sampled, registered_backend, cuda, trace,
                                                    read_text.read_pixels)
    trace('layers.identifier_proposals', frames=sorted(sampled), counts={key: len(rows) for key, rows in proposals.items()})

    def positions(frame):
        image = signals[frame]
        ranked = [verified[frame]] if verified else locations(core, image)
        return [(x, y, score) for x, y, score in ranked
                if score >= max(.62, layer.floor-.08) and agrees_in_halves(core, image, x, y)]

    for identifier, rows in sorted(proposals.items(), key=lambda row: (-len(row[1]), row[0]))[:2]:
        fragment = build_layer(signals, proposals_for(video, rows))
        if fragment is None:
            continue
        text = fragment.motif
        offsets = defaultdict(dict)
        for item, (u, v, _) in diverse_frames(fragment.matches):
            for x, y, score in positions(item.frame):
                if abs(u-x) <= core.width and abs(v-y) <= core.height+text.height:
                    offsets[u-x, v-y].setdefault(item.frame, (item, x, y, score))
        if not offsets:
            continue
        (dx, dy), paired = max(offsets.items(), key=lambda row: len(row[1]))
        if len(paired) < 3:
            continue
        # Keep only the repeated strokes in the recognized text boxes.
        votes = np.zeros(text.mask.shape, np.int32)
        for item, x, y, _ in paired.values():
            a, b, w, h = item.rect
            part, _ = clipped_plane(votes, (a-x-dx-2, b-y-dy-2, w+4, h+4))
            part += 1
        mask = text.mask & (votes >= ceil(2*len(paired)/3))
        try:
            text = Motif(text.signal, text.weights*mask, mask)
        except ValueError:
            continue
        paired = {frame: (item, x, y, score) for frame, (item, x, y, score) in paired.items()
                  if identifier_matches(text, signals[frame], x+dx, y+dy)}
        if len(paired) < 3:
            continue
        checked = {}
        for frame in held_out:
            for x, y, score in positions(frame):
                if identifier_matches(text, signals[frame], x+dx, y+dy):
                    checked[frame] = Observation(frame, visible_box(core, x, y)), x, y, score
                    break
        trace('layers.identifier_alignment', identifier=identifier, offset=[dx, dy], anchors=sorted(paired),
              held_out=sorted(checked))
        if len(checked) < 2:
            continue
        left, top = min(0, dx), min(0, dy)
        width = max(core.width, dx+text.width)-left
        height = max(core.height, dy+text.height)-top
        if width > video.width or height > video.height:
            continue
        padding = ((-top, height-core.height+top), (-left, width-core.width+left))
        signal, weights, union = (np.pad(plane, padding) for plane in (core.signal, core.weights, core.mask))
        u, v = dx-left, dy-top
        region = np.s_[v:v+text.height, u:u+text.width]
        added = (text.mask > 0) & (union[region] == 0)
        if np.count_nonzero(added) < 32:
            continue
        signal[region][added], weights[region][added] = text.signal[added], text.weights[added]
        union[region] |= text.mask
        if np.prod(cv2.boundingRect(union)[2:]) > .18*video.width*video.height:
            continue
        motif = Motif(signal, weights, union)
        matches = {item: (x+left, y+top, score) for item, x, y, score in [*paired.values(), *checked.values()]}
        trace('layers.identifier_mask_proposed', identifier=identifier, added_pixels=int(added.sum()),
              anchors=len(matches), held_out=sorted(checked))
        label = identifier if registered_backend is not None else layer.label
        return Layer(motif, layer.floor, matches, label, layer.reviewed_regions, (text, u, v))
    return None


def overlay_fit(answer, frames):
    fit = answer.get('fit')
    if isinstance(fit, dict) and set(fit) == set(map(str, frames)):
        return ('partial' if 'partial' in fit.values() else
                'complete' if all(value == 'complete' for value in fit.values()) else 'uncertain')
    return fit if fit in ('complete', 'partial') else 'uncertain'


def confirm_layer(video, layer, signals, backend, cuda, trace, fallback, *, prior_watermark=False, review=None):
    evidence, answers, correlation = review if review is not None else layer_review(
        video, layer, signals, backend, cuda, trace)
    decisions = [answer['decision'] for answer in answers]
    identifiers = [candidate_identifiers(answer.get('label')) for answer in answers]
    if (decisions == ['watermark', 'watermark'] and correlation < INDEPENDENT_CONTEXT
            and not identifiers[0].isdisjoint(identifiers[1]) and all(answer.get('fit') == 'complete' for answer in answers)):
        decision = answers[-1]
        trace('layers.local_identity_agreed', context_correlation=round(correlation, 3),
              identifiers=sorted(identifiers[0] & identifiers[1]))
    elif decisions == ['protected', 'protected'] and not prior_watermark:
        decision = answers[-1]
    elif fallback is not None:
        decision = fallback(evidence)
        trace('layers.crop_escalated', context_correlation=round(correlation, 3), answer=decision)
    else:
        return 'unresolved'
    if decision['decision'] == 'watermark':
        label = decision.get('label')
        layer.label = label.strip()[:100] if isinstance(label, str) and label.strip() else 'Repeated visual overlay'
        fit = overlay_fit(decision, (frame for frame, *_ in evidence))
        if fit in ('complete', 'partial'):
            layer.reviewed_regions = tuple((frame, *footprint) for frame, _, footprint, _ in evidence)
            return 'confirmed' if fit == 'complete' else 'partial'
    return 'scene' if decision['decision'] == 'protected' else 'unresolved'


def expand_reviewed_layer(video, layer, signals, backend, cuda, trace, fallback, *, observations=(), initial_consensus=None,
                          read_text=None, identifiers=()):
    for number in range(1, 3):
        consensus = (initial_consensus if number == 1 and initial_consensus is not None
                     else complete_layer(layer, signals, trace, wide=number == 2))
        translucent = complete_translucent_mask(video, consensus or layer, cuda, trace)
        for expanded in (translucent, consensus):
            if expanded is None:
                continue
            status = confirm_layer(video, expanded, signals, backend, cuda, trace, fallback, prior_watermark=True)
            if status in ('confirmed', 'partial'):
                expanded.reviewed_regions = tuple(dict.fromkeys(layer.reviewed_regions+expanded.reviewed_regions))
                trace('layers.expansion_reviewed', pass_number=number, status=status, label=expanded.label,
                      bounds=expanded.motif.bounds)
                if status == 'confirmed':
                    return expanded, status
                layer = expanded
                break
    if video is not None and (expanded := complete_chroma_mask(video, layer, signals, cuda, trace)) is not None:
        status = confirm_layer(video, expanded, signals, backend, cuda, trace, fallback, prior_watermark=True)
        if status in ('confirmed', 'partial'):
            expanded.reviewed_regions = tuple(dict.fromkeys(layer.reviewed_regions+expanded.reviewed_regions))
            trace('layers.chroma_extension_reviewed', status=status, label=expanded.label, bounds=expanded.motif.bounds)
            if status == 'confirmed':
                return expanded, status
            layer = expanded
    expanded = (complete_sampled_layer(video, layer, signals, cuda, trace, observations=observations)
                if video is not None else None)
    names = set(identifiers) | candidate_identifiers(layer.label)
    if read_text is not None and video is not None:
        for registered in (False, True):
            if (registered and not callable(getattr(read_text, 'read_pixels', None))) or (not registered and not names):
                continue
            try:
                text = complete_identifier_layer(video, expanded or layer, signals, cuda, trace, read_text, names,
                                                 registered_backend=backend if registered else None)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('layers.identifier_expansion_unavailable', reason=str(error), registered=registered)
                continue
            if text is not None:
                expected = candidate_identifiers(text.label) if registered else set()
                status = confirm_layer(video, text, signals, backend, cuda, trace, fallback, prior_watermark=True)
                consistent = not expected or candidate_identifiers(text.label) == expected
                trace('layers.identifier_expansion_reviewed', status=status, label=text.label, registered=registered,
                      consistent=consistent)
                if status == 'confirmed' and consistent:
                    return text, status
    if expanded is not None:
        status = confirm_layer(video, expanded, signals, backend, cuda, trace, fallback, prior_watermark=True)
        if status in ('confirmed', 'partial'):
            expanded.reviewed_regions = tuple(dict.fromkeys(layer.reviewed_regions+expanded.reviewed_regions))
            trace('layers.sampled_extension_reviewed', status=status, label=expanded.label, bounds=expanded.motif.bounds)
            return expanded, status
    return layer, 'partial'


def preview_suggests_complete_mask(video, layer, review, backend, cuda, trace):
    evidence, answers, correlation = review
    identifiers = [candidate_identifiers(answer.get('label')) for answer in answers]
    if correlation >= INDEPENDENT_CONTEXT or (all(identifiers) and identifiers[0].isdisjoint(identifiers[1])):
        return False
    left, top, *_ = layer.motif.bounds
    for (frame, rect, footprint, _), answer in zip(evidence, answers, strict=True):
        if answer.get('fit') == 'complete':
            continue
        origin = footprint[0]-left, footprint[1]-top
        try:
            image = filled_mask_evidence(video, frame, cuda, 800, [(layer.motif.mask, *origin)], region=rect)
            result = preview_residue(backend, image, 22)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            result = {'decision': 'uncertain', 'reason': str(error)}
        trace('layers.mask_preview_checked', frame=frame, answer=result)
        if result['decision'] != 'none' or result.get('label') != '':
            return False
    return True


def review_full_layer(video, layer, signals, backend, cuda, trace, fallback, *, observations=(), read_text=None):
    review = layer_review(video, layer, signals, backend, cuda, trace)
    answers = review[1]
    local_partial = (all(a['decision'] == 'watermark' for a in answers)
                     and any(a.get('fit') == 'partial' for a in answers))
    prior_partial = False
    early_review = (local_partial and fallback is not None and video is not None
                    and preview_suggests_complete_mask(video, layer, review, backend, cuda, trace))
    initial_consensus = complete_layer(layer, signals, trace) if early_review else None
    if initial_consensus is not None:
        early_review = False
        trace('layers.mask_preview_has_repeated_extension', frames=[frame for frame, *_ in review[0]],
              bounds=initial_consensus.motif.bounds)
    if early_review:
        trace('layers.mask_review_before_expansion', frames=[frame for frame, *_ in review[0]])
    if not local_partial or early_review:
        classification = confirm_layer(video, layer, signals, backend, cuda, trace, fallback, review=review)
        if classification in ('confirmed', 'scene') or (not local_partial and classification != 'partial'):
            return layer, classification
        prior_partial = classification == 'partial'
    if local_partial:
        trace('layers.local_expansion_requested', frames=[frame for frame, *_ in review[0]])
    identifiers = {name for answer in answers if answer['decision'] == 'watermark'
                   for name in candidate_identifiers(answer.get('label'))}
    candidate, status = expand_reviewed_layer(video, layer, signals, backend, cuda, trace, fallback,
                                              observations=observations, initial_consensus=initial_consensus,
                                              read_text=read_text, identifiers=identifiers)
    if candidate is layer and local_partial and not prior_partial:
        status = confirm_layer(video, layer, signals, backend, cuda, trace, fallback, review=review)
    return candidate, status


def discover_layers(video, rows, backend, cuda, trace, *, fallback=None, required=None, retained=(), save_confirmed=None,
                    read_text=None):
    observations = proposals_for(video, rows)
    frames = sorted({o.frame for o in observations})
    if (not frames or len(rows) < 3) and not retained:
        raise ValueError('There are too few independent local candidates.')
    signals = {i: highpass(video.planes(get_frame(video, i, cuda))[0], 2.2) for i in frames}
    confirmed = []
    for layer in retained:
        if not retain_confirmed_layer(confirmed, layer, signals, video, cuda, trace):
            confirmed.append(replace(layer, matches=dict(layer.matches)))
    partials, unresolved = [], []
    if confirmed:
        prior_regions = defaultdict(list)
        for layer in confirmed:
            for item, (x, y, _) in layer.matches.items():
                prior_regions[item.frame].append(visible_box(layer.motif, x, y))
        observations = [item for item in observations if not any(
            overlap(item.rect, rect) >= .82*item.rect[2]*item.rect[3] for rect in prior_regions[item.frame])]
        trace('layers.confirmed_motifs_retained', labels=[layer.label for layer in confirmed],
              new_candidates=len(observations))

    def merge_fragment(layer, owners):
        for prior in owners:
            if mapped := covered_fragment(layer, prior, signals, video=video, cuda=cuda):
                prior.matches.update(mapped)
                layer.reviewed_regions += tuple((item.frame, *visible_box(prior.motif, x, y))
                                                for item, (x, y, _) in mapped.items())
                return True
        return False

    def require(layer):
        if required is not None:
            for region in map(list, layer.reviewed_regions):
                if region not in required:
                    required.append(region)

    def pending(layer, reason, **details):
        hints = {item.frame: visible_box(layer.motif, *match[:2]) for item, match in diverse_frames(layer.matches)}
        return LayerNeedsEvidence(layer, sorted(hints), reason, hints, confirmed=tuple(confirmed), **details)

    for layer in confirmed:
        require(layer)
    preferred = remainder_observations(video, rows, observations)
    recovery_tried = False
    # The template limit applies to verified edits. Every hypothesis must consume a source proposal.
    while observations:
        layer = build_layer(signals, preferred) if preferred else None
        preferred = ()
        if layer is None:
            layer = build_layer(signals, observations)
        if layer is None and not recovery_tried and not (confirmed or partials or unresolved):
            recovery_tried = True
            layer = recover_layer(video, rows, observations, signals, cuda, trace)
        if layer is None:
            break
        seeds = set(layer.matches)
        if merge_fragment(layer, confirmed):
            classification = 'covered-fragment'
        elif merge_fragment(layer, partials):
            classification = 'partial-fragment'
        else:
            layer, classification = review_full_layer(video, layer, signals, backend, cuda, trace, fallback,
                                                       observations=observations, read_text=read_text)
        require(layer)
        known = defaultdict(list)
        for item, (x, y, _) in layer.matches.items():
            known[item.frame].append(visible_box(layer.motif, x, y))
        trace('layers.hypothesis', classification=classification, label=layer.label,
              anchors=len(known), simultaneous=max(map(len, known.values())), minimum=layer.floor)
        remaining = [o for o in observations if o not in layer.matches and not any(
            overlap(o.rect, rect) >= .82*o.rect[2]*o.rect[3] for rect in known[o.frame])]
        if len(remaining) == len(observations):
            remaining = [o for o in observations if o not in seeds]
        if len(remaining) == len(observations):
            raise pending(layer, 'The recurrent overlay has no matching source proposal.')
        observations = remaining
        if classification == 'confirmed':
            try:
                layer = trim_empty_motif_padding(video, layer)
            except ValueError as error:
                raise pending(layer, str(error)) from error
            if not retain_confirmed_layer(confirmed, layer, signals, video, cuda, trace):
                if len(confirmed) >= MAX_TEMPLATES:
                    raise pending(layer, 'Independently confirmed overlays exceed the repair layer limit.')
                confirmed.append(layer)
            if save_confirmed is not None:
                save_confirmed(confirmed)
        elif classification == 'partial':
            partials.append(layer)
        elif classification == 'unresolved':
            unresolved.append(layer)
    if confirmed and save_confirmed is not None:
        save_confirmed(confirmed)
    for layer in partials:
        if not merge_fragment(layer, confirmed):
            raise pending(layer, 'The proposed mask omits part of a confirmed watermark.', partial_mask=True)
        require(layer)
        trace('layers.deferred_fragment', label=layer.label, anchors=len(layer.matches))
    if unresolved:
        layer = max(unresolved, key=lambda item: len(diverse_frames(item.matches)))
        raise pending(layer, 'Some recurrent overlays still need classification before a safe repair.',
                      needs_classification=True)
    if not confirmed:
        raise ValueError('No distribution watermark was independently confirmed in the proposed regions.')
    return confirmed


def locations(motif, image, limit=MAX_LAYERS+1, *, score_map=None):
    pad_x, pad_y = search_padding(motif)
    if motif.height > image.shape[0]+2*pad_y or motif.width > image.shape[1]+2*pad_x:
        return []
    factor = 1 if image.size <= 520000 else 2
    if score_map is None:
        scores = response(*search_input(motif, image))
    else:
        scores = score_map
    try:
        selected = list(peaks(np.abs(scores), motif.width//factor, motif.height//factor, limit,
                              tolerance=.0001 if score_map is not None else 0.,
                              minimum=.25 if factor == 1 else None))
    except AmbiguousPeak:
        return locations(motif, image, limit)
    ranked = [(factor*x-pad_x, factor*y-pad_y, float(scores[y, x])) for x, y, _ in selected]
    if factor == 2:
        ranked = sorted((motif.local(image, (x, y), 5, 5, reversible=True) for x, y, _ in ranked),
                        key=lambda row: -abs(row[2]))
    elif score_map is not None:
        exact = [motif.local(image, (x, y), 0, 0, reversible=True) for x, y, _ in ranked]
        if any(abs(a[2]-b[2]) > .0001 for a, b in zip(exact, ranked, strict=True)):
            return locations(motif, image, limit)
        ranked = exact
    hits = []
    for x, y, score in ranked:
        box = visible_box(motif, x, y)
        if not any(overlap(box, visible_box(motif, a, b)) > .35*box[2]*box[3] for a, b, _ in hits):
            hits.append((x, y, score))
    return hits


def color_reference(video, layer, cuda):
    matches = diverse_frames(layer.matches)
    if len(matches) < 5:
        return []
    positions = np.zeros((video.count, 2), np.int32)
    for item, (x, y, _) in matches:
        positions[item.frame] = x, y

    cache = {}
    def planes(i):
        if i not in cache:
            cache[i] = video.planes(get_frame(video, i, cuda))
        return cache[i]

    trusted = np.zeros(video.count, bool)
    trusted[[item.frame for item, _ in matches]] = True
    held, (x, y, _) = matches[len(matches)//2]
    trusted[held.frame] = False
    templates = chroma_templates(planes, trusted, positions, layer.motif.mask)
    if templates and chroma_position(templates, chroma_signals(planes(held.frame)), x, y, 1):
        return templates
    if len(matches) < 6:
        return []
    trusted[held.frame] = True
    held = [matches[len(matches)//3], matches[2*len(matches)//3]]
    trusted[[item.frame for item, _ in held]] = False
    templates = chroma_templates(planes, trusted, positions, layer.motif.mask, allow_partial=True)
    checks = [(chroma_signals(planes(item.frame)), x, y) for item, (x, y, _) in held]
    supported = []
    for channel, template in enumerate(templates):
        if template is not None:
            scores = [template.local(signals[channel], (x, y), 2, 2) for signals, x, y in checks]
            if all(score >= .80 for _, _, score in scores):
                supported.append(channel)
    return [template if i in supported else None for i, template in enumerate(templates)] if supported else []


def chroma_position(templates, channels, x, y, radius, *, alongside_luma=False):
    matches = [motif.local(channel, (x, y), radius, radius)
               for motif, channel in zip(templates, channels, strict=True) if motif is not None]
    if len(matches) == 1:
        a, b, score = matches[0]
        return (a, b) if alongside_luma and max(abs(a-x), abs(b-y)) <= 1 and score >= .82 else None
    a, b = matches
    if abs(a[0]-b[0])+abs(a[1]-b[1]) <= 1 and min(a[2], b[2]) >= .67 and max(a[2], b[2]) >= .79:
        return max(matches, key=lambda row: row[2])[:2]
    return None


def global_chroma_point(templates, planes):
    """Propose a position only when two color channels agree on a distinctive peak."""
    if len(templates) != 2 or any(template is None for template in templates):
        return None
    left, right = [locations(template, signal) for template, signal in
                   zip(templates, chroma_signals(planes), strict=True)]
    joint = sorted(((min(score, strength), x, y) for x, y, score in left for a, b, strength in right
                    if abs(x-a)+abs(y-b) <= 1), reverse=True)
    if joint:
        score, x, y = joint[0]
        other = joint[1][0] if len(joint) > 1 else 0.
        if strong_match(score, other):
            return x, y, score
    return None


def recurring_origins(layer):
    positions = [(x, y, item.frame) for item, (x, y, _) in layer.matches.items()]
    groups = []
    for x, y, frame in positions:
        group = next((frames for a, b, frames in groups if max(abs(x-a), abs(y-b)) <= 2), None)
        if group is None:
            groups.append((x, y, {frame}))
        else:
            group.add(frame)
    return [(x, y) for x, y, frames in sorted(groups, key=lambda group: -len(group[2]))[:4] if len(frames) >= 3]


def regional_match(motif, signal, origin, *, radius=2):
    """Verify the same position against two disjoint regions, each with its own contrast."""
    x, y = origin
    search = patch(signal, (x-radius, y-radius, motif.width+2*radius, motif.height+2*radius))
    middle = motif.width//2
    scores = []
    for left, right in ((0, middle), (middle, motif.width)):
        sample, weights = (data[:, left:right] for data in (motif.signal, motif.weights))
        if np.sum(sample*sample*weights) < 120:
            return None
        scores.append(np.abs(response(search[:, left:right+2*radius], sample, weights)))
    left, right = scores
    joint = np.sqrt(left*right)
    joint[np.minimum(left, right) < .42] = 0
    _, score, _, (a, b) = cv2.minMaxLoc(joint)
    return (x+a-radius, y+b-radius, score) if score >= .64 else None


def edge_predictions(evidence, boundaries, motif, size):
    """Continue a measured trajectory only where its confirmed graphic crosses an image edge."""
    dimensions = np.asarray(size)
    left, top, width, height = motif.bounds
    start = np.array([left, top])
    stop = start + [width, height]
    predictions = defaultdict(list)
    for frame, (hits, _, _) in enumerate(evidence):
        if len(hits) != 1:
            continue
        for direction in (-1, 1):
            missing = frame + direction
            if (not 0 <= missing < len(evidence) or evidence[missing][0]
                    or max(frame, missing) in boundaries):
                continue
            previous = []
            for step in range(24):
                i = frame - step*direction
                if not 0 <= i < len(evidence) or len(evidence[i][0]) != 1:
                    break
                if step and max(i, i+direction) in boundaries:
                    break
                previous.append(evidence[i][0][0][:2])
            if len(previous) < 8:
                continue
            times = -np.arange(len(previous), dtype=float)
            positions = np.asarray(previous)
            centered = times - times.mean()
            velocity = centered @ positions / (centered @ centered)
            origin = positions.mean(axis=0) - times.mean()*velocity
            if np.abs(positions - origin - times[:, None]*velocity).max() > 2:
                continue
            axis = int(np.argmax(np.abs(velocity)))
            speed = velocity[axis]
            if abs(speed) < .5:
                continue
            # The known footprint must already touch the edge in the predicted direction.
            towards_edge = (origin[axis]+start[axis] <= 2 if speed < 0 else
                            origin[axis]+stop[axis] >= dimensions[axis]-2)
            if not towards_edge:
                continue
            for i in range(missing, len(evidence) if direction == 1 else -1, direction):
                if evidence[i][0] or max(i, i-direction) in boundaries:
                    break
                point = origin + abs(i-frame)*velocity
                if np.any(point+stop <= 0) or np.any(point+start >= dimensions):
                    break
                if point[axis]+start[axis] < 0 or point[axis]+stop[axis] > dimensions[axis]:
                    predictions[i].append((tuple(map(int, np.rint(point))), axis, (frame, hits[0])))
    return predictions


def clipped_match(motif, signal, origin, axis):
    """Verify two disjoint parts of the visible pattern, excluding off-screen pixels."""
    a, b = origin
    rx, ry = (4, 1) if axis == 0 else (1, 4)
    dimension = 1-axis
    matches = []
    for x in range(a-rx, a+rx+1):
        for y in range(b-ry, b+ry+1):
            bounds = visible_box(motif, x, y)
            if bounds[axis] >= 0 and bounds[axis]+bounds[axis+2] <= signal.shape[dimension]:
                continue
            pixels, region = clipped_plane(signal, (x, y, motif.width, motif.height))
            if min(pixels.shape) < 6 or not motif.mask[region].any():
                continue
            # A near-boundary candidate must have usable pixels on both sides of the visible strip.
            split = pixels.shape[dimension]//2
            fields = zip(*(np.split(field, [split], axis=dimension)
                           for field in (pixels, motif.signal[region], motif.weights[region])), strict=True)
            scores = []
            for observed, reference, weights in fields:
                mass = float(weights.sum())
                energy = float(np.sum(reference*reference*weights))
                if mass < 20 or energy < 120:
                    break
                power = max(4*mass, float(np.sum(observed*observed*weights)))
                scores.append(float(np.sum(observed*reference*weights)/np.sqrt(energy*power)))
            if len(scores) == 2:
                matches.append((float(np.mean(scores)), min(scores), x, y))
    if not matches:
        return None
    score, weakest, x, y = max(matches)
    alternative = max((s for s, _, u, v in matches if max(abs(u-x), abs(v-y)) >= 3), default=0.)
    margin = score-alternative
    if score >= .60 and weakest >= .42 and margin >= .04:
        return (x, y, min(1., score)), score >= .70 and weakest >= .52 and margin >= .08
    return None


def recover_edge_matches(video, layer, evidence, boundaries, size, cuda, trace, *, weak_matches=None):
    predictions = edge_predictions(evidence, boundaries, layer.motif, size)
    if not predictions:
        return 0
    frames = np.array(sorted(predictions), dtype=int)
    matches = {}
    for run in np.split(frames, np.flatnonzero(np.diff(frames) > 1)+1):
        for i, raw in read_frames(video, int(run[0]), int(run[-1])+1, cuda):
            detail = highpass(video.planes(raw)[0], 2.2)
            candidates = [(*match, anchor) for origin, axis, anchor in predictions[i]
                          if (match := clipped_match(layer.motif, detail, origin, axis)) is not None]
            if len({point[:2] for point, _, _ in candidates}) == 1:
                matches[i] = max(candidates, key=lambda row: (row[1], row[0][2], -abs(i-row[2][0])))
    recovered = []
    for i, (point, strong, anchor) in matches.items():
        neighbors = [matches[j] for j in (i-1, i+1) if j in matches and max(i, j) not in boundaries]
        related = [reliable for near, reliable, _ in neighbors
                   if max(abs(a-b) for a, b in zip(point[:2], near[:2], strict=True)) <= 12]
        if not strong or not any(related):
            if related and weak_matches is not None:
                weak_matches[i] = point, anchor
            continue
        hits, weak, _ = evidence[i]
        hits.append(point)
        weak[:] = [candidate for candidate in weak if max(abs(point[0]-candidate[0]), abs(point[1]-candidate[1])) > 6]
        recovered.append(i)
    trace('layers.clipped_motion_checked', label=layer.label, predicted=len(predictions),
          recovered=len(recovered), weak=len(weak_matches) if weak_matches is not None else 0, sample=recovered[:8])
    return len(recovered)


def translucent_reference(video, layer, cuda):
    if (fit := fit_translucency(video, layer, cuda)) is None:
        return None
    slope, offset, support, _, _ = fit
    support &= layer.motif.mask.astype(bool)
    if np.count_nonzero(support) < 80:
        return None
    return slope[support], offset[support], support


def translucent_match(motif, luma, origin, fit):
    slope, offset, support = fit
    pixels = patch(luma, (*origin, motif.width, motif.height))
    background = cv2.medianBlur(pixels, 9)[support].astype(np.float32)
    expected = slope*background+offset
    if np.abs(expected-background).mean() < 3.5:
        return None
    observed = pixels[support]
    null_error = max(1., float(np.abs(observed-background).mean()))
    error = float(np.abs(observed-expected).mean())
    return 1-error/null_error


def temporal_support(evidence, boundaries, limit, *, minimum=.47, radius=8):
    """Corroborate weak pixel matches between two strong matches in the same shot."""
    def distances(order):
        previous, last, result = {}, None, {}
        for i in order:
            if last is not None and max(i, last) in boundaries:
                previous = {}
            hits, weak, _ = evidence[i]
            current = dict.fromkeys(hits, 0)
            for point in weak:
                x, y, score = point
                if score < minimum:
                    continue
                near = [distance for (a, b, _), distance in previous.items()
                        if distance < limit and max(abs(x-a), abs(y-b)) <= radius]
                if len(near) == 1:
                    current[point] = near[0]+1
                    result[i, point] = near[0]+1
            previous, last = current, i
        return result

    indices = range(len(evidence))
    forward, backward = distances(indices), distances(reversed(indices))
    return {(i, point) for (i, point), distance in forward.items()
            if distance+backward.get((i, point), limit+1)-1 <= limit}


def review_weak_matches(layer, evidence, support, boundaries, limit, confirm):
    groups = []
    previous = None
    for i, (hits, weak, _) in enumerate(evidence):
        floor = max(.47, layer.floor-.08) if hits else .47
        candidates = [point for point in weak if point[2] >= floor and (i, point) not in support]
        if len(candidates) != 1:
            continue
        point = candidates[0]
        if (previous is None or i != previous[0]+1 or i in boundaries
                or max(abs(a-b) for a, b in zip(point[:2], previous[1][:2])) > 8):
            groups.append([])
        groups[-1].append((i, point))
        previous = i, point

    probes = [row for group in groups for row in dict.fromkeys([*group[::limit], group[-1]])]
    verified, alternatives = 0, {}
    for index in np.linspace(0, len(probes)-1, min(12, len(probes)), dtype=int):
        frame, point = probes[index]
        result = confirm(layer, frame, point)
        hits, weak, _ = evidence[frame]
        if result == 'matches':
            hits.append(point)
            weak.remove(point)
            verified += 1
        elif result == 'replan':
            weak.remove(point)
            alternatives[frame] = point
    return verified, alternatives


def review_short_weak_gaps(layer, evidence, support, covered, alternatives, screen, confirm, trace, *,
                           proofs=None, save_proofs=None, frames=None, secondary=False):
    scope = range(len(evidence)) if frames is None else frames
    candidates = {}
    for i in scope:
        hits, weak, _ = evidence[i]
        if bool(hits) != secondary or i in covered or i in alternatives:
            continue
        if secondary:
            if not any(score >= max(.73, layer.floor) for _, _, score in hits):
                continue
            weak = [point for point in weak if point[2] >= max(.47, layer.floor-.08) and (i, point) not in support]
            if len(weak) != 1:
                continue
            footprint = visible_box(layer.motif, *weak[0][:2])
            if any(overlap(footprint, visible_box(layer.motif, x, y)) > .35*footprint[2]*footprint[3]
                   for x, y, _ in hits):
                continue
        elif not 0 < len(weak) <= 30 or any((i, point) in support for point in weak):
            continue
        candidates[i] = weak
    if not 0 < len(candidates) <= 80:
        return set(), set()
    references = diverse_frames(layer.matches)
    probes, unclassified = [], set()
    for i, points in candidates.items():
        answers = [(point, screen(layer, i, point)) for point in points]
        for point, decision in answers:
            if decision == 'replan':
                alternatives.setdefault(i, point)
        matching = [point for point, decision in answers if decision == 'matches']
        if len(matching) == 1:
            nearest = min((row for row in references if row[0].frame != i),
                          key=lambda row: (abs(row[0].frame-i), -row[1][2]), default=None)
            if nearest is None:
                continue
            reference, position = nearest
            probes.append((i, matching[0], (reference.frame, position)))
            if any(decision not in ('matches', 'replan') for _, decision in answers):
                unclassified.add(i)
    kind = 'short_secondary' if secondary else 'short_gap'
    recovered = review_anchored_matches(layer, probes, evidence, confirm, trace, kind,
                                       proofs=proofs, save_proofs=save_proofs)
    trace('layers.short_gap_review', label=layer.label, frames=list(candidates),
          confirmed=sorted(recovered), alternatives=sorted(set(candidates) & alternatives.keys()),
          secondary=secondary)
    return recovered, unclassified & recovered


def review_fading_matches(layer, candidates, evidence, cuts, limit, confirm, trace, *, proofs=None, save_proofs=None):
    def close(point, other):
        return max(abs(a-b) for a, b in zip(point[:2], other[:2])) <= 2

    def anchor_for(frame, point):
        options = []
        for direction in (-1, 1):
            previous = frame
            for i in range(frame+direction, frame+direction*(limit+1), direction):
                if not 0 <= i < len(evidence) or max(previous, i) in cuts:
                    break
                hits = [hit for hit in evidence[i][0] if close(point, hit) and hit[2] >= max(.73, layer.floor)]
                if hits:
                    options.append((i, max(hits, key=lambda hit: hit[2])))
                    break
                if i not in candidates or not close(point, candidates[i]):
                    break
                previous = i
        return min(options, key=lambda row: (abs(frame-row[0]), -row[1][2]), default=None)

    probes = [(i, point, anchor) for i, point in candidates.items()
              if not any(close(point, hit) for hit in evidence[i][0]) and (anchor := anchor_for(i, point))]
    return review_anchored_matches(layer, probes, evidence, confirm, trace, 'fading', proofs=proofs,
                                   save_proofs=save_proofs)


def review_anchored_matches(layer, probes, evidence, confirm, trace, kind, *, proofs=None, save_proofs=None):
    recovered, fresh = [], []
    for i, point, anchor in probes:
        frame, (x, y, _) = anchor
        key = f'{kind}:{i}:{point[0]}:{point[1]}:{frame}:{x}:{y}'
        if proofs is not None and proofs.get(key) == [i]:
            recovered.append((i, point))
        else:
            fresh.append((i, point, key, anchor))
    groups = defaultdict(list)
    for i, point, key, anchor in fresh[:120]:
        groups[anchor].append((i, point, key))
    for anchor, rows in groups.items():
        for start in range(0, len(rows), 3):
            group = rows[start:start+3]
            confirmed = confirm(layer, [(i, point) for i, point, _ in group], anchor)
            accepted = [(i, point, key) for i, point, key in group if i in confirmed]
            recovered.extend((i, point) for i, point, _ in accepted)
            if accepted and proofs is not None:
                proofs.update({key: [i] for i, _, key in accepted})
                if save_proofs is not None:
                    save_proofs()
    for i, point in recovered:
        hits, weak, _ = evidence[i]
        hits.append(point)
        weak[:] = [row for row in weak if max(abs(a-b) for a, b in zip(row[:2], point[:2], strict=True)) > 2]
    if probes:
        trace(f'layers.{kind}_matches_checked', label=layer.label, candidates=len(probes),
              checked=min(12, len(fresh)), recovered=[i for i, _ in recovered])
    return {i for i, _ in recovered}


def covered_probe(motif, point, other, position):
    x, y = point[:2]
    u, v = position[:2]
    footprint = visible_box(motif, x, y)
    if overlap(footprint, visible_box(other, u, v)) != footprint[2]*footprint[3]:
        return False
    cover = patch(other.mask, (x-u, y-v, motif.width, motif.height))
    return bool(np.all(cover[motif.mask > 0]))


def overlapping_masks(motif, point, other_hits, *, cover_most=False):
    box = visible_box(motif, *point[:2])
    masks = [(other.motif.mask, x, y) for other, (x, y, score) in other_hits
             if score >= max(.73, other.floor) and overlap(box, visible_box(other.motif, x, y)) >= .78*box[2]*box[3]]
    if cover_most and masks:
        x, y = point[:2]
        union = np.zeros_like(motif.mask)
        for mask, u, v in masks:
            union |= patch(mask, (x-u, y-v, motif.width, motif.height))
        if np.count_nonzero(union & motif.mask) < .90*np.count_nonzero(motif.mask):
            return []
    return masks


def overlapping_identifiers(motif, point, masks, expected, mask_identifiers):
    """Find a separately identified overlay that covers this ambiguous pixel match."""
    if len(expected) != 1:
        return set()
    x, y = point[:2]
    target = motif.mask > 0
    identifiers = set()
    kind = identifier_kind(next(iter(expected)))
    for mask, u, v in masks:
        known = mask_identifiers.get(id(mask), set())
        if len(known) != 1 or not known.isdisjoint(expected) or identifier_kind(next(iter(known))) != kind:
            continue
        coverage = patch(mask, (x-u, y-v, motif.width, motif.height))
        if np.mean(coverage[target]) >= .90:
            identifiers.update(known)
    return identifiers


def masked_proof_keys(regions):
    masks, keys = {}, {}
    for frame, placements in regions.items():
        items = set()
        for mask, x, y in placements:
            identity = id(mask)
            if identity not in masks:
                signature = sha256(repr((mask.shape, mask.dtype.str)).encode('ascii'))
                signature.update(mask.tobytes())
                masks[identity] = signature.hexdigest()
            items.add((int(x), int(y), masks[identity]))
        keys[frame] = sha256(repr(sorted(items)).encode('ascii')).hexdigest()
    return keys


def review_overlapping_matches(layer, candidates, evidence, support, confirm, trace, *, proofs=None, save_proofs=None,
                               covered=None):
    frames = [i for i, (points, _) in candidates.items() if any((i, p) not in support for p in points)]
    keys = {} if proofs is None else {i: f'masked-frame:{i}:{key}' for i, key in masked_proof_keys(
        {i: candidates[i][1] for i in frames}).items()}
    reused = [i for i, key in keys.items() if proofs.get(key) == [i]]
    reused_frames = set(reused)
    frames = [i for i in frames if i not in reused_frames]
    probes = [frames[i] for i in np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int)]
    verified = reused.copy()
    if reused:
        trace('layers.masked_weak_reused', label=layer.label, frames=len(reused), remaining=len(frames))
    for offset in range(0, len(probes), 4):
        group = probes[offset:offset+4]
        trace('layers.masked_weak_visibility', label=layer.label, frames=group)
        checks = confirm(layer, group, hide={i: candidates[i][1] for i in group})
        accepted = [i for i in group if checks.get(i) == 'absent']
        verified.extend(accepted)
        if accepted and proofs is not None:
            proofs.update({keys[i]: [i] for i in accepted})
            if save_proofs is not None:
                save_proofs()
    removed = 0
    for i in verified:
        points, _ = candidates[i]
        hits, weak, _ = evidence[i]
        weak[:] = [point for point in weak if point not in points]
        if covered is not None and not hits and not weak:
            covered.add(i)
        removed += len(points)
    return removed


def absence_identity(layer):
    anchor = max(layer.matches, key=lambda item: layer.matches[item][2])
    motif = layer.motif
    identity = sha256(repr((layer.label, layer.floor, anchor, motif.mask.shape)).encode('utf-8'))
    for field in (motif.signal, motif.weights, motif.mask):
        identity.update(field.tobytes())
    return identity.hexdigest()


def absence_intervals(negatives, borderline, cuts):
    cut_starts = np.fromiter(cuts, dtype=int)

    def contiguous(indices):
        frames = np.asarray(indices, dtype=int)
        if not len(frames):
            return []
        breaks = (np.diff(frames) > 1) | np.isin(frames[1:], cut_starts)
        return np.split(frames, np.flatnonzero(breaks)+1)

    routine, elevated = [], list(borderline)
    for run in contiguous(negatives):
        if len(run) >= 4:
            routine.append(run)
        else:
            elevated.extend(run.tolist())
    return routine, contiguous(sorted(elevated))


def incoherent_peak_intervals(frames, positions, cuts, limit):
    """Keep independent checks close in time and tied to the same rejected pixel position."""
    intervals = []
    run = []
    for i in frames:
        if run and (i != run[-1]+1 or i in cuts or len(run) == limit
                    or max(abs(a-b) for a, b in zip(positions[i], positions[run[0]], strict=True)) > 2):
            intervals.append(np.asarray(run, dtype=int))
            run = []
        run.append(i)
    if run:
        intervals.append(np.asarray(run, dtype=int))
    return intervals


def uncovered_absence(layer, intervals, strengths, confirm, trace, ambiguous, visible, *, borderline=False,
                      proofs=None, save_proofs=None, hidden_keys=None, replan_on_visible=False):
    if not intervals:
        return []
    ordered = sorted(intervals, key=len, reverse=True)
    trace('layers.absence_candidates', label=layer.label, ambiguous=ambiguous,
          interval_count=len(ordered), borderline=borderline,
          longest=[[int(run[0]), int(run[-1])+1] for run in ordered[:4]])

    def key(run):
        interval = f'{int(borderline)}:{int(run[0])}:{int(run[-1])+1}'
        masked = [(int(i), hidden_keys[i]) for i in run if i in hidden_keys] if hidden_keys else []
        if not masked:
            return interval
        signature = sha256(repr(masked).encode('ascii')).hexdigest()
        return f'masked-interval:{interval}:{signature}'

    groups, group, reused = [], [], []
    group_probes = 0
    for run in ordered:
        probes = absence_probes([run], strengths, secondary=borderline) if len(run) >= 4 else run.tolist()
        prior = proofs.get(key(run)) if proofs is not None else None
        if isinstance(prior, list) and prior == probes and all(type(i) is int for i in prior):
            reused.append(run)
            continue
        if group_probes+len(probes) > 4:
            groups.append(group)
            group, group_probes = [], 0
        group.append((run, probes))
        group_probes += len(probes)
    if group:
        groups.append(group)
    if reused:
        trace('layers.absence_reused', label=layer.label, frames=sum(map(len, reused)), intervals=len(reused),
              borderline=borderline)
    pending = []
    verified = len(reused)
    stalled_groups = 0
    for offset, checks in enumerate(groups):
        replan = replan_on_visible and len(visible) >= 3
        if confirm is None or replan or offset >= 80 or stalled_groups >= 30:
            deferred = [i for group in groups[offset:] for run, _ in group for i in run.tolist()]
            pending.extend(deferred)
            if replan:
                trace('layers.absence_deferred_for_replan', label=layer.label, frames=len(deferred),
                      visible_probes=sorted(visible), borderline=borderline)
            elif confirm is not None:
                trace('layers.absence_budget_reached', label=layer.label, frames=len(deferred), groups_checked=offset,
                      stalled_groups=stalled_groups, borderline=borderline)
            break
        if offset == 3:
            trace('layers.absence_continued', label=layer.label, groups_remaining=len(groups)-offset,
                  borderline=borderline)
        probes = [i for _, samples in checks for i in samples]
        reviewed = confirm(layer, probes)
        visible.update(i for i in probes if reviewed.get(i) == 'present')
        accepted = []
        for run, samples in checks:
            if all(reviewed.get(i) == 'absent' for i in samples):
                accepted.append(run)
                if proofs is not None:
                    proofs[key(run)] = samples
            else:
                pending.extend(run.tolist())
        stalled_groups = 0 if accepted else stalled_groups+1
        if accepted:
            verified += len(accepted)
            trace('layers.absence_verified', label=layer.label, frames=sum(map(len, accepted)), intervals=len(accepted),
                  pending=len(ordered)-verified, probes=probes, borderline=borderline, reviewed=reviewed)
            if save_proofs is not None:
                save_proofs()
    return pending


def scan_layers(video, layers, cuda, trace, *, confirm_absence=None, confirm_match=None, confirm_borderline=None,
                confirm_masked=None, confirm_fade=None, confirm_identifiers=None, absence_proofs=None, save_proofs=None,
                replan_on_visible=False, location_cache=None, confirm_edge=None, screen_short=None, confirm_short=None,
                 gpu_search=False, pixel_workers=4, positive_only=False):
    short_review = confirm_short or confirm_fade
    cuts, previous = [], None
    references = [color_reference(video, layer, cuda) for layer in layers]
    origins = [recurring_origins(layer) for layer in layers]
    translucencies = [translucent_reference(video, layer, cuda) if locations_at else None
                      for layer, locations_at in zip(layers, origins, strict=True)]
    evidence = [[] for _ in layers]
    weak_translucency = [[] for _ in layers]
    borderline = [set() for _ in layers]
    exact_probes = [set() for _ in layers]
    incoherent_peaks = [{} for _ in layers]
    inversions = [0 for _ in layers]
    recurring_recoveries = [0 for _ in layers]
    nearby_recoveries = [0 for _ in layers]
    translucent_recoveries = [0 for _ in layers]
    covered_frames = [set() for _ in layers]
    covered_weak = [0 for _ in layers]
    masked_frames = [{} for _ in layers]
    mask_targets = [{} for _ in layers]
    masked_weak = [{} for _ in layers]
    fading = [{} for _ in layers]
    identifiers = [candidate_identifiers(layer.label) for layer in layers]
    mask_identifiers = defaultdict(set)
    for layer, names in zip(layers, identifiers, strict=True):
        mask_identifiers[id(layer.motif.mask)].update(names)
    cached = [None]*len(layers)
    occluded_candidates = [{} for _ in layers]
    occlusion_masks = {id(layer.motif.mask): cv2.dilate(layer.motif.mask, np.ones((5, 5), np.uint8))
                       for layer in layers}
    reused_locations = 0
    if location_cache is not None:
        cached = [location_cache.setdefault((id(video), id(layer.motif), cuda, gpu_search),
                                            (video, layer.motif, {}))[2] for layer in layers]
    from copy import deepcopy
    from wm_runtime import current_runtime

    runtime = current_runtime()
    scan_key = None
    saved = None
    if runtime is not None:
        signature = sha256()
        flags = tuple(callback is not None for callback in
                      (confirm_masked, confirm_fade, confirm_borderline, confirm_identifiers))
        signature.update(repr((str(video.path), video.path.stat().st_mtime_ns, video.count, cuda, gpu_search, flags)).encode())
        for layer in layers:
            signature.update(repr((layer.label, layer.floor,
                                   sorted(layer.matches.items(), key=lambda row: row[0].frame))).encode())
            for array in (layer.motif.signal, layer.motif.weights, layer.motif.mask):
                signature.update(repr((array.shape, array.dtype.str)).encode())
                signature.update(array.tobytes())
        scan_key = signature.hexdigest()
        saved = runtime.pixel_passes.get(scan_key)
    if saved is not None:
        data, masks, identities = saved
        cuts, evidence, weak_translucency, borderline, exact_probes, incoherent_peaks, inversions, recurring_recoveries, nearby_recoveries, translucent_recoveries, covered_frames, covered_weak, masked_frames, mask_targets, masked_weak, fading, occluded_candidates = deepcopy(data, {id(mask): mask for mask in masks})
        mask_identifiers.update(identities)
        trace('layers.pixel_pass_reused', frames=video.count, templates=len(layers))
    else:
        frames = read_frames(video, cuda=cuda)
        motifs = [layer.motif for layer in layers]
        workers = min(max(1, pixel_workers), len(layers))
        with (ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
            for i, planes, signal, score_maps in pixel_search_frames(video, frames, motifs, cached, highpass, trace,
                                                                     gpu=gpu_search and cuda):
                if i % 120 == 0:
                    trace('layers.scan_progress', frame=i, total=video.count, templates=len(layers))
                luma = planes[0]
                current = cv2.resize(luma, (48, 28), interpolation=cv2.INTER_AREA).astype(np.int16)
                cut = previous is not None and np.abs(current-previous).mean() > 23
                if cut:
                    cuts.append(i)
                previous = current
                channels = chroma_signals(planes) if any(references) else None

                def verify_layer(j):
                    layer, templates = layers[j], references[j]
                    motif = layer.motif
                    previous_ranks = cached[j]
                    ranks = previous_ranks.get(i) if previous_ranks is not None else None
                    reused = ranks is not None
                    if ranks is None:
                        ranks = (locations(motif, signal, score_map=score_maps[j]) if j in score_maps else
                                 locations(motif, signal))
                        if previous_ranks is not None:
                            previous_ranks[i] = ranks
                    hits, weak, faint = [], [], []
                    for rank, (x, y, signed) in enumerate(ranks):
                        score = abs(signed)
                        if score < .43:
                            continue
                        if not agrees_in_halves(motif, signal, x, y, -1 if signed < 0 else 1):
                            if (confirm_fade is not None and rank == 0 and len(ranks) > 1
                                    and signed >= max(.58, layer.floor-.12) and signed-abs(ranks[1][2]) >= .15
                                    and any(max(abs(x-u), abs(y-v)) <= 2 for u, v in origins[j])):
                                fading[j][i] = (x, y, score)
                            continue
                        separated = (rank == 0 and len(ranks) > 1 and score >= max(.56, layer.floor-.09)
                                     and score-abs(ranks[1][2]) >= .14 and signed > 0)
                        if score >= max(layer.floor, .64 if signed < 0 else 0) or separated:
                            hits.append((x, y, score))
                            inversions[j] += signed < 0
                            continue
                        if templates:
                            if chroma_position(templates, channels, x, y, 1, alongside_luma=True) is not None:
                                hits.append((x, y, score))
                                continue
                        weak.append((x, y, score))
                    nearby = [] if cut or not evidence[j] else [(x, y) for x, y, _ in evidence[j][-1][0]]
                    searches = [(x, y, 2) for x, y in origins[j]]+[(x, y, 5) for x, y in nearby]
                    for x, y, radius in searches:
                        if any(max(abs(x-a), abs(y-b)) <= 6 for a, b, _ in hits):
                            continue
                        position = None
                        if templates and radius == 2:
                            position = chroma_position(templates, channels, x, y, 3)
                        if position is not None:
                            a, b, score = motif.local(signal, position, 0, 0)
                        elif (regional := regional_match(motif, signal, (x, y), radius=radius)) is not None:
                            a, b, score = regional
                            if radius == 2:
                                recurring_recoveries[j] += 1
                            else:
                                nearby_recoveries[j] += 1
                        elif (radius == 2 and translucencies[j] is not None
                              and (matched := translucent_match(motif, luma, (x, y), translucencies[j])) is not None):
                            if matched < .42:
                                if matched >= .30:
                                    faint.append((x, y, matched))
                                continue
                            a, b, score = x, y, matched
                            translucent_recoveries[j] += 1
                        else:
                            continue
                        hits.append((a, b, abs(score)))
                        weak = [p for p in weak if max(abs(p[0]-a), abs(p[1]-b)) > 6]
                    top = abs(ranks[0][2]) if ranks else 0.
                    evidence[j].append((hits, weak, top))
                    if confirm_borderline is not None and .38 <= top < .47 and not hits and not weak and not faint:
                        coherent = any(agrees_in_halves(motif, signal, x, y, -1 if score < 0 else 1)
                                       for x, y, score in ranks if abs(score) >= .38)
                        if not coherent:
                            borderline[j].add(i)
                        elif top < .43:
                            exact_probes[j].add(i)
                    elif confirm_borderline is not None and top >= .47 and not hits and not weak and not faint and i not in fading[j]:
                        incoherent_peaks[j][i] = ranks[0][:2]
                    if translucencies[j] is not None:
                        faint = [point for point in faint
                                 if not any(max(abs(point[0]-x), abs(point[1]-y)) <= 6 for x, y, _ in hits)]
                        weak_translucency[j].append((hits, faint, 0.))
                    return ranks, reused

                # Each layer owns its temporal history; join before evaluating cross-layer occlusion.
                verified_layers = pool.map(verify_layer, range(len(layers))) if pool else map(verify_layer, range(len(layers)))
                ranked_by_layer = []
                for ranks, reused in verified_layers:
                    ranked_by_layer.append(ranks)
                    reused_locations += reused
                if len(layers) > 1:
                    for j, ranks in enumerate(ranked_by_layer):
                        hits, weak, _ = evidence[j][-1]
                        if (confirm_masked is not None and ranks and not hits and not weak
                                and (i in borderline[j] or .30 <= abs(ranks[0][2]) < .38)):
                            mask_targets[j][i] = ranks[0]
                        other_hits = [(layer, position) for k, layer in enumerate(layers) if k != j
                                      for position in evidence[k][-1][0]]
                        if not other_hits:
                            continue
                        motif = layers[j].motif
                        if not hits:
                            occluders = [(occlusion_masks[id(other.motif.mask)], x, y) for other, (x, y, score) in other_hits
                                         if score >= max(.85, other.floor)]
                            if point := exposed_match(motif, signal, ranks, occluders, prepared=True):
                                occluded_candidates[j][i] = point
                        def covered(point):
                            return any(covered_probe(motif, point, other.motif, position) for other, position in other_hits)
                        uncovered = [point for point in weak if not covered(point)]
                        covered_weak[j] += len(weak)-len(uncovered)
                        weak[:] = uncovered
                        plausible = [point for point in ranks if abs(point[2]) >= .38]
                        if (not hits and plausible and abs(ranks[-1][2]) < .38
                                and all(covered(point) for point in plausible)):
                            covered_frames[j].add(i)
                        elif i in mask_targets[j]:
                            if regions := overlapping_masks(motif, mask_targets[j][i], other_hits):
                                masked_frames[j][i] = regions
                        elif confirm_masked is not None and weak and (hits or len(weak) == 1):
                            points, regions = [], []
                            minimum = max(MASKED_REVIEW_MINIMUM, layers[j].floor-.08) if hits else MASKED_REVIEW_MINIMUM
                            for point in weak:
                                needs_identifier = point[2] < minimum
                                if needs_identifier and (hits or confirm_identifiers is None):
                                    continue
                                matched = overlapping_masks(motif, point, other_hits, cover_most=not hits)
                                if matched and (not needs_identifier or overlapping_identifiers(
                                        motif, point, matched, identifiers[j], mask_identifiers)):
                                    points.append(point)
                                    regions.extend(matched)
                            if points:
                                regions.extend((motif.mask, x, y) for x, y, _ in hits)
                                masked_weak[j][i] = points, regions
        if runtime is not None:
            masks = [layer.motif.mask for layer in layers]
            data = (cuts, evidence, weak_translucency, borderline, exact_probes, incoherent_peaks, inversions, recurring_recoveries, nearby_recoveries, translucent_recoveries, covered_frames, covered_weak, masked_frames, mask_targets, masked_weak, fading, occluded_candidates)
            if len(runtime.pixel_passes) >= 2:
                runtime.pixel_passes.pop(next(iter(runtime.pixel_passes)))
            runtime.pixel_passes[scan_key] = (deepcopy(data, {id(mask): mask for mask in masks}), masks,
                                             dict(mask_identifiers))
    if reused_locations:
        trace('layers.pixel_locations_reused', frame_layer_pairs=reused_locations, frames=video.count)
    tracks = []
    boundaries = set(cuts)
    for j, candidates in enumerate(occluded_candidates):
        recovered = coherent_overlaps(candidates, boundaries)
        for i, (x, y, score, _) in recovered.items():
            hits, weak, _ = evidence[j][i]
            hits.append((x, y, score))
            weak[:] = [point for point in weak if max(abs(point[0]-x), abs(point[1]-y)) > 6]
        if recovered:
            trace('layers.overlap_recovered', label=layers[j].label, frames=sorted(recovered),
                  evidence=[{'frame': i, 'position': list(point[:2]), 'exposed_score': point[2],
                             'occluded_fraction': point[3]} for i, point in sorted(recovered.items())])
    if positive_only:
        partial_tracks = positive_review_tracks(video, layers, evidence, cuts)
        if not partial_tracks:
            raise ValueError('No independently confirmed supplemental placements were found.')
        return partial_tracks
    if runtime is not None and runtime.save_review_checkpoint is not None:
        partial_tracks = positive_review_tracks(video, layers, evidence, cuts)
        if partial_tracks:
            runtime.save_review_checkpoint(partial_tracks, [layer.label for layer in layers])
            runtime.start_coverage_review()
    limit = min(24, max(1, round(float(video.fps)*.7)))
    books = [absence_proofs.setdefault(absence_identity(layer), {}) if absence_proofs is not None else None
             for layer in layers]
    restored = set()
    if confirm_fade is not None:
        for j, layer in enumerate(layers):
            restored.update(review_fading_matches(layer, fading[j], evidence[j], boundaries,
                                                 min(12, max(1, round(float(video.fps)*.4))), confirm_fade, trace,
                                                 proofs=books[j], save_proofs=save_proofs))
    edge_matches = [{} for _ in layers]
    clipped_recoveries = [recover_edge_matches(video, layer, samples, boundaries, (video.width, video.height), cuda, trace,
                                              weak_matches=weak)
                          for layer, samples, weak in zip(layers, evidence, edge_matches, strict=True)]
    if confirm_edge is not None:
        for j, weak in enumerate(edge_matches):
            probes = [(i, point, anchor) for i, (point, anchor) in weak.items()]
            checked = review_anchored_matches(layers[j], probes, evidence[j], confirm_edge, trace, 'clipped',
                                              proofs=books[j], save_proofs=save_proofs)
            clipped_recoveries[j] += len(checked)
            restored.update(checked)
    for j, targets in enumerate(mask_targets):
        for i in restored.intersection(targets):
            other_hits = [(other, point) for k, other in enumerate(layers) if k != j for point in evidence[k][i][0]]
            if regions := overlapping_masks(layers[j].motif, targets[i], other_hits):
                masked_frames[j][i] = regions
    deferred = None
    for j, layer in enumerate(layers):
        proofs = books[j]
        previous_proofs = len(proofs) if proofs is not None else 0

        def check_masked(target, probes, *, hide):
            if confirm_identifiers is None:
                return confirm_masked(target, probes, hide=hide)
            ordinary = [i for i in probes if i not in masked_weak[j] or any(
                point[2] >= MASKED_REVIEW_MINIMUM for point in masked_weak[j][i][0])]
            checks = confirm_masked(target, ordinary, hide={i: hide[i] for i in ordinary}) if ordinary else {}
            candidates = {}
            for i in probes:
                if checks.get(i) == 'absent':
                    continue
                points, _ = masked_weak[j].get(i, ((), ()))
                point = points[0] if len(points) == 1 else mask_targets[j].get(i)
                if point is not None and (other := overlapping_identifiers(
                        target.motif, point, hide[i], identifiers[j], mask_identifiers)):
                    candidates[i] = point, other
            if candidates:
                accepted = confirm_identifiers(target, candidates, hide=hide)
                return {**checks, **dict.fromkeys(accepted.intersection(candidates), 'absent')}
            return checks

        blend_support = temporal_support(weak_translucency[j], boundaries, min(6, limit), minimum=.30, radius=2)
        for i, point in sorted(blend_support):
            hits, weak, _ = evidence[j][i]
            hits.append(point)
            weak[:] = [row for row in weak if max(abs(row[0]-point[0]), abs(row[1]-point[1])) > 6]
        support = temporal_support(evidence[j], boundaries, limit)
        if masked_weak[j] and review_overlapping_matches(layer, masked_weak[j], evidence[j], support,
                                                         check_masked, trace, proofs=proofs, save_proofs=save_proofs,
                                                         covered=covered_frames[j]):
            support = temporal_support(evidence[j], boundaries, limit)
        reviewed, alternatives = 0, {}
        if confirm_match is not None:
            reviewed, alternatives = review_weak_matches(layer, evidence[j], support, boundaries, limit, confirm_match)
            if reviewed or alternatives:
                support = temporal_support(evidence[j], boundaries, limit)
        other_unclassified = set()
        if screen_short is not None and short_review is not None:
            for secondary in (False, True):
                recovered, pending = review_short_weak_gaps(
                    layer, evidence[j], support, covered_frames[j], alternatives, screen_short, short_review, trace,
                    proofs=proofs, save_proofs=save_proofs, secondary=secondary)
                other_unclassified.update(pending)
                if recovered:
                    reviewed += len(recovered)
                    support = temporal_support(evidence[j], boundaries, limit)
        seen, negatives, possible_absence, exact, incoherent = [], [], [], [], []
        uncertain = [*alternatives, *other_unclassified]
        bridged = 0
        dense_verified = 0
        for i, (hits, weak, top) in enumerate(evidence[j]):
            supported = [point for point in weak if (i, point) in support]
            bridged += len(supported)
            confirmed = hits+supported
            if len(confirmed) > 30:
                strong = len(hits) == len(confirmed) and all(point[2] >= max(.73, layer.floor) for point in hits)
                if not strong or len(confirmed) > MAX_LAYERS:
                    uncertain.append(i)
                else:
                    dense_verified += 1
            if any(p[2] >= layer.floor-.08 and p not in supported for p in weak):
                uncertain.append(i)
            if not confirmed and i not in covered_frames[j] and i not in alternatives:
                if i in edge_matches[j]:
                    uncertain.append(i)
                elif top < .38:
                    negatives.append(i)
                elif i in borderline[j]:
                    possible_absence.append(i)
                elif i in exact_probes[j]:
                    exact.append(np.array([i], dtype=int))
                elif i in incoherent_peaks[j]:
                    incoherent.append(i)
                else:
                    uncertain.append(i)
            seen.append(sorted(confirmed, key=lambda row: (row[1], row[0])))
        strengths = [row[2] for row in evidence[j]]
        visible = set(alternatives)
        masked = masked_frames[j]
        other_masks = {}
        if confirm_masked is not None:
            fixed = [(track['mask'], *map(int, track['positions'][0])) for track in tracks
                     if track['present'].all() and np.all(track['positions'] == track['positions'][0])]
            if fixed:
                for i in negatives+possible_absence:
                    masked[i] = [*masked.get(i, ()), *fixed]
                trace('layers.verified_static_masks_hidden', label=layer.label, masks=len(fixed),
                      frames=len(negatives)+len(possible_absence))
            other_layers = [(k, other) for k, other in enumerate(layers) if k != j]
            if other_layers:
                for i in negatives+possible_absence+incoherent+[int(run[0]) for run in exact]:
                    if i in masked:
                        continue
                    masks = [(other.motif.mask, x, y) for k, other in other_layers for x, y, score in evidence[k][i][0]
                             if score >= max(.73, other.floor)]
                    if masks:
                        other_masks[i] = masks

        def confirm_without_other_overlays(target, probes, *, confirm):
            hide = {i: other_masks[i] for i in probes if i in other_masks}
            if not hide:
                return confirm(target, probes)
            trace('layers.other_overlays_hidden', label=target.label, frames=list(hide))
            return confirm_masked(target, probes, hide=hide)

        routine_review, strict_review = confirm_absence, confirm_borderline
        other_keys = masked_proof_keys(other_masks) if other_masks else None
        if other_masks and routine_review is not None:
            routine_review = partial(confirm_without_other_overlays, confirm=confirm_absence)
        if other_masks and strict_review is not None:
            strict_review = partial(confirm_without_other_overlays, confirm=confirm_borderline)
        routine, strict = absence_intervals([i for i in negatives if i not in masked],
                                           [i for i in possible_absence if i not in masked], boundaries)
        uncertain.extend(uncovered_absence(layer, routine, strengths, routine_review, trace, len(set(uncertain)),
                                           visible, proofs=proofs, save_proofs=save_proofs, hidden_keys=other_keys,
                                           replan_on_visible=replan_on_visible))
        if incoherent:
            intervals = incoherent_peak_intervals(incoherent, incoherent_peaks[j], boundaries, min(12, limit))
            trace('layers.incoherent_peak_review', label=layer.label, frames=len(incoherent), intervals=len(intervals))
            strict.extend(intervals)
        uncertain.extend(uncovered_absence(layer, strict+exact, strengths, strict_review, trace,
                                           len(set(uncertain)), visible, borderline=True, proofs=proofs,
                                           save_proofs=save_proofs, hidden_keys=other_keys,
                                           replan_on_visible=replan_on_visible))
        if masked:
            _, intervals = absence_intervals([], [i for i in negatives+possible_absence if i in masked], boundaries)

            def confirm_uncovered(target, probes):
                trace('layers.masked_visibility', label=target.label, frames=probes)
                return check_masked(target, probes, hide={i: masked[i] for i in probes})

            uncertain.extend(uncovered_absence(layer, intervals, strengths, confirm_uncovered, trace,
                                               len(set(uncertain)), visible, borderline=True, proofs=proofs,
                                               save_proofs=save_proofs, hidden_keys=masked_proof_keys(masked),
                                               replan_on_visible=replan_on_visible))
        unresolved = sorted(set(uncertain))
        remainder_masks = {}
        templates = references[j]
        if (screen_short is not None and short_review is not None and confirm_masked is not None
                and len(templates) == 2 and all(template is not None for template in templates)
                and 0 < len(unresolved) <= 80):
            probes = []
            for i in unresolved:
                if i not in visible or i in alternatives or any(evidence[j][i][:2]):
                    continue
                planes = video.planes(get_frame(video, i, cuda))
                if point := global_chroma_point(templates, planes):
                    x, y, color_score = point
                    _, _, score = layer.motif.local(highpass(planes[0], 2.2), (x, y), 0, 0)
                    evidence[j][i][1].append((x, y, abs(score)))
                    probes.append(i)
                    trace('layers.global_chroma_proposal', frame=i, position=[x, y],
                          color_score=color_score, luma_score=score)
            if probes:
                recovered, pending = review_short_weak_gaps(
                    layer, evidence[j], support, covered_frames[j], alternatives, screen_short, short_review, trace,
                    proofs=proofs, save_proofs=save_proofs, frames=probes)
                reviewed += len(recovered)
                for i in recovered:
                    seen[i] = evidence[j][i][0]
                hide = {i: [(layer.motif.mask, x, y) for x, y, _ in seen[i]]
                        for i in recovered - pending - alternatives.keys()}
                if hide:
                    def confirm_remaining(target, frames):
                        checks = confirm_masked(target, frames, hide={i: hide[i] for i in frames})
                        remainder_masks.update({i: hide[i] for i in frames if checks.get(i) == 'present'})
                        return checks

                    intervals = [np.array([i], dtype=int) for i in sorted(hide)]
                    remaining = uncovered_absence(layer, intervals, strengths, confirm_remaining, trace,
                                                  len(unresolved), visible, borderline=True, proofs=proofs,
                                                  save_proofs=save_proofs, hidden_keys=masked_proof_keys(hide))
                    resolved = hide.keys() - set(remaining)
                    unresolved = [i for i in unresolved if i not in resolved]
                    visible.difference_update(resolved)
                    trace('layers.global_chroma_reviewed', confirmed=sorted(recovered), covered=sorted(resolved))
        identifier_gaps = (unverified_identifier_frames(video, layer.identifier_support, seen, cuda, trace)
                           if layer.identifier_support is not None else [])
        if identifier_gaps:
            unresolved = sorted(set(unresolved).union(identifier_gaps))
        copies = max(map(len, seen))
        trace('layers.coverage', label=layer.label, copies=copies, frames=video.count,
              present=sum(bool(row) for row in seen), unresolved=len(unresolved), dense_verified=dense_verified,
              sample=[{'frame': i, 'strong': evidence[j][i][0], 'weak': evidence[j][i][1],
                       'best': evidence[j][i][2]} for i in unresolved[:12]],
              color_reference=bool(references[j]), temporal_matches=bridged, inverted_matches=inversions[j],
              color_channels=sum(template is not None for template in references[j]),
              recurring_recoveries=recurring_recoveries[j], nearby_recoveries=nearby_recoveries[j],
              clipped_recoveries=clipped_recoveries[j],
              translucent_recoveries=translucent_recoveries[j], translucent_temporal=len(blend_support),
              covered_by_other=len(covered_frames[j]), covered_weak=covered_weak[j], locally_verified=reviewed,
              visible_probes=sorted(visible))
        if unresolved or not copies:
            for i in visible.intersection(unresolved):
                known = [*remainder_masks.get(i, ()), *masked.get(i, ()), *other_masks.get(i, ())]
                known.extend((layer.motif.mask, x, y) for x, y, score in evidence[j][i][0]
                             if score >= max(.73, layer.floor))
                if known:
                    remainder_masks[i] = known
            hints = {i: visible_box(layer.motif, *edge_matches[j][i][0][:2])
                     for i in unresolved if i in edge_matches[j]}
            hints.update({i: visible_box(layer.motif, *point[:2]) for i, point in alternatives.items()})
            for i in visible.intersection(incoherent):
                hints.setdefault(i, visible_box(layer.motif, *incoherent_peaks[j][i]))
            for i in identifier_gaps:
                hints.setdefault(i, visible_box(layer.motif, *seen[i][0][:2]))
            for i in unresolved:
                candidates = [point for point in evidence[j][i][1] if (i, point) not in support and point[2] >= .47]
                if candidates and i not in hints:
                    x, y, _ = max(candidates, key=lambda row: row[2])
                    hints[i] = visible_box(layer.motif, x, y)
            gap = LayerNeedsEvidence(layer, unresolved, f'Layer {j+1} has {len(unresolved)} frames without independent coverage.',
                                     hints, visible_frames=sorted(visible), hidden_masks=remainder_masks)
            if not (copies and not hints and not visible and proofs is not None and len(proofs) > previous_proofs):
                raise gap
            deferred = deferred or gap
            trace('layers.coverage_deferred', label=layer.label, unresolved=len(unresolved),
                  remaining_layers=len(layers)-j-1, new_proofs=len(proofs)-previous_proofs)
            continue
        for slot in range(copies):
            points = np.zeros((video.count, 2), np.int32)
            scores = np.zeros(video.count, np.float32)
            active = np.zeros(video.count, bool)
            for i, row in enumerate(seen):
                if slot < len(row):
                    x, y, score = row[slot]
                    points[i], scores[i], active[i] = (x, y), score, True
            track = {'positions': points, 'scores': scores, 'present': active, 'verified': np.ones(video.count, bool),
                     'cuts': np.asarray(cuts, np.int32), 'signal': layer.motif.signal,
                     'weights': layer.motif.weights, 'mask': layer.motif.mask}
            sized = source_sized_tracks(video, track)
            if sized[0] is not track:
                trace('layers.clipped_render_windows', label=layer.label, tracks=len(sized), frames=int(active.sum()))
            tracks.extend(sized)
            if len(tracks) > MAX_TRACKS:
                raise ValueError('The number of simultaneous mask tracks exceeds the repair limit.')
    if deferred is not None:
        raise deferred
    if not 1 <= len(tracks) <= MAX_TRACKS:
        raise ValueError('The number of simultaneous mask tracks exceeds the repair limit.')
    return tracks


def positive_review_tracks(video, layers, evidence, cuts):
    """Keep confirmed edits; unknown presence/absence remains unverified in a review-only candidate."""
    tracks = []
    for layer, samples in zip(layers, evidence, strict=True):
        seen = [sorted(hits, key=lambda point: (point[1], point[0])) for hits, _, _ in samples]
        copies = max(map(len, seen), default=0)
        if not copies or copies > MAX_LAYERS:
            continue
        for slot in range(copies):
            points = np.zeros((video.count, 2), np.int32)
            scores = np.zeros(video.count, np.float32)
            active = np.zeros(video.count, bool)
            for frame, matches in enumerate(seen):
                if slot < len(matches):
                    x, y, score = matches[slot]
                    points[frame], scores[frame], active[frame] = (x, y), score, True
            track = {'positions': points, 'scores': scores, 'present': active, 'verified': active.copy(),
                     'cuts': np.asarray(cuts, np.int32), 'signal': layer.motif.signal,
                     'weights': layer.motif.weights, 'mask': layer.motif.mask}
            tracks.extend(source_sized_tracks(video, track))
            if len(tracks) > MAX_TRACKS:
                return []
    return tracks
