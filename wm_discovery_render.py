"""Apply and audit a discovered mask using the existing local video restoration engine."""

from dataclasses import dataclass
from collections import Counter
from itertools import islice

import cv2
import numpy as np

from wm_discovery import (LUMA_EXTENT_GROUPS, LUMA_EXTENT_OFFSETS, Motif, agrees_in_halves, chroma_signals, color_patch, consensus_fields, discovery_tracks,
                          luma_extent_groups, luma_extent_repairs, observe, pack_tracks, patch)
from wm_inpaint import ProPainter, composite, feather
from wm_media import Video, clipped_plane, get_frame, plane_rects, read_frames, yuv_to_rgb
from wm_repair import windows
from wm_track import highpass, load_track
from workflow import read_object, write_object


class MaskAreaExceeded(ValueError):
    """A frame's proposed edit is larger than the allowed area."""


BROAD_CHROMA_SIGMA = 12.


def put_mask(canvas, mask, x, y):
    view, section = clipped_plane(canvas, (x, y, mask.shape[1], mask.shape[0]))
    np.maximum(view, mask[section], out=view)


def verify_mask_area(video, layers, start, stop):
    limit = .16*video.width*video.height
    upper = np.zeros(stop-start, dtype=np.int64)
    for mask, _, present in layers:
        upper[present[start:stop]] += np.count_nonzero(mask)
    uncertain = np.flatnonzero(upper > limit)
    if not len(uncertain):
        return
    canvas = np.zeros((video.height, video.width), dtype=np.uint8)
    for offset in uncertain:
        i = start+int(offset)
        canvas.fill(0)
        for mask, points, present in layers:
            if present[i]:
                put_mask(canvas, mask, *map(int, points[i]))
        area = np.count_nonzero(canvas)
        if area > limit:
            raise MaskAreaExceeded(f'The discovered watermark mask would cover too much of frame {i} ({area} pixels).')


def context_rect(video, points, mask):
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)+mask.shape[::-1]
    left, top = max(0, (int(left)-34)//2*2), max(0, (int(top)-34)//2*2)
    right = min(video.width, (int(right)+35)//2*2)
    bottom = min(video.height, (int(bottom)+35)//2*2)
    return left, top, right-left, bottom-top


def context_groups(video, active_layers, *, separate_jumps=False):
    groups = []
    for mask, points, active in active_layers:
        selections = (active,)
        if separate_jumps:
            positions = np.unique(points[active], axis=0)
            selections = (active & (points == position).all(axis=1) for position in positions)
        for selected in selections:
            x, y, w, h = context_rect(video, points[selected], mask)
            right, bottom = x+w, y+h
            members = [(mask, points, selected)]
            while True:
                match = next((i for i, ((a, b, c, d), _) in enumerate(groups)
                              if max(x, a) < min(right, c) and max(y, b) < min(bottom, d)), None)
                if match is None:
                    break
                (a, b, c, d), prior = groups.pop(match)
                x, y, right, bottom = min(x, a), min(y, b), max(right, c), max(bottom, d)
                members.extend(prior)
            groups.append(((x, y, right, bottom), members))
    return groups


def visual_signatures(video, track, cuda, consensus=False):
    height, width = track['mask'].shape
    available = np.flatnonzero(track.get('present', np.ones(video.count, bool)))
    if len(available) < 12:
        return None, track['mask']
    count = 20 if consensus else 18
    frames = available[np.linspace(0, len(available)-1, min(count, len(available)), dtype=int)]
    colors, luminance = [], []
    for i in frames:
        raw = get_frame(video, int(i), cuda)
        x, y = map(int, track['positions'][i])
        colors.append(color_patch(video, raw, x, y, width, height))
        if consensus:
            luminance.append(patch(highpass(video.planes(raw)[0], 2.2), (x, y, width, height)))
    aligned = np.stack(colors)
    signal = np.median(aligned, axis=0)
    noise = np.median(np.abs(aligned-signal), axis=0)
    nearby = cv2.dilate(track['mask'], np.ones((13, 13), np.uint8)) > 0
    stable = np.mean(aligned*np.sign(signal) > 2, axis=0) >= .78
    support = nearby&stable&(np.abs(signal) > np.maximum(4, 1.3*noise))
    signal *= support
    energy = float(np.sum(signal*signal))
    color = (signal, energy, support.astype(np.uint8)) if support.sum() >= 80 and energy >= 6000 else None
    mask = consensus_luma(track, luminance) if consensus else track['mask']
    return color, mask


def consensus_luma(track, luminance):
    aligned = np.stack(luminance)
    signal = np.median(aligned, axis=0)
    noise = np.median(np.abs(aligned-signal), axis=0)
    stable = (np.abs(signal) >= np.maximum(3.5, 1.3*noise))&(np.mean(aligned*np.sign(signal)>1.5, axis=0) >= .75)
    support = cv2.dilate(stable.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tight = track['mask']&support
    power = track['signal']**2*track['weights']
    if np.sum(power*tight) >= .93*np.sum(power) and tight.sum() >= 40:
        return tight
    return track['mask']


def restoration_mask(track, color, mask, *, edge=False):
    if color is not None and 'matte_positions' not in track:
        support = cv2.dilate(color[2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = np.maximum(mask, support)
    if edge:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def layer_masks(video, track, cuda, consensus, edge_layer):
    from wm_runtime import source_runtime
    layers = discovery_tracks(track)
    if edge_layer is not None and (type(edge_layer) is not int or not 0 <= edge_layer < len(layers)
                                   or 'matte_positions' in track):
        raise ValueError('An edge retry must select a tracked inpainting layer.')
    # Standalone render/audit workers also need the bounded immutable frame cache.
    # Overlapping templates often sample the same frame; decode it only once.
    with source_runtime({}):
        for index, layer in enumerate(layers):
            color, mask = visual_signatures(video, layer, cuda, consensus)
            yield layer, color, restoration_mask(layer, color, mask, edge=index == edge_layer)


def consensus_layers_for_area(video, track, cuda, trace):
    def area_inputs(layer, mask):
        return mask, layer['positions'], layer['present']

    layers = discovery_tracks(track)
    raw = [area_inputs(layer, layer['mask']) for layer in layers]
    try:
        verify_mask_area(video, raw, 0, video.count)
    except MaskAreaExceeded as error:
        trace('layers.mask_area_consensus_started', reason=str(error), tracks=len(layers))
    else:
        return None
    tightened = [{**layer, 'mask': mask} for layer, _, mask in layer_masks(video, track, cuda, True, None)]
    verify_mask_area(video, [area_inputs(layer, layer['mask']) for layer in tightened], 0, video.count)
    trace('layers.mask_area_consensus_verified', tracks=len(layers), frames=video.count)
    return tightened


def repeated_edge_residue(mask, training, comparisons):
    aligned = np.stack(training)
    signal = np.median(aligned, axis=0)
    deviation = np.median(np.abs(aligned-signal), axis=0)
    agreement = np.mean(aligned*np.sign(signal) > 1.5, axis=0)
    stable = (np.abs(signal) >= np.maximum(3.5, 1.25*deviation)) & (agreement >= .73)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    templates = [signal*stable*band for band in (distance >= 2, (distance > 0)&(distance < 2))]
    if any(np.count_nonzero(template) < 32 for template in templates):
        return {'eligible': False, 'reason': 'Too few repeatable pixels in the core or transition.'}
    energies = [float(np.sum(template**2)) for template in templates]
    samples = []
    for frame, before, after in comparisons:
        source = [float(np.sum(before*template))/energy for template, energy in zip(templates, energies, strict=True)]
        retained = [float(np.sum(after*template))/energy for template, energy in zip(templates, energies, strict=True)]
        core, edge = [value/max(.1, initial) for value, initial in zip(retained, source, strict=True)]
        supported = min(source) >= .50 and abs(core) <= .22
        samples.append({'frame': frame, 'source_core': round(source[0], 3), 'source_edge': round(source[1], 3),
                        'core': round(core, 3), 'edge': round(edge, 3),
                        'matched': supported and .30 <= edge <= .90, 'clean': supported and abs(edge) <= .25})
    return {'eligible': samples[0]['matched'] and sum(row['matched'] for row in samples[1:]) >= 3,
            'cleared': samples[0]['clean'] and sum(row['clean'] for row in samples[1:]) >= 3,
            'samples': samples}


def edge_residue(source, restored, layer, frame, cuda, *, consensus=False):
    active = np.flatnonzero(layer.get('present', np.ones(source.count, bool)) & layer['verified'])
    if type(frame) is not int or len(active) < 17 or frame not in active:
        return {'eligible': False, 'reason': 'Too few verified occurrences for an independent edge comparison.'}
    frames = [int(i) for i in active[np.linspace(0, len(active)-1, 17, dtype=int)] if i != frame]
    held = frames[1::4]
    color, mask = visual_signatures(source, layer, cuda, consensus)
    mask = restoration_mask(layer, color, mask)
    height, width = mask.shape

    def sample(video, index):
        x, y = map(int, layer['positions'][index])
        pixels = video.planes(get_frame(video, index, cuda))[0]
        return patch(highpass(pixels, 5), (x, y, width, height))

    training = [sample(source, i) for i in frames if i not in held]
    compared = [(i, sample(source, i), sample(restored, i)) for i in [frame, *held]]
    return repeated_edge_residue(mask, training, compared)


def chroma_extent_samples(source, restored, positions, signal, frames, cuda, *, sigma=2.2):
    height, width = signal.shape[1:]
    energy = max(1., float(np.sum(signal**2)))

    def amplitude(video, frame):
        x, y = map(int, positions[frame])
        channels = chroma_signals(video.planes(get_frame(video, frame, cuda)), sigma)
        aligned = np.asarray([patch(channel, (x, y, width, height)) for channel in channels])
        return float(np.sum(aligned*signal))/energy

    rows = []
    for frame in map(int, frames):
        before, after = amplitude(source, frame), amplitude(restored, frame)
        rows.append({'frame': frame, 'source': round(before, 4), 'restored': round(after, 4),
                     'ratio': round(abs(after)/max(.1, before), 4)})
    return rows


def chroma_extent_retry(source, candidate, track, settings, index, cuda, trace):
    from wm_layers import Layer, Observation, complete_chroma_mask, visible_box

    if 'matte_positions' in track or 'chroma_extent_repair' in settings:
        return None
    layer = discovery_tracks(track)[index]
    present = layer.get('present', np.ones(source.count, bool))
    active = np.flatnonzero(present & layer['verified'])
    if len(active) < 17:
        return None
    samples = active[np.linspace(0, len(active)-1, min(19, len(active)), dtype=int)]
    training, held = samples[::2].tolist(), samples[1::2]
    color, edit = visual_signatures(source, layer, cuda, settings.get('consensus_mask', False))
    edit = restoration_mask(layer, color, edit, edge=settings.get('edge_repair', {}).get('layer') == index)
    core = Motif(layer['signal'], layer['weights'], edit)
    matches = {}
    for frame in training:
        x, y = map(int, layer['positions'][frame])
        matches[Observation(frame, visible_box(core, x, y))] = x, y, float(layer['scores'][frame])
    original = Layer(core, .51, matches, settings.get('label', ''))
    signals = observe(source, [(frame, None) for frame in training], cuda)
    extension = complete_chroma_mask(source, original, signals, cuda, trace or (lambda *_args, **_kw: None))
    if extension is None:
        return None
    anchor, position = next(iter(extension.matches.items()))
    dx, dy = (position[i]-original.matches[anchor][i] for i in range(2))
    motif = extension.motif
    original_mask = np.zeros(motif.mask.shape, np.uint8)
    put_mask(original_mask, edit, -dx, -dy)
    added = (motif.mask > 0)&(original_mask == 0)
    positions = layer['positions'].copy()
    positions[present] += [dx, dy]

    aligned = []
    for frame in training:
        x, y = map(int, positions[frame])
        channels = chroma_signals(source.planes(get_frame(source, frame, cuda)))
        aligned.append([patch(channel, (x, y, motif.width, motif.height)) for channel in channels])
    signature = np.median(aligned, axis=0).astype(np.float32)
    signature *= (np.abs(signature)>2)&added
    rows = chroma_extent_samples(source, candidate, positions, signature, held, cuda)
    eligible = all(row['source'] >= .55 and row['ratio'] >= .65 for row in rows)
    if trace:
        trace('repair.chroma_extent_evidence', layer=index, added_pixels=int(added.sum()), eligible=eligible,
              training=training, held_out=rows)
    if not eligible:
        return None
    layers = discovery_tracks(track)
    layers[index] = {**layer, 'positions': positions, 'signal': motif.signal, 'weights': motif.weights,
                     'mask': motif.mask}
    updated = pack_tracks(layers)
    updated.pop('profile_sha256', None)
    updated['extent_signal'] = signature
    updated['extent_frames'] = np.asarray(held, dtype=np.int32)
    return updated


def broad_chroma_extent_retry(source, candidate, track, settings, index, frame, cuda, trace):
    """Recover a color silhouette that is visible outside the existing edit in several frames."""
    if 'matte_positions' in track or 'chroma_extent_repair' in settings:
        return None
    layers = discovery_tracks(track)
    layer = layers[index]
    present = layer.get('present', np.ones(source.count, bool))
    active = np.flatnonzero(present & layer['verified'])
    cuts = track['cuts']
    shot = np.searchsorted(cuts, frame, side='right')
    shots = np.searchsorted(cuts, active, side='right')
    distant = active[shots != shot]
    nearby = active[(shots == shot) & (np.abs(active-frame) <= 25)]
    if frame not in nearby or len(distant) < 17 or len(nearby) < 6:
        return None
    samples = distant[np.linspace(0, len(distant)-1, 17, dtype=int)]
    training, held = samples[::2], samples[1::2]
    if len(np.unique(np.searchsorted(cuts, held, side='right'))) < 3:
        return None
    local = np.unique(np.r_[frame, nearby[np.linspace(0, len(nearby)-1, min(8, len(nearby)), dtype=int)]])

    color, edit = visual_signatures(source, layer, cuda, settings.get('consensus_mask', False))
    edit = restoration_mask(layer, color, edit, edge=settings.get('edge_repair', {}).get('layer') == index)
    pad = 34
    edit = np.pad(edit, pad)
    height, width = edit.shape
    positions = layer['positions'].copy()
    positions[present] -= pad

    def aligned(video, i):
        x, y = map(int, positions[i])
        channels = chroma_signals(video.planes(get_frame(video, int(i), cuda)), BROAD_CHROMA_SIGMA)
        return np.stack([patch(channel, (x, y, width, height)) for channel in channels])

    native = np.stack([aligned(source, i) for i in training])
    detail = np.median(native, axis=0)
    noise = np.median(np.abs(native-detail), axis=0)
    power = np.sum(detail**2, axis=0)
    agreement = np.mean(np.sum(native*detail, axis=1) > .45*power, axis=0)
    nearby_edit = cv2.dilate(edit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))) > 0
    stable = (power >= np.maximum(25., 1.96*np.sum(noise**2, axis=0))) & (agreement >= .75)
    outside = stable & nearby_edit & (edit == 0)

    def box(field):
        return cv2.boxFilter(field.astype(np.float32), -1, (19, 19), normalize=False)

    energy = np.maximum(1., box(power*outside))
    count = box(outside)
    contrast = np.sqrt(energy/np.maximum(1., count))

    def projection(pixels):
        product = box(np.sum(pixels*detail, axis=0)*outside)
        norm = box(np.sum(pixels**2, axis=0)*outside)
        return product/energy, product/np.sqrt(np.maximum(1., energy*norm))

    def matches(amplitude, correlation):
        return (amplitude >= .55) & (amplitude <= 2.4) & (correlation >= .75)

    anchor_votes = sum(matches(*projection(aligned(source, i))) for i in held)
    pairs = [(i, aligned(source, i), aligned(candidate, i)) for i in local]
    retained = []
    for _, original, rendered in pairs:
        before, similarity = projection(original)
        after, alignment = projection(rendered)
        retained.append(matches(before, similarity) & (after >= .62*before)
                        & (alignment >= .80) & (after*contrast >= 9.))
    votes = np.stack(retained)
    focus = int(np.searchsorted(local, frame))
    missing = (outside & (count >= 40) & (anchor_votes >= 6) & votes[focus]
               & (votes.sum(axis=0) >= max(5, .60*len(local))))
    grown = cv2.dilate(missing.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    _, components = cv2.connectedComponents(edit | grown)
    attached = np.isin(components, np.unique(components[edit > 0]))
    missing &= attached
    grown &= attached
    added = np.count_nonzero(grown & (edit == 0))
    signature = detail*missing
    strength = max(1., float(np.sum(signature**2)))
    measured = []
    for i, original, rendered in pairs:
        before = float(np.sum(original*signature))/strength
        after = float(np.sum(rendered*signature))/strength
        measured.append({'frame': int(i), 'source': round(before, 4),
                         'ratio': round(max(0., after)/max(.1, before), 4)})
    eligible = (48 <= added <= .35*np.count_nonzero(edit) and all(row['source'] >= .55 for row in measured)
                and measured[focus]['ratio'] >= .62 and sum(row['ratio'] >= .60 for row in measured) >= 6)
    if trace:
        trace('repair.broad_chroma_extent_evidence', layer=index, frame=frame, added_pixels=int(added),
              training=training.tolist(), independent_sources=held.tolist(), samples=measured, eligible=eligible)
    if not eligible:
        return None

    layers[index] = {**layer, 'positions': positions, 'mask': edit | grown,
                     'signal': np.pad(layer['signal'], pad), 'weights': np.pad(layer['weights'], pad)}
    updated = pack_tracks(layers)
    updated.pop('profile_sha256', None)
    return {**updated, 'extent_signal': signature, 'extent_frames': local.astype(np.int32),
            'extent_sigma': np.asarray(BROAD_CHROMA_SIGMA, np.float32)}


def luma_response(image, signal, weights):
    power = max(1., float(np.sum(signal**2*weights)))
    response = float(np.sum(image*signal*weights))
    energy = max(1., float(np.sum(image**2*weights)))
    return response/power, response/(energy*power)**.5


def luma_matches(amplitude, correlation):
    return .40 <= amplitude <= 3.0 and correlation >= .72


def faint_luma_match(image, signal, weights, amplitude, correlation):
    if not (.20 <= amplitude < .40 and correlation >= .90) or np.count_nonzero(weights) < 32:
        return False
    _, _, width, height = cv2.boundingRect((weights > 0).astype(np.uint8))
    axis = int(width >= height)
    energy = np.sum(signal**2*weights, axis=1-axis)
    split = int(np.searchsorted(np.cumsum(energy), energy.sum()/2))+1
    if min(float(energy[:split].sum()), float(energy[split:].sum())) < 120:
        return False
    halves = zip(*(np.split(field, [split], axis=axis) for field in (image, signal, weights)), strict=True)
    return all(amplitude >= .15 and score >= .80 for amplitude, score in (luma_response(*half) for half in halves))


def luma_extent_retry(source, candidate, track, settings, index, frame, cuda, trace):
    if 'matte_positions' in track:
        return None
    retained = luma_extent_proofs(source, track)
    if len(retained) >= len(LUMA_EXTENT_GROUPS):
        return None
    layer = discovery_tracks(track)[index]
    active = layer.get('present', np.ones(source.count, bool)) & layer['verified']
    if not active[frame]:
        return None
    pose = active & (layer['positions'] == layer['positions'][frame]).all(axis=1)
    frames = np.flatnonzero(pose)
    frames = frames[frames != frame]
    if len(frames) < 19:
        return None
    samples = frames[np.linspace(0, len(frames)-1, 19, dtype=int)]
    training, held = samples[::2], np.r_[frame, samples[1::2]]
    shots = np.searchsorted(track['cuts'], held, side='right')
    if len(np.unique(shots)) < 3:
        return None
    color, edit = visual_signatures(source, layer, cuda, settings.get('consensus_mask', False))
    edit = restoration_mask(layer, color, edit, edge=settings.get('edge_repair', {}).get('layer') == index)
    area_limit = .25*np.count_nonzero(edit)
    for extent in retained:
        if extent.layer == index and (extent.registered or pose[extent.frames[0]]):
            edit = edit.copy()
            put_mask(edit, extent.mask, *map(int, extent.positions[frame]-layer['positions'][frame]))
    height, width = edit.shape
    x, y = map(int, layer['positions'][frame])
    rect = x, y, width, height

    def aligned(video, indices):
        return [patch(highpass(video.planes(get_frame(video, int(i), cuda))[0], 2.2), rect) for i in indices]

    signal, weights, mask = consensus_fields(np.stack(aligned(source, training)))
    _, groups, stats, _ = cv2.connectedComponentsWithStats(mask & (edit == 0))
    raw, restored = aligned(source, held), aligned(candidate, held)
    eligible = []
    for component, (_, _, _, _, area) in enumerate(stats[1:], 1):
        extra = groups == component
        support = weights*extra
        power = float(np.sum(signal**2*support))
        if area > area_limit or np.count_nonzero(support) < 10 or power < 400:
            continue
        matched = []
        for before, after in zip(raw, restored, strict=True):
            amplitude, correlation = luma_response(before, signal, support)
            residual, _ = luma_response(after, signal, support)
            matched.append(luma_matches(amplitude, correlation) and abs(residual) >= .65*amplitude)
        if matched[0] and sum(matched) >= 6 and len(np.unique(shots[matched])) >= 3:
            eligible.append((sum(matched), power, extra, support, held[matched]))
    if not eligible:
        if trace:
            trace('repair.luma_extent_evidence', layer=index, frame=frame, eligible=False)
        return None
    _, _, extra, support, confirmed = max(eligible, key=lambda item: item[:2])
    selected = np.zeros(source.count, bool)
    for i, raw in read_frames(source, cuda=cuda):
        if pose[i]:
            image = patch(highpass(source.planes(raw)[0], 2.2), rect)
            selected[i] = luma_matches(*luma_response(image, signal, support))
    if not selected[confirmed].all():
        return None
    if trace:
        trace('repair.luma_extent_evidence', layer=index, frame=frame, eligible=True,
              added_pixels=int(extra.sum()), selected_frames=int(selected.sum()),
              held_out=sorted(map(int, confirmed)))
    updated = {key: value for key, value in track.items() if key != 'profile_sha256'}
    fields = LUMA_EXTENT_GROUPS[len(retained)]
    values = (np.asarray(index, np.int32), extra.astype(np.uint8), signal*extra, support, selected,
              np.sort(confirmed).astype(np.int32))
    updated.update(zip(fields, values, strict=True))
    return updated


def registered_luma_extent_retry(source, candidate, track, settings, index, frame, cuda, trace, backend):
    """Find a moving mask extension whose signature still survives in the rendered candidate."""
    from wm_layers import Layer, Observation, build_layer, proposals_for, registered_identifier_geometry, visible_box

    if 'matte_positions' in track or not callable(backend.get('read_pixels')):
        return None
    retained = luma_extent_proofs(source, track)
    if len(retained) >= len(LUMA_EXTENT_GROUPS):
        return None
    layer = discovery_tracks(track)[index]
    active = layer.get('present', np.ones(source.count, bool)) & layer['verified']
    strong = np.flatnonzero(active & (layer['scores'] >= .73))
    strong = strong[strong != frame]
    if not active[frame] or len(strong) < 12:
        return None
    samples = strong[np.linspace(0, len(strong)-1, min(22, len(strong)), dtype=int)]
    held = np.r_[frame, samples[1::3]]
    training = sorted(set(map(int, samples)).difference(map(int, held)))
    core = Motif(layer['signal'], layer['weights'], layer['mask'])
    positions = layer['positions']
    anchors = {}
    for i in training:
        x, y = map(int, positions[i])
        anchor = Observation(i, visible_box(core, x, y))
        anchors[i] = anchor, (x, y, float(layer['scores'][i]))
    original = Layer(core, .73, dict(anchors.values()))
    evidence = registered_identifier_geometry(source, original, anchors, backend, cuda, trace, backend['read_pixels'])
    if evidence is None or evidence[2] is None:
        return None
    name, _, (left, top, right, bottom) = evidence
    proposals = {i: [{'rect': [max(0, x+left), max(0, y+top),
                               min(source.width, x+right), min(source.height, y+bottom)]}]
                 for i, (_, (x, y, _)) in anchors.items()}
    signals = observe(source, [(i, None) for i in training], cuda)
    learned = build_layer(signals, proposals_for(source, proposals))
    if learned is None:
        return None
    offsets = Counter((x-int(positions[item.frame, 0]), y-int(positions[item.frame, 1]))
                      for item, (x, y, _) in learned.matches.items())
    (dx, dy), count = offsets.most_common(1)[0]
    if count < max(3, (len(training)+1)//2):
        return None
    color, edit = visual_signatures(source, layer, cuda, settings.get('consensus_mask', False))
    edit = restoration_mask(layer, color, edit, edge=settings.get('edge_repair', {}).get('layer') == index)
    motif = learned.motif
    existing = patch(edit, (dx, dy, motif.width, motif.height))
    for extent in retained:
        if extent.layer == index:
            x, y = extent.positions[frame]-positions[frame]
            put_mask(existing, extent.mask, int(x)-dx, int(y)-dy)
    mask = motif.mask & (existing == 0)
    added = int(mask.sum())
    weights = motif.weights*mask
    if added > .25*np.count_nonzero(edit) or np.count_nonzero(weights) < 32 or np.sum(motif.signal**2*weights) < 400:
        return None
    component = Motif(motif.signal*mask, weights, mask)

    def measure(i, raw, restored, *, test_offset=False):
        x, y = map(int, positions[i])
        x, y = x+dx, y+dy
        detail = highpass(source.planes(raw)[0], 2.2)
        output = highpass(candidate.planes(restored)[0], 2.2)
        size = component.width, component.height
        before, correlation = luma_response(patch(detail, (x, y, *size)), component.signal, weights)
        after, _ = luma_response(patch(output, (x, y, *size)), component.signal, weights)
        ratio = abs(after)/max(.1, before)
        matched = luma_matches(before, correlation) and agrees_in_halves(component, detail, x, y)
        if matched and test_offset:
            alternative = max(abs(luma_response(patch(detail, (x+a, y+b, *size)), component.signal, weights)[1])
                              for a, b in ((12, 0), (-12, 0), (0, 12), (0, -12)))
            matched = correlation-alternative >= .12
        return matched, ratio

    checks = []
    for i in map(int, held):
        matched, ratio = measure(i, get_frame(source, i, cuda), get_frame(candidate, i, cuda), test_offset=True)
        checks.append({'frame': i, 'matched': matched, 'ratio': round(ratio, 4)})
    confirmed = [row['frame'] for row in checks if row['matched'] and row['ratio'] >= .65]
    shots = np.searchsorted(track['cuts'], confirmed, side='right')
    eligible = frame in confirmed and len(confirmed) >= 4 and len(np.unique(shots)) >= 2
    trace('repair.registered_extent_evidence', layer=index, frame=frame, identifier=name, added_pixels=added,
          training=training, held_out=checks, eligible=eligible)
    if not eligible:
        return None
    selected = np.zeros(source.count, bool)
    pairs = zip(read_frames(source, cuda=cuda), read_frames(candidate, cuda=cuda), strict=True)
    for (i, raw), (_, restored) in pairs:
        if active[i]:
            matched, ratio = measure(i, raw, restored)
            selected[i] = matched and ratio >= .65
    if not selected[confirmed].all():
        return None
    updated = {key: value for key, value in track.items() if key != 'profile_sha256'}
    number = len(retained)
    values = (np.asarray(index, np.int32), mask, component.signal, weights, selected,
              np.sort(confirmed).astype(np.int32))
    updated.update(zip(LUMA_EXTENT_GROUPS[number], values, strict=True))
    updated[LUMA_EXTENT_OFFSETS[number]] = np.array([dx, dy], np.int32)
    trace('repair.registered_extent_verified', layer=index, frame=frame, active_frames=int(selected.sum()))
    return updated


@dataclass(frozen=True)
class LumaExtent:
    layer: int
    positions: np.ndarray
    mask: np.ndarray
    signal: np.ndarray
    weights: np.ndarray
    present: np.ndarray
    frames: np.ndarray
    registered: bool = False


def luma_extent_proof(video, layers, index, mask, signal, weights, present, frames, offset=None):
    if index.ndim or index.dtype.kind not in 'iu' or not 0 <= int(index) < len(layers):
        raise ValueError('The conditional luminance repair must select a tracked layer.')
    layer = layers[int(index)]
    registered = offset is not None
    if registered and (offset.shape != (2,) or offset.dtype != np.int32
                       or np.any(np.abs(offset.astype(np.int64)) >= [video.width, video.height])):
        raise ValueError('The registered luminance repair has an invalid alignment offset.')
    shape = mask.shape if registered else layer['mask'].shape
    if (mask.shape != shape or (registered and (mask.ndim != 2 or min(shape) < 8
                                                or shape[0] > video.height or shape[1] > video.width))
            or mask.dtype != np.uint8 or np.any(mask > 1)
            or signal.shape != shape or signal.dtype != np.float32 or not np.isfinite(signal).all()
            or weights.shape != shape or weights.dtype != np.float32 or not np.isfinite(weights).all()
            or np.any((weights < 0) | (weights > 1)) or np.any(weights[mask == 0])
            or np.count_nonzero(weights) < 10 or np.sum(signal**2*weights) < 400
            or present.shape != (video.count,) or present.dtype != np.bool_
            or np.any(present & ~(layer.get('present', np.ones(video.count, bool)) & layer['verified']))
            or frames.ndim != 1 or frames.dtype != np.int32 or not (4 if registered else 6) <= len(frames) <= 10
            or np.any(np.diff(frames) <= 0) or frames[0] < 0 or frames[-1] >= video.count
            or not present[frames].all()
            or (not registered and not (layer['positions'][present] == layer['positions'][frames[0]]).all())):
        raise ValueError('The conditional luminance repair has invalid pixels or source frames.')
    positions = layer['positions']+offset if registered else layer['positions']
    if registered:
        x, y, w, h = cv2.boundingRect(mask)
        left, top = (positions[present]+[x, y]).T
        if np.any((left >= video.width)|(top >= video.height)|(left+w <= 0)|(top+h <= 0)):
            raise ValueError('The registered luminance repair is outside the visible source.')
    return LumaExtent(int(index), positions, mask, signal, weights, present, frames, registered)


def luma_extent_proofs(video, track):
    groups = luma_extent_groups(track)
    if groups and 'matte_positions' in track:
        raise ValueError('The conditional luminance repair requires an inpainting track.')
    layers = discovery_tracks(track)
    return [luma_extent_proof(video, layers, *(track[key] for key in fields)) for fields in groups]


def extend_faint_runs(present, faint, cuts):
    frames = np.flatnonzero(present | faint)
    breaks = (np.diff(frames) > 1) | np.isin(frames[1:], cuts)
    extended = present.copy()
    for run in np.split(frames, np.flatnonzero(breaks)+1):
        if present[run].any():
            extended[run] = True
    return extended


def refresh_luma_visibility(video, track, cuda):
    extents = luma_extent_proofs(video, track)
    if all(extent.registered for extent in extents):
        return
    layers = discovery_tracks(track)
    poses = [(extent.positions == extent.positions[extent.frames[0]]).all(axis=1)
             & layers[extent.layer].get('present', np.ones(video.count, bool)) for extent in extents]
    faint = [np.zeros(video.count, bool) for _ in extents]
    probes = [(extent, pose & ~extent.present, matches)
              for extent, pose, matches in zip(extents, poses, faint, strict=True) if not extent.registered]
    for i, raw in read_frames(video, cuda=cuda):
        active = [(extent, matches) for extent, pending, matches in probes if pending[i]]
        if not active:
            continue
        detail = highpass(video.planes(raw)[0], 2.2)
        for extent, matches in active:
            x, y = map(int, extent.positions[i])
            h, w = extent.mask.shape
            image = patch(detail, (x, y, w, h))
            matches[i] = faint_luma_match(image, extent.signal, extent.weights,
                                          *luma_response(image, extent.signal, extent.weights))
    for number, (extent, matches, fields) in enumerate(zip(extents, faint, luma_extent_groups(track), strict=True), 1):
        if extent.registered:
            continue
        selected = extend_faint_runs(extent.present, matches, track['cuts'])
        track[fields[4]] = selected
        added = int(np.count_nonzero(selected & ~extent.present))
        if added:
            print(f'Confirmed {added} additional faint source frames for luminance repair {number}.', flush=True)


def registered_donor_match(motif, detail, position, radius):
    x, y = position
    if radius:
        x, y, _ = motif.local(detail, position, radius, radius)
    w, h = motif.width, motif.height
    image = patch(detail, (x, y, w, h))
    strength, correlation = luma_response(image, motif.signal, motif.weights)
    if not .20 <= strength <= 3.0 or correlation < .72 or not agrees_in_halves(motif, detail, x, y):
        return None
    alternatives = (patch(detail, (x+dx, y+dy, w, h)) for dx, dy in ((12, 0), (-12, 0), (0, 12), (0, -12)))
    rival = max(abs(luma_response(sample, motif.signal, motif.weights)[1]) for sample in alternatives)
    return (x, y) if correlation-rival >= .12 else None


def luma_reference_masks(video, buffer, masks, members, origin, extents, *, parents=None):
    """Hide faint source fragments from temporal propagation without committing those edits."""
    left, top = origin
    context_height, context_width = masks.shape[1:]
    context = masks
    details = {}
    for mask, _, edited in members:
        if id(mask) not in extents:
            continue
        extent, layer_present = extents[id(mask)]
        if extent.registered:
            if not parents or extent.layer not in parents:
                continue
            core, parent_positions = parents[extent.layer]
            signature = Motif(extent.signal, extent.weights, extent.mask)
            radius = min(24, max(8, min(signature.width, signature.height)//4))
            edited_offsets = np.flatnonzero(edited)
            if not len(edited_offsets):
                continue
            anchor_frame = buffer[int(edited_offsets[0])][0]
            dx, dy = map(int, extent.positions[anchor_frame]-parent_positions[anchor_frame])
            for offset, (frame, raw) in enumerate(buffer):
                if edited[offset]:
                    continue
                if offset not in details:
                    details[offset] = highpass(video.planes(raw)[0], 2.2)
                detail = details[offset]
                nearest = int(edited_offsets[np.argmin(np.abs(edited_offsets-offset))])
                reference = frame if layer_present[frame] else buffer[nearest][0]
                position = tuple(map(int, extent.positions[reference]))
                search = 0 if layer_present[frame] else min(72, radius+3*abs(nearest-offset))
                match = registered_donor_match(signature, detail, position, search)
                if match is None:
                    continue
                x, y = match
                if (x >= left+context_width or y >= top+context_height
                        or x+signature.width <= left or y+signature.height <= top):
                    continue
                if context is masks:
                    context = masks.copy()
                put_mask(context[offset], mask, x-left, y-top)
                u, v, score = core.local(detail, (x-dx, y-dy), 1, 1)
                if score >= .73 and agrees_in_halves(core, detail, u, v):
                    put_mask(context[offset], core.mask, u-left, v-top)
            continue
        x, y = map(int, extent.positions[extent.frames[0]])
        height, width = mask.shape
        for offset, (frame, raw) in enumerate(buffer):
            if edited[offset] or not layer_present[frame] or tuple(extent.positions[frame]) != (x, y):
                continue
            if offset not in details:
                details[offset] = highpass(video.planes(raw)[0], 2.2)
            image = patch(details[offset], (x, y, width, height))
            amplitude, correlation = luma_response(image, extent.signal, extent.weights)
            if .20 <= amplitude <= 3.0 and correlation >= .72:
                if context is masks:
                    context = masks.copy()
                put_mask(context[offset], mask, x-left, y-top)
    return context


def reference_matte(track, track_path):
    from wm_matte import Matte

    height, width = track['mask'].shape
    positions, precise = track['matte_positions'], track['matte_precise']
    if (positions.shape != track['positions'].shape or positions.dtype != np.int32
            or precise.shape != positions.shape or not np.array_equal(precise, track['positions'])
            or np.any(positions < 0) or np.any(positions % 2)
            or np.any((precise-positions < 0)|(precise-positions > 1))):
        raise ValueError('The inverse-compositing plate has invalid placement.')
    for j in range(3):
        fields = track.get(f'fields{j}')
        if fields is None or fields.shape != (2, height, width) or not np.isfinite(fields).all():
            raise ValueError('The inverse-compositing plate is incomplete.')
    profile = {'translucent': {'reference_rect': [0, 0, width, height]}}
    matte = Matte(profile, {**track, 'positions': positions, 'precise': precise}, track_path)
    return ReferencePatches(matte, track) if 'reference_indices' in track else matte


class ReferencePatches:
    def __init__(self, matte, track):
        self.matte, self.track = matte, track
        indices = track['reference_indices']
        self.height, self.width = track['mask'].shape
        self.patches = [track.get(f'reference{j}') for j in range(3)]
        if (indices.ndim != 1 or indices.dtype != np.int32 or np.any(np.diff(indices) <= 0)
                or np.any((indices < 0)|(indices >= len(track['positions'])))
                or any(p is None or p.dtype != np.uint8 or p.shape != (len(indices), self.height//scale, self.width//scale)
                       for p, scale in zip(self.patches, (1, 2, 2), strict=True))):
            raise ValueError('The same-shot reference patches are incomplete.')
        self.by_frame = {int(frame): row for row, frame in enumerate(indices)}

    def restore(self, video, frames, start):
        self.matte.restore(video, frames, start)
        for i, frame in enumerate(frames, start):
            row = self.by_frame.get(i)
            if row is None:
                continue
            x, y = self.track['matte_positions'][i]
            dx, dy = self.track['positions'][i]-[x, y]
            mask = cv2.warpAffine(self.track['mask'], np.float32([[1, 0, dx], [0, 1, dy]]),
                                  (self.width, self.height)).astype(np.float32)
            rect = int(x), int(y), self.width, self.height
            for plane, patch in zip(plane_rects(video.planes(frame), rect), self.patches, strict=True):
                active = cv2.resize(mask, plane.shape[::-1], interpolation=cv2.INTER_AREA) > 0
                plane[active] = patch[row, active]


def restore_frames(video, track, start, stop, cuda, scale, raft_iterations, model_dir=None, *, consensus=False,
                   separate_jumps=False, edge_layer=None):
    layers = [(mask, layer['positions'], layer.get('present', np.ones(video.count, bool)))
              for layer, _, mask in layer_masks(video, track, cuda, consensus, edge_layer)]
    luma = luma_extent_proofs(video, track)
    references = {id(extent.mask): (extent, layers[extent.layer][2]) for extent in luma}
    parents = {}
    if any(extent.registered for extent in luma):
        tracked = discovery_tracks(track)
        for index in {extent.layer for extent in luma if extent.registered}:
            layer = tracked[index]
            parents[index] = Motif(layer['signal'], layer['weights'], layer['mask']), layer['positions']
        for extent in luma:
            if extent.registered:
                references[id(extent.mask)] = extent, layers[extent.layer][2] & tracked[extent.layer]['verified']
    layers.extend((extent.mask, extent.positions, extent.present) for extent in luma)
    verify_mask_area(video, layers, start, stop)
    cuts = track['cuts']
    cuts = cuts[(cuts > start)&(cuts < stop)]-start
    model = None
    source = iter(read_frames(video, start, stop, cuda))
    buffer = []
    first = end = 0
    for window in windows(stop-start, cuts, 24, 4):
        buffer = buffer[window.in_start-first:]
        buffer.extend(islice(source, window.in_stop-end))
        first, end = window.in_start, window.in_stop
        inputs = slice(start+first, start+end)
        outputs = range(window.start-first, window.stop-first)
        active_layers = [(mask, points[inputs], present[inputs]) for mask, points, present in layers
                         if present[inputs].any()]
        restored = {}
        for (x, y, right, bottom), members in context_groups(video, active_layers, separate_jumps=separate_jumps):
            w, h = right-x, bottom-y
            region = x, y, w, h
            masks = np.zeros((len(buffer), h, w), np.uint8)
            for mask, points, active in members:
                for offset in np.flatnonzero(active):
                    left, top = map(int, points[offset])
                    put_mask(masks[offset], mask, left-x, top-y)
            active_outputs = [offset for offset in outputs if masks[offset].any()]
            if not active_outputs:
                continue
            originals = [plane_rects(video.planes(frame), region) for _, frame in buffer]
            rgb = np.stack([yuv_to_rgb(parts, video.matrix) for parts in originals])
            if model is None:
                options = {'model_dir': model_dir} if model_dir else {}
                model = ProPainter(scale=scale, raft_iterations=raft_iterations, **options)
            model.scale = min(scale, (108000/(w*h))**.5)
            context = luma_reference_masks(video, buffer, masks, members, (x, y), references,
                                           parents=parents) if references else masks
            predicted = model.predict(rgb, context)
            for offset in active_outputs:
                if offset not in restored:
                    restored[offset] = buffer[offset][1].copy()
                composite(plane_rects(video.planes(restored[offset]), region), predicted[offset],
                          feather(masks[offset], 2), video.matrix)
        for offset in outputs:
            i, original = buffer[offset]
            yield i, restored.get(offset, original)
    next(source, None)


def summarize_layer(layer, motif, color, mask, present, residuals, color_residuals):
    active_frames = np.flatnonzero(present)
    ranked = active_frames[np.argsort(-residuals[active_frames])]
    first = int(ranked[0])
    displacement = np.abs(layer['positions'][ranked]-layer['positions'][first])
    separated = ranked[(displacement > np.asarray(motif.mask.shape[::-1])//3).any(axis=1)]
    worst = list(dict.fromkeys([first, *map(int, separated[:1]), *map(int, ranked[:8])]))[:8]
    p95 = float(np.quantile(residuals[active_frames], .95))
    maximum = float(max(residuals))
    covered = int(np.count_nonzero(layer['verified']))
    report = dict(discovery_verified_frames=covered, discovery_residual_p95=round(p95, 5),
                  discovery_residual_max=round(maximum, 5),
                  discovery_worst_frames=worst,
                  discovery_mask_added_pixels=int(np.count_nonzero(mask&(motif.mask==0))),
                  discovery_mask_removed_pixels=int(np.count_nonzero(motif.mask&(mask==0))),
                  discovery_passed=covered == len(present) and p95 <= .35 and maximum <= .55)
    if color_residuals:
        c95 = float(np.quantile(color_residuals, .95))
        cmax = max(color_residuals)
        color_frame = int(active_frames[np.argmax(color_residuals)])
        report.update(discovery_color_residual_p95=round(c95, 5), discovery_color_residual_max=round(cmax, 5),
                      discovery_color_support_pixels=int(color[2].sum()))
        if cmax > .30:
            report['discovery_worst_frames'] = list(dict.fromkeys([first, color_frame, *worst]))[:8]
        report['discovery_passed'] &= c95 <= .35 and cmax <= .60
    return report


def projected_residual(motif, source, restored, x, y, *, regional=False):
    weighted = motif.signal*motif.weights
    rect = (x, y, motif.width, motif.height)
    visible, section = clipped_plane(source, rect)
    if visible.shape != weighted.shape:
        # An off-screen part of the reference cannot reduce the measured visible residual.
        clipped = np.zeros_like(weighted)
        clipped[section] = weighted[section]
        weighted = clipped
    sample = patch(source, rect)
    search = patch(restored, (x-2, y-2, motif.width+4, motif.height+4))

    def measure(left, right, *, component=False):
        template = weighted[:, left:right]
        energy = max(1., float(np.sum(motif.signal[:, left:right]*template)))
        before = abs(float(np.sum(sample[:, left:right]*template)))
        if component and (energy < 120 or before < .25*energy):
            return 0.
        after = float(np.max(np.abs(cv2.matchTemplate(search[:, left:right+4], template, cv2.TM_CCORR))))
        return after/max(.25*energy, before)

    score = measure(0, motif.width)
    if regional:
        middle = motif.width//2
        score = max(score, measure(0, middle, component=True), measure(middle, motif.width, component=True))
    return score


def audit_discovery(source, output, profile_path, workdir, report_path, cuda):
    track = load_track(source, profile_path, workdir/'track.npz')
    settings = read_object(profile_path)['discovery']
    consensus = settings.get('consensus_mask', False) and 'matte_positions' not in track
    edge_repair = settings.get('edge_repair', {})
    edge_layer = edge_repair.get('layer')
    metric = settings.get('residual_metric')
    projection = metric in ('amplitude', 'regional-amplitude')
    layers = []
    for layer, color, mask in layer_masks(source, track, cuda, consensus, edge_layer):
        motif = Motif(layer['signal'], layer['weights'], layer['mask'])
        present = layer.get('present', np.ones(source.count, bool))
        layers.append((layer, motif, color, mask, present, np.zeros(source.count), []))
    any_present = np.logical_or.reduce([row[4] for row in layers])
    luma = luma_extent_proofs(source, track)
    luma_settings = luma_extent_repairs(settings)
    if (len(luma) != len(luma_settings) or any(setting['layer'] != extent.layer or setting['frame'] not in extent.frames
                                               for extent, setting in zip(luma, luma_settings, strict=True))):
        raise ValueError('The conditional luminance repair does not match the audited profile.')
    luma_samples = [[] for _ in luma]
    checked_luma = list(zip(luma, luma_samples, strict=True))
    if 'matte_positions' not in track:
        edits = [(mask, layer['positions'], present) for layer, _, _, mask, present, _, _ in layers]
        edits.extend((extent.mask, extent.positions, extent.present) for extent in luma)
        verify_mask_area(source, edits, 0, source.count)
    outside, absent_changes = [], []
    step = max(1, source.count//45)
    streams = zip(read_frames(source, cuda=cuda), read_frames(output, cuda=cuda), strict=True)
    for (i, raw), (_, restored) in streams:
        if i and i % 300 == 0:
            print(f'Audited {i}/{source.count} frames.', flush=True)
        before, after = source.planes(raw)[0], output.planes(restored)[0]
        if not any_present[i]:
            absent_changes.append(float(np.abs(before.astype(np.float32)-after).mean()))
            continue
        detail = highpass(after, 2.2)
        luma_active = [(extent, samples) for extent, samples in checked_luma if extent.present[i]]
        original_detail = highpass(before, 2.2) if projection or luma_active else None
        protected = np.zeros(before.shape, np.uint8) if i%step == 0 else None
        for layer, motif, color, mask, present, residuals, color_residuals in layers:
            if not present[i]:
                continue
            x, y = map(int, layer['positions'][i])
            if projection:
                residuals[i] = projected_residual(motif, original_detail, detail, x, y,
                                                  regional=metric == 'regional-amplitude')
            else:
                _, _, score = motif.local(detail, (x, y), 2, 2)
                residuals[i] = max(0., score)/max(.1, float(layer['scores'][i]))
            if color is not None:
                expected, energy, _ = color
                height, width = mask.shape
                source_color = color_patch(source, raw, x, y, width, height)
                target_color = color_patch(output, restored, x, y, width, height)
                strength = float(np.sum(expected*source_color)/energy)
                residual = float(np.sum(expected*target_color)/energy)
                color_residuals.append(abs(residual)/max(.45, abs(strength)))
            if protected is not None:
                put_mask(protected, mask, x, y)
        for extent, samples in luma_active:
            x, y = map(int, extent.positions[i])
            height, width = extent.mask.shape
            rect = x, y, width, height
            source_image = patch(original_detail, rect)
            initial, correlation = luma_response(source_image, extent.signal, extent.weights)
            residual, _ = luma_response(patch(detail, rect), extent.signal, extent.weights)
            matched = luma_matches(initial, correlation) or faint_luma_match(
                source_image, extent.signal, extent.weights, initial, correlation)
            samples.append({'frame': i, 'matched': matched,
                            'ratio': abs(residual)/max(.1, initial)})
            if protected is not None:
                put_mask(protected, extent.mask, x, y)
        if protected is not None:
            delta = before.astype(np.float32)-after
            outside.append(float(np.abs(delta[protected == 0]).mean()))
    summaries = [summarize_layer(*row) for row in layers]
    outside_mean = float(np.mean(outside)) if outside else 0.
    report = {**summaries[0], 'frames_audited': source.count, 'outside_mask_luma_mae': round(outside_mean, 4)}
    if edge_layer is not None:
        report['discovery_edge_layer'] = edge_layer
    if projection:
        report['discovery_residual_metric'] = metric
    report['discovery_passed'] = outside_mean <= 1.8 and all(row['discovery_passed'] for row in summaries)
    if edge_layer is not None:
        checked = edge_residue(source, output, layers[edge_layer][0], edge_repair.get('frame'), cuda)
        report['discovery_edge_evidence'] = checked
        report['discovery_passed'] &= checked.get('cleared', False)
    if (extent := settings.get('chroma_extent_repair')) is not None:
        index = extent.get('layer')
        if type(index) is not int or not 0 <= index < len(layers):
            raise ValueError('The chroma audit must select a verified layer.')
        layer, _, _, mask, active, *_ = layers[index]
        signal, frames = track['extent_signal'], track['extent_frames']
        if (signal.shape != (2, *mask.shape) or signal.dtype != np.float32 or not np.isfinite(signal).all()
                or not np.any(signal) or np.any(signal[:, mask == 0])
                or frames.ndim != 1 or frames.dtype != np.int32 or not 6 <= len(frames) <= 9
                or np.any(np.diff(frames) <= 0) or frames[0] < 0 or frames[-1] >= source.count or not active[frames].all()):
            raise ValueError('The chroma audit proof has invalid pixels or sample frames.')
        if 'extent_sigma' in track:
            scale = track['extent_sigma']
            if scale.ndim != 0 or float(scale) != BROAD_CHROMA_SIGMA or extent.get('frame') not in frames:
                raise ValueError('The broad chroma repair has an invalid audit scale or focus frame.')
            samples = chroma_extent_samples(source, output, layer['positions'], signal, frames, cuda, sigma=float(scale))
        else:
            samples = chroma_extent_samples(source, output, layer['positions'], signal, frames, cuda)
        ratios = [row['ratio'] for row in samples if row['source'] >= .55]
        cleared = len(ratios) >= 6 and float(np.quantile(ratios, .95)) <= .35 and max(ratios) <= .60
        report['discovery_chroma_extent'] = {'layer': index, 'samples': samples, 'cleared': cleared}
        report['discovery_passed'] &= cleared
    if luma:
        summaries_luma = []
        for extent, samples in checked_luma:
            ratios = [row['ratio'] for row in samples]
            p95 = float(np.quantile(ratios, .95))
            worst = max(samples, key=lambda row: row['ratio'])
            cleared = all(row['matched'] for row in samples) and p95 <= .35 and worst['ratio'] <= .60
            summaries_luma.append({'layer': extent.layer, 'frames': len(ratios), 'cleared': cleared,
                                   'residual_p95': round(p95, 5), 'worst_frame': worst['frame'],
                                   'residual_max': round(worst['ratio'], 5)})
        report['discovery_luma_extents'] = summaries_luma
        if len(summaries_luma) == 1:
            report['discovery_luma_extent'] = summaries_luma[0]
        report['discovery_passed'] &= all(row['cleared'] for row in summaries_luma)
    if len(layers) > 1:
        report['discovery_layers'] = summaries
        worst = list(dict.fromkeys(i for row in summaries for i in row['discovery_worst_frames'][:2]))
        report['discovery_worst_frames'] = worst
        for key in ('discovery_residual_p95', 'discovery_residual_max', 'discovery_color_residual_p95',
                    'discovery_color_residual_max'):
            values = [row[key] for row in summaries if key in row]
            if values:
                report[key] = max(values)
    if absent_changes:
        absent = np.flatnonzero(~any_present)
        ranges = np.split(absent, np.flatnonzero(np.diff(absent)>1)+1)
        worst_absent = max(absent_changes)
        report.update(discovery_unmarked_frames=len(absent), unmarked_frame_luma_max=round(worst_absent, 4),
                      discovery_absence_samples=[int(run[len(run)//2]) for run in ranges])
        report['discovery_passed'] &= worst_absent <= 1.8
    write_object(report_path, report)
    print(report, flush=True)
