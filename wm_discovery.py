"""Learn a watermark signature from visual evidence and verify it on the whole source."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from wm_media import clipped_plane, digest, get_frame, read_frames
from wm_track import highpass, save_npz


LAYER_FIELDS = ('positions', 'scores', 'present', 'verified', 'signal', 'weights', 'mask')
LUMA_EXTENT_FIELDS = ('luma_layer', 'luma_mask', 'luma_signal', 'luma_weights', 'luma_present', 'luma_frames')
LUMA_EXTENT_GROUPS = (LUMA_EXTENT_FIELDS, *(tuple(f'{key}_{i}' for key in LUMA_EXTENT_FIELDS) for i in range(2, 31)))
LUMA_EXTENT_OFFSETS = ('luma_offset', *(f'luma_offset_{i}' for i in range(2, 31)))
MAX_LAYERS = 60  # Simultaneous verified placements of one template.
MAX_TEMPLATES = 120
MAX_TRACKS = 360


def luma_extent_groups(track):
    remaining = {key for key in track if key.startswith('luma_')}
    groups = []
    for fields, offset in zip(LUMA_EXTENT_GROUPS, LUMA_EXTENT_OFFSETS, strict=True):
        if not remaining:
            break
        if not remaining.issuperset(fields):
            raise ValueError('The conditional luminance repair proof is incomplete.')
        if offset in remaining:
            fields = (*fields, offset)
        groups.append(fields)
        remaining.difference_update(fields)
    if remaining:
        raise ValueError('The conditional luminance repair exceeds the component limit.')
    return groups


def luma_extent_repairs(settings):
    repairs = settings.get('luma_extent_repair', [])
    if isinstance(repairs, dict):
        repairs = [repairs]
    if (not isinstance(repairs, list) or len(repairs) > len(LUMA_EXTENT_GROUPS)
            or any(not isinstance(row, dict) or type(row.get('layer')) is not int or type(row.get('frame')) is not int
                   for row in repairs)):
        raise ValueError('The conditional luminance repairs need valid layers and frames.')
    return repairs


def pack_tracks(tracks):
    primary, *additional = tracks
    if not additional:
        return primary
    if len(additional) >= MAX_TRACKS:
        raise ValueError('The discovery track contains too many watermark placements.')
    return {**primary, 'layer_count': np.asarray(len(tracks), np.int32),
            **{f'layer{i}_{key}': layer[key] for i, layer in enumerate(additional, 1) for key in LAYER_FIELDS}}


def discovery_tracks(track):
    stored = np.asarray(track.get('layer_count', 1))
    if stored.ndim != 0 or stored.dtype.kind not in 'iu':
        raise ValueError('The discovery track layer count must be an integer.')
    count = int(stored)
    if not 1 <= count <= MAX_TRACKS or (count > 1 and 'matte_positions' in track):
        raise ValueError('The discovery track contains an unsupported layer count.')
    try:
        return [track, *({key: track[f'layer{i}_{key}'] for key in LAYER_FIELDS} for i in range(1, count))]
    except KeyError as error:
        raise ValueError('A discovery track layer is incomplete.') from error


def patch(image, rect):
    x, y, w, h = rect
    result = np.zeros((h, w), image.dtype)
    view, section = clipped_plane(image, (x, y, w, h))
    result[section] = view
    return result


def response(image, signal, weights):
    rows, columns = image.shape[0]-signal.shape[0]+1, image.shape[1]-signal.shape[1]+1
    if rows > 0 and columns > 0 and rows*columns <= 49:
        # Tiny local searches need only a few dot products, not two full FFT correlations.
        windows = np.lib.stride_tricks.sliding_window_view(image, signal.shape)
        numerator = np.einsum('ijxy,xy->ij', windows, signal*weights, optimize=False)
        energy = np.einsum('ijxy,ijxy,xy->ij', windows, windows, weights, optimize=False)
    else:
        numerator = cv2.matchTemplate(image, signal*weights, cv2.TM_CCORR)
        energy = cv2.matchTemplate(image*image, weights, cv2.TM_CCORR)
    reference = float(np.sum(signal*signal*weights))
    floor = max(1., 4*float(weights.sum()))
    return np.clip(numerator/np.sqrt(np.maximum(energy, floor)*max(reference, 1.)), -1, 1)


class AmbiguousPeak(ValueError):
    pass


def peaks(scores, width, height, count, *, tolerance=0., minimum=None):
    scores = scores.copy()
    for number in range(count):
        _, value, _, (x, y) = cv2.minMaxLoc(scores)
        if (tolerance and value > max(.01, minimum or 0.)
                and cv2.countNonZero((scores >= value-tolerance).astype(np.uint8)) > 1):
            raise AmbiguousPeak('Two search locations are indistinguishable at GPU precision.')
        yield x, y, float(value)
        # Keep a runner-up and the first rejected peak as evidence of completeness.
        if minimum is not None and number >= 1 and value < minimum:
            return
        rx, ry = max(3, width//3), max(3, height//3)
        scores[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1] = -1


def strong_match(score, alternative):
    return score >= .76 and score-alternative >= .18


@dataclass
class Motif:
    signal: np.ndarray
    weights: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        shape = self.signal.shape
        if (self.signal.ndim != 2 or min(shape) < 8 or self.weights.shape != shape or self.mask.shape != shape
                or self.signal.dtype != np.float32 or self.weights.dtype != np.float32
                or self.mask.dtype != np.uint8 or np.any(self.mask > 1)
                or not np.isfinite(self.signal).all() or not np.isfinite(self.weights).all()
                or np.any((self.weights < 0) | (self.weights > 1))
                or np.count_nonzero(self.weights) < 32 or np.count_nonzero(self.mask) < 32):
            raise ValueError('The repeated watermark has too little consistent image evidence.')
        self.height, self.width = shape
        self.bounds = cv2.boundingRect(self.mask.astype(np.uint8))
        size = (self.width//2, self.height//2)
        self.small_weights = cv2.resize(self.weights, size, interpolation=cv2.INTER_AREA)
        self.small_signal = cv2.resize(self.signal*self.weights, size, interpolation=cv2.INTER_AREA)/np.maximum(
            self.small_weights, .001)

    def local(self, image, rect, radius_x, radius_y, *, reversible=False):
        x, y = rect
        left, top = x-radius_x, y-radius_y
        region = patch(image, (left, top, self.width+2*radius_x, self.height+2*radius_y))
        scores = response(region, self.signal, self.weights)
        _, _, _, (a, b) = cv2.minMaxLoc(np.abs(scores) if reversible else scores)
        return left+a, top+b, float(scores[b, a])

    def locate(self, image, *, coarse=False):
        # Padding permits a learned margin outside the picture while the foreground stays in view.
        px, py = max(8, self.width//6), max(8, self.height//6)
        padded = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
        if not coarse and image.size <= 520000:
            ranked = list(peaks(response(padded, self.signal, self.weights), self.width, self.height, 2))
            x, y, score = ranked[0]
            return x-px, y-py, score, ranked[1][2]
        size = (padded.shape[1]//2, padded.shape[0]//2)
        small = cv2.resize(padded, size, interpolation=cv2.INTER_AREA)
        guesses = peaks(response(small, self.small_signal, self.small_weights),
                        self.width//2, self.height//2, 4)
        ranked = sorted((self.local(image, (2*x-px, 2*y-py), 5, 5) for x, y, _ in guesses),
                        key=lambda item: -item[2])
        x, y, score = ranked[0]
        other = next((s for a, b, s in ranked[1:]
                      if abs(a-x) > self.width//3 or abs(b-y) > self.height//3), -1.)
        return x, y, score, other


def observe(video, observations, cuda):
    return {i: highpass(video.planes(get_frame(video, i, cuda))[0], 2.2) for i, _ in observations}


def learned_motif(observations, signals):
    training = observations[:-1]
    candidates = []
    for i, (x, y, w, h) in training:
        mx, my = max(8, w//7), max(8, h//5)
        rect = x-mx, y-my, w+2*mx, h+2*my
        template = patch(signals[i], rect)
        strength = np.abs(template)
        weights = np.minimum(strength/12, 1).astype(np.float32)
        if np.count_nonzero(weights>.2) < 32:
            continue
        guide = Motif(template, weights, (weights>.2).astype(np.uint8))
        aligned, scores = [(i, rect[0], rect[1])], []
        for j, _ in training:
            if i == j:
                continue
            u, v, score, other = guide.locate(signals[j])
            if score >= .40 and score-other >= .075:
                aligned.append((j, u, v))
                scores.append(score)
        if scores:
            candidates.append((len(scores), min(scores), guide.width, guide.height, aligned))
    if not candidates:
        raise ValueError('The visual observations do not align to a repeated graphic.')
    _, _, width, height, placements = max(candidates, key=lambda row: row[:2])
    motif, _ = aligned_consensus(width, height, placements, signals)
    return motif


def aligned_consensus(width, height, placements, signals):
    """Return the consensus and its crop origin relative to the guide placements."""
    mx, my = max(6, width//8), max(6, height//7)
    aligned = np.stack([patch(signals[i], (x-mx, y-my, width+2*mx, height+2*my))
                        for i, x, y in placements])
    signal, weights, mask = consensus_fields(aligned)
    if not mask.any():
        raise ValueError('The observations do not define a coherent watermark mask.')
    x, y, w, h = cv2.boundingRect(mask)
    left, top = max(0, x-3), max(0, y-3)
    rect = np.s_[top:y+h+3, left:x+w+3]
    return Motif(signal[rect].copy(), weights[rect].copy(), mask[rect].copy()), (left-mx, top-my)


def consensus_fields(aligned, visible=None):
    if visible is None:
        signal = np.median(aligned, axis=0).astype(np.float32)
        noise = np.median(np.abs(aligned-signal), axis=0)
        agreement = np.mean(aligned*np.sign(signal) > 1.5, axis=0)
    else:
        count = visible.sum(axis=0)
        measured = count >= 3
        samples = np.where(visible[:, measured], aligned[:, measured], np.nan)
        median = np.nanmedian(samples, axis=0)
        signal = np.zeros(aligned.shape[1:], np.float32)
        noise, agreement = np.zeros_like(signal), np.zeros_like(signal)
        signal[measured] = median
        noise[measured] = np.nanmedian(np.abs(samples-median), axis=0)
        agreement[measured] = np.sum(samples*np.sign(median) > 1.5, axis=0)/count[measured]
    strength = np.abs(signal)
    reliable = (strength >= 3.5) & (strength >= 1.35*noise) & (agreement >= .75)
    weights = np.minimum(strength/12, 1).astype(np.float32)*reliable
    support = cv2.morphologyEx(reliable.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(support.shape, np.uint8)
    cv2.drawContours(mask, [c for c in contours if cv2.contourArea(c) >= 2], -1, 1, cv2.FILLED)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return signal, weights, mask


def evidence_threshold(motif, observations, signals):
    scores, distractors, verified = [], [], []
    a, b, w, h = motif.bounds
    for i, (left, top, width, height) in observations:
        x, y, score, other = motif.locate(signals[i])
        x, y = x+a, y+b
        overlap = max(0, min(left+width, x+w)-max(left, x))*max(0, min(top+height, y+h)-max(top, y))
        agrees = overlap >= .55*min(width*height, w*h)
        if score < .43 or score-other < .055 or (not agrees and not strong_match(score, other)):
            raise ValueError(f'Full-frame matching did not independently verify the watermark at frame {i}.')
        scores.append(score)
        distractors.append(other)
        verified.append(dict(frame=i, visual_box=[left, top, width, height], verified_box=[x,y,w,h],
                             corroborates_visual_box=agrees))
    if sum(row['corroborates_visual_box'] for row in verified) < 2:
        raise ValueError('At least two visual observations must corroborate the learned graphic.')
    floor = max(.38, max(distractors)+.035, min(scores)*.78)
    if floor >= min(scores)-.02:
        raise ValueError('Watermark matches cannot be separated from similar scene content.')
    return floor, scores, distractors, verified


def propose_boxes(video, observations, cuda):
    """Find repeatable regions that a visual model can confirm from the actual pixels."""
    if len(observations) < 3:
        return {}
    signals = observe(video, observations, cuda)
    try:
        motif = learned_motif(observations, signals)
    except ValueError:
        return {}
    left, top, width, height = motif.bounds
    proposals = {}
    for i, _ in observations:
        x, y, score, other = motif.locate(signals[i])
        x, y = x+left, y+top
        x0, y0 = max(0, x), max(0, y)
        w, h = min(video.width, x+width)-x0, min(video.height, y+height)-y0
        if strong_match(score, other) and min(w, h) >= 8 and w*h <= .16*video.width*video.height:
            proposals[i] = (x0, y0, w, h)
    return proposals


def existing_motifs(directory, video, source_digest):
    def modified(path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    paths = sorted(directory.glob('*.npz'), key=modified, reverse=True)[:120]
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data['source_sha256']) == source_digest:
                    continue
                signal = data['signal']
                if signal.ndim != 2 or signal.shape[0] > video.height or signal.shape[1] > video.width:
                    continue
                motif = Motif(signal, data['weights'], data['mask'])
        except (OSError, ValueError, KeyError, cv2.error):
            continue
        yield path, motif


def remembered_proposals(video, sampled, library, identity, trace):
    proposals = {}
    signals = None
    for path, motif in existing_motifs(library, video, identity):
        if signals is None:
            signals = {i: highpass(planes[0], 2.2) for i, planes in sampled}
        locations = []
        left, top, width, height = motif.bounds
        for i, signal in signals.items():
            x, y, score, other = motif.locate(signal, coarse=True)
            if strong_match(score, other) and agrees_in_halves(motif, signal, x, y):
                x, y = x+left, y+top
                locations.append((i, [x, y, x+width, y+height]))
        if len(locations) < 3 or locations[-1][0]-locations[0][0] < .20*(video.count-1):
            continue
        trace('discovery.memory_proposals', prototype=str(path), frames=[i for i, _ in locations])
        for i, rect in locations:
            proposals.setdefault(i, []).append({'rect': rect, 'provider': 'verified-library'})
    return proposals


def scan_source(video, motif, floor, cuda, margin=.045, *, fast=False):
    positions = np.empty((video.count, 2), np.int32)
    scores = np.empty(video.count, np.float32)
    verified = np.empty(video.count, bool)
    cuts = []
    previous = None
    for i, raw in read_frames(video, cuda=cuda):
        luma = video.planes(raw)[0]
        signal = highpass(luma, 2.2)
        coarse = fast and 150000 < signal.size <= 520000 and min(motif.width, motif.height) >= 16
        x, y, score, other = motif.locate(signal, coarse=coarse)
        if coarse and (score < floor or not strong_match(score, other)):
            x, y, score, other = motif.locate(signal)
        positions[i] = x, y
        scores[i] = score
        verified[i] = score >= floor and score-other >= margin
        current = cv2.resize(luma, (48, 28), interpolation=cv2.INTER_AREA).astype(np.int16)
        if previous is not None and np.abs(current-previous).mean() > 23:
            cuts.append(i)
        previous = current
        if (i+1)%600 == 0:
            print(f'Located the learned watermark on {i+1}/{video.count} source frames.', flush=True)
    return dict(positions=positions, scores=scores, verified=verified, cuts=np.asarray(cuts, np.int32),
                signal=motif.signal, weights=motif.weights, mask=motif.mask)


def chroma_signals(planes, sigma=2.2):
    h, w = planes[0].shape
    return [highpass(cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR), sigma) for p in planes[1:]]


def color_patch(video, raw, x, y, width, height):
    _, u, v = video.planes(raw)
    left, top = max(0, (x-24)//2*2), max(0, (y-24)//2*2)
    right, bottom = min(video.width, (x+width+25)//2*2), min(video.height, (y+height+25)//2*2)
    if left >= right or top >= bottom:
        return np.zeros((height, width), np.float32)
    roi = np.s_[top//2:bottom//2, left//2:right//2]
    values = cv2.resize(v[roi].astype(np.float32)-u[roi], (right-left, bottom-top), interpolation=cv2.INTER_LINEAR)
    return patch(highpass(values, 6), (x-left, y-top, width, height))


def chroma_templates(read_planes, verified, positions, support, *, allow_partial=False):
    height, width = support.shape
    frames = np.flatnonzero(verified)
    samples = []
    for i in frames[::max(1, len(frames)//16)]:
        x, y = map(int, positions[i])
        samples.append([patch(s, (x, y, width, height)) for s in chroma_signals(read_planes(int(i)))])
    templates = []
    for signal in np.median(samples, axis=0).astype(np.float32):
        weights = np.minimum(np.abs(signal)/4, 1)*support
        if np.count_nonzero(weights) < 32 or np.sum(signal*signal*weights) < 400:
            if not allow_partial:
                return []
            templates.append(None)
        else:
            templates.append(Motif(signal, weights, support))
    return templates if any(template is not None for template in templates) else []


def chroma_position(planes, templates):
    if not templates:
        return None
    matches = [m.locate(s) for m, s in zip(templates, chroma_signals(planes), strict=True)]
    x, y, _, _ = matches[0]
    if any(abs(x-a)+abs(y-b) > 1 or not strong_match(score, other) for a, b, score, other in matches):
        return None
    return (int(x), int(y)), matches


def recover_chroma(video, track, cuda, trace):
    missing = np.flatnonzero(~track['verified'])
    if not len(missing) or len(missing) > .4*video.count:
        return

    def read_planes(i):
        return video.planes(get_frame(video, i, cuda))

    templates = chroma_templates(read_planes, track['verified'], track['positions'], track['mask'])
    if not templates:
        return
    motif = Motif(track['signal'], track['weights'], track['mask'])
    fixed = []
    for i, raw in read_frames(video, int(missing[0]), int(missing[-1])+1, cuda):
        if track['verified'][i]:
            continue
        planes = video.planes(raw)
        if result := chroma_position(planes, templates):
            position, scores = result
            _, _, score = motif.local(highpass(planes[0], 2.2), position, 0, 0)
            track['positions'][i] = position
            track['scores'][i] = score
            track['verified'][i] = True
            fixed.append(i)
            if len(fixed) <= 12:
                trace('discovery.chroma_frame', frame=i, position=list(position), matches=scores)
    trace('discovery.chroma_recovery', missing=len(missing), recovered=len(fixed), remaining=len(missing)-len(fixed))


def agrees_in_halves(motif, image, x, y, polarity=1):
    target = patch(image, (x, y, motif.width, motif.height))
    split = motif.width//2
    for columns in (slice(None, split), slice(split, None)):
        signal, weights, pixels = (a[:, columns] for a in (motif.signal, motif.weights, target))
        ref = float(np.sum(signal*signal*weights))
        energy = max(float(np.sum(pixels*pixels*weights)), 4*float(weights.sum()))
        if ref < 120 or polarity*float(np.sum(signal*pixels*weights)) < .36*np.sqrt(ref*energy):
            return False
    return True


def absence_probes(intervals, scores, *, secondary=False):
    probes = []
    for run in intervals:
        middle = int(run[len(run)//2])
        peak = max(map(int, run), key=lambda i: (scores[i], -abs(i-middle)))
        probes.append(peak)
        if secondary:
            probes.append(int(run[0] if peak >= middle else run[-1]))
        elif len(intervals) == 1 and len(run) >= 12:
            fraction = 1 if peak >= middle else 2
            probes.append(int(run[fraction*len(run)//3]))
    return probes


def verify_presence(video, track, cuda, trace, *, confirm_absence=None):
    """Resolve weak matches at recurring locations and verify low-response intervals."""
    good = track['verified']
    missing = np.flatnonzero(~good)
    if not len(missing) or len(missing) > .20*video.count or np.median(track['scores'][good]) < .72:
        return
    points = track['positions'][good]
    unique, counts = np.unique(points, axis=0, return_counts=True)
    major = np.argsort(-counts)[:4]
    positions = unique[major[counts[major] >= max(12, .04*len(points))]]
    near = np.abs(points[:, None]-positions[None]).max(axis=2) <= 2
    if near.any(axis=1).mean() >= .93:
        uncertain = missing[track['scores'][missing] >= .42]
        motif = Motif(track['signal'], track['weights'], track['mask'])
        recovered = 0
        if len(uncertain):
            for i, raw in read_frames(video, int(uncertain[0]), int(uncertain[-1])+1, cuda):
                if good[i] or track['scores'][i] < .42:
                    continue
                image = highpass(video.planes(raw)[0], 2.2)
                ranks = sorted((motif.local(image, tuple(map(int, pos)), 2, 2) for pos in positions),
                               key=lambda row: -row[2])
                x, y, score = ranks[0]
                other = max((s for a, b, s in ranks[1:] if abs(x-a)+abs(y-b) > 4), default=-1.)
                if score >= .52 and score-other >= .16 and score+.05 >= track['scores'][i] and agrees_in_halves(motif, image, x, y):
                    track['positions'][i], track['scores'][i], good[i] = (x, y), score, True
                    recovered += 1
        trace('discovery.recurring_positions', locations=positions.tolist(), recovered=recovered)
    missing = np.flatnonzero(~good)
    intervals = [run for run in np.split(missing, np.flatnonzero(np.diff(missing)>1)+1)
                 if len(run) >= 4 and np.max(track['scores'][run]) < .38]
    if intervals and len(intervals) <= 40 and confirm_absence is not None:
        probes = absence_probes(intervals, track['scores'])
        if confirm_absence(probes):
            present = track.setdefault('present', np.ones(video.count, bool))
            for run in intervals:
                good[run] = True
                present[run] = False
            trace('discovery.unmarked_intervals', ranges=[[int(run[0]), int(run[-1])+1] for run in intervals],
                  count=sum(map(len, intervals)), maximum_score=.38, probes=probes)


def recall(video, frames, library, source_digest, cuda, trace):
    """Propose an approved motif using only current pixels, with no prior coordinates."""
    matches = []
    signals = {}
    for path, motif in existing_motifs(library, video, source_digest):
        if not signals:
            signals = observe(video, [(i, None) for i in frames], cuda)
        validation = [motif.locate(signals[i])[2:] for i in frames]
        strong = all(strong_match(score, other) for score, other in validation)
        trace('discovery.memory.screened', prototype=str(path), frames=frames,
              validation=validation, strong=strong)
        if strong:
            matches.append((path, motif, validation))
    if len(matches) != 1:
        return None
    path, motif, validation = matches[0]
    floor = max(.61, min(score for score, _ in validation)*.82)
    track = scan_source(video, motif, floor, cuda, margin=.12)
    trace('discovery.memory.checked', prototype=str(path), verified=int(track['verified'].sum()),
          threshold=floor, margin=.12)
    return (track, str(path)) if track['verified'].all() else None


def enroll(video, observations, library, source_digest, cuda, trace, *, tolerate_outlier=False):
    signals = observe(video, observations, cuda)
    for prior, motif in existing_motifs(library, video, source_digest):
        try:
            floor, scores, distractors, verified = evidence_threshold(motif, observations, signals)
        except (ValueError, cv2.error):
            continue
        track = scan_source(video, motif, floor, cuda, fast=tolerate_outlier)
        trace('discovery.memory.checked', prototype=str(prior), verified=int(track['verified'].sum()),
              validation=scores, distractors=distractors, observations=verified)
        if track['verified'].all():
            return track, str(prior)
    groups = [(None, observations)]
    if tolerate_outlier and len(observations) >= 4:
        groups.extend((row[0], observations[:j]+observations[j+1:]) for j, row in enumerate(observations))
    for dropped, group in groups:
        try:
            motif = learned_motif(group, signals)
            floor, scores, distractors, verified = evidence_threshold(motif, group, signals)
            if dropped is not None and sum(row['corroborates_visual_box'] for row in verified) < .75*len(group):
                raise ValueError('A reduced sample group still disagrees on the watermark location.')
        except ValueError as error:
            if dropped is None:
                failure = error
            continue
        trace('discovery.learned', training=[i for i, _ in group[:-1]], held_out=group[-1][0],
              validation=scores, distractors=distractors, observations=verified, threshold=floor,
              mask_pixels=int(motif.mask.sum()), dropped_observation=dropped)
        return scan_source(video, motif, floor, cuda, fast=tolerate_outlier), None
    raise failure


def import_discovery(video, profile, profile_path, output, cuda=True):
    location = Path(profile['discovery']['track'])
    if digest(location) != profile['discovery']['sha256']:
        raise ValueError('The discovered track changed after visual planning.')
    source_digest = digest(video.path)
    with np.load(location, allow_pickle=False) as saved:
        track = dict(saved)
    if str(track['source_sha256']) != source_digest:
        raise ValueError('The discovered watermark belongs to a different source.')
    layers = discovery_tracks(track)
    review_only = profile['discovery'].get('coverage_review_only') is True
    if len(layers) != profile['discovery'].get('layer_count', 1):
        raise ValueError('The discovery profile and saved track disagree about the number of layers.')
    for layer in layers:
        mask = layer['mask']
        if (layer['positions'].shape != (video.count, 2) or layer['positions'].dtype != np.int32
                or layer['scores'].shape != (video.count,) or not np.isfinite(layer['scores']).all()
                or layer['verified'].shape != (video.count,) or layer['verified'].dtype != np.bool_
                or (not layer['verified'].all() and not review_only) or mask.ndim != 2 or mask.shape[0] > video.height
                or mask.shape[1] > video.width):
            raise ValueError('A discovered watermark layer has invalid geometry or unverified frames.')
        if 'present' in layer and (layer['present'].shape != (video.count,) or layer['present'].dtype != np.bool_
                                  or not layer['present'].any()):
            raise ValueError('The watermark visibility record is invalid.')
        if review_only and ('present' not in layer or np.any(layer['present'] & ~layer['verified'])):
            raise ValueError('A review candidate can edit only independently verified placements.')
        motif = Motif(layer['signal'], layer['weights'], mask)
        x, y, w, h = motif.bounds
        points = layer['positions'][layer['present']] if 'present' in layer else layer['positions']
        left, top = points.T.astype(np.int64)
        if np.any((left+x >= video.width)|(top+y >= video.height)|(left+x+w <= 0)|(top+y+h <= 0)):
            raise ValueError('A verified watermark is outside the image.')
    if luma_extent_groups(track):
        from wm_discovery_render import refresh_luma_visibility
        refresh_luma_visibility(video, track, cuda)
    save_npz(output, **track, profile_sha256=digest(profile_path))
    print(f'Imported {len(layers)} watermark tracks covering {video.count} frames.', flush=True)


def remember(library, attempt):
    with np.load(attempt/'track.npz', allow_pickle=False) as saved:
        track = dict(saved)
    for layer in discovery_tracks(track):
        content = {name: layer[name] for name in ('signal', 'weights', 'mask')}
        key = sha256()
        for samples in content.values():
            key.update(repr(samples.shape).encode('ascii'))
            key.update(samples.tobytes())
        path = library/(key.hexdigest()+'.npz')
        try:
            save_npz(path, **content, source_sha256=track['source_sha256'])
        except FileExistsError:
            yield path, False
        else:
            yield path, True
