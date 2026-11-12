"""Prepare a verified inverse-compositing recipe from whole-frame watermark detections."""

import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from wm_director import job_trace, write_discovery_recipe
from wm_discovery import (Motif, chroma_position, chroma_templates, evidence_threshold, learned_motif,
                          observe, patch, scan_source)
from wm_matte import affine_between, exclusion, fit_fields, sampling_grid, training_pair, warp_fields
from wm_media import Video, digest, get_frame, plane_rects, read_frames
from wm_track import highpass
from workflow import read_object, write_object


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))


def area(rect):
    return max(0., rect[2]-rect[0])*max(0., rect[3]-rect[1])


def region(box, video):
    a, b, c, d = box
    a, b = max(0, int(np.floor(a))), max(0, int(np.floor(b)))
    c, d = min(video.width, int(np.ceil(c))), min(video.height, int(np.ceil(d)))
    return (a, b, c-a, d-b)


def selected_anchors(video, candidates):
    anchors = []
    for block in np.array_split(np.arange(video.count), min(8, video.count)):
        choices = [(i, candidates[i][0]) for i in block if candidates[i]]
        if not choices:
            continue
        i, choice = max(choices, key=lambda pair: pair[1]['confidence'])
        box = region(choice['rect'], video)
        if min(box[2:]) >= 8 and box[2]*box[3] <= .16*video.width*video.height:
            anchors.append((int(i), box))
    return anchors


def detector_supported_track(video, candidates, cuda, trace):
    anchors = selected_anchors(video, candidates)
    if len(anchors) < 4:
        raise ValueError('At least four separated observations are needed to fit a moving watermark.')
    signals = observe(video, anchors, cuda)
    motif = learned_motif(anchors, signals)
    floor, scores, distractors, checks = evidence_threshold(motif, anchors, signals)
    agreement = sum(row['corroborates_visual_box'] for row in checks)
    trace('reference.anchors_checked', agreement=agreement, total=len(checks), threshold=floor,
          scores=scores, distractors=distractors, observations=checks)
    if agreement < .75*len(checks):
        raise ValueError(f'Detector boxes and the pixel pattern disagree on the target in {len(checks)-agreement}/{len(checks)} samples.')
    track = scan_source(video, motif, floor, cuda)
    a, b, w, h = motif.bounds
    offsets = []
    for i in np.flatnonzero(track['verified']):
        x, y = track['positions'][i]
        outer = np.array([x+a, y+b, x+a+w, y+b+h])
        matches = [row for row in candidates[i] if area(row['rect']) > .12*w*h
                   and intersection(row['rect'], outer) >= .72*area(row['rect'])]
        if matches:
            offsets.append(np.asarray(matches[0]['rect'])-[x, y, x, y])
    if len(offsets) < 8:
        raise ValueError('The detector and repeated pixel signature lack a shared position.')
    footprint = np.median(offsets, axis=0)
    previous_count = int(track['verified'].sum())
    for i in np.flatnonzero(~track['verified']):
        if not candidates[i]:
            continue
        signal = highpass(video.planes(get_frame(video, int(i), cuda))[0], 2.2)
        possibilities = []
        for proposal in candidates[i]:
            box = np.asarray(proposal['rect'])
            origin = np.rint(((box[:2]+box[2:])-(footprint[:2]+footprint[2:]))/2).astype(int)
            x, y, score = motif.local(signal, origin, 6, 6)
            predicted = footprint+[x, y, x, y]
            overlap = intersection(box, predicted)
            union = area(box)+area(predicted)-overlap
            if score >= .42 and overlap >= .5*union:
                possibilities.append((score, x, y))
        possibilities.sort(reverse=True)
        if not possibilities:
            continue
        score, x, y = possibilities[0]
        if any(s > score-.08 and abs(x-u)+abs(y-v) > 6 for s, u, v in possibilities[1:]):
            continue
        track['positions'][i] = x, y
        track['scores'][i] = score
        track['verified'][i] = True
    trace('reference.detector_agreement', anchors=[i for i, _ in anchors], matched_boxes=len(offsets),
          pixel_only=previous_count, verified=int(track['verified'].sum()), total=video.count)
    if int(track['verified'].sum()) < .6*video.count:
        raise ValueError(f'Only {int(track["verified"].sum())}/{video.count} frames have verified watermark locations.')
    return track, footprint


def calibration_pairs(video, profile, plate, cuts, verified, cuda):
    positions = plate['precise']
    width, height = profile['translucent']['reference_rect'][2:]
    displacement = np.abs(np.diff(positions, axis=0)).max(axis=1)
    jumps = [int(i) for i in np.flatnonzero((displacement > .24*width)&verified[1:]&verified[:-1])+1
             if i not in cuts]
    if len(jumps) > 24:
        jumps = [jumps[i] for i in np.linspace(0, len(jumps)-1, 24, dtype=int)]
    edges = np.r_[0, cuts, video.count]
    rows = []
    for jump in jumps:
        shot = np.searchsorted(edges, jump, side='right')-1
        lo, hi = max(int(edges[shot]), jump-8), min(int(edges[shot+1]), jump+9)
        frames = dict(read_frames(video, lo, hi, cuda))
        for offset in (-4, -3, -2, -1, 0, 1, 2, 3):
            i, j = jump+offset, jump if offset < 0 else jump-1
            x, y = plate['positions'][i] if i in frames else (-1, -1)
            if j not in frames or not (0 <= x <= video.width-width and 0 <= y <= video.height-height) or not verified[i]:
                continue
            row = training_pair(video, profile, plate, i, j, frames[i], frames[j], 1/(1+.18*abs(i-j)))
            if row is not None and np.count_nonzero(row[2][0]) > .35*width*height:
                margin = np.ones((height, width), bool)
                margin[4:-4, 4:-4] = False
                good = margin&(row[2][0] > 0)
                delta = np.abs(row[0][0].astype(np.float32)-row[1][0])
                if good.sum() > 32 and np.quantile(delta[good], .7) <= 4:
                    rows.append((i, j, row))
    if len(rows) < 12 or len({i for i, _, _ in rows}) < 10:
        raise ValueError('Too few usable same-shot references where the watermark moved away.')
    return rows


def measured_support(rows, phases, width, height):
    evidence = []
    for (source, clean, weight), (dx, dy) in zip(rows, phases, strict=True):
        channels = []
        for j, (a, b, w) in enumerate(zip(source, clean, weight, strict=True)):
            delta = np.abs(a.astype(np.float32)-b)
            delta[w <= 0] = 0
            if j:
                delta = cv2.resize(delta, (width, height), interpolation=cv2.INTER_LINEAR)
            channels.append(delta if not j else delta*1.35)
        aligned = cv2.warpAffine(np.maximum.reduce(channels), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                                (width, height))
        evidence.append(aligned)
    mask = (np.median(evidence, axis=0) >= 3.4).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    mask = cv2.dilate(keep[labels].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    if np.count_nonzero(mask) < 40:
        raise ValueError('The same-shot references did not isolate a repeatable transparent layer.')
    return mask


@torch.inference_mode()
def holdout_error(rows, phases, fields, support):
    device = torch.device('cuda')
    height, width = support.shape
    shifts = torch.from_numpy(phases).to(device)
    before, after = 0., 0.
    for j, scale in enumerate((1, 2, 2)):
        src, ref, confidence = [torch.from_numpy(np.stack([row[k][j] for row in rows])).to(device).float()
                                for k in range(3)]
        grid = sampling_grid(shifts, width, height, scale)
        mask = warp_fields(torch.from_numpy(support[None]).to(device).float(), grid)[:, 0] > .45
        d, k = warp_fields(torch.from_numpy(fields[j]).to(device), grid).unbind(1)
        prediction = ((src-128*(1+k))/(1+d)+128).clamp(0, 255)
        weight = mask*confidence
        total = weight.sum().clamp_min(1)
        before += float(((src-ref).abs()*weight).sum()/total)
        after += float(((prediction-ref).abs()*weight).sum()/total)
    return before/3, after/3


def aligned_planes(planes, matrix):
    height, width = planes[0].shape
    result = []
    for plane, scale in zip(planes, (1, 2, 2), strict=True):
        transform = matrix.copy()
        transform[:, 2] /= scale
        result.append(cv2.warpAffine(plane, transform, (width//scale, height//scale)))
    return result


def full_color(planes):
    h, w = planes[0].shape
    return np.stack([planes[0], *(cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in planes[1:])])


def patch_agreement(marked, clean, d, k, inside, outside):
    scale = np.asarray([1., .7, .7], np.float32)[:, None]
    delta = (marked[:, inside]-clean[:, inside])*scale
    signature = (d[:, inside]*(clean[:, inside]-128)+128*k[:, inside])*scale
    size = float(np.linalg.norm(delta)*np.linalg.norm(signature))
    correlation = float(np.sum(delta*signature)/max(1., size))
    texture_error = float(np.mean(np.abs(marked[:, outside]-clean[:, outside])*scale))
    strength = float(np.mean(np.abs(delta)))
    inverse = np.clip((marked[:, inside]-128*(1+k[:, inside]))/(1+d[:, inside])+128, 0, 255)
    residual = float(np.mean(np.abs(inverse-clean[:, inside])*scale))
    return correlation, texture_error, strength, residual


def reference_position(video, profile, plate, frame, options, footprint, references, read_planes, support, fields, trace):
    if not options:
        return None
    height, width = support.shape
    foreground = support.astype(bool)
    ring = cv2.dilate(support, np.ones((7, 7), np.uint8)).astype(bool)&~foreground
    d, k = np.stack(fields).transpose(1, 0, 2, 3)
    src = full_color(read_planes(frame))
    ranked = []
    best = None
    for proposal in options:
        box = np.asarray(proposal['rect'])
        if not .52*area(footprint) <= area(box) <= 1.8*area(footprint):
            continue
        origin = np.rint(((box[:2]+box[2:])-(footprint[:2]+footprint[2:]))/2).astype(int)
        x, y = origin
        box_with_margin = (x-8, y-8, x+width+8, y+height+8)
        good_refs = []
        for i in references:
            u, v = plate['precise'][i]
            if not intersection(box_with_margin, (u-8, v-8, u+width+8, v+height+8)):
                good_refs.append(i)
        if len(good_refs) < 2:
            continue
        plate['positions'][frame] = origin//2*2
        target_good = exclusion(video, profile, plate, frame)
        refs = []
        for i in sorted(good_refs, key=lambda j: abs(frame-j))[:3]:
            ref = read_planes(i)
            ref_good = exclusion(video, profile, plate, i)
            matrix = affine_between(ref[0], src[0], ref_good, target_good)
            if np.max(np.abs(matrix[:, :2]-np.eye(2))) < .001 and np.max(np.abs(matrix[:, 2]-np.rint(matrix[:, 2]))) < .15:
                matrix = np.rint(matrix)
            planes = aligned_planes(ref, matrix)
            valid = cv2.warpAffine(ref_good, matrix, (video.width, video.height)) == 255
            refs.append((i, full_color(planes), valid, planes))
        for top in range(max(0, y-5), min(video.height-height, y+5)+1):
            for left in range(max(0, x-5), min(video.width-width, x+5)+1):
                sl = np.s_[top:top+height, left:left+width]
                marked = src[:, top:top+height, left:left+width].astype(np.float32)
                evidence = []
                for i, aligned, valid, planes in refs:
                    keep = valid[sl]
                    inside, outside = foreground&keep, ring&keep
                    if not keep[foreground].all() or outside.sum() < .8*ring.sum():
                        continue
                    clean = aligned[:, top:top+height, left:left+width]
                    corr, texture, before, after = patch_agreement(marked, clean, d, k, inside, outside)
                    rank = corr-max(0., texture-3)/10
                    entry = (rank, i, left, top, corr, texture, before, after)
                    if best is None or entry > best:
                        best = entry
                    matching_background = texture <= .9 and before >= 6*texture and corr >= .15
                    if before >= 2.5 and (matching_background or (corr >= .45 and texture <= 4)):
                        evidence.append((i, corr, texture, before, after, clean, planes))
                for index, first in enumerate(evidence):
                    for second in evidence[index+1:]:
                        disagreement = float(np.abs(first[5][:, foreground].astype(np.float32)
                                                    -second[5][:, foreground]).mean())
                        if disagreement <= 3.0:
                            ranked.append((min(first[1], second[1]), left, top, first, second, disagreement))
    if not ranked:
        trace('reference.frame_unconfirmed', frame=int(frame), best=best)
        return None
    score, x, y, first, second, disagreement = max(ranked, key=lambda row: row[0])
    if any(other >= score-.06 and abs(x-u)+abs(y-v) > 3 for other, u, v, *_ in ranked):
        trace('reference.frame_unconfirmed', frame=int(frame), best=best, reason='ambiguous-position')
        return None
    rect = x//2*2, y//2*2, width, height
    patches = [tuple(p.copy() for p in plane_rects(row[6], rect)) for row in (first, second)]
    replacement = [np.rint((a.astype(np.float32)+b)*.5).astype(np.uint8) for a, b in zip(*patches, strict=True)]
    trace('reference.frame_confirmed', frame=int(frame), position=[x, y],
          comparisons=[first[:5], second[:5]], donor_error=round(disagreement, 4))
    return (x, y), replacement


def confirm_unverified(video, profile, plate, track, candidates, footprint, support, fields, cuda, trace):
    cuts = np.r_[0, track['cuts'], video.count]
    verified = track['verified'].copy()
    replacements = {}

    @lru_cache(maxsize=28)
    def read_planes(i):
        return video.planes(get_frame(video, int(i), cuda))

    templates = None
    for i in np.flatnonzero(~verified):
        scene = np.searchsorted(cuts, i, side='right')-1
        lo, hi = max(int(cuts[scene]), i-24), min(int(cuts[scene+1]), i+25)
        references = [j for j in range(lo, hi) if verified[j]]
        result = reference_position(video, profile, plate, int(i), candidates[i], footprint, references,
                                    read_planes, support, fields, trace)
        if result is not None:
            position, replacements[int(i)] = result
        else:
            if templates is None:
                templates = chroma_templates(read_planes, verified, plate['precise'], support)
            match = chroma_position(read_planes(int(i)), templates)
            if match is None:
                continue
            position, scores = match
            trace('reference.chroma_confirmed', frame=int(i), position=list(position), matches=scores)
        plate['precise'][i] = position
        plate['positions'][i] = np.asarray(position)//2*2
        track['verified'][i] = True
    trace('reference.full_coverage', verified=int(track['verified'].sum()), total=video.count)
    if not track['verified'].all():
        raise ValueError(f'Only {int(track["verified"].sum())}/{video.count} frames have verified watermark locations.')
    return replacements


def prepare_reference(video, candidates, cuda, trace):
    track, footprint = detector_supported_track(video, candidates, cuda, trace)
    left, top = np.floor(footprint[:2]).astype(int)-5
    right, bottom = np.ceil(footprint[2:]).astype(int)+6
    width, height = (int(right-left+2)//2*2, int(bottom-top+2)//2*2)
    exact = track['positions']+np.asarray([left, top], np.int32)
    plate = dict(positions=(exact//2*2).astype(np.int32), precise=exact.astype(np.float32))
    profile = {'translucent': {'reference_rect': [0, 0, width, height]}}
    rows = calibration_pairs(video, profile, plate, track['cuts'], track['verified'], cuda)
    indices, _, data = zip(*rows, strict=True)
    phases = (plate['precise']-plate['positions'])[list(indices)].astype(np.float32)
    check = np.arange(len(rows))%5 == 2
    train = [row for row, test in zip(data, check, strict=True) if not test]
    test = [row for row, test in zip(data, check, strict=True) if test]
    support = measured_support(train, phases[~check], width, height)
    fields = fit_fields(train, phases[~check], support)
    before, after = holdout_error(test, phases[check], fields, support)
    trace('reference.holdout', pairs=len(rows), holdout=len(test), changed_pixels=int(support.sum()),
          average_error_before=round(before, 4), average_error_after=round(after, 4))
    if before < 3 or after > min(4.5, .48*before):
        raise ValueError(f'Inverse compositing did not pass independent references ({before:.2f} -> {after:.2f}).')
    replacements = confirm_unverified(video, profile, plate, track, candidates, footprint-[left, top, left, top],
                                      support, fields, cuda, trace)
    exact = plate['precise'].astype(np.int32)
    if (exact < 0).any():
        raise ValueError('An inverse-compositing plate extends beyond the supported frame origin.')
    signal = patch(track['signal'], (left, top, width, height))
    weights = patch(track['weights'], (left, top, width, height))*support
    motif = Motif(signal, weights, support)
    scores = []
    for i, raw in read_frames(video, cuda=cuda):
        _, _, score = motif.local(highpass(video.planes(raw)[0], 2.2), tuple(map(int, exact[i])), 0, 0)
        scores.append(score)
    track.update(signal=signal, weights=weights, mask=support, positions=exact, scores=np.asarray(scores, np.float32),
                 matte_positions=plate['positions'], matte_precise=plate['precise'],
                 **{f'fields{j}': field for j, field in enumerate(fields)})
    if replacements:
        track['reference_indices'] = np.asarray(list(replacements), np.int32)
        for j, planes in enumerate(zip(*replacements.values(), strict=True)):
            track[f'reference{j}'] = np.stack(planes)
    return track


def make_recipe(source, detections, directory, label, cuda):
    video = Video.open(source)
    video.check_timestamps()
    candidates = [row['detections'] for row in detections]
    if len(candidates) != video.count or any(row['frame'] != i for i, row in enumerate(detections)):
        raise ValueError('The detection report must cover each frame of the source in order.')
    directory.mkdir(parents=True, exist_ok=False)
    trace = job_trace(directory/'events.jsonl')
    trace('reference.started', source=str(video.path), size=[video.width, video.height], frames=video.count,
          frames_with_candidates=sum(bool(rows) for rows in candidates), cuda_decode=cuda)
    try:
        track = prepare_reference(video, candidates, cuda, trace)
    except ValueError as error:
        trace('reference.rejected', reason=str(error))
        raise
    plan = {}
    recipe = write_discovery_recipe(video, directory, digest(video.path), track, None, label, plan, trace)
    write_object(directory/'plan.json', plan)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('detections', type=Path)
    parser.add_argument('job', type=Path)
    parser.add_argument('--clip')
    parser.add_argument('--label')
    parser.add_argument('--cpu-decode', action='store_true')
    args = parser.parse_args()
    report = read_object(args.detections)
    if args.clip:
        report = next(row for row in report['clips'] if row['id'] == args.clip)
    if report.get('source_sha256') and report['source_sha256'] != digest(args.source):
        raise ValueError('The detections belong to a different source.')
    try:
        recipe = make_recipe(args.source.resolve(strict=True), report['rows'], args.job.resolve(),
                             args.label, not args.cpu_decode)
    except ValueError as error:
        parser.exit(1, f'Recipe not created: {error}\n')
    print(recipe, flush=True)


if __name__ == '__main__':
    main()
