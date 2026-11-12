"""Build a repair recipe from near-black source frames and a verified animation layout."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from wm_animation_audit import signature
from wm_media import Video, get_frame, plane_rects, planes, read_frames, yuv_to_rgb
from wm_motion_paths import paired_paths
from wm_opening import brand_colors
from wm_track import Tracker, crop, highpass, template_peak
from workflow import read_object, write_object as write_report


def padded_rect(bounds, video, pad):
    left, top, right, bottom = bounds
    x, y = max(0, (left-pad)//2*2), max(0, (top-pad)//2*2)
    r, b = min(video.width, (right+pad+1)//2*2), min(video.height, (bottom+pad+1)//2*2)
    return [int(x), int(y), int(r-x), int(b-y)]


def dark_candidates(video, cuda):
    factor = min(1, 320/max(video.width, video.height))
    size = tuple(max(2, round(value*factor/2)*2) for value in (video.width, video.height))
    candidates = []
    for i, frame in read_frames(video, cuda=cuda, size=size, step=4):
        y = planes(frame, *size)[0]
        middle = y[size[1]//7:]
        dark = float(np.mean(middle <= 22))
        if dark >= .94:
            candidates.append((dark, i))
    selected = []
    for _, index in sorted(candidates, reverse=True):
        if all(abs(index-previous) >= 12 for previous in selected):
            selected.append(index)
        if len(selected) == 20:
            break
    print(f'Near-black candidates from source: {selected}', flush=True)
    return selected


def badge_geometry(parts, video):
    y, u, v = parts
    height, width = video.height, video.width
    cap, half = height//6, width//2
    color = cv2.resize(v.astype(np.float32)-u.astype(np.float32), (width, height))
    sides = []
    for side in (0, 1):
        start, stop = side*half, (side+1)*half
        hue = (color[:cap, start:stop] > 35) & (y[:cap, start:stop] > 80)
        blobs = [row for row in cv2.connectedComponentsWithStats(hue.astype(np.uint8))[2][1:]
                 if row[3] >= 18 and .45 <= row[2]/row[3] <= 1.65 and row[1] < height*.07]
        if not blobs:
            continue
        x, top, w, h, area = max(blobs, key=lambda row: row[4])
        if area < max(80, width*height*.00065) or min(w, h) < 15:
            continue
        features = padded_rect((start+x, top, start+x+w, top+h), video, 8)
        keep = min(cap, top+h+max(18, h//3))
        rr, cc = np.where(y[:keep, start:stop] > 39)
        if len(cc):
            bounds = (start+int(cc.min()), int(rr.min()), start+int(cc.max())+1, int(rr.max())+1)
            sides.append((side, features, bounds))
    return sides


def white_shapes(parts, video, badges):
    y, u, v = parts
    color = cv2.resize(np.abs(u.astype(np.float32)-128)+np.abs(v.astype(np.float32)-128),
                       (video.width, video.height))
    mask = ((y > 34) & (color < 14)).astype(np.uint8)
    for _, _, bounds in badges:
        x, t, w, h = padded_rect(bounds, video, 12)
        mask[t:t+h, x:x+w] = 0
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (29, 23)))
    components = cv2.connectedComponentsWithStats(support)[2][1:]
    shapes = []
    for x, y0, w, h, count in sorted(components, key=lambda row: -row[4]):
        if count < video.width*video.height*.004:
            break
        rows, cols = np.where(mask[y0:y0+h, x:x+w])
        if len(cols) < 400:
            continue
        width, height = int(np.ptp(cols))+1, int(np.ptp(rows))+1
        if not (video.width*.14 <= width <= video.width*.65 and 1.5 <= width/height <= 5):
            continue
        slope = float(np.cov(cols, rows)[0, 1] / max(1, np.var(cols)))
        rect = (int(x+cols.min()), int(y0+rows.min()), int(x+cols.max()+1), int(y0+rows.max()+1))
        shapes.append((rect, slope))
    return shapes


def corner_config(video, anchors):
    out = {'references': [], 'search_rects': [], 'regions': [], 'masks': [], 'minimum_score': .52,
           'chroma_template': True}
    for _, (index, feature, bounds) in sorted(anchors.items()):
        x, y, w, h = padded_rect(bounds, video, 8)
        out['references'].append({'frame': index, 'rect': feature})
        a, b, c, d = feature
        out['search_rects'].append(padded_rect((a, b, a+c, b+d), video, 12))
        out['regions'].append(padded_rect((x, y, x+w, y+h), video, 66))
        out['masks'].append([[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]])
    return out


def equal_plates(bounds, video):
    padded = [padded_rect(rect, video, 10) for rect in bounds]
    width = max(rect[2] for rect in padded)
    height = max(rect[3] for rect in padded)
    if any(x+width > video.width or y+height > video.height for x, y, _, _ in padded):
        raise ValueError('Dark references do not fit the same template size.')
    return [[x, y, width, height] for x, y, _, _ in padded]


def colored_plates(video, anchors, reference, neutral_rects, cuda):
    pictures = [video.planes(get_frame(video, reference[side][0], cuda)) for side in (0, 1)]
    has_color = []
    for parts, rect in zip(pictures, neutral_rects, strict=True):
        y, u, v = plane_rects(parts, rect)
        chroma = cv2.resize(np.abs(u.astype(np.float32)-128)+np.abs(v.astype(np.float32)-128), y.shape[::-1])
        bright = y > 34
        has_color.append(np.count_nonzero(bright & (chroma > 14)) > .12*np.count_nonzero(bright))
    if not any(has_color):
        return None
    fixed = [padded_rect(anchors[side][2], video, 4) for side in (0, 1)]
    references = []
    for target, (x, y, w, h) in zip(pictures, neutral_rects, strict=True):
        rect = padded_rect((x-w//2-24, y-h//2, x+w+38, y+h+h//2), video, 0)
        search = crop(target[0], rect).astype(np.float32)
        detail = highpass(search, 3)
        matches = []
        for side, (parts, opaque_rect) in enumerate(zip(pictures, fixed, strict=True)):
            patch = crop(parts[0], opaque_rect).astype(np.float32)
            a, b, score = template_peak(detail, highpass(patch, 3))
            if score < .88:
                continue
            width, height = opaque_rect[2:]
            observed = cv2.getRectSubPix(search, (width, height), (a+(width-1)/2, b+(height-1)/2))-16
            signal = patch-16
            strong = signal > 50
            gain = float(np.median(observed[strong]/signal[strong]))
            error = float(np.median(np.abs(observed[strong]-gain*signal[strong])))
            if .15 <= gain <= .65 and error <= 4:
                matches.append((score, side, a+rect[0], b+rect[1], gain, error))
        if not matches:
            raise ValueError('The colored moving plate has no verified opaque copy of the same artwork.')
        references.append(max(matches))
    bounds = []
    for _, side, x, y, _, _ in references:
        w, h = fixed[side][2:]
        bounds.append((int(np.floor(x)), int(np.floor(y)), int(np.ceil(x+w)), int(np.ceil(y+h))))
    rectangles = equal_plates(bounds, video)
    plates = []
    for i, (rect, (score, side, x, y, gain, error)) in enumerate(zip(rectangles, references, strict=True)):
        plate = {'frame': reference[i][0], 'rect': rect, 'max_alpha': min(.65, gain+.03),
                 'opacity_reference': {'frame': anchors[side][0], 'rect': fixed[side],
                                       'offset': [round(x-rect[0], 4), round(y-rect[1], 4)],
                                       'gain': round(gain, 6), 'correlation': round(score, 5),
                                       'median_luma_error': round(error, 4)}}
        print(f'Verified colored plate {plate["frame"]}, opacity {gain:.4f}, correlation {score:.4f}, '
              f'median luma error {error:.3f}.', flush=True)
        plates.append(plate)
    return plates


def animation_config(video, layout, cuda):
    if video.fps != 30 or abs(video.width/video.height - 1080/1922) > .012:
        raise ValueError('This animation layout requires a matching 30 fps portrait source.')

    def scale(rect):
        x, y, w, h = rect
        left = round(x*video.width/1080/2)*2
        top = round(y*video.height/1922/2)*2
        right = round((x+w)*video.width/1080/2)*2
        bottom = round((y+h)*video.height/1922/2)*2
        return [left, top, right-left, bottom-top]

    config = {k: v for k, v in layout.items() if k not in ('signature_starts', 'jobs', 'extra_cuts')}
    for key in ('region', 'mask_region'):
        config[key] = scale(config[key])
    config['age_regions'] = [[lo, hi, *scale(rect)] for lo, hi, *rect in config['age_regions']]
    ages = [36, 44, 56, 62, 66]
    starts = list(range(0, video.count-ages[-1], 1800))
    if len(starts) < 3 or starts[2]+config['template_frames'] > video.count:
        raise ValueError('The repeated animation needs at least three full occurrences.')
    region = config['mask_region']
    samples = signature(video, starts[:2], ages, region, cuda)
    for start in starts:
        error = []
        for age, (coords, values) in zip(ages, samples, strict=True):
            patch = plane_rects(video.planes(get_frame(video, start+age, cuda)), region)
            error.extend(float(np.abs(p[at]-want).mean()) for p, at, want in zip(patch, coords, values, strict=True))
        distance = float(np.mean(error))
        print(f'Animation anchor {start}, mean color distance {distance:.3f}.', flush=True)
        if distance > 6:
            raise ValueError('The minute animation does not match every expected source anchor.')
    config['signature_starts'] = [i for i in starts if i+config['template_frames'] <= video.count][:6]
    config['jobs'] = {f'frame-{i}': {'repeat_start': i, 'context': [i+18, min(i+138, video.count)],
                                    'active': [i+26, min(i+120, video.count)]} for i in starts}
    return config, {'name': 'minute-animation', 'starts': starts[:2], 'ages': ages, 'region': region,
                    'occurrences': len(starts)}


def minute_color_scan(video, cuda):
    period, step = 1800, 4
    scale = min(1, 160/video.width, 280/video.height)
    width, height = (max(2, round(length*scale/2)*2) for length in (video.width, video.height))
    top, rows = height//4, height//2
    votes = np.zeros((period//step, 2, rows, width), np.uint16)
    for index, frame in read_frames(video, cuda=cuda, size=(width, height), step=step):
        rgb = yuv_to_rgb(planes(frame, width, height), video.matrix)[top:top+rows]
        cyan, yellow, _ = brand_colors(rgb)
        age = (index % period)//step
        votes[age, 0] += cyan
        votes[age, 1] += yellow
    maximum = int(np.count_nonzero(votes >= 3, axis=(2, 3)).max())
    starts = list(range(72, video.count, period))
    samples = [starts[i] for i in np.linspace(0, len(starts)-1, min(len(starts), 6), dtype=int)]
    return {'step': step, 'period': period, 'minimum_periods': 3, 'size': [width, height],
            'max_common_pixels': maximum, 'limit': 8, 'review_frames': samples}


def portrait_recipe(video, directory, layout, cuda=True, *, allow_missing_animation=False):
    anchors, reference = {}, {}
    for i in dark_candidates(video, cuda):
        parts = video.planes(get_frame(video, i, cuda))
        badges = badge_geometry(parts, video)
        shapes = white_shapes(parts, video, badges)
        if len(badges) == len(shapes) == 1 and abs(shapes[0][1]) < .18:
            side, feature, bounds = badges[0]
            if side not in anchors:
                anchors[side] = i, feature, bounds
                reference[side] = i, shapes[0][0]
    if sorted(anchors) != [0, 1]:
        raise ValueError('No clean left and right dark plates were found.')
    first, other = (reference[i][0] for i in (0, 1))
    rect, alternate = equal_plates([reference[i][1] for i in (0, 1)], video)
    tracking = rect
    plates = colored_plates(video, anchors, reference, (rect, alternate), cuda)
    match = [6, 4, (rect[2]-16)//2*2, (rect[3]*3//4)//2*2]
    if plates:
        for plate, inner in zip(plates, (rect, alternate), strict=True):
            plate['tracking_rect'] = inner
        rect = plates[0]['rect']
    else:
        plates = [{'frame': i, 'rect': region, 'max_alpha': .25} for i, region in ((first, rect), (other, alternate))]
    corners = corner_config(video, anchors)
    tracker = Tracker(video, {'corners': corners}, cuda)
    interval = round(30*float(video.fps))
    starts = [0]
    for turn, center in enumerate(range(interval, video.count, interval), 1):
        state = turn % 2
        samples = [(i, tracker.locate_corners(*video.planes(raw)))
                   for i, raw in read_frames(video, center-6, min(center+8, video.count), cuda)]
        accepted = [i for i, scores in samples if scores[state] >= .55 and scores[state]-scores[1-state] > .24]
        if not accepted or accepted[0] <= center-6:
            raise ValueError(f'Corner switch could not be located near frame {center}.')
        starts.append(accepted[0])
    alternate_ranges = []
    for slot, (start, end) in enumerate(zip(starts, [*starts[1:], video.count], strict=True)):
        for i in (start, (start+end)//2, end-1):
            scores = tracker.locate_corners(*video.planes(get_frame(video, i, cuda)))
            if scores[slot % 2] < .54 or scores[slot % 2]-scores[1-slot % 2] < .24:
                raise ValueError(f'Corner switch pattern is unverified at frame {i}.')
        if slot % 2:
            alternate_ranges.append([start, end])
    print(f'Corner changes verified at frames {starts[1:]}', flush=True)
    moving = {'reference_frame': first, 'reference_rect': rect,
              'match_rect': match,
              'minimum_score': .25, 'local_score': .51, 'downsample': 4, 'bridge_gaps': 8,
              'continuous_motion': {'max_gap': 60, 'max_step': 6}, 'phase_step': .25,
              'white_plate': {**plates[0], 'variants': [{'ranges': alternate_ranges, 'plate': plates[1]}]}}
    colored = 'opacity_reference' in plates[0]
    if colored:
        moving['tracking_rect'] = tracking
    profile = {'size': [video.width, video.height], 'cut_score': 23, 'corners': corners, 'translucent': moving}
    recipe = {'profile': 'profile.json', 'shared_render': ['--model-scale', '0.62', '--core', '64', '--overlap', '7',
                                                          '--dilation', '10'],
              'final_render': ['--edges', 'white-flat' if colored else 'off', '--align-white',
                               '--recover-corner-overlap', '--cq', '16']}
    try:
        animation, check = animation_config(video, layout, cuda)
    except ValueError as error:
        if not allow_missing_animation or video.fps != 30 or video.count < 5400:
            raise
        evidence = minute_color_scan(video, cuda)
        print(f'Minute color scan: {evidence}', flush=True)
        if evidence['max_common_pixels'] >= evidence['limit']:
            raise ValueError(f'{error} Unresolved periodic color evidence remains.') from error
        profile['minute_color_scan'] = evidence
        detected = 'portrait-switching'
    else:
        write_report(directory/'animation.json', animation)
        recipe.update(patches=[
            {'command': 'animation-template', 'config': 'animation.json', 'output': 'patches/template.npz'},
            {'command': 'animation-patch', 'config': 'animation.json', 'jobs': 'all',
             'output': 'patches/animation-{job}.npz', 'args': ['--template', '{work}/patches/template.npz'],
             'attach': 'animation'}], animation_checks=[check])
        detected = 'portrait-minute'
    write_report(directory/'profile.json', profile)
    write_report(directory/'recipe.json', recipe)
    print(f'Created {detected} recipe from source plates {first} and {other}.', flush=True)
    return detected


def paired_recipe(video, directory, cuda=True):
    for i in dark_candidates(video, cuda):
        parts = video.planes(get_frame(video, i, cuda))
        badges = badge_geometry(parts, video)
        shapes = sorted(white_shapes(parts, video, badges), key=lambda row: row[1])
        full = [(bounds, slope) for bounds, slope in shapes
                if 14 <= bounds[0] < bounds[2] <= video.width-14
                and 14 <= bounds[1] < bounds[3] <= video.height-14]
        if len(badges) == 1 and len(full) == 2 and full[0][1] < -.16 and full[1][1] > .16:
            break
    else:
        raise ValueError('No dark source frame contains both unobstructed slanted marks.')
    rects = [padded_rect(bounds, video, 10) for bounds, _ in full]
    print(f'Dark frame {i}, angle slopes {[round(s, 3) for _, s in full]}, template rectangles {rects}.', flush=True)
    paths, gate = paired_paths(video, rects, parts[0], cuda)
    moving = [{'reference_frame': i, 'reference_rect': rect, 'motion_path': path, 'phase_step': .125,
               'white_plate': {'frame': i, 'rect': rect, 'max_alpha': .52}}
              for rect, path in zip(rects, paths, strict=True)]
    moving[0].update(verify_paired_motion=True, motion_floor=gate)
    corners = corner_config(video, {side: (i, feature, bounds) for side, feature, bounds in badges})
    tracker = Tracker(video, {'corners': corners}, cuda)
    for frame in np.linspace(0, video.count-1, 24, dtype=int):
        if tracker.locate_corners(*video.planes(get_frame(video, int(frame), cuda)))[0] < .52:
            raise ValueError(f'The single opaque badge is unverified at frame {frame}.')
    write_report(directory/'profile.json', {'size': [video.width, video.height], 'cut_score': 23,
                 'corners': corners, 'translucent': moving[0], 'extra_white': moving[1:]})
    write_report(directory/'recipe.json', {'profile': 'profile.json',
                 'shared_render': ['--model-scale', '0.62', '--core', '64', '--overlap', '7', '--dilation', '10'],
                 'final_render': ['--edges', 'white-flat', '--align-white', '--cq', '16']})
    print(f'Created dual motion recipe from source frame {i}.', flush=True)


def detect_known_layout(video, directory, cuda=True):
    if video.width < video.height:
        layout = Path(__file__).parent/'profiles'/'portrait-minute-layout.json'
        return portrait_recipe(video, directory, read_object(layout), cuda, allow_missing_animation=True)
    paired_recipe(video, directory, cuda)
    return 'paired-motion'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('directory', type=Path)
    layout = parser.add_mutually_exclusive_group(required=True)
    layout.add_argument('--animation-layout', type=Path)
    layout.add_argument('--paired-motion', action='store_true')
    parser.add_argument('--cpu-decode', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    video = Video.open(args.source)
    args.directory.mkdir(parents=True)
    if args.paired_motion:
        paired_recipe(video, args.directory, not args.cpu_decode)
    else:
        portrait_recipe(video, args.directory, read_object(args.animation_layout), not args.cpu_decode)


if __name__ == '__main__':
    main()
