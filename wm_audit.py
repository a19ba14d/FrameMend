import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from wm_matte import frame_variants, logo_rect, phase_offsets
from wm_media import Video, clipped_plane, read_frames, yuv_to_rgb
from wm_track import Tracker, corner_activity, crop, extra_tracks, highpass, load_profile, load_track
from wm_temporal import CapsuleRim


def quantiles(values):
    return {name: round(float(np.quantile(values, q)), 5)
            for name, q in (('median', .50), ('p95', .95), ('p99', .99), ('max', 1))}


class WhiteResponse:
    def __init__(self, alpha):
        self.signal = highpass(self.small(-np.log1p(-alpha)), 2.5)
        self.rows = []

    @staticmethod
    def small(image):
        return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_AREA)

    def sample(self, index, before, after, shift=None):
        signal = self.signal
        if shift is not None:
            dx, dy = shift / 2
            signal = cv2.warpAffine(signal, np.float32([[1, 0, dx], [0, 1, dy]]), signal.shape[::-1])
        before, after = [self.small(plane.astype(np.float32)) for plane in (before, after)]
        weight = (after < 210) / (1 + (highpass(after, 2.5)/10)**2)
        energy = float(np.sum(signal**2 * weight))
        if energy < .1:
            return
        score = []
        for image in (before, after):
            distance = -np.log(np.clip((235-image)/219, .05, 1.1))
            score.append(float(np.sum(highpass(distance, 2.5)*signal*weight)/energy))
        self.rows.append((index, *score))

    def report(self):
        if not self.rows:
            return {'sampled_frames': 0}
        samples = np.asarray(self.rows)
        before, after = np.median(samples[:, 1:], axis=0)
        return {'sampled_frames': len(samples), 'before_median': round(float(before), 5),
                'after_median': round(float(after), 5),
                'absolute_median_residual_ratio': round(float(abs(after) / max(.05, abs(before))), 5),
                'worst_frames': [{'frame': int(i), 'after_response': round(float(value), 5)}
                                 for i, _, value in sorted(self.rows, key=lambda item: -abs(item[2]))[:6]]}


def audit(source, output, profile_path, workdir, report_path, sheet_path=None, sample_frames=(), cuda=True):
    source, output = Video.open(source), Video.open(output)
    if (source.width, source.height, source.count, source.fps) != (output.width, output.height, output.count, output.fps):
        raise ValueError('Audit requires aligned input and output frames.')
    profile = load_profile(source, profile_path)
    directory = Path(workdir)
    if profile.get('discovery'):
        from wm_discovery_render import audit_discovery
        audit_discovery(source, output, profile_path, directory, Path(report_path), cuda)
        return
    track_file = directory/'track-recovered.npz'
    track = load_track(source, profile_path, track_file if track_file.is_file() else directory/'track.npz')
    tracker = Tracker(source, profile, cuda)
    logo = bool(profile.get('translucent'))
    white_logo = bool(profile.get('translucent', {}).get('white_plate'))
    fixed_layers = profile.get('stationary', [])
    fixed, white_moving, companions = [], [], []
    rim = None
    if logo or fixed_layers:
        with np.load(Path(workdir) / 'matte.npz', allow_pickle=False) as data:
            for j, layer in enumerate(fixed_layers):
                active = frame_variants(source.count, [layer['ranges']]) > 0 if 'ranges' in layer else np.ones(source.count, bool)
                fixed.append((layer['rect'], active, WhiteResponse(data[f'stationary{j}'])))
            if logo:
                fields = data['fields0']
                variants = data['variants'] if 'variants' in data else np.zeros(source.count, np.int16)
            for layer, points, key in extra_tracks(profile, track):
                companions.append((layer, points, phase_offsets(layer, points), WhiteResponse(-data[key+'fields0'][0])))
    if logo:
        if fields.ndim == 3:
            fields = fields[None]
        if white_logo:
            white_moving = [WhiteResponse(-field[0]) for field in fields]
        phases = phase_offsets(profile, track)
        margins = profile['translucent'].get('capsule_margins')
        if margins:
            rim = CapsuleRim(*profile['translucent']['reference_rect'][2:], margins)
    ratios, corners, luma_error, luma_square = [], [], [], []
    rim_errors = []
    rows = {}
    pairs = zip(read_frames(source, cuda=cuda), read_frames(output, cuda=cuda), strict=True)
    for (i, first), (_, second) in pairs:
        original = source.planes(first)
        restored = output.planes(second)
        before, after = original[0], restored[0]
        logo_region = logo_rect(profile, track, i) if logo else None
        in_frame = (logo_region and 0 <= logo_region[0] <= source.width-logo_region[2]
                    and 0 <= logo_region[1] <= source.height-logo_region[3])
        if in_frame:
            x, y, w, h = logo_region
            a, b = crop(before, logo_region), crop(after, logo_region)
            dx, dy = phases[i]
            if white_moving and i % 7 == 0:
                white_moving[variants[i]].sample(i, a, b, phases[i])
            gain, tint = fields[variants[i]]
            contrast = gain * (float(np.median(b)) - 128) + tint * 128
            shifted = cv2.warpAffine(contrast, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
            model = highpass(shifted, 4)
            raw = float(np.mean(highpass(a, 4) * model))
            clean = float(np.mean(highpass(b, 4) * model))
            if abs(raw) > 3:
                ratios.append((i, abs(clean / raw)))
            if rim:
                maps, band, fraction = rim.samples(float(dx), float(dy), 1)
                inner, outer = [cv2.remap(b, *maps[j], cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32) for j in (0, 3)]
                flat = (band > 0.9) & (np.abs(inner - outer) < 7)
                if flat.sum() >= 100:
                    prediction = inner + fraction * (outer - inner)
                    rim_errors.append(float(np.abs(b - prediction)[flat].mean()))
        if i % 7 == 0:
            for layer, points, offsets, response in companions:
                rect = logo_rect(layer, points, i)
                if rect and 0 <= rect[0] <= source.width-rect[2] and 0 <= rect[1] <= source.height-rect[3]:
                    response.sample(i, crop(before, rect), crop(after, rect), offsets[i])
        if fixed and i % 13 == 0:
            for rect, active, response in fixed:
                if active[i]:
                    response.sample(i, crop(before, rect), crop(after, rect))
        if profile.get('corners'):
            scores = tracker.locate_corners(*restored)
            active = corner_activity(profile['corners'], track['corners'][i])
            corners.extend(float(score) for flag, score in zip(active, scores, strict=True) if flag)
        if i > 150 and i % 180 == 0:
            known = np.ones_like(before, dtype=np.uint8)
            regions = list(profile.get('corners', {}).get('regions', []))
            regions.extend(rect for rect, active, _ in fixed if active[i])
            if logo_region:
                regions.append(logo_region)
            regions.extend(rect for layer, points, _, _ in companions if (rect := logo_rect(layer, points, i)))
            for region in regions:
                clipped_plane(known, region)[0][:] = 0
            delta = (before[known > 0].astype(np.float32) - after[known > 0])
            luma_error.append(float(np.abs(delta).mean()))
            luma_square.append(float(np.square(delta).mean()))
        if i in sample_frames:
            rows[i] = (yuv_to_rgb(original, source.matrix), yuv_to_rgb(restored, output.matrix))
        if (i+1) % 1200 == 0:
            print(f'Audited {i+1}/{source.count} frames', flush=True)

    report = {'frames_audited': source.count}
    if logo:
        report['translucent_tracked_frames'] = int(np.count_nonzero(track['positions'][:, 0] >= 0))
        if 'interpolated' in track:
            report['translucent_interpolated_frames'] = int(track['interpolated'].sum())
        if 'motion_recovered' in track:
            report['translucent_motion_recovered_frames'] = int(track['motion_recovered'].sum())
    if white_moving:
        report['white_moving_layers'] = [response.report() for response in [*white_moving, *(row[3] for row in companions)]]
    if companions:
        report['additional_white_tracked_frames'] = [int(np.count_nonzero(points['positions'][:, 0] >= 0))
                                                     for _, points, _, _ in companions]
    if fixed:
        report['white_stationary_layers'] = [response.report() for _, _, response in fixed]
    if ratios:
        report['url_relative_luma_template_response'] = quantiles([v for _, v in ratios])
        report['url_frames_measured'] = len(ratios)
        report['url_worst_frames'] = [{'frame': i, 'ratio': round(v, 5)}
                                      for i, v in sorted(ratios, key=lambda item: -item[1])[:12]]
    if corners:
        report['active_corner_template_score_after'] = quantiles(corners)
        report['corner_frames_at_original_detection_threshold'] = sum(
            v >= profile['corners'].get('minimum_score', 0.7) for v in corners)
    if rim_errors:
        report['flat_rim_luma_error'] = quantiles(rim_errors)
        report['flat_rim_frames_measured'] = len(rim_errors)
    if luma_error:
        report['outside_regions_sampled_frames'] = len(luma_error)
        report['outside_regions_luma_mean_absolute_error'] = round(float(np.mean(luma_error)), 4)
        mse = float(np.mean(luma_square))
        report['outside_regions_luma_psnr_db'] = round(10*np.log10(255**2/mse), 3) if mse else None
    if sheet_path:
        comparison_sheet(rows, source.fps, sheet_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def comparison_sheet(rows, fps, path):
    if not rows:
        raise ValueError('Choose at least one valid comparison frame.')
    rgb = next(iter(rows.values()))[0]
    w, h = 640, round(640 * rgb.shape[0] / rgb.shape[1])
    sheet = Image.new('RGB', (2*w, (h+26)*len(rows)), (25, 25, 25))
    pen = ImageDraw.Draw(sheet)
    for j, (i, pair) in enumerate(sorted(rows.items())):
        y = j * (h+26)
        for col, (title, data) in enumerate(zip(('Source', 'RTX 4090 restored'), pair, strict=True)):
            pen.text((w*col+9, y+6), f'{title}   {float(i/fps):.2f}s   frame {i}', fill='white')
            sheet.paste(Image.fromarray(data).resize((w, h), Image.Resampling.LANCZOS), (w*col, y+26))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)
