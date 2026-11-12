import argparse
from contextlib import contextmanager, nullcontext
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from wm_media import Video, encode, executable, probe, read_frames
from wm_track import load_profile, load_track, recover_corner_track, save_npz, scan


def frame_range(value, video):
    if value is None:
        return 0, video.count
    start, stop = map(int, value.split(':'))
    if not 0 <= start < stop <= video.count:
        raise ValueError(f'Expected 0 <= start < stop <= {video.count}.')
    return start, stop


def work_paths(args):
    root = Path(args.workdir)
    return root / 'track.npz', root / 'matte.npz', root / 'corners', root / 'opening.npz'


@contextmanager
def discovery_render_slot():
    from workflow import capacity_slot

    count = int(os.environ.get('WATERMARK_RENDER_SLOTS', '1'))
    if count not in (1, 2):
        raise ValueError('GPU rendering supports one or two shared slots.')
    root = Path(tempfile.gettempdir())/'watermark-gpu-render-v1'

    def waiting(seconds):
        print(f'Waiting for a GPU render slot ({seconds:.0f}s)', flush=True)

    with capacity_slot(root, count, 3600, waiting=waiting) as (slot, seconds):
        print(f'GPU render slot {slot} ready after {seconds:.1f}s', flush=True)
        yield


def processing(args):
    from wm_matte import Matte, calibrate
    from wm_opening import Opening
    from wm_repair import Corners

    video = Video.open(args.input)
    if args.output and (args.output.resolve() == video.path or args.output.exists()):
        raise ValueError('Choose a new MP4 output path.')
    config = load_profile(video, args.profile)
    start, stop = frame_range(args.frames, video)
    cuda = not args.cpu_decode
    track_file, matte_file, corner_dir, opening_file = work_paths(args)
    if args.command == 'run' and not track_file.exists():
        scan(video, config, args.profile, track_file, cuda)
    track = load_track(video, args.profile, track_file)
    has_matte = bool(config.get('translucent') or config.get('stationary'))
    if has_matte and not matte_file.exists():
        if args.command != 'run':
            raise ValueError(f'Missing {matte_file}. Run calibrate or run.')
        pairs = json.loads(Path(args.pairs).read_text(encoding='utf-8')) if args.pairs else None
        calibrate(video, config, track, matte_file, cuda, pairs)
    context = Matte(config, track, matte_file) if has_matte else None
    corners = None
    if config.get('corners'):
        corners = Corners(video, config, track, corner_dir, scale=args.model_scale, core=args.core,
                          overlap=args.overlap, dilation=args.dilation, raft_iterations=args.raft_iterations,
                          context_matte_path=matte_file if has_matte else None, model_dir=args.model_dir,
                          phases=args.corner_phases)
        if args.command in ('run', 'prepare'):
            corners.prepare(start, stop, context, cuda)

    opening = None
    if args.opening:
        opening = Opening(video, config, track, args.opening, opening_file)
        if start >= opening.settings.get('corner', {}).get('stop', opening.settings['stop']):
            opening = None
        elif args.command in ('run', 'prepare'):
            if corners and opening.settings.get('corner'):
                seed = opening.settings['corner']['plate_frame']
                corners.prepare(seed, seed+1, context, cuda)
            opening.prepare(context, cuda, args.model_dir, corners)
        else:
            opening.load()
    if args.command == 'prepare':
        return
    if not args.output:
        raise ValueError('Supply an output MP4 path.')
    if args.recover_corner_overlap:
        if start != 0 or stop != video.count:
            raise ValueError('Corner overlap recovery is supported on full renders.')
        recovered_file = Path(args.workdir)/'track-recovered.npz'
        track = (load_track(video, args.profile, recovered_file) if recovered_file.exists()
                 else recover_corner_track(video, config, track, cuda))
        if not recovered_file.exists():
            save_npz(recovered_file, **track)
    conservative = args.edges == 'temporal-conservative'
    temporal = bool(config.get('translucent')) and args.edges in ('temporal', 'temporal-rim', 'temporal-conservative')
    white_flat = args.edges == 'white-flat'
    if (args.align_white or white_flat) and not config.get('translucent', {}).get('white_plate'):
        raise ValueError('White alignment and edge cleanup need a moving white plate calibration.')
    matte = Matte(config, track, matte_file, clean_edges=args.edges != 'off',
                  narrow_edges=conservative or white_flat, align_white=args.align_white,
                  white_flat=white_flat) if has_matte else None
    retouch = None
    if args.retouch:
        from wm_retouch import Retouch
        retouch = Retouch(video, args.retouch, matte)
    caption_patches = []
    if args.caption_patch:
        from wm_caption_inpaint import CaptionPatch
        caption_patches = [CaptionPatch(video, track, path) for path in args.caption_patch]
    animation_patches = []
    if args.animation_patch:
        from wm_animation import AnimationPatch
        animation_patches = [AnimationPatch(video, track, path) for path in args.animation_patch]
    if args.caption_band is not None and (not matte or not temporal):
        raise ValueError('Caption reference recovery requires a translucent matte and temporal edges.')
    if args.caption_context and args.caption_band is None:
        raise ValueError('Caption context repair requires --caption-band.')
    render_slot = nullcontext()
    if temporal and matte:
        from wm_temporal import temporal_frames
        frames = temporal_frames(video, matte, track, start, stop, cuda,
                                 rim=args.edges == 'temporal-rim', caption_band=args.caption_band,
                                 caption_context=args.caption_context, conservative=conservative)
    elif config.get('discovery'):
        from wm_discovery_render import reference_matte, restore_frames
        if 'matte_positions' in track:
            frames = inverse_frames(video, reference_matte(track, track_file), start, stop, cuda)
        else:
            render_slot = discovery_render_slot()
            frames = restore_frames(video, track, start, stop, cuda, args.model_scale, args.raft_iterations, args.model_dir,
                                    consensus=config['discovery'].get('consensus_mask', False),
                                    separate_jumps=config['discovery'].get('separate_jumps', False),
                                    edge_layer=config['discovery'].get('edge_repair', {}).get('layer'))
    else:
        frames = inverse_frames(video, matte, start, stop, cuda)
    with render_slot, encode(video, args.output, start, stop, args.cq) as stream:
        began = time.monotonic()
        for i, frame in frames:
            override = opening and opening.restore_corner(i, frame)
            if corners and not override:
                corners.restore(i, frame)
            if opening:
                opening.restore(i, frame)
            if retouch:
                retouch.apply(i, frame)
            for patch in caption_patches:
                patch.apply(i, frame)
            for patch in animation_patches:
                patch.apply(i, frame)
            stream.write(frame)
            if (i-start+1) % 300 == 0:
                elapsed = time.monotonic() - began
                print(f'Encoded {i-start+1}/{stop-start} frames ({(i-start+1)/elapsed:.1f} fps)', flush=True)
    protected = [patch.checked for patch in caption_patches if patch.layer_frames is None]
    if protected:
        print(f'Protected subtitle YUV samples verified before encoding: {sum(protected)}', flush=True)
    if corners and corners.dark_fallback_frames:
        dark = np.unique(corners.dark_fallback_frames)
        runs = np.split(dark, np.flatnonzero(np.diff(dark) > 1)+1)
        ranges = ', '.join(f'{run[0]}:{run[-1]+1}' for run in runs)
        print(f'Used local dark plates for {len(dark)} frames in ranges {ranges}.', flush=True)
    shifts = ([shift for layer in (matte, *matte.extra) for _, shift in layer.white_alignment_changes] if matte else [])
    if shifts:
        print(f'Realigned {len(shifts)} white overlay edges, median {np.median(shifts):.3f} px, '
              f'maximum {max(shifts):.3f} px.', flush=True)
    print(f'Saved {Path(args.output).resolve()}', flush=True)


def inverse_frames(video, matte, start, stop, cuda):
    frames = iter(read_frames(video, start, stop, cuda))
    while batch := list(itertools.islice(frames, 24)):
        if matte:
            matte.restore(video, [frame for _, frame in batch], batch[0][0])
        yield from batch


def render_patches(args):
    from wm_animation import AnimationPatch
    from wm_retouch import Retouch

    source, base = Video.open(args.input), Video.open(args.base)
    if (source.width, source.height, source.fps, source.count, source.offset, source.matrix) != (
            base.width, base.height, base.fps, base.count, base.offset, base.matrix):
        raise ValueError('Patch rendering requires an aligned base with matching color format.')
    if args.output.resolve() == source.path:
        raise ValueError('Choose a new MP4 output path.')
    track = load_track(source, args.profile, Path(args.workdir)/'track.npz')
    patches = [AnimationPatch(base, track, path) for path in args.animation_patch]
    retouch = Retouch(base, args.retouch) if args.retouch else None
    if not patches and retouch is None:
        raise ValueError('Supply an animation patch or a retouch configuration.')
    start, stop = frame_range(args.frames, base)
    with encode(base, args.output, start, stop, args.cq) as stream:
        for i, frame in read_frames(base, start, stop, not args.cpu_decode):
            for patch in patches:
                patch.apply(i, frame)
            if retouch:
                retouch.apply(i, frame)
            stream.write(frame)
    print(f'Saved {args.output.resolve()}', flush=True)


def verify(source, output, report):
    original, restored = Video.open(source), Video.open(output)
    original.check_timestamps()
    restored.check_timestamps()
    same_video = (original.width, original.height, original.fps, original.count, original.offset) == (
        restored.width, restored.height, restored.fps, restored.count, restored.offset)
    streams = []
    for path in (original.path, restored.path):
        data = probe(path, '-select_streams', 'a', '-show_packets', '-show_data_hash', 'SHA256',
                     '-show_entries', 'packet=stream_index,pts_time,duration_time,data_hash')
        audio = {}
        for packet in data.get('packets', []):
            audio.setdefault(packet['stream_index'], []).append(
                (packet.get('pts_time'), packet.get('duration_time'), packet.get('data_hash')))
        streams.append([audio[i] for i in sorted(audio)])
    same_audio = streams[0] == streams[1]
    subprocess.run([executable('ffmpeg'), '-hide_banner', '-v', 'error', '-xerror', '-nostdin',
                    '-i', str(restored.path), '-map', '0:v:0', '-map', '0:a?', '-f', 'null', '-'], check=True)
    result = dict(frames=restored.count, size=[restored.width, restored.height], fps=str(restored.fps),
                  video_offset_seconds=restored.offset, same_video_timing_and_size=same_video,
                  audio_packets=sum(map(len, streams[1])), same_audio_payload_and_timing=same_audio, full_decode_ok=True)
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not same_video or not same_audio:
        raise ValueError('Source/output timing, video dimensions or audio packets differ.')


def add_input(parser):
    parser.add_argument('input', type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--cpu-decode', action='store_true', help='Use CPU decode; repair/encode still require CUDA/NVENC.')


def main():
    parser = argparse.ArgumentParser(description='Local NVIDIA video watermark repair.')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('scan', 'calibrate', 'prepare', 'render', 'run'):
        sub = commands.add_parser(command)
        add_input(sub)
        if command in ('run', 'calibrate'):
            sub.add_argument('--pairs', type=Path, help='Optional JSON [target frame, clean same-shot frame, weight] entries.')
        if command in ('prepare', 'render', 'run'):
            if command != 'prepare':
                sub.add_argument('output', type=Path)
            else:
                sub.set_defaults(output=None)
            sub.add_argument('--frames', help='Preview range as START:STOP, zero-based, stop exclusive.')
            sub.add_argument('--opening', type=Path, help='Optional clip-specific opening mask configuration.')
            sub.add_argument('--retouch', type=Path, help='Optional frame-ranged particle retouch regions.')
            sub.add_argument('--caption-band', type=int, help='Top row of the burned-in caption band for temporal recovery.')
            sub.add_argument('--caption-context', action='store_true', help='Also repair verified stable background near captions.')
            sub.add_argument('--caption-patch', type=Path, action='append', help='GPU patch cache for a subtitle overlap. Repeat to apply more than one.')
            sub.add_argument('--animation-patch', type=Path, action='append', help='Apply a cached animation repair.')
            sub.add_argument('--model-dir', type=Path)
            sub.add_argument('--model-scale', type=float, default=0.625)
            sub.add_argument('--corner-phases', action='store_true',
                             help='Use shifted inpaint predictions to reduce the model grid while retaining structure.')
            sub.add_argument('--core', type=int, default=56)
            sub.add_argument('--overlap', type=int, default=6)
            sub.add_argument('--dilation', type=int, default=14)
            sub.add_argument('--raft-iterations', type=int, default=6)
            sub.add_argument('--cq', type=int, choices=range(1, 32), default=15, metavar='1..31')
            sub.add_argument('--edges', choices=('off', 'spatial', 'white-flat', 'temporal', 'temporal-rim',
                                                 'temporal-conservative'),
                             default='temporal')
            sub.add_argument('--align-white', action='store_true', help='Refine weak white-overlay alignment at render time.')
            sub.add_argument('--recover-corner-overlap', action='store_true',
                             help='Recheck a moving mark that is temporarily covered by a corner badge.')
    check = commands.add_parser('verify')
    check.add_argument('source', type=Path)
    check.add_argument('output', type=Path)
    check.add_argument('--report', type=Path)
    qa = commands.add_parser('audit')
    add_input(qa)
    qa.add_argument('output', type=Path)
    qa.add_argument('--report', required=True, type=Path)
    qa.add_argument('--sheet', type=Path)
    qa.add_argument('--samples', type=int, nargs='*', default=())
    captions = commands.add_parser('caption-audit')
    add_input(captions)
    captions.add_argument('baseline', type=Path)
    captions.add_argument('output', type=Path)
    captions.add_argument('--band', type=int, required=True)
    captions.add_argument('--samples', type=int, nargs='+', required=True)
    captions.add_argument('--report', type=Path, required=True)
    caption_plan = commands.add_parser('caption-plan')
    add_input(caption_plan)
    caption_plan.add_argument('base', type=Path)
    caption_plan.add_argument('--band', type=int, required=True)
    background = commands.add_parser('caption-inpaint')
    add_input(background)
    background.add_argument('base', type=Path)
    background.add_argument('output', type=Path)
    background.add_argument('--config', type=Path, required=True)
    reveal = commands.add_parser('url-reveal', help='Recover a moving translucent mark from visible same-shot pixels.')
    add_input(reveal)
    reveal.add_argument('base', type=Path)
    reveal.add_argument('output', type=Path, help='New repair .npz file.')
    reveal.add_argument('--config', type=Path, required=True)
    reveal.add_argument('--job', help='Job name when the configuration contains multiple shots or intervals.')
    reveal.add_argument('--report', type=Path)
    local = commands.add_parser('local-patch', help='Refine a tracked translucent watermark in one short region.')
    add_input(local)
    local.add_argument('output', type=Path, help='New repair .npz file.')
    interval = local.add_mutually_exclusive_group(required=True)
    interval.add_argument('--frames', help='START:STOP, zero-based, stop exclusive.')
    interval.add_argument('--seconds', help='START:STOP in seconds from the start of the video.')
    interval.add_argument('--config', type=Path, help='JSON configuration for a manually guided shot.')
    local.add_argument('--job', help='Job name when the configuration contains multiple shots.')
    local.add_argument('--region', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help='Limit repair to this rectangle. Defaults to the tracked watermark path.')
    local.add_argument('--protect', type=int, nargs=4, action='append', default=[], metavar=('X', 'Y', 'W', 'H'),
                       help='Preserve this rectangle. Repeat for multiple regions.')
    local.add_argument('--report', type=Path)
    animation = commands.add_parser('animation-patch')
    add_input(animation)
    animation.add_argument('base', type=Path, nargs='?',
                           help='Aligned reference video. Omit to repair directly from source windows.')
    animation.add_argument('output', type=Path)
    animation.add_argument('--config', type=Path, required=True)
    animation.add_argument('--job', help='Job name when the configuration contains multiple occurrences.')
    animation.add_argument('--template', type=Path, help='Mask learned from multiple occurrences of the same animation.')
    animation_template = commands.add_parser('animation-template')
    add_input(animation_template)
    animation_template.add_argument('output', type=Path)
    animation_template.add_argument('--config', type=Path, required=True)
    region_patch = commands.add_parser('region-patch', help='Repair an opaque mark attached to one scene.')
    add_input(region_patch)
    region_patch.add_argument('output', type=Path)
    region_patch.add_argument('--config', type=Path, required=True)
    region_patch.add_argument('--job')
    wipe_template = commands.add_parser('wipe-template')
    add_input(wipe_template)
    wipe_template.add_argument('output', type=Path)
    wipe_template.add_argument('--config', type=Path, required=True)
    wipe_patch = commands.add_parser('wipe-patch')
    add_input(wipe_patch)
    wipe_patch.add_argument('base', type=Path)
    wipe_patch.add_argument('output', type=Path)
    wipe_patch.add_argument('--config', type=Path, required=True)
    wipe_patch.add_argument('--template', type=Path, required=True)
    wipe_patch.add_argument('--anchor-bottom', type=int, nargs=2, action='append', metavar=('AGE','ROWS'),
                            help='Exclude a thin bottom edge when projecting a reference. Repeat as needed.')
    wipe_patch.add_argument('--job', required=True)
    wipe_patch.add_argument('--animation-dir', type=Path, help='Prior wordmark masks used to prefer observed anchor pixels.')
    pose = commands.add_parser('pose-patch')
    add_input(pose)
    pose.add_argument('base', type=Path)
    pose.add_argument('output', type=Path)
    pose.add_argument('--config', type=Path, required=True)
    pose.add_argument('--wordmark', type=Path, required=True, help='Existing wordmark patch providing the affected area.')
    pose.add_argument('--wipe', type=Path, required=True, help='Existing wipe patch providing the affected area.')
    pose.add_argument('--template', type=Path, required=True, help='Wipe template used to identify visible original pixels.')
    corners = commands.add_parser('corner-reference')
    add_input(corners)
    corners.add_argument('base', type=Path)
    corners.add_argument('output', type=Path, help='New folder for patch caches and the keyframe plan.')
    corners.add_argument('--config', type=Path, required=True)
    corners.add_argument('--animation-template', type=Path)
    trim = commands.add_parser('corner-trim', help='Extend a visible straight edge across a corner watermark.')
    add_input(trim)
    trim.add_argument('output', type=Path, help='New corner repair .npz file.')
    trim.add_argument('--config', type=Path, required=True)
    patch_render = commands.add_parser('patch-render')
    add_input(patch_render)
    patch_render.add_argument('base', type=Path)
    patch_render.add_argument('output', type=Path)
    patch_render.add_argument('--frames', required=True)
    patch_render.add_argument('--animation-patch', type=Path, action='append', default=[])
    patch_render.add_argument('--retouch', type=Path, help='Apply a final frame-ranged dark or cyan cleanup.')
    patch_render.add_argument('--cq', type=int, choices=range(1, 32), default=14, metavar='1..31')
    grid = commands.add_parser('corner-degrid', help='Clean the stationary texture left by corner inpainting.')
    add_input(grid)
    grid.add_argument('base', type=Path)
    grid.add_argument('output', type=Path)
    grid.add_argument('--frames', help='Optional preview range as START:STOP.')
    grid.add_argument('--model-scale', type=float, default=.625)
    grid.add_argument('--dilation', type=int, default=14)
    grid.add_argument('--strength', type=float, default=1)
    grid.add_argument('--protect-first', type=int, default=0, help='Exclude this many custom opening frames.')
    grid.add_argument('--protect-patch', type=Path, action='append', default=[],
                      help='Preserve frames with original corner pixels restored by this patch. Repeat as needed.')
    grid.add_argument('--animation-patch', type=Path, action='append', default=[],
                      help='Apply a verified patch after grid filtering. Repeat as needed.')
    grid.add_argument('--retouch', type=Path, help='Apply a final frame-ranged dark or cyan cleanup.')
    grid.add_argument('--cq', type=int, choices=range(1, 32), default=14, metavar='1..31')
    grid.add_argument('--report', type=Path)
    animation_audit = commands.add_parser('animation-audit')
    animation_audit.add_argument('source', type=Path)
    animation_audit.add_argument('output', type=Path)
    animation_audit.add_argument('--starts', type=int, nargs=2, required=True)
    animation_audit.add_argument('--ages', type=int, nargs='+', required=True)
    animation_audit.add_argument('--region', type=int, nargs=4, required=True)
    animation_audit.add_argument('--report', type=Path, required=True)
    animation_audit.add_argument('--threshold', type=float, default=6)
    animation_audit.add_argument('--cpu-decode', action='store_true')
    join = commands.add_parser('splice')
    join.add_argument('base', type=Path)
    join.add_argument('patch', type=Path)
    join.add_argument('output', type=Path)
    join.add_argument('--frames', required=True, help='Keyframe-aligned START:STOP to replace.')
    join.add_argument('--report', type=Path)
    commands.add_parser('self-test')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.command == 'verify':
        verify(args.source, args.output, args.report)
    elif args.command == 'audit':
        from wm_audit import audit
        audit(args.input, args.output, args.profile, args.workdir, args.report, args.sheet,
              args.samples, not args.cpu_decode)
    elif args.command == 'caption-audit':
        from wm_caption_audit import audit_captions
        audit_captions(args.input, args.baseline, args.output, args.profile, args.workdir,
                       args.band, args.samples, args.report, not args.cpu_decode)
    elif args.command == 'caption-plan':
        from wm_caption import caption_plan
        print(json.dumps(caption_plan(args.input, args.base, args.profile, args.workdir, args.band), indent=2))
    elif args.command == 'caption-inpaint':
        from wm_caption_inpaint import prepare_caption_patch
        prepare_caption_patch(args.input, args.base, args.output, args.profile, args.workdir,
                              args.config, not args.cpu_decode)
    elif args.command == 'url-reveal':
        from wm_url_reveal import prepare_url_reveal
        prepare_url_reveal(args.input, args.base, args.output, args.profile, args.workdir, args.config,
                           args.report, not args.cpu_decode, args.job)
    elif args.command == 'local-patch':
        from wm_local_patch import prepare_local_patch
        cv2.setNumThreads(2)
        prepare_local_patch(args)
    elif args.command == 'animation-patch':
        from wm_animation import prepare_animation_patch
        prepare_animation_patch(args.input, args.base, args.output, args.profile, args.workdir,
                                args.config, not args.cpu_decode, args.job, args.template)
    elif args.command == 'animation-template':
        from wm_animation import build_animation_template
        build_animation_template(args.input, args.output, args.profile, args.workdir, args.config, not args.cpu_decode)
    elif args.command == 'region-patch':
        from wm_region_patch import prepare_region_patch
        prepare_region_patch(args.input, args.output, args.profile, args.workdir, args.config, not args.cpu_decode, args.job)
    elif args.command == 'wipe-template':
        from wm_wipe import build_wipe_template
        build_wipe_template(args.input, args.output, args.profile, args.workdir, args.config, not args.cpu_decode)
    elif args.command == 'wipe-patch':
        from wm_wipe import prepare_wipe_patch
        prepare_wipe_patch(args.input, args.base, args.output, args.profile, args.workdir, args.config,
                           args.template, args.job, not args.cpu_decode, args.animation_dir, args.anchor_bottom)
    elif args.command == 'pose-patch':
        from wm_pose import prepare_pose_patch
        prepare_pose_patch(args.input, args.base, args.output, args.profile, args.workdir, args.config,
                           args.wordmark, args.wipe, args.template, not args.cpu_decode)
    elif args.command == 'corner-reference':
        from wm_corner_reference import prepare_corner_references
        prepare_corner_references(args.input, args.base, args.output, args.profile, args.workdir, args.config,
                                  args.animation_template, not args.cpu_decode)
    elif args.command == 'corner-trim':
        from wm_corner_trim import prepare_trim_patch
        prepare_trim_patch(args.input, args.output, args.profile, args.workdir, args.config, not args.cpu_decode)
    elif args.command == 'patch-render':
        render_patches(args)
    elif args.command == 'corner-degrid':
        from wm_corner_grid import degrid_video
        degrid_video(args, *frame_range(args.frames, Video.open(args.base)))
    elif args.command == 'animation-audit':
        from wm_animation_audit import audit_animation
        audit_animation(args.source, args.output, args.starts, args.ages, args.region, args.report,
                        args.threshold, not args.cpu_decode)
    elif args.command == 'splice':
        from wm_splice import splice
        start, stop = frame_range(args.frames, Video.open(args.base))
        splice(args.base, args.patch, args.output, start, stop, args.report)
    elif args.command == 'self-test':
        from wm_selftest import self_test
        self_test()
    elif args.command in ('scan', 'calibrate'):
        video = Video.open(args.input)
        config = load_profile(video, args.profile)
        track_file, matte_file, _, _ = work_paths(args)
        cuda = not args.cpu_decode
        if args.command == 'scan':
            scan(video, config, args.profile, track_file, cuda)
        else:
            from wm_matte import calibrate
            track = load_track(video, args.profile, track_file)
            pairs = json.loads(args.pairs.read_text(encoding='utf-8')) if args.pairs else None
            calibrate(video, config, track, matte_file, cuda, pairs)
    else:
        processing(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
