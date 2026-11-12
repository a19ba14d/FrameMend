"""Replace a keyframe-aligned interval and verify the copied packet payloads."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from wm_media import Video, executable, probe, read_frames


PACKET_FIELDS = 'packet=pts,dts,duration,flags,data_hash'


def video_packets(path):
    data = probe(path, '-select_streams', 'v:0', '-show_streams', '-show_packets',
                 '-show_data_hash', 'SHA256', '-show_entries',
                 f'stream=codec_name,time_base,extradata_hash:{PACKET_FIELDS}')
    return data['streams'][0], data['packets']


def audio_packets(path):
    return probe(path, '-select_streams', 'a', '-show_packets', '-show_data_hash', 'SHA256',
                 '-show_entries', 'packet=stream_index,pts,dts,duration,data_hash').get('packets', [])


def verify_boundary_frames(base, patch, candidate, start, stop):
    candidate = Video.open(candidate)
    head = min(6, patch.count)
    for (_, expected), (_, actual) in zip(read_frames(patch, 0, head, cuda=False),
                                         read_frames(candidate, start, start+head, cuda=False), strict=True):
        if not np.array_equal(expected, actual):
            raise ValueError('The start of the replacement does not decode independently.')
    if stop == base.count:
        return
    # Decode across the boundary from inside both preceding GOPs to catch
    # dependencies on pictures that a packet hash alone cannot detect.
    lo, hi = max(start, stop-6), min(base.count, stop+8)
    for (i, expected), (_, actual) in zip(read_frames(base, lo, hi, cuda=False),
                                         read_frames(candidate, lo, hi, cuda=False), strict=True):
        if i >= stop and not np.array_equal(expected, actual):
            raise ValueError('The copied tail does not decode independently after the splice.')


def splice(base, patch, output, start, stop, report=None):
    base, patch = Video.open(base), Video.open(patch)
    output = Path(output).resolve()
    if output.exists() or output.suffix.lower() != '.mp4':
        raise ValueError('Choose a new MP4 output path.')
    if report and Path(report).resolve() in (base.path, patch.path, output):
        raise ValueError('Splice report must be a separate file.')
    if not 0 <= start < stop <= base.count or patch.count != stop-start:
        raise ValueError('Patch frame count must match the replacement interval.')
    if (base.width, base.height, base.fps) != (patch.width, patch.height, patch.fps):
        raise ValueError('Patch must have the same size and frame rate as the base.')
    base_stream, original = video_packets(base.path)
    patch_stream, replacement = video_packets(patch.path)
    ticks = 90000 / base.fps
    if (base_stream != patch_stream or base_stream.get('codec_name') != 'h264'
            or base_stream.get('time_base') != '1/90000' or ticks.denominator != 1):
        raise ValueError('Both inputs need identical H.264 headers and a 90 kHz frame grid.')
    ticks = int(ticks)
    positions = sorted(int(packet['pts']) for packet in original)
    if len(original) != base.count or len(replacement) != patch.count or any(
            pts != positions[0]+i*ticks for i, pts in enumerate(positions)):
        raise ValueError('The base has an irregular packet timeline.')
    for boundary in (start, stop):
        if boundary < base.count and (int(original[boundary]['pts']) != positions[boundary]
                                      or 'K' not in original[boundary].get('flags', '')):
            raise ValueError(f'Frame {boundary} must be an independently decodable keyframe.')
    if 'K' not in replacement[0].get('flags', ''):
        raise ValueError('The patch must start on a keyframe.')

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = [executable('ffmpeg'), '-hide_banner', '-v', 'error', '-nostdin', '-xerror', '-n']
    stream_copy = ['-map', '0:v:0', '-c', 'copy', '-video_track_timescale', '90000', '-movie_timescale', '90000']
    with TemporaryDirectory(prefix='.watermark-splice-', dir=output.parent) as temp:
        work = Path(temp)
        parts = []
        if start:
            subprocess.run(ffmpeg + ['-i', str(base.path), '-frames:v', str(start)]
                           + stream_copy + [str(work/'prefix.mp4')], check=True)
            parts.append(('prefix.mp4', start))
        subprocess.run(ffmpeg + ['-i', str(patch.path)] + stream_copy + [str(work/'patch.mp4')], check=True)
        parts.append(('patch.mp4', stop-start))
        if stop < base.count:
            # The stream-copy seek must land on the keyframe itself. Seeking
            # a few microseconds earlier can pull in an entire previous GOP.
            seek = (round(base.offset*90000) + stop*ticks) / 90000
            subprocess.run(ffmpeg + ['-ss', f'{seek:.6f}', '-i', str(base.path)]
                           + stream_copy + [str(work/'tail.mp4')], check=True)
            parts.append(('tail.mp4', base.count-stop))
        manifest = work / 'segments.ffconcat'
        manifest.write_text('ffconcat version 1.0\n' + ''.join(
            f'file {name}\nduration {frames/float(base.fps):.6f}\n' for name, frames in parts), encoding='ascii')
        candidate = work / 'candidate.mp4'
        subprocess.run(ffmpeg + ['-itsoffset', f'{base.offset:.6f}', '-f', 'concat', '-auto_convert', '0',
                                 '-safe', '1', '-i', str(manifest), '-i', str(base.path), '-map', '0:v:0',
                                 '-map', '1:a?', '-map', '1:s?', '-map_metadata', '1', '-map_chapters', '1',
                                 '-c', 'copy', '-video_track_timescale', '90000', '-movie_timescale', '90000',
                                 '-movflags', '+faststart', str(candidate)], check=True)
        joined_stream, joined = video_packets(candidate)
        if (joined_stream != base_stream or sorted(int(p['pts']) for p in joined) != positions
                or joined[:start] != original[:start] or joined[stop:] != original[stop:]
                or [p['data_hash'] for p in joined[start:stop]] != [p['data_hash'] for p in replacement]):
            raise ValueError('Splice changed the expected video packets or presentation times.')
        if audio_packets(candidate) != audio_packets(base.path):
            raise ValueError('Splice changed the audio packet payload or timing.')
        verify_boundary_frames(base, patch, candidate, start, stop)
        if output.exists():
            raise FileExistsError(output)
        candidate.rename(output)
    result = dict(start_frame=start, stop_frame_exclusive=stop, replacement_video_packets=stop-start,
                  unchanged_video_packets=base.count-(stop-start), packet_payloads_and_timing_verified=True,
                  decoded_join_boundaries_verified=True)
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
