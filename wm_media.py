import hashlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from wm_runtime import check_budget, bounded_timeout, current_runtime


ROOT = Path(__file__).resolve().parent
READ_RETRIES = 2    # restarts of a decoder that stopped short at one frame before the read fails
# Frames between two wanted ones that are decoded and passed over rather than starting another decoder for the second:
# starting one (the more so with CUDA) takes as long as decoding a few hundred frames in a row.
READ_GAP = 150


def tools_directory(root):
    if configured := os.environ.get('WATERMARK_TOOLS_DIR'):
        return Path(configured).expanduser().resolve()
    for parent in (root, *root.parents):
        candidate = parent/'.watermark-tools'
        if candidate.is_dir():
            return candidate
    return root/'.watermark-tools'


TOOLS = tools_directory(ROOT)


@lru_cache(maxsize=None)
def _installed(name):
    # Looked up once per process: every read of frames starts a decoder, and walking the tools folder is slow where
    # it lies on a shared drive.
    # setup.ps1 and setup.sh unpack their builds side by side; each platform takes only its own.
    suffix = '.exe' if os.name == 'nt' else ''
    for path in sorted(TOOLS.glob(f'ffmpeg-*/bin/{name}{suffix}'), reverse=True):
        return str(path)
    return shutil.which(name)


def executable(name):
    configured = os.environ.get(f'WATERMARK_{name.upper()}')
    if configured:
        return configured
    path = _installed(name)
    if path is None:
        raise ValueError(f'{name} not found. Run setup.ps1 (Windows) or setup.sh (Linux) or set WATERMARK_{name.upper()}.')
    return path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(path, *entries):
    return json.loads(subprocess.check_output([
        executable('ffprobe'), '-v', 'error', *entries, '-of', 'json', str(path),
    ], timeout=bounded_timeout(90)))


@dataclass(frozen=True)
class Video:
    path: Path
    width: int
    height: int
    fps: Fraction
    count: int
    offset: float
    sar: str
    matrix: str
    transfer: str
    primaries: str
    start: float = 0.   # where the file's timeline starts (an empty edit before every stream when positive)

    @classmethod
    def open(cls, path):
        path = Path(path).resolve(strict=True)
        info = probe(path, '-show_streams', '-show_format')
        video = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        if video is None or video.get('pix_fmt') != 'yuv420p':
            raise ValueError('This pipeline expects 8-bit yuv420p video.')
        width, height = video['width'], video['height']
        if width % 2 or height % 2 or video.get('field_order') not in (None, 'unknown', 'progressive'):
            raise ValueError('This pipeline expects even dimensions and progressive frames.')
        if any(s.get('rotation', 0) % 360 for s in video.get('side_data_list', [])):
            raise ValueError('Normalize video rotation before making a profile.')
        if video.get('color_range') not in (None, 'unknown', 'tv'):
            raise ValueError('This pipeline expects limited-range SDR video.')
        fps = Fraction(video['r_frame_rate'])
        if fps <= 0 or fps != Fraction(video['avg_frame_rate']) or not video.get('nb_frames', '').isdigit():
            raise ValueError('This pipeline expects a constant frame rate and known frame count.')
        matrix = video.get('color_space')
        if matrix in (None, 'unknown'):
            matrix = 'bt709' if height >= 720 else 'smpte170m'
        transfer = video.get('color_transfer', 'unknown')
        if matrix not in ('bt709', 'bt470bg', 'smpte170m') or transfer in ('smpte2084', 'arib-std-b67'):
            raise ValueError('Use an SDR BT.709 or BT.601 source.')
        start = float(info['format'].get('start_time', '0'))
        offset = float(video.get('start_time', '0')) - start
        sar = video.get('sample_aspect_ratio') or '1:1'
        return cls(path, width, height, fps, int(video['nb_frames']), offset, sar,
                   matrix, transfer, video.get('color_primaries', 'unknown'), start)

    def check_timestamps(self):
        info = probe(self.path, '-select_streams', 'v:0', '-show_packets', '-show_entries', 'packet=pts_time')
        pts = np.sort([float(packet['pts_time']) for packet in info['packets']])
        if len(pts) != self.count or np.max(np.abs(pts - pts[0] - np.arange(self.count) / float(self.fps))) > 0.0002:
            raise ValueError('Presentation timestamps are variable or incomplete. Normalize to CFR first.')

    def planes(self, frame):
        return planes(frame, self.width, self.height)


def planes(frame, width, height):
    end_y = width * height
    end_u = end_y * 5 // 4
    return (frame[:end_y].reshape(height, width),
            frame[end_y:end_u].reshape(height // 2, width // 2),
            frame[end_u:].reshape(height // 2, width // 2))


def plane_rects(video_planes, rect):
    x, y, width, height = rect
    return tuple(p[y//s:(y+height)//s, x//s:(x+width)//s] for p, s in zip(video_planes, (1, 2, 2)))


def clipped_plane(plane, rect):
    x, y, w, h = rect
    left, top = max(0, x), max(0, y)
    right, bottom = min(plane.shape[1], x+w), min(plane.shape[0], y+h)
    right, bottom = max(left, right), max(top, bottom)
    return plane[top:bottom, left:right], np.s_[top-y:bottom-y, left-x:right-x]


def _seek_time(video, start):
    # Seek just before the desired PTS so sub-microsecond rounding never skips it.
    return f'{max(0, video.offset + start / float(video.fps) - 0.000005):.6f}'


def read_frames(video, start=0, stop=None, cuda=True, *, size=None, step=1):
    check_budget()
    stop = video.count if stop is None else stop
    if not 0 <= start < stop <= video.count:
        raise ValueError(f'Frame range [{start}, {stop}) is out of bounds.')
    width, height = size if size is not None else (video.width, video.height)
    if (type(step) is not int or step < 1 or not 2 <= width <= video.width or not 2 <= height <= video.height
            or width % 2 or height % 2):
        raise ValueError('Sampling needs an even in-frame resolution and a positive frame step.')
    filters = []
    if step > 1:
        filters.append(f'select=not(mod(n\\,{step}))')
    if (width, height) != (video.width, video.height):
        filters.append(f'scale_cuda={width}:{height}:interp_algo=bilinear' if cuda else f'scale={width}:{height}:flags=area')
    if cuda:
        filters.extend(('hwdownload', 'format=nv12', 'format=yuv420p'))
    indices = range(start, stop, step)
    frame_size = width * height * 3 // 2
    # A read of the file can go wrong once (a network or shared drive under load hands the demuxer bad bytes) and
    # the same frames decode on the next try: the decoder is restarted at the frame it stopped at, a few times.
    position, failed_at, failures = 0, None, 0
    while True:
        first = indices[position]
        cmd = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-xerror']
        if first:
            cmd += ['-ss', _seek_time(video, first)]
        if cuda:
            cmd += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
        cmd += ['-noautorotate', '-i', str(video.path), '-map', '0:v:0']
        if filters:
            cmd += ['-vf', ','.join(filters)]
        cmd += ['-frames:v', str(len(indices)-position), '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p',
                '-c:v', 'rawvideo', '-threads:v', '1', '-f', 'rawvideo', 'pipe:1']
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
            try:
                while position < len(indices):
                    check_budget()
                    frame = bytearray(frame_size)
                    if process.stdout.readinto(frame) != frame_size:
                        break
                    yield indices[position], np.frombuffer(frame, dtype=np.uint8)
                    position += 1
                else:
                    if process.wait():
                        raise RuntimeError('FFmpeg decoder failed.')
                    return
            finally:
                if process.poll() is None:
                    process.terminate()
        failures = failures+1 if failed_at == position else 1
        failed_at = position
        if failures > READ_RETRIES:
            raise RuntimeError(f'FFmpeg stopped decoding at frame {indices[position]}.')


def get_frame(video, index, cuda=True):
    check_budget()
    runtime = current_runtime()
    key = None
    if runtime is not None:
        stat = video.path.stat()
        key = (str(video.path), stat.st_size, stat.st_mtime_ns, index, cuda, video.width, video.height)
        with runtime.frame_lock:
            cached = runtime.frames.get(key)
            if cached is not None:
                runtime.frame_cache_hits += 1
                runtime.frames.move_to_end(key)
        if cached is not None:
            return cached.copy()
    for _, frame in read_frames(video, index, index + 1, cuda):
        result = frame
    if runtime is not None and result.nbytes <= 128*1024**2:
        with runtime.frame_lock:
            previous = runtime.frames.pop(key, None)
            if previous is not None:
                runtime.frame_bytes -= previous.nbytes
            while runtime.frames and runtime.frame_bytes+result.nbytes > 128*1024**2:
                _, oldest = runtime.frames.popitem(last=False)
                runtime.frame_bytes -= oldest.nbytes
            runtime.frames[key] = result.copy()
            runtime.frame_bytes += result.nbytes
    return result


def read_selected(video, indices, cuda=True, *, gap=READ_GAP):
    """Yield (index, frame) for the given frames, in ascending order. Frames no more than `gap` apart are read by
    one decoder, which passes over the frames between them: the frames are the ones get_frame returns for each."""
    wanted = sorted(set(int(i) for i in indices))
    first = 0
    for k in range(1, len(wanted)+1):
        if k == len(wanted) or wanted[k]-wanted[k-1] > gap:
            run = set(wanted[first:k])
            for i, frame in read_frames(video, wanted[first], wanted[k-1]+1, cuda):
                if i in run:
                    yield i, frame
            first = k


def yuv_to_rgb(parts, matrix):
    y, u, v = parts
    rows, cols = y.shape
    # Chroma is reconstructed within the ROI, then only the masked YUV samples are written back.
    u = cv2.resize(u, (cols, rows), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    v = cv2.resize(v, (cols, rows), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    y = (y.astype(np.float32) - 16) * (255 / 219)
    if matrix == 'bt709':
        rgb = (y + 1.7927 * v, y - 0.2132 * u - 0.5329 * v, y + 2.1124 * u)
    else:
        rgb = (y + 1.5960 * v, y - 0.3918 * u - 0.8130 * v, y + 2.0172 * u)
    return np.clip(np.rint(np.stack(rgb, axis=-1)), 0, 255).astype(np.uint8)


def rgb_to_yuv(rgb, matrix):
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    kr, kb = (0.2126, 0.0722) if matrix == 'bt709' else (0.299, 0.114)
    luma = kr * r + (1 - kr - kb) * g + kb * b
    u = (b - luma) * (112 / (255 * (1 - kb)))
    v = (r - luma) * (112 / (255 * (1 - kr)))
    size = (rgb.shape[1] // 2, rgb.shape[0] // 2)
    result = (16 + luma * (219 / 255), 128 + cv2.resize(u, size, interpolation=cv2.INTER_AREA),
              128 + cv2.resize(v, size, interpolation=cv2.INTER_AREA))
    return tuple(np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in result)


@contextmanager
def encode(video, output, start, stop, cq):
    output = Path(output).resolve()
    if output.suffix.lower() != '.mp4' or output.exists() or output == video.path:
        raise ValueError('Choose a new .mp4 output path.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.stem}.{os.getpid()}.partial.mp4')
    cmd = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-xerror']
    # Raw YUV has no color tags. Declare the existing matrix and range on input
    # so FFmpeg passes the samples through instead of converting their colors.
    cmd += ['-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', f'{video.width}x{video.height}',
            '-framerate', str(video.fps), '-color_range', 'tv', '-colorspace', video.matrix, '-i', 'pipe:0']
    if start:
        cmd += ['-ss', _seek_time(video, start)]
    if stop != video.count:
        cmd += ['-t', f'{(stop-start) / float(video.fps):.6f}']
    cmd += ['-i', str(video.path), '-map', '0:v:0', '-map', '1:a?', '-map', '1:s?',
            '-map_metadata', '1', '-map_chapters', '1' if start == 0 and stop == video.count else '-1',
            '-c:a', 'copy', '-c:s', 'copy', '-c:v', 'h264_nvenc', '-preset', 'p6', '-tune', 'hq',
            '-rc', 'vbr', '-cq', str(cq), '-b:v', '0', '-multipass', 'fullres',
            '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '24', '-bf', '3',
            '-fps_mode', 'passthrough', '-enc_time_base', '1:90000', '-video_track_timescale', '90000',
            '-color_range', 'tv', '-colorspace', video.matrix]
    if video.transfer != 'unknown':
        cmd += ['-color_trc', video.transfer]
    if video.primaries != 'unknown':
        cmd += ['-color_primaries', video.primaries]
    filters = ['settb=1/90000']
    # A file whose timeline starts late (an empty edit before its streams) keeps that start: FFmpeg moves the copied
    # streams to zero and adds the output offset back, in microseconds rounded to each stream's ticks. The video's
    # shift is chosen so that it lands on the tick of its own first presentation time.
    lead = round(max(0., video.start)*1e6) if start == 0 else 0
    shift = round((lead/1e6+video.offset)*90000)-round(lead*90000/1e6) if start == 0 else 0
    if shift > 0:
        filters.append(f'setpts=PTS+{shift}')
    if video.sar not in ('1:1', '0:1', 'N/A'):
        filters.append('setsar=' + video.sar.replace(':', '/'))
    # A small positive presentation offset can be lost when an MP4 with B-frames
    # uses only unsigned composition offsets. Signed offsets retain its empty edit.
    movflags = '+faststart+negative_cts_offsets' if start == 0 and video.offset > 0 else '+faststart'
    if lead:
        cmd += ['-output_ts_offset', f'{lead/1e6:.6f}']
    cmd += ['-vf', ','.join(filters), '-movie_timescale', '90000', '-movflags', movflags, str(temporary)]
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
            try:
                yield process.stdin
                process.stdin.close()
                if process.wait():
                    raise RuntimeError('FFmpeg NVENC encoding failed.')
            finally:
                if process.poll() is None:
                    process.terminate()
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
