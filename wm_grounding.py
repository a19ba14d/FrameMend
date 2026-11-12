"""Shared single-frame watermark evidence and an optional local visual backend."""

import base64
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import cv2
import numpy as np
from PIL import Image, ImageDraw

from wm_media import Video, clipped_plane, digest, get_frame as decode_frame, yuv_to_rgb
from workflow import CapacityTimeout, capacity_slot, read_object, write_object


DECISIONS = ('found', 'no_mark', 'review')
LOCAL_PARALLEL_REQUESTS = 2
LOCAL_ADMISSION_SECONDS = 180
CANDIDATE_DECISIONS = ('watermark', 'protected', 'uncertain')
CANDIDATE_PROMPT = (
    'Classify the outlined graphic or lettering in the candidate crop. Use "watermark" for a '
    'distribution logo, channel branding or promotional URL overlaid on the video. Use "protected" '
    'for dialogue captions, story titles, credits, physical scene content, or an empty crop. Use '
    '"uncertain" when the crop is insufficient. Read visible wording and describe any icon. '
    'Return only JSON with decision (watermark, protected or uncertain), label (visible wording or icon), '
    'and reason (a brief explanation of how you distinguished these categories).'
)
FIT_INSTRUCTION = (' An orange rectangle outlines the proposed mask bounds. For a watermark, also return '
                   '"fit" as "complete" only if all parts of the overlay are covered, or "partial" '
                   'if any part such as an icon or wording is missed. In a mask preview, the cyan tint is '
                   'the proposed edit support. An uncolored part inside the orange rectangle is still '
                   'missing. Ignore antialiasing and compression fringes at the border.')
CANDIDATE_SCHEMA = {
    'type': 'object',
    'properties': {'decision': {'enum': list(CANDIDATE_DECISIONS)},
                   'fit': {'enum': ['complete', 'partial', 'uncertain']},
                   'label': {'type': 'string'}, 'reason': {'type': 'string'}},
    'required': ['decision', 'fit', 'label', 'reason'],
    'additionalProperties': False,
}


def prompt_for(label, confirm=False, compact=False):
    name = json.dumps(label, ensure_ascii=False)
    if compact:
        task = 'Confirm the entire watermark is inside the crop.' if confirm else (
            'Locate the watermark with a tight [left,top,right,bottom] rect, integers 0–1000 for this image.'
        )
        return (
            f'Use only this image. Target: {name}. {task} '
            'Return JSON with decision "found", "no_mark" or "review" and a brief reason. '
            'If unclear or cropped, use "review". Answer directly without tools.'
        )
    if confirm:
        return (
            f'Does this crop contain the entire overlaid watermark {name}? '
            'Inspect the actual pixels, including the icon and small text. Scene signs and subtitles are '
            'content to keep. Return only JSON with decision "found" if the entire target is visible, '
            '"no_mark" if absent, or "review" if clipped or unclear, and a short "reason".'
        )
    return (
        f'Locate this watermark in the single image: {name}. '
        'Look for the actual overlaid mark, including its icon and small text. '
        'Leave scene signs, intentional titles and subtitles alone. '
        'Give a tight box in this image: [left, top, right, bottom], with integer coordinates '
        'from 0 to 1000 relative to the entire image. '
        'Return only JSON: {"decision":"found","rect":[left,top,right,bottom],"reason":"short reason"}. '
        'If absent, use decision "no_mark" and rect null. If uncertain, use decision "review" and rect null.'
    )


def jpeg(image, max_side):
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    data = BytesIO()
    image.save(data, 'JPEG', quality=95, subsampling=0)
    return data.getvalue()


def crop_detail(image, region):
    x, y, w, h = region
    crop = image.crop((x, y, x+w, y+h))
    scale = min(3, max(1, 480/max(w, h)))
    if scale > 1:
        crop = crop.resize((round(w*scale), round(h*scale)), Image.Resampling.BICUBIC)
    return crop


@lru_cache(maxsize=16)
def cached_evidence_frame(video, frame, cuda, _file_version):
    pixels = decode_frame(video, frame, cuda)
    pixels.flags.writeable = False
    return pixels


def get_frame(video, frame, cuda=True):
    """Share immutable decoded pixels across the full image, crop and mask preview."""
    if isinstance(video, Video) and video.width*video.height <= 1920*1080:
        stat = video.path.stat()
        version = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        return cached_evidence_frame(video, frame, cuda, version)
    return decode_frame(video, frame, cuda)


def frame_tiles(video, frame, cuda, max_side=1440):
    image = Image.fromarray(yuv_to_rgb(video.planes(get_frame(video, frame, cuda)), video.matrix))
    w, h = (3*video.width+4)//5, (3*video.height+4)//5
    for y in (0, video.height-h):
        for x in (0, video.width-w):
            region = (x, y, w, h)
            yield region, (f'Frame {frame}, single cropped view', jpeg(crop_detail(image, region), max_side))


def frame_evidence(video, frame, cuda, max_side=1280, region=None, outline=None, *, mask_at=None, hide=()):
    parts = video.planes(get_frame(video, frame, cuda))
    image = Image.fromarray(yuv_to_rgb(parts, video.matrix))
    for mask, x, y in hide:
        height, width = mask.shape
        image.paste((16, 36, 48), (x, y, x+width, y+height), Image.fromarray(mask*255))
    if mask_at is not None:
        x, y, w, h = region
        raw = image.crop((x, y, x+w, y+h))
        mask, u, v = mask_at
        alpha = Image.new('L', (w, h))
        alpha.paste(Image.fromarray(mask*108), (u-x, v-y))
        preview = Image.composite(Image.new('RGB', (w, h), (0, 220, 245)), raw, alpha)
        a, b, width, height = outline
        ImageDraw.Draw(preview).rectangle((a-x-2, b-y-2, a-x+width+1, b-y+height+1),
                                          outline=(255, 98, 24), width=2)
        scale = min(3, max(1, 510/max(w, h)))
        size = round(w*scale), round(h*scale)
        w, h = size
        vertical = w >= 2*h
        panel = Image.new('RGB', (w, 2*(h+18)) if vertical else (2*w, h+18), (24, 29, 35))
        pen = ImageDraw.Draw(panel)
        for title, tile, (left, top) in [('Original', raw, (0, 0)),
                                         ('Mask preview', preview, (0, h+18) if vertical else (w, 0))]:
            pen.text((left+4, top+2), title, fill='white')
            panel.paste(tile.resize(size, Image.Resampling.BICUBIC), (left, top+18))
        return f'Frame {frame}, original crop and cyan mask preview', jpeg(panel, max_side)
    if outline is not None:
        x, y, w, h = outline
        ImageDraw.Draw(image).rectangle((x-2, y-2, x+w+1, y+h+1), outline=(255, 98, 24), width=2)
    if region is None:
        context = 'with other verified edit pixels hidden' if hide else 'single full image'
        return f'Frame {frame}, {context}', jpeg(image, max_side)
    return f'Frame {frame}, candidate crop', jpeg(crop_detail(image, region), max_side)


def filled_mask_evidence(video, frame, cuda, max_side, hide, *, region=None):
    """Show the untouched pixels without drawing a readable silhouette of a known overlay."""
    source = yuv_to_rgb(video.planes(get_frame(video, frame, cuda)), video.matrix)
    mask = np.zeros(source.shape[:2], np.uint8)
    for marks, x, y in hide:
        part, portion = clipped_plane(mask, (x, y, marks.shape[1], marks.shape[0]))
        part |= marks[portion]
    preview = cv2.inpaint(source, mask, 3, cv2.INPAINT_TELEA)
    image = Image.fromarray(preview)
    if region is not None:
        return f'Frame {frame}, current crop', jpeg(crop_detail(image, region), max_side)
    return f'Frame {frame}, known overlay pixels preview-filled', jpeg(image, max_side)


def check_backend(settings):
    if not isinstance(settings, dict):
        raise ValueError('grounding_local must specify an endpoint and a model.')
    endpoint, model = settings.get('endpoint'), settings.get('model')
    if not isinstance(endpoint, str) or not isinstance(model, str) or not model.strip():
        raise ValueError('grounding_local must specify an endpoint and a model.')
    url = urlsplit(endpoint)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or url.username or url.password or url.path != '/v1/chat/completions' or url.query or url.fragment):
        raise ValueError('The local grounding endpoint must be a loopback HTTP /v1/chat/completions URL.')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def evidence_identity(settings):
    """Bind cached responses to the installed model and the serving runtime."""
    launch = settings.get('launcher') if settings else None
    if not launch:
        return None
    endpoint = urlsplit(settings['endpoint'])
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(
                f'{endpoint.scheme}://{endpoint.netloc}/props', timeout=3) as response:
            properties = json.load(response)
        if (properties.get('model_alias') != settings['model']
                or Path(properties['model_path']).resolve() != Path(launch['weights']).resolve()
                or not properties.get('build_info')):
            return None
        files = {key: digest(Path(launch[key])) for key in ('executable', 'weights', 'projector')}
    except (OSError, KeyError, TypeError, ValueError):
        return None
    settings_signature = json.dumps({key: properties.get(key) for key in (
        'build_info', 'chat_template', 'default_generation_settings', 'modalities')}, sort_keys=True)
    return {'endpoint': settings['endpoint'], 'model': settings['model'], 'device': launch['device'],
            'model_files': files, 'server_sha256': sha256(settings_signature.encode()).hexdigest()}


class LocalEvidenceCache:
    def __init__(self, path, scope, trace, *, prior_paths=()):
        self.path, self.scope, self.trace = path, scope, trace
        self.entries = {}
        self.imported_paths = []
        self.lock = Lock()
        backend = scope.get('backend')
        source = scope.get('source_sha256')
        self.shared_root = None
        if (isinstance(backend, dict) and backend.get('server_sha256') and backend.get('model_files')
                and isinstance(source, str) and len(source) == 64 and all(c in '0123456789abcdef' for c in source)):
            identity = sha256(json.dumps(scope, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
            self.shared_root = Path(tempfile.gettempdir())/'watermark-local-evidence-v1'/identity
        paths = [*dict.fromkeys(prior for prior in prior_paths if prior != path), path]
        for candidate in paths:
            if not candidate.is_file():
                continue
            try:
                saved = read_object(candidate)
                if saved.get('scope') == scope and isinstance(saved.get('entries'), dict):
                    for key, reply in saved['entries'].items():
                        self.entries.pop(key, None)
                        self.entries[key] = reply
                    self.entries = dict(list(self.entries.items())[-512:])
                    if candidate != path:
                        self.imported_paths.append(candidate)
                        trace('vision.local_evidence_imported', path=str(candidate), requests=len(saved['entries']))
            except (OSError, ValueError) as error:
                trace('vision.local_cache_rejected', path=str(candidate), reason=str(error))

    def lookup(self, key):
        with self.lock:
            return self.entries.get(key)

    def store(self, key, reply):
        with self.lock:
            self.entries[key] = reply
            if len(self.entries) > 512:
                self.entries.pop(next(iter(self.entries)))
            try:
                write_object(self.path, {'scope': self.scope, 'entries': self.entries})
            except OSError as error:
                self.trace('vision.local_cache_write_failed', reason=str(error))


def resolve_launcher(settings, directory):
    launch = settings.get('launcher')
    if launch is None:
        return
    keys = ('executable', 'weights', 'projector', 'device')
    if not isinstance(launch, dict) or any(not isinstance(launch.get(k), str) or not launch[k].strip() for k in keys):
        raise ValueError('The local model launcher needs executable, weights, projector and device.')
    settings['launcher'] = {**launch, **{key: str((directory/Path(launch[key])).resolve()) for key in keys[:3]}}


@contextmanager
def local_service(settings, log, trace):
    launch = settings.get('launcher') if settings else None
    if launch is None:
        yield
        return
    endpoint = urlsplit(settings['endpoint'])
    base = f'{endpoint.scheme}://{endpoint.netloc}'
    opener = build_opener(ProxyHandler({}), NoRedirect())

    def ready():
        try:
            with opener.open(base+'/v1/models', timeout=2) as response:
                models = json.load(response)['data']
        except (URLError, TimeoutError, KeyError, ValueError):
            return False
        if any(row.get('id') == settings['model'] for row in models):
            return True
        raise ValueError('The local port is serving a different vision model.')

    if ready():
        trace('batch.local_vision_reused', model=settings['model'], endpoint=settings['endpoint'])
        yield
        return
    executable, weights, projector = (Path(launch[key]) for key in ('executable', 'weights', 'projector'))
    if not all(path.is_file() for path in (executable, weights, projector)):
        raise ValueError('The local vision executable, weights or projector is missing.')
    args = [str(executable), 'serve', '-m', str(weights), '--mmproj', str(projector), '--alias', settings['model'],
            '--host', endpoint.hostname, '--port', str(endpoint.port or 80), '--device', launch['device'], '-ngl', '999',
            '-c', '8192', '--parallel', str(LOCAL_PARALLEL_REQUESTS), '--image-min-tokens', '1024', '--reasoning-budget', '0',
            '--no-cache-prompt', '--no-webui']
    with log.open('ab') as stream:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=executable.parent, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            trace('batch.local_vision_starting', pid=process.pid, model=settings['model'], log=str(log))
            deadline = monotonic()+75
            while not ready():
                if process.poll() is not None or monotonic() >= deadline:
                    raise RuntimeError(f'The local vision service failed to start. See {log}.')
                sleep(1)
            trace('batch.local_vision_ready', pid=process.pid, model=settings['model'])
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            trace('batch.local_vision_stopped', pid=process.pid, exit_code=process.returncode)


def local_request_body(settings, prompt, image, *, max_tokens, json_mode, schema, reference):
    content = []
    for title, picture in ((reference, image) if reference is not None else (image,)):
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(picture).decode('ascii')
        content.extend(({'type': 'text', 'text': title if content else f'{prompt}\n{title}'},
                        {'type': 'image_url', 'image_url': {'url': image_url}}))
    body = {
        'model': settings['model'], 'temperature': 0, 'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': content}],
    }
    if schema is not None:
        body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'watermark_candidate', 'schema': schema, 'strict': True}}
    elif json_mode:
        body['response_format'] = {'type': 'json_object'}
    return json.dumps(body).encode('utf-8')


def parse_local_reply(result):
    try:
        reply = result['choices'][0]['message']['content'].strip()
        if reply.startswith('```') and reply.endswith('```'):
            reply = reply.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        return json.loads(reply)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('The local model did not return a grounding JSON object.') from error


def wait_for_local_backend(settings, opener, wait_seconds, trace):
    if 'launcher' not in settings:
        return
    endpoint = urlsplit(settings['endpoint'])
    slots = f'{endpoint.scheme}://{endpoint.netloc}/slots'
    started = monotonic()
    heartbeat = started+15
    while True:
        try:
            with opener.open(slots, timeout=2) as response:
                state = json.load(response)
        except (OSError, ValueError):
            # A different local server may omit the managed runtime's status endpoint.
            return
        if (not isinstance(state, list) or not state
                or any(not isinstance(row, dict) or type(row.get('is_processing')) is not bool for row in state)):
            return
        if any(not row['is_processing'] for row in state):
            return
        now = monotonic()
        if now-started >= wait_seconds:
            raise CapacityTimeout('The local backend is still processing earlier requests.')
        if trace and now >= heartbeat:
            trace('vision.local_backend_waiting', seconds=round(now-started, 2))
            heartbeat = now+15
        sleep(.25)


def local_json(settings, prompt, image, timeout, *, max_tokens, json_mode=False, schema=None, reference=None,
               cacheable=None):
    from wm_runtime import bounded_timeout
    timeout = bounded_timeout(timeout)
    check_backend(settings)
    body = local_request_body(settings, prompt, image, max_tokens=max_tokens, json_mode=json_mode,
                              schema=schema, reference=reference)
    cache = settings.get('evidence_cache') if cacheable is not None else None
    shared = None
    if cache is not None:
        key = sha256(body).hexdigest()
        if cache.shared_root is not None:
            shared = cache.shared_root/key
        cached = cache.lookup(key)
        if cached is not None and cacheable(cached):
            if shared is not None and not (shared/'response.json').is_file():
                try:
                    with capacity_slot(shared, 1, 0):
                        if not (shared/'response.json').is_file():
                            write_object(shared/'response.json', {'reply': cached})
                except CapacityTimeout:
                    pass
                except OSError as error:
                    cache.trace('vision.local_shared_cache_write_failed', reason=str(error))
            cache.trace('vision.local_evidence_reused', request_sha256=key)
            return cached

    def shared_reply():
        if shared is None:
            return None
        try:
            reply = read_object(shared/'response.json').get('reply')
        except (OSError, ValueError):
            return None
        if cacheable(reply):
            cache.store(key, reply)
            cache.trace('vision.local_shared_evidence_reused', request_sha256=key)
            return reply
        return None

    cached = shared_reply()
    if cached is not None:
        return cached
    request = Request(settings['endpoint'], data=body,
                      headers={'Content-Type': 'application/json'})
    port = urlsplit(settings['endpoint']).port or 80
    root = Path(tempfile.gettempdir())/'watermark-local-vision-v1'/str(port)
    trace = cache.trace if cache is not None else None
    waiting = (lambda seconds: trace('vision.local_slot_waiting', seconds=round(seconds, 2))) if trace else None
    proposal_window = settings.get('proposal_window')
    source_budget = settings.get('source_admission_budget')
    limits = [limit for limit in (source_budget, settings.get('admission_budget'), proposal_window) if limit is not None]
    admission_started = monotonic()
    budget = bounded_timeout(min([LOCAL_ADMISSION_SECONDS, *(limit['remaining'] for limit in limits)]))
    if not budget:
        if proposal_window is not None:
            proposal_window['deferred'] = True
            raise RuntimeError('The optional local proposal window ended.')
        scope = 'source turn' if source_budget is not None and not source_budget['remaining'] else 'visual review'
        reason = f'The local admission budget for this {scope} was used.'
        if trace:
            trace('vision.local_transport_unavailable', reason=reason)
        raise RuntimeError(reason)
    inference_started = None
    try:
        request_lock = capacity_slot(shared, 1, budget) if shared is not None else nullcontext((None, 0.))
        with request_lock as (_, request_wait):
            cached = shared_reply()
            if cached is not None:
                return cached
            available = max(0., budget-request_wait)
            with capacity_slot(root, LOCAL_PARALLEL_REQUESTS, available, waiting=waiting) as (slot, seconds):
                if trace and seconds >= .2:
                    trace('vision.local_slot_acquired', slot=slot, seconds=round(seconds, 2))
                opener = build_opener(ProxyHandler({}), NoRedirect())
                wait_for_local_backend(settings, opener, max(0., available-seconds), trace)
                inference_started = monotonic()
                if trace:
                    trace('vision.local_request_started', request_sha256=key)
                with opener.open(request, timeout=bounded_timeout(timeout)) as response:
                    result = json.load(response)
                if trace:
                    trace('vision.local_request_finished', request_sha256=key,
                          seconds=round(monotonic()-inference_started, 2))
            reply = parse_local_reply(result)
            if cache is not None and cacheable(reply):
                if shared is not None:
                    try:
                        write_object(shared/'response.json', {'reply': reply})
                    except OSError as error:
                        trace('vision.local_shared_cache_write_failed', reason=str(error))
                cache.store(key, reply)
    except CapacityTimeout as error:
        if proposal_window is not None:
            proposal_window['deferred'] = True
        elif trace:
            trace('vision.local_transport_unavailable', reason=str(error))
        raise RuntimeError('The local vision server is still busy after waiting for a shared slot.') from error
    except HTTPError as error:
        if trace and error.code in (429, 502, 503, 504):
            trace('vision.local_transport_unavailable', reason=f'HTTP {error.code}')
        raise RuntimeError(f'The local vision server returned HTTP {error.code}.') from error
    except (URLError, TimeoutError) as error:
        if trace:
            trace('vision.local_transport_unavailable', reason=str(error))
        raise RuntimeError(f'The local vision server did not respond: {error}.') from error
    finally:
        if limits:
            admission_end = inference_started if inference_started is not None else monotonic()
            elapsed = admission_end-admission_started
            for limit in limits:
                limit['remaining'] = max(0., limit['remaining']-elapsed)
    return reply


def ask_local(settings, prompt, image, timeout, decisions=DECISIONS, *, max_tokens=240, json_mode=False, schema=None,
              reference=None):
    def cacheable(answer):
        return (isinstance(answer, dict) and answer.get('decision') in decisions
                and answer['decision'] not in ('uncertain', 'review')
                and isinstance(answer.get('reason'), str) and bool(answer['reason'].strip())
                and (answer['decision'] != 'watermark' or 'fit' not in answer
                     or answer.get('fit') in ('complete', 'partial')))

    decision = local_json(settings, prompt, image, timeout, max_tokens=max_tokens, json_mode=json_mode, schema=schema,
                          reference=reference, cacheable=cacheable)
    if not isinstance(decision, dict) or decision.get('decision') not in decisions:
        raise ValueError('The local model returned an invalid grounding decision.')
    if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise ValueError('The local model returned an empty grounding explanation.')
    return decision


def classify_candidate(settings, image, timeout):
    prompt = CANDIDATE_PROMPT+FIT_INSTRUCTION+' Set "fit" to "uncertain" for protected or uncertain candidates.'
    try:
        return ask_local(settings, prompt, image, timeout, CANDIDATE_DECISIONS, schema=CANDIDATE_SCHEMA)
    except RuntimeError as error:
        cause = error.__cause__
        if not isinstance(cause, HTTPError) or cause.code not in (400, 422):
            raise
        return ask_local(settings, prompt, image, timeout, CANDIDATE_DECISIONS, json_mode=True)


def preview_residue(settings, image, timeout):
    prompt = ('Inspect this single crop as it appears now. Look only for visible, coherent parts of an overlaid '
              'distribution watermark, such as readable promotional text, an account ID, a URL, or a recognizable '
              'distribution emblem. Do not infer a watermark from blur, a flat fill, scene details, dialogue, or title text. '
              'Set decision visible only if you can quote the visible wording or describe a recognizable emblem, '
              'none if only scene content or fill is visible, and uncertain if a fragment cannot be identified. '
              'For none use an empty label. Return JSON with decision, label, and a brief reason.')
    decisions = ('visible', 'none', 'uncertain')
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['decision', 'label', 'reason'],
              'properties': {'decision': {'enum': list(decisions)}, 'label': {'type': 'string'},
                             'reason': {'type': 'string'}}}
    return ask_local(settings, prompt, image, timeout, decisions, max_tokens=160, schema=schema)


def located_boxes(response):
    rows = response if isinstance(response, list) else [response]
    result = []
    for row in rows[:60]:
        box = row.get('bbox_2d') if isinstance(row, dict) else None
        if (isinstance(box, list) and len(box) == 4 and all(type(v) is int and 0 <= v <= 1000 for v in box)
                and box[0] < box[2] and box[1] < box[3] and box not in result):
            result.append(box)
    return result


def locate_reference_overlay(settings, image, reference, timeout):
    prompt = ('The FIRST image is a reference for a distribution overlay. Inspect only the SECOND image for any '
              'clearly recognizable part of that same overlay. Identify its actual location in the SECOND image. '
              'An inpainting patch, scene object, or generic blur does not establish presence. '
              'Return a JSON object with decision present, absent, or uncertain, bbox_2d, and reason. '
              'If present, give a tight [left,top,right,bottom] rectangle with integer coordinates 0 to 1000 within '
              'the SECOND image, and describe visible strokes or artwork. If absent or uncertain, give []. '
              'Do not use the reference crop itself as evidence that an overlay exists in the SECOND image.')
    answer = ask_local(settings, prompt, image, timeout, ('present', 'absent', 'uncertain'),
                       max_tokens=200, json_mode=True, reference=reference)
    return located_boxes(answer) if answer['decision'] == 'present' else []


SOURCE_LABEL_PROMPT = (
    'Find visible overlaid distribution watermarks in this single source image. Ignore objects and signs '
    'in the scene, dialogue subtitles and editorial text. Draw one tight box per complete watermark '
    'copy, including the icon and adjacent lettering together. Return a JSON array. For each detection '
    'set bbox_2d to four integers in left, top, right, bottom order, normalized from 0 to 1000 within '
    'this image. Set label to the actual visible URL, account, wording or a short description of the '
    'visible emblem. Copy legible characters exactly. Use an icon description when the text cannot '
    'be read. Return [] when no distribution watermark is visible.'
)


def locate_source_overlays(settings, image, timeout, *, read_labels=False):
    prompt = ('Locate any visible overlaid distribution watermark in this single source frame. Ignore scene signs, '
              'dialogue subtitles and editorial text. Include the entire overlay, including the icon and '
              'lettering, in one tight box. Report every clearly distinguishable distribution overlay or copy. '
              'Return JSON [{"bbox_2d":[left,top,right,bottom],"label":"visible wording or icon"}], '
              'coordinates 0 to 1000 relative to this image. Return [] if no visible distribution watermark.')
    if read_labels:
        prompt = SOURCE_LABEL_PROMPT

    def labeled_boxes(reply):
        rows = reply if isinstance(reply, list) else [reply]
        found = []
        for row in rows[:60]:
            boxes = located_boxes(row)
            label = row.get('label') if isinstance(row, dict) else None
            if boxes and isinstance(label, str) and label.strip():
                label = label.strip()[:100]
                found.append(('' if label.casefold() == 'visible wording or icon' else label, boxes[0]))
        return found

    reply = local_json(settings, prompt, image, timeout, max_tokens=2400, cacheable=labeled_boxes)
    return labeled_boxes(reply)


def locate_named(settings, label, image, timeout, *, tight=False):
    if label:
        prompt = f'Detect {label} in the image, output the bbox coordinates in JSON format. The coordinates are in [0,1000].'
        if tight:
            prompt += ' Use a tight box around the whole visible overlay. Use [] if the target is absent or unclear.'
    else:
        prompt = ('Locate any visible overlaid distribution watermark in this original crop. Ignore scene signs, '
                  'dialogue subtitles and editorial text. Include the entire overlay, including the icon and '
                  'lettering, in one tight box. Return JSON [{"bbox_2d":[left,top,right,bottom],'
                  '"label":"visible wording or icon"}], coordinates 0 to 1000 relative to this image. '
                  'Return [] if no visible distribution watermark.')
    return located_boxes(local_json(settings, prompt, image, timeout, max_tokens=2400, cacheable=located_boxes))
