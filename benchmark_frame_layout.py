"""Compare vision requests with one, two, or four of the same labeled video frames."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import monotonic
from types import SimpleNamespace

from PIL import Image

from benchmark_grounding import iou, score
from wm_director import ask, box, contact_sheets, ground_observation, job_trace, load_settings
from wm_grounding import ask_local, check_backend, jpeg
from workflow import read_object, write_object


DECISIONS = ('track', 'no_mark', 'review')
MODES = {'four': (4, 400), 'two': (2, 400), 'one': (1, 400), 'native': (1, None),
         'direct400': (1, 400), 'direct_native': (1, None)}


def layout_prompt(target):
    return (
        f'Find this watermark in each labeled frame: {json.dumps(target, ensure_ascii=False)}. '
        'Each tile is one frame. Return a tight [left,top,right,bottom] for the entire mark, '
        'integer coordinates 0 to 1000 relative to the pixels of that frame only. '
        'Keep subtitles and scene signs. Omit frames where the watermark is absent. '
        'Return only JSON: {"decision":"track","reason":"brief reason","observations":'
        '[{"frame":12,"rect":[left,top,right,bottom]}]}. '
        'Use no_mark or review with an empty observations list when appropriate.'
    )


def evaluate_request(item, root, config, trace):
    mode, group, frames = item
    width = MODES[mode][1]
    images, sizes, rows = [], {}, {}
    for frame in frames:
        index = frame['frame']
        with Image.open(root/frame['image']) as source:
            sizes[index] = SimpleNamespace(width=source.width, height=source.height)
            picture = source.convert('RGB')
        if width:
            picture.thumbnail((width, width), Image.Resampling.LANCZOS)
        images.append((f'Frame {index}', picture))
        rows[index] = {'id': frame['id'], 'frame': index, 'group': group['id'],
                       'truth': frame['rect_px'], 'rect': None}
    started = monotonic()
    request = {'mode': mode, 'group': group['id'], 'frames': list(rows)}
    try:
        if mode.startswith('direct'):
            label, picture = images[0]
            evidence = (f'{label}, single full image', jpeg(picture, max(picture.size)))
            reply = ground_observation(config, group['target'], evidence, trace)
            observations = ([{'frame': frames[0]['frame'], 'rect': reply.get('rect')}]
                            if reply['decision'] == 'found' else [])
        else:
            evidence = contact_sheets(images, max_columns=2)[0]
            prompt = layout_prompt(group['target'])
            if backend := config.get('grounding_local'):
                reply = ask_local(backend, prompt, evidence, config['timeout_seconds'], decisions=DECISIONS)
            else:
                reply = ask(config, prompt, [evidence], DECISIONS, trace)
            observations = reply.get('observations', []) if reply['decision'] == 'track' else []
        request['reply'] = reply
        if not isinstance(observations, list):
            raise ValueError('The model did not return a list of observations.')
        for observation in observations:
            index = observation.get('frame') if isinstance(observation, dict) else None
            if type(index) is not int or index not in rows or rows[index]['rect'] is not None:
                raise ValueError('The model returned a duplicated or undisplayed frame.')
            rows[index]['rect'] = list(box(observation.get('rect'), sizes[index]))
        for row in rows.values():
            row['decision'] = {'decision': 'found' if row['rect'] else reply['decision']}
    except (OSError, RuntimeError, ValueError) as error:
        request['error'] = str(error)
        for row in rows.values():
            row['error'] = str(error)
            row['rect'] = None
    request['seconds'] = round(monotonic()-started, 3)
    for row in rows.values():
        row['seconds'] = request['seconds']
        if row['truth'] is not None:
            row['iou'] = round(iou(row['rect'], row['truth']), 4)
    trace('layout.request', result=request)
    return request, list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--grok-config', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--group', action='append')
    parser.add_argument('--workers', choices=(1, 2), type=int, default=1)
    parser.add_argument('--timeout', type=int, default=105)
    args = parser.parse_args()
    if args.grok_config and not args.endpoint and not args.model:
        config = load_settings(args.grok_config)
        config.pop('grounding_local', None)
        model = config['model']
    elif args.endpoint and args.model and not args.grok_config:
        backend = {'endpoint': args.endpoint, 'model': args.model}
        check_backend(backend)
        config, model = {'grounding_local': backend}, args.model
    else:
        parser.error('Use --grok-config or both --endpoint and --model.')
    config['timeout_seconds'] = args.timeout
    groups = [g for g in read_object(args.manifest)['groups'] if not args.group or g['id'] in args.group]
    if not groups:
        parser.error('The manifest has no matching groups.')
    args.output.mkdir(parents=True, exist_ok=False)
    log = job_trace(args.output/'events.jsonl')
    lock = Lock()

    def trace(event, **data):
        with lock:
            log(event, **data)

    jobs = [(mode, group, group['frames'][start:start+MODES[mode][0]])
            for group in groups for mode in args.modes
            for start in range(0, len(group['frames']), MODES[mode][0])]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda job: evaluate_request(job, args.manifest.parent, config, trace), jobs))
    summary = {}
    for mode in args.modes:
        requests = [req for req, _ in results if req['mode'] == mode]
        rows = [row for req, data in results if req['mode'] == mode for row in data]
        metrics = score(rows)
        metrics.update(requests=len(requests), request_errors=sum('error' in req for req in requests),
                       negative_frames_without_detection=sum(row['truth'] is None and not row['rect']
                                                             and 'error' not in row for row in rows),
                       median_answer_seconds=round(median(req['seconds'] for req in requests), 2),
                       total_request_seconds=round(sum(req['seconds'] for req in requests), 2))
        summary[mode] = {'metrics': metrics, 'requests': requests, 'frames': rows}
        print(mode, metrics, flush=True)
    write_object(args.output/'result.json', {'model': model, 'manifest': str(args.manifest), 'modes': summary})


if __name__ == '__main__':
    main()
