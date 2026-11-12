"""Score Grok or a local visual model on the same labeled single images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median
from threading import Lock
from time import monotonic
from types import SimpleNamespace

from PIL import Image

from wm_director import box, ground_observation, job_trace, load_settings
from wm_grounding import check_backend, jpeg
from workflow import read_object, write_object


def iou(left, right):
    if left is None:
        return 0.
    w = max(0, min(left[2], right[2])-max(left[0], right[0]))
    h = max(0, min(left[3], right[3])-max(left[1], right[1]))
    a = (left[2]-left[0])*(left[3]-left[1])
    b = (right[2]-right[0])*(right[3]-right[1])
    return w*h/(a+b-w*h)


def score(rows):
    positive = [row for row in rows if row['truth'] is not None]
    negative = [row for row in rows if row['truth'] is None]
    answered = [row for row in rows if 'error' not in row]
    return {
        'cases': len(rows), 'positives': len(positive), 'negatives': len(negative),
        'located_iou_50': sum(row['iou'] >= .5 for row in positive),
        'mean_positive_iou': round(mean(row['iou'] for row in positive), 4) if positive else None,
        'false_positive': sum(row['rect'] is not None for row in negative),
        'confirmed_absent': sum(row.get('decision', {}).get('decision') == 'no_mark' for row in negative),
        'errors': len(rows)-len(answered),
        'median_answer_seconds': round(median(row['seconds'] for row in answered), 2) if answered else None,
    }


def run_case(case, manifest_dir, config, log, lock):
    def trace(event, **details):
        with lock:
            log(event, case=case['id'], **details)

    path = manifest_dir/case['image']
    with Image.open(path) as image:
        size = SimpleNamespace(width=image.width, height=image.height)
        picture = path.read_bytes() if image.format == 'JPEG' else jpeg(image.convert('RGB'), max(image.size))
    image = (case['image_label'], picture)
    truth = case['rect_px']
    candidate = case.get('candidate_rect_px')
    row = {'id': case['id'], 'group': case['group'], 'truth': truth, 'rect': None}
    started = monotonic()
    try:
        reply = ground_observation(config, case['target'], image, trace, confirm=candidate is not None)
        row['decision'] = reply
        if reply['decision'] == 'found':
            row['rect'] = candidate if candidate is not None else list(box(reply.get('rect'), size))
    except (OSError, RuntimeError, ValueError) as error:
        row['error'] = str(error)
    row['seconds'] = round(monotonic()-started, 3)
    if truth is not None:
        row['iou'] = round(iou(row['rect'], truth), 4)
    trace('benchmark.case', result=row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--grok-config', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=int, default=105)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.grok_config and not args.endpoint and not args.model:
        config = load_settings(args.grok_config)
        config.pop('grounding_local', None)
        provider, model = 'grok', config['model']
    elif args.endpoint and args.model and not args.grok_config:
        backend = {'endpoint': args.endpoint, 'model': args.model}
        check_backend(backend)
        config = {'grounding_local': backend}
        provider, model = 'local', args.model
    else:
        parser.error('Use --grok-config or both --endpoint and --model.')
    config['timeout_seconds'] = args.timeout
    cases = read_object(args.manifest)['cases']
    args.output.mkdir(parents=True, exist_ok=False)
    trace = job_trace(args.output/'events.jsonl')
    lock = Lock()
    started = monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda case: run_case(case, args.manifest.parent, config, trace, lock), cases))
    groups = sorted({case['group'] for case in results})
    summary = {'provider': provider, 'model': model, 'total_seconds': round(monotonic()-started, 2),
               'overall': score(results),
               'groups': {group: score([row for row in results if row['group'] == group]) for group in groups},
               'results': results}
    write_object(args.output/'result.json', summary)
    print(summary['overall'], flush=True)


if __name__ == '__main__':
    main()
