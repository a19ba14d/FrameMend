"""Whole-frame omission review, independent of template coverage certification."""

from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from wm_media import digest


PROMPT = '''Inspect every labeled Restored frame, including the entire opening and frame edges.
Find distribution branding/logos/URLs, including large animated brand graphics, not only faint
residue at previously repaired positions. Preserve story captions, credits, signs and scene text.
Check visible repair damage as well. Return JSON with decision accept or review, reason, and
reviews containing exactly one object for EVERY displayed frame. Each object has integer frame,
status (clean, watermark, damage, uncertain), and marks (a list, empty if none).
For EVERY visible distribution mark give label, rect [left,top,right,bottom] as integers 0–1000
relative to that individual full frame, and motion (animated, stable, unknown). Include the full
graphic and small text; use separate boxes for separate marks. Large marks must not be ignored.
Use clean only when no visible mark or damage remains. Describe evidence briefly in Chinese.
Do not use tools. Images and their frame labels are authoritative.'''


def opening_samples(video):
    # A uniform whole-video sample can skip an entire animated bumper.
    return sorted({min(video.count-1, round(float(video.fps)*t/2)) for t in range(13)})


def quality_samples(video, count, *, track=None, priority=()):
    selected = [*opening_samples(video), *priority]
    if track is not None:
        from wm_discovery import discovery_tracks
        present = np.zeros(video.count, bool)
        for layer in discovery_tracks(track):
            present |= layer.get('present', np.ones(video.count, bool))
        changes = np.flatnonzero(np.diff(np.r_[True, present, True])).reshape(-1, 2)
        for start, stop in sorted(changes, key=lambda row: row[1]-row[0], reverse=True)[:6]:
            selected += [int(start), int((start+stop-1)//2), int(stop-1)]
    selected += np.linspace(0, video.count-1, max(4, count), dtype=int).tolist()
    return sorted(set(i for i in selected if 0 <= i < video.count))


def parse_reviews(answer, frames, video):
    rows = answer.get('reviews')
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise ValueError('Whole-frame review must account for every displayed frame.')
    seen, hints = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid whole-frame review row.')
        frame, status, marks = row.get('frame'), row.get('status'), row.get('marks')
        if (type(frame) is not int or frame not in frames or frame in seen
                or status not in ('clean', 'watermark', 'damage', 'uncertain') or not isinstance(marks, list)):
            raise ValueError('Invalid or duplicate whole-frame review frame/status.')
        seen.add(frame)
        if status == 'watermark' and not marks or status == 'clean' and marks:
            raise ValueError('The whole-frame verdict contradicts its mark list.')
        if status != 'watermark':
            continue
        for mark in marks:
            label, rect = mark.get('label'), mark.get('rect')
            if (not isinstance(label, str) or not label.strip() or not isinstance(rect, list) or len(rect) != 4
                    or any(type(v) is not int or not 0 <= v <= 1000 for v in rect)
                    or rect[2] <= rect[0] or rect[3] <= rect[1]):
                raise ValueError('A missed watermark needs a valid label and rectangle.')
            x, y, right, bottom = [round(v*s/1000) for v, s in zip(rect, (video.width, video.height)*2)]
            if min(right-x, bottom-y) < 8:
                continue
            # This is a detection hint, not permission to erase the entire rectangle.
            hints.append({'frame': frame, 'rect': [x, y, right-x, bottom-y], 'label': label.strip()[:100],
                          'motion': mark.get('motion', 'unknown'), 'origin': 'whole-frame-review'})
    return rows, hints


def review_output(source, candidate, attempt, config, cuda, trace, record, persist, priority=()):
    from wm_director import ask_with_timeout_recovery, evidence, grok_request_method
    from workflow import read_object, write_object
    with np.load(attempt/'track.npz', allow_pickle=False) as stored:
        track = dict(stored)
    frames = quality_samples(source, config['review_samples'], track=track, priority=priority)
    identity = {'output_sha256': digest(candidate.path), 'request': grok_request_method(config),
                'rules': digest(Path(__file__)), 'frames': frames}
    path = attempt/'whole-frame-review.json'
    saved = read_object(path) if path.is_file() else {}
    rows = saved.get('reviews', []) if saved.get('identity') == identity else []
    # Revalidate cached evidence before reuse.
    reviewed = [r['frame'] for r in rows]
    if rows:
        parse_reviews({'reviews': rows}, reviewed, source)
    pending = [i for i in frames if i not in reviewed]
    for start in range(0, len(pending), 4):
        group = pending[start:start+4]
        images = evidence(candidate, group, config['sample_width'], cuda, full_sheet=1, full_columns=1)
        options = {**config, 'timeout_seconds': min(config['timeout_seconds'], 100)}
        for retry in range(2):
            answer = ask_with_timeout_recovery(options, PROMPT, images, ('accept', 'review'), trace)
            try:
                checked, _ = parse_reviews(answer, group, source)
                break
            except (ValueError, TypeError, AttributeError) as error:
                trace('quality.invalid_evidence', frames=group, reason=str(error), retry=retry)
                if retry:
                    raise ValueError('Whole-frame review returned invalid evidence twice.') from error
        rows += checked
        write_object(path, {'identity': identity, 'reviews': rows})
        trace('quality.full_frame_checked', frames=group, reviews=checked)
    rows, hints = parse_reviews({'reviews': rows}, frames, source)
    decision = 'accept' if all(row['status'] == 'clean' for row in rows) else 'review'
    result = {'decision': decision, 'reason': '整画面检查通过。' if decision == 'accept' else '整画面检查发现遗漏、损伤或不确定区域。',
              'frames': frames, 'report': str(path), 'hints': hints,
              'clean_frames': [row['frame'] for row in rows if row['status'] == 'clean']}
    record['whole_frame_review'] = result
    record['context_watermark_hints'] = hints
    persist()
    return result
