"""Keep source-side location proposals available between automatic attempts."""

from workflow import read_object, write_object


FILENAME = 'refined-proposals.json'
PROVIDERS = {'qwen-visible-remainder', 'qwen-crop', 'qwen-native', 'qwen-dense-replan', 'temporal-pixels'}


def merge_proposals(previous, fresh):
    result = dict(previous)
    for frame, rows in fresh.items():
        unique = {}
        for row in [*rows, *previous.get(frame, ())]:
            unique.setdefault(tuple(row['rect']), row)
        result[frame] = list(unique.values())
    return result


def validated_rows(rows, size):
    count, width, height = size
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        frame, rect, provider = row.get('frame'), row.get('rect'), row.get('provider')
        if (type(frame) is not int or not 0 <= frame < count or not isinstance(provider, str)
                or provider not in PROVIDERS or not isinstance(rect, list)
                or len(rect) != 4 or any(type(n) is not int for n in rect)):
            continue
        left, top, right, bottom = rect
        if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                or min(right-left, bottom-top) < 8 or (right-left)*(bottom-top) > .18*width*height):
            continue
        result.setdefault(frame, []).append({'rect': rect, 'provider': provider})
    return merge_proposals({}, result)


def read_refinements(path, identity, *, size=None):
    try:
        saved = read_object(path)
    except (OSError, ValueError):
        return {}
    recorded, rows = saved.get('source_size'), saved.get('proposals')
    if (saved.get('source_sha256') != identity or saved.get('version') != 1
            or not isinstance(recorded, list) or len(recorded) != 3
            or any(type(n) is not int or n <= 0 for n in recorded)
            or (size is not None and list(size) != recorded) or not isinstance(rows, list)):
        return {}
    return validated_rows(rows, recorded)


def save_refinements(path, identity, size, previous, proposals):
    rows = ({'frame': frame, **row} for frame, items in proposals.items() for row in items)
    new = validated_rows(rows, size)
    updated = merge_proposals(previous, new)
    if updated != previous:
        write_object(path, {'version': 1, 'source_sha256': identity, 'source_size': list(size),
                            'proposals': [{'frame': frame, **row} for frame, items in sorted(updated.items())
                                          for row in items]})
    return updated
