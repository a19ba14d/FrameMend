"""Train and evaluate a small watermark detector on disjoint synthetic video scenes."""

import argparse
from bisect import bisect_right
from contextlib import nullcontext
from functools import lru_cache
import math
import os
from pathlib import Path
from random import Random
import subprocess
import sys
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from benchmark_grounding import iou
from wm_media import Video, digest, executable
from workflow import read_object, write_object


WIDTH, HEIGHT, COUNT = 640, 360, 120
WINDOWS_FONTS = Path(os.environ.get('WINDIR', r'C:\Windows'))/'Fonts'
TRAIN_WORDS = ('VEX CHANNEL', 'LUMA LIVE', 'BAY STREAM', 'SCENE LAB', 'ORBIT MEDIA',
               'WAVE CLIP', '墨线影像', '留影频道', '行云记录', '光束剧场', 'play.zen.tv', 'tape:031')
VAL_WORDS = ('NIMBUS CLIPS', '远镜放映', 'SHORE RECORDS', 'fox.view')
HELD_OUT = (
    ('arc-tide', 'ARC TIDE beside two overlapping turquoise rings', 'ARC TIDE', 6, .68),
    ('wind-text', 'the overlaid Chinese words 风径影像 with a small yellow icon', '风径影像', 7, .42),
    ('hex-dots', 'an open pale blue hexagon with three dots inside, no text', '', 8, .50),
    ('clean', 'ARC TIDE beside two overlapping turquoise rings', '', None, 0.),
)
SUBTITLES = ('Wait for the next frame.', 'I found the right place.', '请看前面的路标。', '这段保留原有字幕。')
SCENE_SIGNS = ('STUDIO 8', 'HALL 12', 'LOCAL NEWS', 'WAY OUT', '开放展厅')


@lru_cache(maxsize=64)
def font(name, size):
    return ImageFont.truetype(str(WINDOWS_FONTS/name), size)


def make_mark(rng, symbol, words, strength, *, held_out=False):
    size = rng.randint(15, 26) if not held_out else (18 if symbol == 7 else 22)
    face = font('msyh.ttc' if any(ord(c) > 127 for c in words)
                else rng.choice(('arial.ttf', 'calibrib.ttf', 'consola.ttf')), size)
    icon = rng.randint(21, 35) if not held_out else (25 if symbol == 7 else 34)
    if symbol is None:
        icon = 0
    pad = 6
    width = int(face.getlength(words))+(icon+6 if icon else 0)+2*pad
    height = max(icon, size+7)+2*pad
    canvas = Image.new('RGBA', (max(14, width), height))
    draw = ImageDraw.Draw(canvas)
    alpha = round(strength*255)
    rgb = (80, 215, 207) if symbol == 6 else (249, 205, 81) if symbol == 7 else (160, 216, 249) if symbol == 8 else (
        rng.choice(((245, 245, 245), (41, 46, 52), (100, 215, 235), (243, 194, 90))))
    color = (*rgb, alpha)
    if icon:
        x, y, r = pad, (height-icon)//2, icon-1
        bounds = (x, y, x+r, y+r)
        thick = max(2, r//9)
        if symbol == 6:
            draw.ellipse((x, y+3, x+r*.67, y+r*.67+3), outline=color, width=thick)
            draw.ellipse((x+r*.32, y+r*.27, x+r, y+r), outline=color, width=thick)
        elif symbol == 7:
            draw.line([(x+3, y+r-2), (x+r*.42, y+3), (x+r-1, y+r*.62)], fill=color, width=thick)
            draw.ellipse((x+r*.71, y+2, x+r*.95, y+r*.27), fill=color)
        elif symbol == 8:
            pts = [(x+r*(.5+.47*math.cos(j*math.pi/3)), y+r*(.5+.47*math.sin(j*math.pi/3)))
                   for j in range(6)]
            draw.line(pts[:5], fill=color, width=thick)
            for j in (.30, .50, .70):
                draw.ellipse((x+r*j-2, y+r*.5-2, x+r*j+2, y+r*.5+2), fill=color)
        elif symbol == 0:
            draw.ellipse(bounds, outline=color, width=thick)
            draw.line((x+r*.28, y+r*.72, x+r*.72, y+r*.26), fill=color, width=thick)
        elif symbol == 1:
            draw.polygon([(x+r/2, y), (x+r, y+r/2), (x+r/2, y+r), (x, y+r/2)], fill=color)
        elif symbol == 2:
            draw.polygon([(x, y+r), (x+r/2, y), (x+r, y+r)], outline=color, width=thick)
            draw.line((x+r*.25, y+r*.73, x+r*.75, y+r*.73), fill=color, width=thick)
        elif symbol == 3:
            draw.rounded_rectangle(bounds, radius=r//4, outline=color, width=thick)
            draw.arc((x+r*.22, y+r*.22, x+r*.8, y+r*.8), 0, 250, fill=color, width=thick)
        elif symbol == 4:
            draw.line([(x, y+r*.35), (x+r*.45, y+r*.78), (x+r, y+r*.1)], fill=color, width=thick)
            draw.ellipse((x+r*.7, y+r*.65, x+r, y+r*.95), fill=color)
        else:
            draw.ellipse(bounds, fill=color)
            draw.arc((x+r*.25, y+r*.25, x+r*.75, y+r*.75), 35, 280, fill=(25, 30, 35, alpha), width=thick)
    if words:
        text_color = (245, 245, 245, alpha) if held_out else (*rng.choice((rgb, (248, 248, 248))), alpha)
        draw.text((pad+(icon+6 if icon else 0), (height-size)//2), words, font=face,
                  fill=text_color, stroke_width=0 if held_out else rng.choice((0, 0, 1)),
                  stroke_fill=(15, 22, 35, alpha//2))
    mask = canvas.getchannel('A').point(lambda a: 255 if a > 8 else 0)
    return canvas.crop(mask.getbbox())


def add_scene_sign(image, rng):
    draw = ImageDraw.Draw(image)
    message = rng.choice(SCENE_SIGNS)
    face = font('msyh.ttc' if message == '开放展厅' else 'arialbd.ttf', rng.randint(15, 25))
    x, y = rng.randint(32, 405), rng.randint(24, 183)
    width = int(face.getlength(message))+18
    palette = rng.choice((((96, 82, 69), (235, 226, 210)), ((207, 213, 199), (28, 46, 46)),
                          ((42, 67, 89), (228, 233, 239))))
    draw.rectangle((x, y, x+width, y+34), fill=palette[0])
    draw.text((x+9, y+3), message, font=face, fill=palette[1])
    return [x, y, x+width, y+35]


def add_subtitle(image, rng):
    message = rng.choice(SUBTITLES)
    face = font('msyh.ttc' if any(ord(c) > 127 for c in message) else 'arialbd.ttf', 19)
    width = int(face.getlength(message))
    x, y = (WIDTH-width)//2, HEIGHT-rng.randint(28, 41)
    draw = ImageDraw.Draw(image)
    draw.text((x, y), message, font=face, fill='white', stroke_width=2, stroke_fill='black')
    return [x-2, y-2, x+width+3, min(HEIGHT, y+25)]


def background(path, rng):
    with Image.open(path) as source:
        return ImageOps.fit(source.convert('RGB'), (WIDTH, HEIGHT), Image.Resampling.BICUBIC,
                            centering=(rng.uniform(.1, .9), rng.uniform(.1, .9)))


def overlay(image, mark, x, y):
    image.paste(mark, (x, y), mark)
    return [x, y, x+mark.width, y+mark.height]


def training_examples(root, photos, split, words, seed, repeats):
    image_dir, label_dir = root/'images'/split, root/'labels'/split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rng = Random(seed)
    total = 0
    for index, photo in enumerate(photos):
        for sample in range(repeats):
            image = background(photo, rng)
            if rng.random() < .67:
                add_scene_sign(image, rng)
            if rng.random() < .8:
                add_subtitle(image, rng)
            text = ''
            if rng.random() < .74:
                kind = rng.randrange(4)
                mark = make_mark(rng, None if kind == 0 else rng.randrange(6),
                                 '' if kind == 1 else rng.choice(words), rng.uniform(.26, .95))
                x, y = rng.randint(0, WIDTH-mark.width), rng.randint(0, HEIGHT-mark.height)
                a, b, c, d = overlay(image, mark, x, y)
                box = ((a+c)/(2*WIDTH), (b+d)/(2*HEIGHT), (c-a)/WIDTH, (d-b)/HEIGHT)
                text = '0 '+' '.join(f'{v:.6f}' for v in box)+'\n'
            stem = f'{index:03d}-{sample:02d}'
            image.save(image_dir/(stem+'.jpg'), quality=rng.randint(84, 96))
            (label_dir/(stem+'.txt')).write_text(text, encoding='ascii')
            total += 1
    return total


def test_clip(root, spec, photos, number):
    name, label, words, symbol, strength = spec
    rng = Random(28037+number)
    mark = make_mark(rng, symbol, words, strength, held_out=True) if strength else None
    scenes = []
    for path in photos[:4]:
        image = background(path, rng)
        scenes.append((image, add_scene_sign(image, rng)))
    changes = (0, 17, 34, 50, 67, 84, 103)
    zones = ((.08, .13), (.68, .69), (.55, .14), (.15, .57), (.62, .36), (.30, .74), (.24, .27))
    if mark:
        positions = [(round((WIDTH-mark.width)*a), round((HEIGHT-mark.height)*b)) for a, b in zones]
    test = root/'test'
    path = test/(name+'.mp4')
    cmd = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{WIDTH}x{HEIGHT}', '-r', '24', '-i', '-', '-c:v', 'libx264', '-preset', 'fast',
           '-crf', '18', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
           '-color_trc', 'bt709', str(path)]
    rows = []
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        for i in range(COUNT):
            base, sign = scenes[i//30]
            shift = round(4*math.sin(i/6))
            image = ImageChops.offset(base, shift, 0)
            subtitle = add_subtitle(image, Random(91+number+i//18))
            truth = None
            present = mark is not None and not (number == 1 and 51 <= i < 62)
            if present:
                x, y = positions[bisect_right(changes, i)-1]
                truth = overlay(image, mark, x+round(4*math.sin(i/5)), y+round(3*math.cos(i/4)))
            process.stdin.write(image.tobytes())
            rows.append({'frame': i, 'rect_px': truth, 'protected': [[sign[0]+shift, sign[1], sign[2]+shift, sign[3]], subtitle],
                         'reacquisition': bool(present and i > 0 and (i in changes or rows[-1]['rect_px'] is None))})
        process.stdin.close()
        if process.wait():
            raise RuntimeError(f'Encoding failed for {name}.')
    return {'id': name, 'video': str(path.relative_to(root)), 'target': label, 'frames': rows,
            'backgrounds': [p.name for p in photos[:4]]}


def prepare(root, backgrounds):
    photos = sorted(backgrounds.glob('*.jpg'))
    if len(photos) < 112:
        raise ValueError('At least 112 source backgrounds are required for the disjoint splits.')
    Random(4173).shuffle(photos)
    root.mkdir(parents=True, exist_ok=False)
    (root/'test').mkdir()
    train, val, test = photos[:80], photos[80:96], photos[96:112]
    sizes = {'train': training_examples(root, train, 'train', TRAIN_WORDS, 813, 10),
             'val': training_examples(root, val, 'val', VAL_WORDS, 1497, 7)}
    write_object(root/'dataset.yaml', {'path': str(root), 'train': 'images/train', 'val': 'images/val',
                                      'names': ['watermark']})
    clips = [test_clip(root, spec, test[i*4:i*4+4], i) for i, spec in enumerate(HELD_OUT)]
    write_object(root/'manifest.json', {'counts': sizes, 'train_backgrounds': [p.name for p in train],
                                       'val_backgrounds': [p.name for p in val], 'clips': clips})
    print(sizes, 'held-out clips', len(clips), flush=True)


def frame_batches(path, batch):
    decoder = cv2.VideoCapture(str(path))
    try:
        while True:
            frames = []
            for _ in range(batch):
                ok, frame = decoder.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            yield frames
    finally:
        decoder.release()


def predict_video(model, path, count, confidence, *, prediction_lock=None, device=0):
    import torch

    rows = []
    started = monotonic()
    for images in frame_batches(path, 16):
        with prediction_lock if prediction_lock is not None else nullcontext():
            batch = model.predict(images, imgsz=640, batch=16, device=device, half=device != 'cpu',
                                  conf=confidence, max_det=120, verbose=False)
        for result in batch:
            detections = result.boxes.data.detach().cpu().numpy()
            detections = [{'rect': [round(float(x), 2) for x in box[:4]], 'confidence': round(float(box[4]), 4)}
                          for box in detections]
            rows.append({'frame': len(rows), 'detections': detections})
    if device != 'cpu':
        torch.cuda.synchronize()
    elapsed = monotonic()-started
    if len(rows) != count:
        raise ValueError(f'{path.name} decoded {len(rows)} of {count} expected frames.')
    return {'source_sha256': digest(path), 'rows': rows, 'seconds': round(elapsed, 3),
            'decoded_detection_fps': round(len(rows)/elapsed, 1)}


def metrics(rows):
    true_frames = hits = top_hits = false_boxes = protected_boxes = 0
    negative_frames = false_frames = jumps = jumps_hit = 0
    for row in rows:
        truth = row['rect_px']
        detections = row['detections']
        if truth is None:
            negative_frames += 1
            false_frames += bool(detections)
            overlaps = []
        else:
            true_frames += 1
            overlaps = [iou(d['rect'], truth) for d in detections]
            found = any(value >= .5 for value in overlaps)
            hits += found
            top_hits += bool(overlaps and overlaps[0] >= .5)
            if row['reacquisition']:
                jumps += 1
                jumps_hit += found
        accepted = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps and max(overlaps) >= .5 else -1
        for j, item in enumerate(detections):
            if j != accepted:
                false_boxes += 1
                rect = item['rect']
                area = (rect[2]-rect[0])*(rect[3]-rect[1])
                if area > 0 and any(max(0, min(rect[2], sign[2])-max(rect[0], sign[0]))
                                    * max(0, min(rect[3], sign[3])-max(rect[1], sign[1])) >= .25*area
                                    for sign in row['protected']):
                    protected_boxes += 1
    return {'frames': len(rows), 'watermark_frames': true_frames, 'hits_iou_50': hits,
            'top_hits_iou_50': top_hits, 'negative_frames': negative_frames,
            'negative_frames_flagged': false_frames, 'false_boxes': false_boxes,
            'false_boxes_on_subtitles_or_signs': protected_boxes,
            'reacquisition_frames': jumps, 'reacquisition_first_frame_hits': jumps_hit}


def detector(weights, confidence, *, device=0):
    from ultralytics import YOLO

    import torch

    model = YOLO(str(weights))
    warm = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    model.predict([warm]*16, imgsz=640, batch=16, device=device, half=device != 'cpu',
                  conf=confidence, verbose=False)
    if device != 'cpu':
        torch.cuda.synchronize()
    return model


def evaluate(root, weights, confidence):
    import torch

    model = detector(weights, confidence)
    source = read_object(root/'manifest.json')
    clips = []
    for item in source['clips']:
        truth = item['frames']
        clip = predict_video(model, root/item['video'], len(truth), confidence)
        clip.update(id=item['id'], rows=[{**tag, **row} for row, tag in zip(clip['rows'], truth, strict=True)])
        clip['metrics'] = metrics(clip['rows'])
        clips.append(clip)
        print(clip['id'], clip['metrics'], clip['decoded_detection_fps'], flush=True)
    report = {'model': str(weights), 'device': torch.cuda.get_device_name(0), 'confidence': confidence,
              'overall': metrics([row for clip in clips for row in clip['rows']]), 'clips': clips}
    write_object(root/'yolo-result.json', report)
    print('Overall', report['overall'], flush=True)


def train(root, weights, epochs):
    from ultralytics import YOLO, settings

    import torch

    settings.update({'sync': False})
    torch.set_float32_matmul_precision('high')
    model = YOLO(str(weights))
    model.train(data=str(root/'dataset.yaml'), epochs=epochs, imgsz=640, batch=24, device=0,
                workers=0, amp=False, cache='ram', seed=4173, project=str(root), name='training',
                plots=False, mosaic=.25, close_mosaic=3, fliplr=.1, hsv_s=.25)
    return Path(model.trainer.best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'evaluate', 'detect'))
    parser.add_argument('path', type=Path, help='Benchmark directory, or source MP4 for detect.')
    parser.add_argument('--backgrounds', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--dependencies', type=Path)
    parser.add_argument('--report', type=Path, help='New JSON output for detect.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--confidence', type=float, default=.25)
    args = parser.parse_args()
    root = args.path.resolve()
    if args.action == 'prepare':
        if not args.backgrounds:
            parser.error('prepare needs --backgrounds.')
        prepare(root, args.backgrounds.resolve())
        return
    if not args.weights:
        parser.error(f'{args.action} needs --weights.')
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies.resolve()))
    weights = args.weights.resolve(strict=True)
    if args.action == 'detect':
        if not args.report or args.report.suffix.lower() != '.json' or args.report.exists():
            parser.error('detect needs a new JSON path in --report.')
        video = Video.open(root)
        video.check_timestamps()
        report = predict_video(detector(weights, args.confidence), video.path, video.count, args.confidence)
        write_object(args.report, {'model': str(weights), 'confidence': args.confidence, **report})
        print(f'{video.count} frames scanned at {report["decoded_detection_fps"]} fps. Saved {args.report}', flush=True)
        return
    if args.action == 'train':
        weights = train(root, weights, args.epochs)
    evaluate(root, weights, args.confidence)


if __name__ == '__main__':
    main()
