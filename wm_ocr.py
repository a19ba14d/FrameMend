"""Optional proposal source for text that recurs across independent video samples."""

from collections import defaultdict
from threading import Lock
import unicodedata

import cv2
import numpy as np

from wm_media import get_frame, yuv_to_rgb


class TextProposals:
    def __init__(self):
        from rapidocr import RapidOCR

        self.engine = RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads': 2,
                                       'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                                       'Global.use_cls': False, 'Global.log_level': 'error'})
        self.lock = Lock()

    def read(self, image, minimum):
        with self.lock:
            result = self.engine(image, use_cls=False, text_score=minimum)
        if result.boxes is None:
            return []
        return list(zip(result.boxes, result.txts, result.scores, strict=True))

    def read_pixels(self, image):
        h, w = image.shape[:2]
        larger = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        grey = cv2.cvtColor(larger, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(grey)
        rows = []
        for pixels in (larger, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)):
            for quad, text, _ in self.read(pixels, .35):
                left, top, width, height = cv2.boundingRect(np.asarray(quad, np.float32)/2)
                key = unicodedata.normalize('NFKC', text).strip()
                rows.append((key, [left, top, left+width, top+height]))
        return rows

    def crop(self, video, frame, region, cuda):
        x, y, w, h = region
        planes = video.planes(get_frame(video, frame, cuda))
        image = yuv_to_rgb(planes, video.matrix)[y:y+h, x:x+w, ::-1]
        return [(text, [x+a, y+b, x+c, y+d]) for text, (a, b, c, d) in self.read_pixels(image)]

    def __call__(self, video, frame, region, cuda):
        return self.crop(video, frame, region, cuda)

    def sample(self, video, frames, trace):
        readings = defaultdict(list)
        for i, planes in frames:
            image = yuv_to_rgb(planes, video.matrix)[..., ::-1].copy()
            for quad, text, score in self.read(image, .55):
                key = ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum() or c in '@.')
                if len(key) < 4:
                    continue
                x, y, w, h = cv2.boundingRect(np.asarray(quad, np.float32))
                readings[key].append((i, {'rect': [x, y, x+w, y+h], 'provider': 'local-ocr',
                                          'confidence': float(score)}))
        proposals, labels = defaultdict(list), []
        for key, matches in readings.items():
            samples = sorted({i for i, _ in matches})
            if len(samples) < 3 or samples[-1]-samples[0] < .35*(video.count-1):
                continue
            labels.append({'text': key, 'samples': len(samples)})
            for i, match in matches:
                proposals[i].append(match)
        trace('layers.ocr_sampled', sampled_frames=len(frames), recurrent=labels)
        return {i: sorted(rows, key=lambda row: -row['confidence'])[:30] for i, rows in proposals.items()}
