"""Batch the pixel correlation search without changing how candidates are verified."""

from itertools import islice

import cv2
import numpy as np


def search_padding(motif):
    return max(8, motif.width//6), max(8, motif.height//6)


def search_input(motif, signal):
    px, py = search_padding(motif)
    padded = cv2.copyMakeBorder(signal, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
    if signal.size <= 520000:
        return padded, motif.signal, motif.weights
    size = padded.shape[1]//2, padded.shape[0]//2
    return cv2.resize(padded, size, interpolation=cv2.INTER_AREA), motif.small_signal, motif.small_weights


class GpuPixelSearch:
    def __init__(self):
        import torch

        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 512*1024**2:
            raise RuntimeError('CUDA pixel search has insufficient available memory.')
        self.torch = torch
        self.kernels = {}

    def scores(self, motif, signals):
        torch = self.torch
        images = [search_input(motif, signal) for signal in signals]
        plane, target, weights = images[0]
        h, w = plane.shape
        th, tw = target.shape
        if h < th or w < tw:
            return [None]*len(images)
        shape = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
        if id(motif) not in self.kernels:
            reference = torch.as_tensor(target*weights, device='cuda')
            support = torch.as_tensor(weights, device='cuda')
            self.kernels[id(motif)] = (torch.fft.rfft2(reference, s=shape).conj(),
                                       torch.fft.rfft2(support, s=shape).conj(),
                                       max(1., 4*float(weights.sum())),
                                       max(1., float(np.sum(target*target*weights))))
        reference, support, floor, power = self.kernels[id(motif)]
        pixels = torch.from_numpy(np.stack([image for image, _, _ in images])).to('cuda')
        valid = np.s_[:, :h-th+1, :w-tw+1]
        correlation = torch.fft.irfft2(torch.fft.rfft2(pixels, s=shape)*reference, s=shape)[valid]
        energy = torch.fft.irfft2(torch.fft.rfft2(pixels.square(), s=shape)*support, s=shape)[valid]
        return (correlation/(energy.clamp_min(floor)*power).sqrt()).clamp(-1, 1).cpu().numpy()


def pixel_search_frames(video, frames, motifs, cached, detail, trace, *, gpu=False):
    frames = iter(frames)
    accelerator = None
    count = 0
    if gpu and video.count >= 32:
        try:
            accelerator = GpuPixelSearch()
        except (ImportError, OSError, RuntimeError) as error:
            trace('layers.gpu_pixel_search_fallback', reason=str(error))
    while batch := list(islice(frames, 12 if accelerator is not None else 1)):
        data = [(i, video.planes(raw)) for i, raw in batch]
        signals = [detail(planes[0], 2.2) for _, planes in data]
        maps = [{} for _ in batch]
        if accelerator is not None:
            try:
                for layer, (motif, saved) in enumerate(zip(motifs, cached, strict=True)):
                    pending = [k for k, (i, _) in enumerate(data) if saved is None or i not in saved]
                    if pending:
                        scores = accelerator.scores(motif, [signals[k] for k in pending])
                        for k, matrix in zip(pending, scores, strict=True):
                            if matrix is not None:
                                maps[k][layer] = matrix
                                count += 1
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                trace('layers.gpu_pixel_search_fallback', reason=str(error))
                accelerator = None
        for (i, planes), signal, score_maps in zip(data, signals, maps, strict=True):
            yield i, planes, signal, score_maps
    if count:
        trace('layers.gpu_pixel_search', frame_layer_pairs=count, batch_size=12)
