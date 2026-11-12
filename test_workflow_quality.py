import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import tempfile
import json

import cv2
import numpy as np

from wm_quality import parse_reviews, quality_samples
from wm_dynamic import animated_group, segment_overlay


class WholeFrameQualityTests(unittest.TestCase):
    def test_standalone_render_shares_reference_frames_across_templates(self):
        from wm_discovery import pack_tracks
        from wm_discovery_render import layer_masks
        from wm_runtime import current_runtime
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'source.mp4'
            path.write_bytes(b'fixture')
            video = SimpleNamespace(path=path, count=20, width=20, height=20)
            layer = {'positions': np.zeros((20,2),np.int32), 'scores': np.ones(20,np.float32),
                     'present': np.ones(20,bool), 'verified': np.ones(20,bool),
                     'signal': np.zeros((10,10),np.float32), 'weights': np.ones((10,10),np.float32),
                     'mask': np.ones((10,10),np.uint8), 'cuts': np.array([],np.int32)}
            with patch('wm_media.read_frames', side_effect=lambda v,a,b,c: iter([(a,np.zeros(600,np.uint8))])) as decode, \
                 patch('wm_discovery_render.color_patch', return_value=np.zeros((2,10,10),np.float32)):
                result = list(layer_masks(video, pack_tracks([layer,layer]), False, False, None))
            self.assertEqual(decode.call_count, 18)
            self.assertEqual(len(result), 2)
            self.assertIsNone(current_runtime())

    def test_one_uncertain_animation_anchor_does_not_discard_confirmed_frames(self):
        from wm_dynamic import dynamic_tracks
        video = SimpleNamespace(count=95, fps=30, width=120, height=80,
                                planes=lambda raw: (raw, raw, raw), matrix='bt709')
        hints = [{'frame': i, 'rect': [20, 20, 30, 20], 'label': 'brand', 'motion': 'animated'}
                 for i in (30, 45, 60, 75)]
        def pixels(_video, i, _cuda):
            return np.full((80, 120), i, np.uint8)
        def segment(rgb, _bounds):
            return None if rgb[0, 0, 0] == 75 else (20, 20, np.ones((12, 20), np.uint8))
        with patch('wm_dynamic.get_frame', side_effect=pixels), \
             patch('wm_dynamic.read_frames', side_effect=lambda v,a,b,c: ((i,pixels(v,i,c)) for i in range(a,b))), \
             patch('wm_dynamic.yuv_to_rgb', side_effect=lambda planes,matrix: np.repeat(planes[0][...,None],3,axis=2)), \
             patch('wm_dynamic.segment_overlay', side_effect=segment):
            tracks, proof = dynamic_tracks(video, video, hints, {'cuts': np.array([],np.int32)}, False, Mock())
        self.assertIn(75, proof['unresolved_anchors'])
        self.assertTrue({30,45,60}.issubset(proof['frames']))
        self.assertTrue(all(not layer['present'][75] for layer in tracks))

    def test_deferred_quality_review_resumes_despite_incomplete_coverage(self):
        from wm_batch import awaiting_visual_review
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'run.json').write_text(json.dumps({'status': 'complete'}))
            verification = {key: True for key in (
                'same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok')}
            (root/'verification.json').write_text(json.dumps(verification))
            report = {'status': 'failed', 'error': 'Grok review deferred until the remaining sources have had a first turn.',
                      'attempts': [{'directory': str(root), 'local_findings': ['Coverage incomplete']}]}
            self.assertTrue(awaiting_visual_review(report))
            verification['full_decode_ok'] = False
            (root/'verification.json').write_text(json.dumps(verification))
            self.assertFalse(awaiting_visual_review(report))

    def test_opening_and_unedited_intervals_are_always_sampled(self):
        video = SimpleNamespace(count=3600, fps=30)
        active = np.ones(3600, bool)
        active[:120] = False
        frames = quality_samples(video, 10, track={'present': active})
        self.assertTrue({30, 60, 90, 105, 119, 120, 3599}.issubset(frames))
        self.assertGreater(len(frames), 10)

    def test_missing_frames_and_contradictory_verdicts_are_rejected(self):
        video = SimpleNamespace(width=480, height=854)
        row = {'frame': 3, 'status': 'clean', 'marks': []}
        with self.assertRaises(ValueError):
            parse_reviews({'reviews': [row]}, [3, 4], video)
        with self.assertRaises(ValueError):
            parse_reviews({'reviews': [row, row]}, [3, 4], video)
        with self.assertRaises(ValueError):
            parse_reviews({'reviews': [{**row, 'status': 'watermark'}]}, [3], video)

    def test_large_marks_are_reported_but_damage_does_not_become_an_erase_hint(self):
        video = SimpleNamespace(width=480, height=854)
        mark = {'label': 'large distribution logo', 'rect': [0, 200, 1000, 700], 'motion': 'animated'}
        rows = [{'frame': 3, 'status': 'watermark', 'marks': [mark]},
                {'frame': 4, 'status': 'damage', 'marks': [mark]}]
        _, hints = parse_reviews({'reviews': rows}, [3, 4], video)
        self.assertEqual([r['frame'] for r in hints], [3])
        self.assertEqual(hints[0]['rect'], [0, 171, 480, 427])

    def test_dynamic_group_stops_at_abrupt_change_to_small_unrelated_mark(self):
        hints = [{'frame': i, 'rect': [20, 40, 200, 80], 'label': 'brand', 'motion': 'animated'}
                 for i in (30, 45, 60)]
        hints += [{'frame': 75, 'rect': [20, 40, 12, 12], 'label': 'different', 'motion': 'animated'}]
        group = animated_group(hints, 30)
        self.assertEqual([r['frame'] for r in group], [30, 45, 60])

    def test_local_segmentation_excludes_bright_background_outside_logo_band(self):
        image = np.full((300, 400, 3), (35, 45, 65), np.uint8)
        cv2.circle(image, (100, 100), 25, (255, 225, 0), -1)
        cv2.putText(image, 'BRAND', (140, 110), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        cv2.rectangle(image, (180, 205), (210, 230), (255, 245, 200), -1)
        found = segment_overlay(image, (55, 60, 230, 185))
        self.assertIsNotNone(found)
        x, y, mask = found
        full = np.zeros(image.shape[:2], np.uint8)
        full[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
        self.assertTrue(full[100, 100])
        self.assertFalse(full[215, 195])
        self.assertIsNone(segment_overlay(np.zeros_like(image), (50, 60, 100, 50)))

    def test_source_confirmed_overlaps_can_be_unioned_without_erasing_old_tracks(self):
        from wm_batch import join_feedback_tracks, FeedbackTrackOverlap
        def layer():
            return {'positions': np.zeros((3, 2), np.int32), 'scores': np.ones(3, np.float32),
                    'present': np.ones(3, bool), 'verified': np.ones(3, bool),
                    'signal': np.ones((10, 10), np.float32), 'weights': np.ones((10, 10), np.float32),
                    'mask': np.ones((10, 10), np.uint8), 'cuts': np.array([], np.int32)}
        prior, new = layer(), layer()
        with self.assertRaises(FeedbackTrackOverlap):
            join_feedback_tracks(prior, [new], 3)
        merged, first = join_feedback_tracks(prior, [new], 3, source_verified=True)
        self.assertEqual(first, 1)
        self.assertEqual(int(merged['layer_count']), 2)
        new['verified'][1] = False
        with self.assertRaises(FeedbackTrackOverlap):
            join_feedback_tracks(prior, [new], 3, source_verified=True)

    def test_incomplete_coverage_still_triggers_full_frame_review(self):
        from wm_director import review_candidate
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'recipe.json').write_text(json.dumps({'profile': 'profile.json'}))
            (root/'profile.json').write_text(json.dumps({'discovery': {'coverage_review_only': True}}))
            (root/'verification.json').write_text(json.dumps({key: True for key in
                ('same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok')}))
            config = {'model': 'fake', 'reasoning_effort': 'xhigh', 'review_samples': 10,
                      'max_url_p95': .08, 'batch_routing': True}
            record = {}
            result = {'decision': 'review', 'reason': 'visible large mark',
                      'hints': [{'frame': 90, 'label': 'brand', 'rect': [20, 40, 300, 100]}]}
            with patch('wm_director.corner_overlap_regions', return_value=[]), \
                 patch('wm_director.review_inputs', return_value=([90], {})), \
                 patch('wm_director.numeric_findings', return_value=['Coverage incomplete']), \
                 patch('wm_director.overlay_coverage_findings', return_value=[]), \
                 patch('wm_director.Video.open', return_value=SimpleNamespace(path=root/'result.mp4')), \
                 patch('wm_quality.review_output', return_value=result) as review:
                verdict = review_candidate(SimpleNamespace(), root, root/'recipe.json', config, False,
                                           Mock(), record, Mock(), repairs_left=0)
            review.assert_called_once()
            self.assertEqual(verdict['watermark_hint']['frame'], 90)
            self.assertEqual(record['local_findings'], ['Coverage incomplete'])


if __name__ == '__main__':
    unittest.main()
