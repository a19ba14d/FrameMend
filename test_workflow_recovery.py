import subprocess
import sys
import time
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wm_overlap import coherent_overlaps, exposed_match, review_samples
from wm_runtime import WorkflowLimit, bounded_timeout, processing_phase, source_runtime


class OverlapRecoveryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(108)
        target = rng.normal(0, 35, (24, 80)).astype(np.float32)
        self.motif = SimpleNamespace(signal=target, weights=np.ones_like(target), width=80, height=24)
        self.image = rng.normal(0, 8, (120, 200)).astype(np.float32)
        self.image[50:74, 40:120] = target
        self.mask = np.ones((34, 60), np.uint8)
        self.image[45:79, 80:140] = rng.normal(0, 80, (34, 60))

    def test_exposed_fragment_recovers_occluded_location(self):
        found = exposed_match(self.motif, self.image, [(40, 50, .32), (10, 30, .35)], [(self.mask, 80, 45)])
        self.assertEqual(found[:2], (40, 50))
        self.assertGreater(found[2], .99)
        self.assertTrue(.4 < found[3] < .7)

    def test_missing_fragment_or_missing_occluder_is_not_accepted(self):
        self.assertIsNone(exposed_match(self.motif, self.image, [(40, 50, .32)], []))
        blank = self.image.copy()
        blank[50:74, 40:80] = 0
        self.assertIsNone(exposed_match(self.motif, blank, [(40, 50, .32)], [(self.mask, 80, 45)]))
        full = np.ones((50, 100), np.uint8)
        self.assertIsNone(exposed_match(self.motif, self.image, [(40, 50, .32)], [(full, 35, 40)]))

    def test_time_support_rejects_isolated_match_jump_and_cut(self):
        group = {i: (40+i, 50, .9, .5) for i in range(3)}
        self.assertEqual(set(coherent_overlaps(group, set())), {0, 1, 2})
        self.assertFalse(coherent_overlaps(group, {1}))
        self.assertFalse(coherent_overlaps({0: group[0]}, set()))
        group[1] = (100, 50, .95, .5)
        self.assertFalse(coherent_overlaps(group, set()))

    def test_risk_samples_cover_both_overlap_intervals(self):
        self.assertEqual(review_samples([10, 11, 12, 20, 21, 22]), [10, 11, 12, 20, 21, 22])

    def test_peak_cutoff_keeps_dense_matches_and_one_rejected_peak(self):
        from wm_discovery import peaks
        scores = np.zeros((100, 100), np.float32)
        for y in range(5, 95, 10):
            for x in range(5, 95, 10):
                scores[y, x] = .9 - (x+y)*.0001
        dense = list(peaks(scores, 9, 9, 61, minimum=.25))
        self.assertEqual(len(dense), 61)
        scores[:] = .1
        scores[50, 50] = .9
        sparse = list(peaks(scores, 9, 9, 61, minimum=.25))
        self.assertEqual(len(sparse), 2)
        self.assertGreater(sparse[0][2], .8)
        self.assertLess(sparse[-1][2], .25)

    def test_local_correlation_matches_original_math(self):
        import cv2
        from wm_discovery import response

        rng = np.random.default_rng(22)
        for h, w in ((16, 32), (40, 251), (128, 264)):
            signal = rng.normal(0, 20, (h, w)).astype(np.float32)
            weights = rng.random((h, w), dtype=np.float32)
            for radius in (0, 1, 2, 3, 5):
                image = rng.normal(0, 20, (h+2*radius, w+2*radius)).astype(np.float32)
                numerator = cv2.matchTemplate(image, signal*weights, cv2.TM_CCORR)
                energy = cv2.matchTemplate(image*image, weights, cv2.TM_CCORR)
                expected = np.clip(numerator/np.sqrt(np.maximum(energy, max(1, 4*float(weights.sum())))
                                                    *max(1, float(np.sum(signal*signal*weights)))), -1, 1)
                np.testing.assert_allclose(response(image, signal, weights), expected, atol=1e-6)


class WorkflowBudgetTests(unittest.TestCase):
    def test_coverage_review_deadline_cannot_restart_with_new_proposals(self):
        with patch('wm_runtime.monotonic', return_value=100.) as clock:
            with source_runtime({'source_budget_seconds': 1000, 'coverage_budget_seconds': 30}) as state:
                state.start_coverage_review()
                clock.return_value = 120.
                state.start_coverage_review()
                self.assertEqual(bounded_timeout(90), 10)
                clock.return_value = 131.
                with self.assertRaisesRegex(WorkflowLimit, 'Coverage review'):
                    bounded_timeout(90)
                with processing_phase(allow_planning_exhausted=True):
                    self.assertGreater(bounded_timeout(1000), 900)

    def test_review_checkpoint_preserves_unknown_frames_and_rejects_unverified_edits(self):
        from wm_discovery import Motif, import_discovery, pack_tracks
        from wm_director import numeric_findings, publish_result, write_discovery_recipe
        from wm_layers import Layer, positive_review_tracks
        from wm_media import digest
        from wm_track import save_npz

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, profile_path, asset = root/'source.mp4', root/'profile.json', root/'track.npz'
            source.write_bytes(b'controlled source fixture')
            profile_path.write_text('{}')
            plane = np.ones((16, 16), np.float32)
            motif = Motif(plane*20, plane, plane.astype(np.uint8))
            layer = Layer(motif, .6, {}, 'known overlay')
            video = SimpleNamespace(path=source, width=64, height=64, count=3)
            evidence = [[([(4, 5, .9)], [], .9), ([], [(5, 5, .55)], .55), ([(6, 5, .9)], [], .9)]]
            tracks = positive_review_tracks(video, [layer], evidence, [])
            np.testing.assert_array_equal(tracks[0]['present'], [True, False, True])
            np.testing.assert_array_equal(tracks[0]['verified'], [True, False, True])
            directory = root/'planned'
            directory.mkdir()
            plan = {}
            recipe = write_discovery_recipe(video, directory, digest(source), pack_tracks(tracks), None,
                                            'known overlay', plan, lambda *a, **k: None,
                                            retained_settings={'coverage_review_only': True})
            self.assertTrue(recipe.is_file())
            self.assertEqual(plan['verified_frames'], 0)
            self.assertEqual(plan['verified_placements'], 2)
            save_npz(asset, **pack_tracks(tracks), source_sha256=digest(source))
            profile = {'discovery': {'track': str(asset), 'sha256': digest(asset)}}
            with self.assertRaisesRegex(ValueError, 'unverified'):
                import_discovery(video, profile, profile_path, root/'strict.npz', False)
            profile['discovery']['coverage_review_only'] = True
            import_discovery(video, profile, profile_path, root/'review.npz', False)
            tracks[0]['present'][1] = True
            save_npz(asset, **pack_tracks(tracks), source_sha256=digest(source), replace=True)
            profile['discovery']['sha256'] = digest(asset)
            with self.assertRaisesRegex(ValueError, 'independently verified'):
                import_discovery(video, profile, profile_path, root/'unsafe.npz', False)
            (root/'verification.json').write_text(json.dumps(dict.fromkeys(
                ['same_video_timing_and_size', 'same_audio_payload_and_timing', 'full_decode_ok'], True)))
            findings = numeric_findings(root, profile, {'discovery_passed': True}, .08)
            self.assertTrue(any('Coverage is incomplete' in row for row in findings))
            with self.assertRaisesRegex(ValueError, 'cannot be published'):
                publish_result(root, root, {'planning': {'coverage_review_only': True}}, lambda *a, **k: None)

    def test_review_fallback_uses_remaining_total_budget_after_planning_expires(self):
        with patch('wm_runtime.monotonic', return_value=100.) as clock:
            with source_runtime({'source_budget_seconds': 100, 'planning_budget_seconds': 10}) as state:
                clock.return_value = 111.
                with processing_phase(allow_planning_exhausted=True):
                    self.assertEqual(bounded_timeout(90), 89)
                self.assertLess(state.planning_remaining, 0)
                clock.return_value = 201.
                with self.assertRaises(WorkflowLimit):
                    with processing_phase(allow_planning_exhausted=True):
                        pass

    def test_pixel_pass_cache_is_invalidated_when_identifier_changes(self):
        from dataclasses import replace
        from wm_discovery import Motif
        from wm_layers import Layer, scan_layers

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.mp4'
            path.write_bytes(b'fixture')
            plane = np.ones((32, 32), np.float32)
            motif = Motif(plane*20, plane, plane.astype(np.uint8))
            layer = Layer(motif, .6, {}, 'one.example')
            video = SimpleNamespace(path=path, width=64, height=64, count=3, fps=30)
            image = np.zeros((64, 64), np.float32)
            events = []
            def frames(*args, **kwargs):
                return iter((i, (image,), image, {}) for i in range(3))
            with source_runtime({}) as state, \
                 patch('wm_layers.color_reference', return_value=[]), \
                 patch('wm_layers.read_frames', return_value=[]), \
                 patch('wm_layers.pixel_search_frames', side_effect=frames), \
                 patch('wm_layers.locations', return_value=[(0, 0, .95)]), \
                 patch('wm_layers.agrees_in_halves', return_value=True):
                trace = lambda event, **data: events.append(event)
                scan_layers(video, [layer], False, trace)
                scan_layers(video, [replace(layer, label='two.example')], False, trace)
                self.assertNotIn('layers.pixel_pass_reused', events)
                self.assertEqual(len(state.pixel_passes), 2)

    def test_frame_cache_isolated_from_mutations_and_invalidated_on_file_change(self):
        from wm_media import get_frame

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.mp4'
            path.write_bytes(b'one')
            video = SimpleNamespace(path=path, width=8, height=8)
            pixels = np.arange(96, dtype=np.uint8)
            with source_runtime({}) as state, patch('wm_media.read_frames', return_value=[(0, pixels)]) as decode:
                first = get_frame(video, 0, False)
                first[:] = 0
                self.assertGreater(get_frame(video, 0, False).sum(), 0)
                self.assertEqual(decode.call_count, 1)
                path.write_bytes(b'changed')
                get_frame(video, 0, False)
                self.assertEqual(decode.call_count, 2)
                self.assertEqual(state.frame_cache_hits, 1)

    def test_concurrent_frame_loads_keep_cache_size_consistent(self):
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context
        from threading import Barrier
        from wm_media import get_frame

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.mp4'
            path.write_bytes(b'fixture')
            video = SimpleNamespace(path=path, width=8, height=8)
            pixels = np.arange(96, dtype=np.uint8)
            barrier = Barrier(2)
            def decode(*args):
                barrier.wait(timeout=5)
                return [(0, pixels)]
            with source_runtime({}) as state, patch('wm_media.read_frames', side_effect=decode):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(copy_context().run, get_frame, video, 0, False) for _ in range(2)]
                    for future in futures:
                        np.testing.assert_array_equal(future.result(), pixels)
                self.assertEqual(len(state.frames), 1)
                self.assertEqual(state.frame_bytes, pixels.nbytes)

    def test_repeated_gap_does_not_count_as_progress(self):
        with source_runtime({}) as state:
            self.assertTrue(state.gap_progress('overlay', [1, 2, 3]))
            self.assertTrue(state.gap_progress('overlay', [1, 2]))
            self.assertFalse(state.gap_progress('overlay', [1, 2]))
            self.assertTrue(state.stalled)

    def test_planning_and_processing_budgets_are_distinct(self):
        with patch('wm_runtime.monotonic', return_value=100.) as clock:
            with source_runtime({'source_budget_seconds': 100, 'planning_budget_seconds': 10}) as state:
                clock.return_value = 105.
                self.assertEqual(bounded_timeout(90), 5)
                with processing_phase():
                    clock.return_value = 150.
                    self.assertEqual(bounded_timeout(90), 50)
                clock.return_value = 151.
                self.assertEqual(bounded_timeout(90), 4)
                clock.return_value = 155.
                with self.assertRaises(WorkflowLimit):
                    state.remaining()

    def test_silent_worker_can_be_stopped_at_budget(self):
        from workflow import stop_worker, worker_lines
        import os

        with source_runtime({'source_budget_seconds': .15, 'planning_budget_seconds': 2}):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                                     stdout=subprocess.PIPE, text=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                     start_new_session=os.name != 'nt')
            started = time.monotonic()
            try:
                with self.assertRaises(WorkflowLimit):
                    list(worker_lines(child))
            finally:
                stop_worker(child)
                child.wait(timeout=5)
                child.stdout.close()
            self.assertLess(time.monotonic()-started, 5)

    def test_review_export_requires_technical_checks_and_does_not_publish(self):
        from wm_director import export_review_candidate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root/'attempt-01'
            attempt.mkdir()
            (attempt/'result.mp4').write_bytes(b'tested candidate fixture')
            report = {'status': 'needs_review', 'attempts': [{'directory': str(attempt),
                       'local_findings': ['Residual check requires review'], 'decision': {'decision': 'review'}}]}
            export_review_candidate(root, report, lambda *args, **kwargs: None)
            self.assertNotIn('review_candidate', report)
            (attempt/'verification.json').write_text(json.dumps({
                'same_video_timing_and_size': True, 'same_audio_payload_and_timing': True, 'full_decode_ok': True}))
            export_review_candidate(root, report, lambda *args, **kwargs: None)
            self.assertEqual(Path(report['review_candidate']).read_bytes(), b'tested candidate fixture')
            self.assertEqual(report['status'], 'needs_review')
            self.assertNotIn('result', report)


if __name__ == '__main__':
    unittest.main()
