import tempfile
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from wm_matte import Matte, fit_matte, logo_rect, phase_offsets, sampling_grid, warp_fields
from wm_media import Video, plane_rects
from wm_temporal import CapsuleRim


def self_test():
    from fractions import Fraction

    if not torch.cuda.is_available():
        raise ValueError('CUDA PyTorch is required.')
    w, h, n = 320, 180, 40
    rw, rh = 96, 44
    video = Video(Path('self-test.mp4'), w, h, Fraction(30), n, 0, '1:1', 'bt709', 'bt709', 'bt709')
    logo = np.zeros((rh, rw), np.uint8)
    cv2.rectangle(logo, (7, 6), (rw-8, rh-7), 80, -1)
    cv2.putText(logo, 'QA', (19, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.83, 255, 2, cv2.LINE_AA)
    a = logo.astype(np.float32) / 255 * 0.55
    profile = {'translucent': {'reference_rect': [30, 24, rw, rh], 'phase_step': 0.25}}
    phases = np.float32([[(i % 8 - 4)/4, ((i * 3) % 8 - 4)/4] for i in range(n)])
    positions = np.int16([[30 + 2*(i % 34), 24 + 2*((i*5) % 31)] for i in range(n)])
    track = dict(positions=positions, precise=positions+phases, source_sha256='synthetic', profile_sha256='synthetic')
    device = torch.device('cuda')
    fields = [torch.from_numpy(np.stack((-a, a*(value - 128)/128))).to(device) for value in (223, 78, 176)]
    grids = [sampling_grid(torch.from_numpy(phase_offsets(profile, track)).to(device), rw, rh, s) for s in (1, 2, 2)]
    shifted = [warp_fields(field, grid).cpu().numpy() for field, grid in zip(fields, grids, strict=True)]
    rng = np.random.default_rng(459)
    originals, marked, rows = [], [], []
    for i in range(n):
        clean = np.empty(w*h*3//2, dtype=np.uint8)
        for j, p in enumerate(video.planes(clean)):
            yy, xx = np.mgrid[:p.shape[0], :p.shape[1]]
            background = 127 + 72*np.sin(xx*0.034 + yy*0.021 + i*0.55 + j*0.8)
            p[:] = np.rint(background + rng.normal(0, 2, p.shape)).astype(np.uint8)
        frame = clean.copy()
        rect = logo_rect(profile, track, i)
        original = plane_rects(video.planes(clean), rect)
        changed = plane_rects(video.planes(frame), rect)
        for src, dst, model in zip(original, changed, shifted, strict=True):
            d, k = model[i]
            dst[:] = np.rint((1+d)*(src.astype(np.float32)-128) + 128*(1+k)).astype(np.uint8)
        rows.append((tuple(p.copy() for p in changed), original, tuple(np.ones_like(p) for p in original)))
        originals.append(clean)
        marked.append(frame)

    with tempfile.TemporaryDirectory(prefix='watermark-qa-') as root:
        path = Path(root) / 'matte.npz'
        fit_matte(rows[:32], phases[:32], profile, track, path, rw, rh)
        matte = Matte(profile, track, path)
        batch = [f.copy() for f in marked[32:]]
        matte.restore(video, batch, 32)
        marked_mae, restored_mae = [], []
        for i, result in enumerate(batch, 32):
            rect = logo_rect(profile, track, i)
            clean = plane_rects(video.planes(originals[i]), rect)
            before = plane_rects(video.planes(marked[i]), rect)
            after = plane_rects(video.planes(result), rect)
            marked_mae.extend(np.abs(a.astype(np.int16)-b).mean() for a, b in zip(clean, before, strict=True))
            restored_mae.extend(np.abs(a.astype(np.int16)-b).mean() for a, b in zip(clean, after, strict=True))
            outside = result.copy()
            for dst, ref in zip(plane_rects(video.planes(outside), rect), clean, strict=True):
                dst[:] = ref
            if not np.array_equal(outside, originals[i]):
                raise ValueError('Pixels outside the restoration ROI were changed.')
        before, after = float(np.mean(marked_mae)), float(np.mean(restored_mae))
        if after > 1.8 or after > before * 0.18:
            raise ValueError(f'Synthetic calibration did not meet the error target: {before:.3f} -> {after:.3f}.')
        print(f'Synthetic held-out ROI MAE {before:.3f} -> {after:.3f} on {torch.cuda.get_device_name(0)}.')
        print('All synthetic samples matched the clean source outside the ROI before encoding.')
        test_encoder_colors(video, Path(root))
        test_mux_offset(video, Path(root))
    test_capsule_rim()
    test_caption_references()
    test_caption_context()
    test_caption_patch_guard()
    test_caption_layer()
    test_stripe_retouch()
    test_aligned_edge()
    test_corner_grid()
    test_stationary_white()
    test_white_alignment()
    test_discovery_jumps()
    test_sparse_discovery_sampling()
    test_layer_alignment()
    test_motif_checkpoint_partial_reuse()
    test_complementary_motif_checkpoints()
    test_layer_pixel_seed()
    test_local_library_without_detector()
    test_local_boundary_proposals()
    test_residual_proposal_bounds()
    test_dense_replan_sampling()
    test_mask_replan_after_coverage_gap()
    test_layer_completion()
    test_sampled_layer_completion()
    test_translucent_mask_completion()
    test_chroma_mask_completion()
    test_layer_review_anchors()
    test_local_layer_expansion_review()
    test_registered_identifier_proposals()
    test_weak_match_replan()
    test_alternate_identifier_review()
    test_layer_temporal_support()
    test_clipped_layer_motion()
    test_regional_match_and_residual()
    test_moving_regional_scan()
    test_variant_track_coverage()
    test_fading_match_coverage()
    test_global_chroma_gap()
    test_identifier_gate()
    test_dense_copy_tracking()
    test_retained_layer_capacity()
    test_verified_stationary_mask_review()
    test_variant_mask_review()
    test_absence_budget()
    test_masked_proof_resume()
    test_prior_absence_checkpoint_scope()
    test_layer_fragment_coverage()
    test_matching_templates_keep_both_masks()
    test_discovery_composite()
    test_clipped_render_windows()
    test_discovery_context_review()
    test_pending_context_resume()
    test_multiple_context_watermark_hints()
    test_mislocalized_target_recheck()
    test_feedback_track_join()
    test_feedback_observed_locations()
    test_independent_comparisons()
    test_comparison_proof_resume()
    test_grok_api_transport()
    test_grok_prompt_recovery()
    test_shared_grok_slots()
    test_detail_retry_mask()
    test_luma_extent_phase()
    test_registered_extent_geometry()
    test_luma_reference_conditioning()
    test_chroma_extent_recipe()
    test_repeated_edge_residue()
    test_candidate_review_resume()
    test_batch_prior_evidence()
    test_failed_render_reuse()
    test_batch_learning_release_integrity()
    test_batch_evidence_requeue()
    test_batch_resume_scheduler()
    test_expanded_workflow_limits()
    test_batch_resume_input_isolation()
    test_pending_visual_review_routing()
    test_layer_review_cache()
    test_review_rules_stay_bound_to_loaded_worker()
    test_grok_visibility_cache()
    test_lone_local_visibility_conflict()
    test_local_vision_evidence_cache()
    test_local_admission_before_inference()
    test_shared_local_admission_budget()
    test_optional_local_proposals_keep_exact_cache()
    test_evidence_frame_cache()
    test_director_local_cache_logging()


def test_expanded_workflow_limits():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import restore_scheduler, run_batch
    from wm_director import load_settings, make_known_recipe, profile_from_plan
    from wm_discovery import LUMA_EXTENT_GROUPS, discovery_tracks, pack_tracks
    from wm_discovery_render import luma_extent_proofs
    from wm_grounding import located_boxes
    from wm_track import save_npz
    from workflow import build_plan, read_object, write_object

    trace = lambda *_args, **_kwargs: None
    with tempfile.TemporaryDirectory(prefix='wm-expanded-limits-') as name:
        root = Path(name)
        settings = read_object(Path(__file__).parent/'profiles'/'local-grok-director.json')
        settings['discovery_local'].pop('launcher')
        settings.update(max_repairs=30, evidence_retries=30, local_proposal_frames=120,
                        scan_samples=240, review_samples=240,
                        review_regions=[[i, 0, 0, 16, 16] for i in range(80)])
        config_path = root/'config.json'
        write_object(config_path, settings)
        config = load_settings(config_path)
        assert config['max_repairs'] == 30 and len(config['review_regions']) == 80
        boxes = [{'bbox_2d': [i*10, 10, i*10+8, 30]} for i in range(61)]
        assert len(located_boxes(boxes)) == 60

        video = SimpleNamespace(path=root/'source.mp4', count=6, width=1000, height=1000, fps=30,
                                check_timestamps=lambda: None)
        video.path.write_bytes(b'fixture')
        signal = np.full((12, 20), 8., np.float32)
        track = {'positions': np.full((video.count, 2), 20, np.int32),
                 'scores': np.ones(video.count, np.float32), 'present': np.ones(video.count, bool),
                 'verified': np.ones(video.count, bool), 'signal': signal,
                 'weights': np.ones_like(signal), 'mask': np.ones(signal.shape, np.uint8)}
        asset = root/'tracks.npz'
        save_npz(asset, **pack_tracks([track]*360))
        with np.load(asset, allow_pickle=False) as saved:
            assert len(discovery_tracks(dict(saved))) == 360
        assert discovery_tracks(track)[0] is track
        with np.testing.assert_raises(ValueError):
            pack_tracks([track]*361)
        extent_values = (np.asarray(0, np.int32), track['mask'], signal, track['weights'],
                         track['present'], np.arange(video.count, dtype=np.int32))
        expanded = dict(track)
        for fields in LUMA_EXTENT_GROUPS:
            expanded.update(zip(fields, extent_values, strict=True))
        assert len(luma_extent_proofs(video, expanded)) == 30

        moving = [{'frame': 0, 'rect': [40+40*i, 40, 60+40*i, 80], 'shape': 'graphic',
                   'calibration': 'visibility', 'label': f'mark {i}',
                   'observations': [{'frame': frame, 'rect': [40+40*i, 40, 60+40*i, 80]}
                                    for frame in (0, 2, 5)]} for i in range(10)]
        single = profile_from_plan(video, {'moving': moving[0]}, [0, 2, 5])
        assert single == profile_from_plan(video, {'moving': moving[:1]}, [0, 2, 5])
        config.update(detect_known_layout=False)
        decision = {'decision': 'process', 'reason': 'fixture', 'moving': moving}
        report = {}
        with (patch('wm_director.ask', return_value=decision),
              patch('wm_director.evidence', return_value=[]),
              patch('wm_discovery.enroll', side_effect=lambda *_a, **_k: (dict(track), None)) as enroll,
              patch('wm_discovery.recover_chroma'), patch('wm_director.verify_discovery_presence')):
            recipe = make_known_recipe(video, root, config, False, report, trace)
        assert enroll.call_count == 10 and report['planning']['moving_designs'] == 10
        profile = read_object(recipe.parent/'profile.json')
        assert profile['discovery']['layer_count'] == 10
        build_plan(video.path, root/'attempt', recipe, True)

        source = root/'input'
        source.mkdir()
        (source/'one.mp4').write_bytes(b'input')
        batch = root/'batch'
        calls = []

        def unavailable(item, *_args, **_kwargs):
            calls.append(item['index'])
            if len(calls) in (4, 8):
                raise KeyboardInterrupt
            return {**item, 'status': 'needs_review', 'transient_local_vision': True}

        with patch('wm_batch.run_source', side_effect=unavailable):
            for resume in (False, True):
                with np.testing.assert_raises(KeyboardInterrupt):
                    run_batch(source, batch, {'model': 'test', 'reasoning_effort': 'high'},
                              None, False, 1, 1, resume, True, evidence_retries=30)
                saved = read_object(batch/'batch.json')
                expected = 3 if not resume else 6
                assert saved['scheduler']['transport_retries'] == {'1': expected}
                del saved['scheduler']
                write_object(batch/'batch.json', saved)
            run_batch(source, batch, {'model': 'test', 'reasoning_effort': 'high'},
                      None, False, 1, 1, True, True, evidence_retries=30)
        saved = read_object(batch/'batch.json')
        assert len(calls) == 13 and saved['scheduler']['transport_retries'] == {'1': 10}
        legacy = {'scheduler': {'running': [1], 'waiting': [], 'evidence_retries': {'1': 1},
                                'transport_retried': [1]}}
        assert restore_scheduler(legacy, batch/'batch-events.jsonl')['transport_retries'] == {'1': 1}
    print('Expanded capacities load, store and render-plan correctly, including legacy and repeated-interruption resumes.')


def test_batch_evidence_requeue():
    import json
    from unittest.mock import patch

    from wm_batch import overlay_regions, run_batch
    from wm_media import digest
    from wm_refinements import FILENAME, read_refinements, save_refinements
    from workflow import read_object, write_object

    config = {'model': 'test', 'reasoning_effort': 'xhigh'}
    calls = []

    def run(item, root, *_args, **_kw):
        index = item['index']
        calls.append(index)
        directory = root/'jobs'/f'{index:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        identity = item['source_sha256'] if index == 1 else 'wrong source'
        write_object(directory/'absence-evidence.json', {'source_sha256': identity,
                     'proofs': {'mask': {'frame': [7]}}})
        reason = 'Grok timed out during the API visual request.' if index == 3 else 'fixture'
        return {**item, 'status': 'needs_review', 'reason': reason}

    with tempfile.TemporaryDirectory(prefix='wm-evidence-requeue-') as directory:
        source, root = Path(directory)/'input', Path(directory)/'batch'
        source.mkdir()
        (source/'a.mp4').write_bytes(b'a')
        (source/'b.mp4').write_bytes(b'b')
        (source/'c.mp4').write_bytes(b'c')
        with patch('wm_batch.run_source', side_effect=run):
            assert run_batch(source, root, config, None, False, 1, 1, False, False, evidence_retries=2) == 2
        events = [json.loads(line) for line in (root/'batch-events.jsonl').read_text('utf-8').splitlines()]
        assert calls == [1, 2, 3, 1, 3, 3]
        retries = [row for row in events if row['event'] == 'batch.source_requeued']
        assert [(row['index'], row['transient_vision_error']) for row in retries] == [(1, False), (3, True), (3, True)]

        calls.clear()
        proposal = {1: [{'rect': [10, 12, 30, 24], 'provider': 'qwen-visible-remainder'}]}
        size = (4, 80, 60)

        def record_proposal(item, batch, *_args, **_kw):
            index = item['index']
            calls.append(index)
            job = batch/'jobs'/f'{index:03d}'
            job.mkdir(parents=True, exist_ok=True)
            path = job/FILENAME
            identity = item['source_sha256']
            previous = read_refinements(path, identity, size=size)
            if index == 1 and calls.count(index) == 2:
                assert previous == proposal
                assert not read_refinements(path, identity, size=(4, 81, 60))
            save_refinements(path, identity if index == 1 else 'wrong source', size, previous, proposal)
            return {**item, 'status': 'needs_review', 'reason': 'fixture'}

        root = root.with_name('refinement-batch')
        with patch('wm_batch.run_source', side_effect=record_proposal):
            assert run_batch(source, root, config, None, False, 1, 1, False, False, evidence_retries=2) == 2
        events = [json.loads(line) for line in (root/'batch-events.jsonl').read_text('utf-8').splitlines()]
        retries = [row for row in events if row['event'] == 'batch.source_requeued']
        assert calls == [1, 2, 3, 1]
        assert [(row['index'], row['new_regions'], row['new_proofs'], row['new_refinements']) for row in retries] == [
            (1, 0, 0, 1)]

        calls.clear()

        def yield_first(item, *_args, **_kw):
            index = item['index']
            if not calls:
                calls.append(index)
                return {**item, 'status': 'needs_review', 'reason': 'Grok review deferred.', 'visual_deferred': True}
            calls.append(index)
            return {**item, 'status': 'complete'}

        root = root.with_name('yield-batch')
        with patch('wm_batch.run_source', side_effect=yield_first):
            assert run_batch(source, root, config, None, False, 1, 1, False, False) == 0
        events = [json.loads(line) for line in (root/'batch-events.jsonl').read_text('utf-8').splitlines()]
        assert calls == [1, 2, 3, 1]
        assert [row['index'] for row in events if row['event'] == 'batch.source_yielded'] == [1]

        calls.clear()
        backend = {'model': 'local', 'endpoint': 'http://127.0.0.1:18082/v1/chat/completions'}
        local_config = {**config, 'discovery_local': backend, 'grounding_local': backend}

        def local_first_turn(item, _root, source_config, *_args, **_kw):
            index = item['index']
            calls.append(index)
            discovery = source_config['discovery_local'].get('source_admission_budget')
            assert discovery is source_config['grounding_local'].get('source_admission_budget')
            if calls.count(index) == 1:
                assert discovery and discovery['remaining'] > 0
                discovery['remaining'] = 0.
                if index == 1:
                    return {**item, 'status': 'needs_review', 'transient_local_vision': True}
            else:
                assert discovery is None
            return {**item, 'status': 'complete'}

        root = root.with_name('local-admission-batch')
        identity = {**backend, 'server_sha256': 'fixture-server'}
        with (patch('wm_batch.run_source', side_effect=local_first_turn),
              patch('wm_batch.evidence_identity', return_value=identity)):
            assert run_batch(source, root, local_config, None, False, 1, 1, False, False, evidence_retries=1) == 0
        assert calls == [1, 2, 3, 1] and 'source_admission_budget' not in backend

        calls.clear()
        prior = root.with_name('peer-batch')
        peer = prior/'jobs/001'
        peer.mkdir(parents=True)
        sample = source/'a.mp4'
        identity = digest(sample)
        write_object(prior/'manifest.json', {'sources': [{'index': 1, 'source': str(sample), 'source_sha256': identity}]})
        regions = [[5, 2, 3, 12, 10]]

        def peer_finishes(item, batch, *_args, **_kw):
            calls.append(item['index'])
            if item['index'] == 1 and calls.count(1) == 1:
                write_object(peer/'overlay-evidence.json', {'source_sha256': identity, 'regions': regions})
            elif item['index'] == 1:
                assert overlay_regions(batch/'jobs/001', identity) == regions
            return {**item, 'status': 'complete', 'result': 'fixture', 'result_sha256': 'same-result'}

        root = root.with_name('late-evidence-batch')
        with (patch('wm_batch.run_source', side_effect=peer_finishes),
              patch('wm_batch.completed_route', return_value=None) as reviewed):
            assert run_batch(source, root, config, None, False, 1, 1, False, False,
                             prior_runs=[prior], evidence_retries=1) == 0
            assert reviewed.call_count == 1 and reviewed.call_args.args[2] == regions
        events = [json.loads(line) for line in (root/'batch-events.jsonl').read_text('utf-8').splitlines()]
        assert calls == [1, 2, 3, 1] and read_object(root/'batch.json')['counts'] == {'complete': 3}
        assert [row['index'] for row in events if row['event'] == 'batch.acceptance_withheld'] == [1]
    print('Batch retry follows new same-source evidence and keeps temporary failures within the retry budget.')


def test_batch_resume_scheduler():
    from unittest.mock import patch

    from wm_batch import run_batch
    from workflow import read_object, write_object

    config = {'model': 'test', 'reasoning_effort': 'xhigh'}
    calls = []

    def process(item, *_args, **_kwargs):
        index = item['index']
        calls.append(index)
        if index == 2 and calls.count(2) == 1:
            raise KeyboardInterrupt
        if index == 1:
            return {**item, 'status': 'needs_review', 'transient_local_vision': True}
        return {**item, 'status': 'complete'}

    with tempfile.TemporaryDirectory(prefix='wm-scheduler-resume-') as name:
        directory = Path(name)
        source = directory/'input'
        source.mkdir()
        for index in range(1, 4):
            (source/f'{index}.mp4').write_bytes(bytes([index]))
        for legacy in (False, True):
            root = directory/f'batch-{legacy}'
            calls.clear()
            with patch('wm_batch.run_source', side_effect=process):
                with np.testing.assert_raises(KeyboardInterrupt):
                    run_batch(source, root, config, None, False, 1, 1, False, False, evidence_retries=2)
                state = read_object(root/'batch.json')
                assert state['scheduler'] == {'running': [2], 'waiting': [3, 1], 'evidence_retries': {'1': 1},
                                              'transport_retried': [1], 'transport_retries': {'1': 1}}
                if legacy:
                    del state['scheduler']
                    write_object(root/'batch.json', state)
                assert run_batch(source, root, config, None, False, 1, 1, True, True, evidence_retries=2) == 2
            assert calls == [1, 2, 2, 3, 1, 1]
            assert read_object(root/'batch.json')['counts'] == {'complete': 2, 'needs_review': 1}
    print('Interrupted batches restore queue order and preserve consumed transport retry counts.')


def test_batch_resume_input_isolation():
    from unittest.mock import patch

    from wm_batch import run_batch
    from wm_media import digest
    from workflow import read_object

    config = {'model': 'test', 'reasoning_effort': 'xhigh'}
    calls = []

    def process(item, root, *_args, **_kwargs):
        index = item['index']
        calls.append(index)
        if index == 2:
            raise KeyboardInterrupt
        result = root/f'result-{index}.mp4'
        result.write_bytes(b'accepted')
        return {**item, 'status': 'complete', 'result': str(result), 'result_sha256': digest(result)}

    with tempfile.TemporaryDirectory(prefix='wm-batch-input-') as name:
        directory = Path(name)
        source, root = directory/'input', directory/'batch'
        source.mkdir()
        for index in range(1, 5):
            (source/f'{index}.mp4').write_bytes(bytes([index]))
        with patch('wm_batch.run_source', side_effect=process):
            with np.testing.assert_raises(KeyboardInterrupt):
                run_batch(source, root, config, None, False, 1, 1, False, False)
            manifest = (root/'manifest.json').read_bytes()
            (source/'2.mp4').unlink()
            (source/'3.mp4').write_bytes(b'replaced')
            (source/'0.mp4').write_bytes(b'added after intake')
            (root/'result-1.mp4').write_bytes(b'replaced result')
            assert run_batch(source, root, config, None, False, 1, 1, True, False) == 2
        state = read_object(root/'batch.json')
        assert calls == [1, 2, 4] and (root/'manifest.json').read_bytes() == manifest
        assert [row.get('failure_stage') for row in state['jobs']] == ['release_validation', 'intake', 'intake', None]
        assert [row['index'] for row in state['jobs'] if row['status'] == 'complete'] == [4]
        assert state['metrics']['failed_rate'] == .75 and state['metrics']['timed_sources'] == 2
        assert state['jobs'][3]['timing']['attempts'] == 1
    print('Resume keeps the original manifest and isolates missing inputs, replacements and changed releases.')


def test_pending_visual_review_routing():
    from threading import BoundedSemaphore
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from wm_batch import run_source
    from wm_media import digest
    from workflow import write_object

    with tempfile.TemporaryDirectory(prefix='wm-pending-review-') as name:
        root = Path(name)
        source = root/'source.mp4'
        source.write_bytes(b'source')
        item = {'index': 1, 'source': str(source), 'source_sha256': digest(source)}
        attempt = root/'candidate'
        attempt.mkdir()
        write_object(attempt/'run.json', {'status': 'complete'})
        last = {'directory': str(attempt), 'local_findings': []}
        report = {'status': 'failed', 'error': 'Grok timed out during the API visual request.', 'attempts': [last]}
        local = Mock()
        video = SimpleNamespace(check_timestamps=lambda: None)
        with (patch('wm_batch.Video.open', return_value=video),
              patch('wm_batch.resume_candidate_review', return_value=report),
              patch('wm_batch.run_route', return_value={'status': 'needs_review'}) as route):
            result = run_source(item, root, {}, local, False, BoundedSemaphore(1))
            assert result['status'] == 'failed' and result['reason'] == report['error']
            local.recipe.assert_not_called()
            route.assert_not_called()
            last['local_findings'] = ['Repeated watermark edge remains.']
            assert run_source(item, root, {}, local, False, BoundedSemaphore(1))['status'] == 'needs_review'
            local.recipe.assert_called_once()
            assert [call.args[2] for call in route.call_args_list] == ['reference', 'adaptive']
    print('A verified candidate survives a Grok transport failure while an actual quality failure can change strategy.')


def test_layer_review_cache():
    from unittest.mock import Mock, patch

    from wm_batch import cached_layer_reviewer, motif_review_method
    from workflow import read_object, write_object

    config = {'model': 'test', 'reasoning_effort': 'xhigh', 'discovery_local': {'model': 'local'},
              'absence_source_sha256': 'original'}
    method = motif_review_method(config)
    proof = {'decision': 'watermark', 'fit': 'complete', 'label': 'verified mark', 'reason': 'visible'}
    uncertain = {'decision': 'uncertain', 'reason': 'timeout'}
    evidence = [(1, (0, 0, 40, 40), (5, 5, 20, 20), ('mask preview', b'original mask'))]
    changed = [(*evidence[0][:3], ('mask preview', b'changed mask'))]
    trace = Mock()
    with tempfile.TemporaryDirectory(prefix='wm-layer-review-') as directory:
        path = Path(directory)/'reviews.json'
        with (patch('wm_batch.motif_review_method', side_effect=lambda settings: {
                   **method, 'effort': settings['reasoning_effort']}),
              patch('wm_batch.classify_layer', side_effect=[proof, uncertain, proof, proof, proof]) as provider):
            review = cached_layer_reviewer(None, config, False, trace, path)
            assert review(evidence) == proof
            resumed = cached_layer_reviewer(None, config, False, trace, path)
            assert resumed(evidence) == proof
            inherited = cached_layer_reviewer(None, config, False, trace, path.with_name('new.json'), prior_paths=(path,))
            assert inherited(evidence) == proof
            peer = path.with_name('peer.json')
            ongoing = cached_layer_reviewer(None, config, False, trace, path.with_name('ongoing.json'),
                                            prior_paths=(peer, path))
            assert ongoing(evidence) == proof
            earlier = read_object(path)
            key = next(iter(earlier['entries']))
            partial = {**proof, 'fit': 'partial', 'label': 'example.com'}
            write_object(peer, {**earlier, 'entries': {key: partial}})
            assert ongoing(evidence) == partial
            write_object(peer, {**earlier, 'entries': {key: {**partial, 'decision': 'protected'}}})
            assert ongoing(evidence)['decision'] == 'uncertain'
            write_object(peer, {**earlier, 'source_sha256': 'different', 'entries': {key: partial}})
            assert ongoing(evidence) == proof
            assert provider.call_count == 1
            assert resumed(changed) == uncertain
            assert resumed(changed) == proof
            altered = cached_layer_reviewer(None, {**config, 'reasoning_effort': 'high'}, False, trace, path)
            assert altered(evidence) == proof
            assert provider.call_count == 4
            different = cached_layer_reviewer(None, {**config, 'absence_source_sha256': 'another'}, False, trace,
                                              path.with_name('another.json'), prior_paths=(path,))
            assert different(evidence) == proof and provider.call_count == 5

        unclear = {'decision': 'uncertain', 'reason': 'The shape has no identifiable details.'}

        def unclear_response(_video, _evidence, *, config, cuda, trace):
            trace('grok.decision', response=unclear)
            return unclear

        empty = path.with_name('ambiguity.json')
        with (patch('wm_batch.motif_review_method', return_value=method),
              patch('wm_batch.classify_layer', side_effect=unclear_response) as provider):
            review = cached_layer_reviewer(None, config, False, trace, empty)
            assert review(evidence) == unclear and review(evidence) == unclear
            assert provider.call_count == 1 and not empty.exists()
            assert review(changed) == unclear and provider.call_count == 2
            restarted = cached_layer_reviewer(None, config, False, trace, empty)
            assert restarted(evidence) == unclear and provider.call_count == 3
    print('Stored crop verdicts match the exact image, original and model settings.')


def test_grok_visibility_cache():
    from io import BytesIO
    from unittest.mock import Mock, patch

    from PIL import Image
    from wm_director import ask_cached_visibility, compact_images
    from wm_grounding import LocalEvidenceCache

    trace = Mock()
    frames = [7]
    decisions = ('present', 'absent', 'uncertain')
    proof = {'decision': 'present', 'reason': 'The overlay is visible.', 'checks': [{'frame': 7, 'decision': 'present'}]}
    uncertain = {'decision': 'uncertain', 'reason': 'The crop is unclear.',
                 'checks': [{'frame': 7, 'decision': 'uncertain'}]}
    with tempfile.TemporaryDirectory(prefix='wm-grok-visibility-') as directory:
        path = Path(directory)
        config = {'grok_api_base_url': 'http://127.0.0.1:8317/v1', 'model': 'test', 'reasoning_effort': 'xhigh'}
        scope = {'source_sha256': 'source'}
        cache_path = path/'review.json'

        def review(pixels, settings=config, provenance=scope):
            cache = LocalEvidenceCache(cache_path, provenance, trace)
            return ask_cached_visibility(settings, 'unchanged prompt', [('source', pixels)], decisions,
                                         trace, frames=frames, cache=cache)

        with patch('wm_director.ask', side_effect=[proof, uncertain, proof, proof, proof, proof]) as provider:
            assert review(b'original') == proof
            assert review(b'original') == proof and provider.call_count == 1
            assert review(b'changed') == uncertain
            assert review(b'changed') == proof and provider.call_count == 3
            assert review(b'changed') == proof and provider.call_count == 3
            assert review(b'original', {**config, 'reasoning_effort': 'high'}) == proof
            assert review(b'original', provenance={'source_sha256': 'different'}) == proof
            assert review(b'original', {**config, 'grok_api_base_url': 'http://127.0.0.1:8318/v1'}) == proof
            assert provider.call_count == 6
        buffer = BytesIO()
        Image.new('RGB', (800, 640), 'blue').save(buffer, format='JPEG')
        images = [('source', buffer.getvalue())]
        cache = LocalEvidenceCache(cache_path, scope, trace)

        def request(prompt, payload=images, *, reuse_compact=False):
            return ask_cached_visibility(config, prompt, payload, decisions, trace, frames=frames, cache=cache,
                                         reuse_compact=reuse_compact)

        compact = compact_images(images, 640)
        absent = {'decision': 'absent', 'reason': 'No mark.', 'checks': [{'frame': 7, 'decision': 'absent'}]}
        with patch('wm_director.ask', side_effect=[proof, proof, absent, proof, proof]) as provider:
            assert request('same request') == request('compact positive', compact) == proof
            assert request('compact negative', compact) == absent
            assert request('same request') == request('compact positive', reuse_compact=True) == proof
            assert provider.call_count == 3
            assert request('compact negative', reuse_compact=True) == proof and provider.call_count == 4
            with patch('wm_director.GROK_VISUAL_ID', 'different protocol'):
                assert request('same request') == proof and provider.call_count == 5
    print('Grok visibility reuses exact protocol evidence and rechecks unresolved or compact negative frames.')


def test_lone_local_visibility_conflict():
    from unittest.mock import patch

    from wm_director import review_watermark_visibility

    config = {'discovery_local': {'model': 'test'}, 'timeout_seconds': 60}
    frame, reference = 4, [0, 0, 0, 20, 20]
    positive = {'decision': 'present', 'reason': 'appears similar at full size'}
    absent = {'decision': 'absent', 'reason': 'a different graphic is visible',
              'reference_identifier': 'ref.example.com', 'observed_identifiers': ['other.example.com'],
              'checks': [{'frame': frame, 'decision': 'absent'}]}
    with (patch('wm_director.wm_grounding.frame_evidence', return_value=('frame', b'jpeg')),
          patch('wm_director.wm_grounding.frame_tiles', return_value=[((0, 0, 20, 20), ('crop', b'crop'))]),
          patch('wm_director.wm_grounding.ask_local', return_value=positive),
          patch('wm_director.local_visibility_crops', return_value=True) as local,
          patch('wm_director.ask_with_timeout_recovery', return_value=absent) as cloud):
        def review():
            return review_watermark_visibility(None, 'target ref.example.com', reference, [frame], config, False,
                                               lambda *_a, **_k: None, require_local=True)

        assert review() == {frame: 'absent'}
        assert local.call_count == cloud.call_count == 1
        local.return_value = False
        assert review() == {frame: 'present'} and cloud.call_count == 1
        local.return_value = True
        cloud.return_value = {'decision': 'uncertain', 'reason': 'still unclear'}
        assert review() == {frame: 'present'} and cloud.call_count == 2
        cloud.return_value = {**absent, 'observed_identifiers': ['ref.example.com']}
        assert review() == {frame: 'present'} and cloud.call_count == 3
    print('A lone local visibility claim is overturned only when local and cloud detail checks agree.')


def test_local_vision_evidence_cache():
    from io import StringIO
    import json
    from unittest.mock import Mock, patch

    from wm_grounding import LocalEvidenceCache, ask_local

    backend = {'endpoint': 'http://127.0.0.1:18082/v1/chat/completions', 'model': 'test'}
    confirmed = {'decision': 'present', 'reason': 'The same pattern is visible.'}
    uncertain = {'decision': 'uncertain', 'reason': 'The scene is unclear.'}
    image = ('frame', b'original')
    trace = Mock()

    def response(answer):
        return StringIO(json.dumps({'choices': [{'message': {'content': json.dumps(answer)}}]}))

    with tempfile.TemporaryDirectory(prefix='wm-local-evidence-') as directory:
        path = Path(directory)/'responses.json'
        scope = {'source_sha256': 'source', 'backend': 'known runtime'}
        opener = Mock()
        opener.open.side_effect = [response(confirmed), response(uncertain), response(confirmed), response(confirmed)]
        with (patch('wm_grounding.build_opener', return_value=opener),
              patch('wm_grounding.tempfile.gettempdir', return_value=directory)):
            def call(cache, picture):
                return ask_local({**backend, 'evidence_cache': cache}, 'Find the target', picture, 5,
                                 ('present', 'uncertain'), json_mode=True)

            cache = LocalEvidenceCache(path, scope, trace)
            assert call(cache, image) == confirmed
            resumed = LocalEvidenceCache(path, scope, trace)
            assert call(resumed, image) == confirmed
            imported = LocalEvidenceCache(Path(directory)/'new-run.json', scope, trace, prior_paths=(path,))
            assert call(imported, image) == confirmed
            assert imported.imported_paths == [path]
            from_other_source = LocalEvidenceCache(Path(directory)/'other-source.json', {**scope, 'source_sha256': 'other'},
                                                   trace, prior_paths=(path,))
            assert not from_other_source.entries and not from_other_source.imported_paths
            assert opener.open.call_count == 1
            changed = ('frame', b'different pixels')
            assert call(resumed, changed) == uncertain
            assert call(resumed, changed) == confirmed
            assert call(LocalEvidenceCache(path, {**scope, 'backend': 'updated runtime'}, trace), image) == confirmed
            assert opener.open.call_count == 4
    print('Exact local requests resume across jobs and retry an uncertain or changed response.')


def test_local_admission_before_inference():
    from contextlib import contextmanager
    from io import StringIO
    import json
    from unittest.mock import Mock, patch

    from wm_grounding import ask_local

    @contextmanager
    def queued(root, count, wait_seconds, *, waiting=None):
        assert root.name == '18082' and count == 2 and wait_seconds == 180
        yield 1, 60

    answer = {'decision': 'found', 'reason': 'visible'}
    opener = Mock()
    opener.open.side_effect = [StringIO('[{"is_processing":true},{"is_processing":true}]'),
                               StringIO('[{"is_processing":true},{"is_processing":false}]'),
                               StringIO(json.dumps({'choices': [{'message': {'content': json.dumps(answer)}}]}))]
    backend = {'model': 'local', 'endpoint': 'http://127.0.0.1:18082/v1/chat/completions', 'launcher': {}}
    with (patch('wm_grounding.capacity_slot', side_effect=queued),
          patch('wm_grounding.build_opener', return_value=opener),
          patch('wm_grounding.sleep')):
        assert ask_local(backend, 'Inspect this frame', ('original', b'pixels'), 22) == answer
    requests = opener.open.call_args_list
    assert [call.args[0] for call in requests[:2]] == ['http://127.0.0.1:18082/slots']*2
    assert requests[2].args[0].full_url == backend['endpoint'] and requests[2].kwargs['timeout'] == 22
    print('Local queue admission waits for a free slot before the inference deadline starts.')


def test_shared_local_admission_budget():
    from contextlib import contextmanager
    from io import StringIO
    import json
    from unittest.mock import Mock, patch

    from wm_grounding import LocalEvidenceCache, ask_local
    from workflow import CapacityTimeout

    now = [0.]
    waits = []

    @contextmanager
    def queued(_root, _count, wait_seconds, *, waiting=None):
        waits.append(wait_seconds)
        if len(waits) > 1:
            now[0] += wait_seconds
            raise CapacityTimeout('All slots are busy.')
        now[0] += 8.
        yield 1, 8.

    answer = {'decision': 'found', 'reason': 'visible'}

    def respond(_request, *, timeout):
        assert timeout == 22
        now[0] += 20.
        return StringIO(json.dumps({'choices': [{'message': {'content': json.dumps(answer)}}]}))

    opener = Mock()
    opener.open.side_effect = respond
    budget = {'remaining': 10.}
    source_budget = {'remaining': 11.}
    with tempfile.TemporaryDirectory(prefix='wm-admission-budget-') as root:
        cache = LocalEvidenceCache(Path(root)/'reviews.json', {}, lambda *_args, **_kwargs: None)
        settings = {'model': 'local', 'endpoint': 'http://127.0.0.1:18082/v1/chat/completions',
                    'admission_budget': budget, 'source_admission_budget': source_budget, 'evidence_cache': cache}
        with (patch('wm_grounding.capacity_slot', side_effect=queued),
              patch('wm_grounding.wait_for_local_backend'),
              patch('wm_grounding.build_opener', return_value=opener),
              patch('wm_grounding.monotonic', side_effect=lambda: now[0])):
            assert ask_local(settings, 'Check visibility', ('frame', b'first'), 22) == answer
            assert budget['remaining'] == 2. and source_budget['remaining'] == 3.
            for pixels in (b'second', b'third'):
                try:
                    ask_local(settings, 'Check visibility', ('frame', pixels), 22)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('An uncached review exceeded the shared admission budget.')
            assert ask_local(settings, 'Check visibility', ('frame', b'first'), 22) == answer
            alternate = {**settings, 'admission_budget': {'remaining': 5.}}
            with np.testing.assert_raises_regex(RuntimeError, 'still busy'):
                ask_local(alternate, 'Check another region', ('frame', b'fourth'), 22)
            with np.testing.assert_raises_regex(RuntimeError, 'source turn'):
                ask_local(alternate, 'Check another region', ('frame', b'fifth'), 22)
    assert waits == [10., 2., 1.] and budget['remaining'] == source_budget['remaining'] == 0.
    assert opener.open.call_count == 1
    print('A visual review shares one admission budget without charging inference or discarding exact cache hits.')


def test_review_rules_stay_bound_to_loaded_worker():
    from unittest.mock import patch

    from wm_batch import absence_review_method, motif_review_method

    config = {'model': 'remote', 'reasoning_effort': 'high', 'discovery_local': {'model': 'local'}}
    methods = (motif_review_method, absence_review_method)
    initial = [method(config) for method in methods]
    with patch('wm_batch.getsource', side_effect=AssertionError('The running worker reread a changed file.')):
        assert [method(config) for method in methods] == initial
        for method, scope in zip(methods, initial, strict=True):
            assert method({**config, 'model': 'other'}) == {**scope, 'model': 'other'}
    print('A batch keeps its loaded rule identities while model configuration remains part of the proof.')


def test_optional_local_proposals_keep_exact_cache():
    from hashlib import sha256
    from unittest.mock import patch

    from wm_grounding import LocalEvidenceCache, ask_local, local_request_body

    backend = {'model': 'local', 'endpoint': 'http://127.0.0.1:18082/v1/chat/completions'}
    prompt = 'Find the watermark'
    image = 'native crop', b'exact source pixels'
    answer = {'decision': 'found', 'reason': 'visible'}
    key = sha256(local_request_body(backend, prompt, image, max_tokens=240, json_mode=False,
                                   schema=None, reference=None)).hexdigest()
    with tempfile.TemporaryDirectory(prefix='wm-optional-proposals-') as root:
        cache = LocalEvidenceCache(Path(root)/'proposals.json', {}, lambda *_args, **_kwargs: None)
        cache.store(key, answer)
        window = {'remaining': 0.}
        with patch('wm_grounding.capacity_slot') as slot:
            settings = {**backend, 'proposal_window': window, 'evidence_cache': cache}
            assert ask_local(settings, prompt, image, 22) == answer
            assert not window.get('deferred')
            try:
                ask_local(settings, prompt, (image[0], b'different pixels'), 22)
            except RuntimeError as error:
                assert 'optional local proposal' in str(error)
            else:
                raise AssertionError('An uncached optional proposal waited beyond its window.')
            assert window['deferred'] and not slot.called
    print('An optional proposal window retains exact responses and defers only uncached work.')


def test_evidence_frame_cache():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_grounding import cached_evidence_frame, get_frame

    with tempfile.TemporaryDirectory(prefix='wm-evidence-frame-') as root:
        path = Path(root)/'source.mp4'
        path.write_bytes(b'original')
        video = Video(path, 32, 24, Fraction(30), 3, 0, '1:1', 'bt709', 'bt709', 'bt709')

        def decode(source, index, cuda):
            return np.full(32*24*3//2, source.path.stat().st_size+index+int(cuda), dtype=np.uint8)

        try:
            with patch('wm_grounding.decode_frame', side_effect=decode) as reader:
                first = get_frame(video, 1, True)
                assert get_frame(video, 1, True) is first and not first.flags.writeable
                assert get_frame(video, 1, False)[0] == first[0]-1
                path.write_bytes(b'changed source')
                assert get_frame(video, 1, True)[0] != first[0]
                assert reader.call_count == 3
        finally:
            cached_evidence_frame.cache_clear()
    print('Visual evidence shares immutable source pixels and invalidates them when the file changes.')


def test_director_local_cache_logging():
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_director import run_job
    from wm_grounding import LocalEvidenceCache

    with tempfile.TemporaryDirectory(prefix='wm-local-director-') as name:
        root = Path(name)
        cache = LocalEvidenceCache(root/'cache.json', {}, lambda *_args, **_kw: None)
        settings = {'model': 'fixture', 'reasoning_effort': 'high',
                    'discovery_local': {'model': 'local', 'evidence_cache': cache}}
        video = SimpleNamespace(width=80, height=64, fps=24, count=16, offset=0, matrix='bt709')
        job = root/'job'
        with (patch('wm_director.Video.open', return_value=video),
              patch('wm_director.make_recipe', return_value=None) as plan):
            assert run_job(root/'input.mp4', job, None, settings, False)['status'] == 'needs_review'
        event = json.loads((job/'events.jsonl').read_text(encoding='utf-8').splitlines()[0])
        assert event['config']['discovery_local'] == {'model': 'local'}
        assert plan.call_args.args[2]['discovery_local']['evidence_cache'] is cache
    print('The director keeps the live local cache available without writing it into JSON logs.')


def test_sparse_discovery_sampling():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import joined_component_boxes, merge_visual_proposals
    from wm_director import (cloud_discovery_samples, distinct_native_boxes, make_discovery_recipe, native_tile_boxes,
                             sparse_discovery_samples)

    source = SimpleNamespace(count=1800, fps=30)
    seen = {0, 500, 910, 1600}
    frames, nearby = sparse_discovery_samples(source, seen, {910: object()})
    assert nearby == [880, 940] and len(set(frames)) == 4 and not set(frames).intersection(seen)
    assert frames[2] < 400 and frames[3] > 1650
    handoff = cloud_discovery_samples(source, seen, [915], [], {500: object(), 910: object()}, 6)
    assert handoff[:3] == [500, 910, 915] and len(set(handoff)) == 6 and handoff[-1] == 1799

    def discovery(_source, frames, _width, _cuda, _config, _prompt, _trace, attempt, _label):
        if attempt >= 2:
            raise RuntimeError('Cloud planning unavailable')
        return {'decision': 'track', 'label': 'fixture', 'observations': [
            {'frame': frame, 'rect': [30, 40, 140, 130]} for frame in frames]}

    source.path, source.width, source.height, source.check_timestamps = Path('fixture.mp4'), 640, 360, lambda: None
    config = {'batch_routing': True, 'local_layers': True, 'timeout_seconds': 60, 'sample_width': 440}
    report = {}
    trace = lambda *_a, **_kw: None
    with (patch('wm_director.wm_grounding.frame_tiles', return_value=[((256, 144, 384, 216), ('crop', b'image'))]),
          patch('wm_director.wm_grounding.locate_named', return_value=[[100, 100, 300, 250]])):
        tile = native_tile_boxes(source, 910, False, {}, trace)[0]
        assert tile == [460, 460, 580, 550]
        assert distinct_native_boxes(source, [[470, 465, 575, 545], tile, [0, 0, 1000, 1000]]) == [tile]
    assert joined_component_boxes([[800, 970, 950, 990], [890, 940, 950, 966], [20, 30, 120, 60]]) == [
        [800, 940, 950, 990]]
    with (patch('wm_director.digest', return_value='fixture'),
          patch('wm_director.discovery_decision', side_effect=discovery) as provider,
          patch('wm_director.refine_discovery', return_value={}),
          patch('wm_discovery.propose_boxes', return_value={}),
          patch('wm_discovery.enroll', side_effect=ValueError('The single track does not cover the source'))):
        assert make_discovery_recipe(source, Path('.'), config, False, report, trace) is None
        assert report['planning']['local_layer_refinement'] and len(report['planning']['rounds']) == 2
        provider.reset_mock()
        recipe, report = Path('verified-layers.json'), {}
        assert make_discovery_recipe(source, Path('.'), config, False, report, trace,
                                     local_refinement=lambda plan: recipe if len(plan['rounds']) == 2 else None) == recipe
        assert provider.call_count == 2 and report['planning']['recognition'] == 'local-layer-refinement'

    def sparse(_source, frames, _width, _cuda, _config, _prompt, _trace, attempt, _label):
        assert attempt < 2
        if attempt == 0:
            return {'decision': 'track', 'label': 'fixture', 'observations': [
                {'frame': frames[0], 'rect': [30, 40, 140, 130]}]}
        return {'decision': 'review', 'observations': []}

    def try_pixels(plan):
        sampled = {i for round_ in plan['rounds'] for i in round_['frames']}
        proposals = merge_visual_proposals(source, {}, plan, include_samples=True)
        assert set(proposals) == sampled and sum(map(len, proposals.values())) == 1
        assert len(plan['local_seed_observations']) == 1
        return recipe

    with (patch('wm_director.digest', return_value='fixture'),
          patch('wm_director.discovery_decision', side_effect=sparse),
          patch('wm_director.refine_discovery', return_value={}),
          patch('wm_discovery.propose_boxes', return_value={})):
        assert make_discovery_recipe(source, Path('.'), config, False, {}, trace, local_refinement=try_pixels) == recipe
    source.count = 4
    hybrid = {**config, 'discovery_local': {'model': 'local'}, 'sample_width': 640, 'scan_samples': 12}
    report = {}
    with (patch('wm_director.digest', return_value='fixture'),
          patch('wm_director.native_discovery', return_value={'decision': 'review'}),
          patch('wm_director.evidence', return_value=[]) as images,
          patch('wm_director.ask_with_timeout_recovery', return_value={'decision': 'no_mark'}) as provider):
        assert make_discovery_recipe(source, Path('.'), hybrid, False, report, trace) is None
        provider.assert_called_once()
        assert images.call_args.args[1:3] == ([0, 1, 2, 3], 640)
        assert images.call_args.kwargs['full_sheet'] == 2
        assert report['planning']['reason'] == 'Grok did not see a watermark in the inspected source frames.'
    print('Sparse discovery keeps local observations and gives independent layer refinement priority over cloud planning.')


def test_discovery_jumps():
    from wm_discovery import evidence_threshold, learned_motif
    from wm_track import highpass

    rng = np.random.default_rng(17)
    mark = np.zeros((30, 86), np.uint8)
    cv2.circle(mark, (12, 15), 8, 215, -1)
    cv2.putText(mark, 'JUMP', (23, 22), cv2.FONT_HERSHEY_SIMPLEX, .66, 255, 2, cv2.LINE_AA)
    placements = [(104, 20), (25, 89), (197, 64), (101, 131)]
    observations, signals, clean = [], {}, {}
    for i, (x, y) in enumerate(placements):
        frame = cv2.resize(rng.integers(55, 160, (9, 16), np.uint8), (320, 180))
        clean[i] = highpass(frame, 2.2)
        region = frame[y:y+30, x:x+86]
        region[:] = np.rint(region + (230-region.astype(np.float32))*mark.astype(np.float32)/470)
        signals[i] = highpass(frame, 2.2)
        observations.append((i, (x+55 if i == 1 else x, y, 86, 30)))
    motif = learned_motif(observations, signals)
    _, _, _, checks = evidence_threshold(motif, observations, signals)
    assert not checks[1]['corroborates_visual_box']
    for i, check in enumerate(checks):
        left, top, _, _ = check['verified_box']
        x, y = placements[i]
        assert abs(left-(x+4)) < 8 and abs(top-(y+6)) < 8
        assert motif.locate(clean[i])[2] < .76
    print('Full-frame discovery corrected a displaced observation and rejected clean lookalikes.')


def test_layer_completion():
    from wm_discovery import Motif, agrees_in_halves, patch
    from wm_layers import Layer, Observation, build_layer, complete_layer, covered_fragment, locations
    from wm_track import highpass

    rng = np.random.default_rng(862)
    mark = np.zeros((60, 114), np.uint8)
    cv2.putText(mark, 'DEMO.TV', (4, 52), cv2.FONT_HERSHEY_SIMPLEX, .64, 235, 2, cv2.LINE_AA)
    icon = np.zeros_like(mark)
    cv2.circle(icon, (60, 15), 9, 235, 2, cv2.LINE_AA)
    cv2.line(icon, (57, 10), (66, 16), 235, 2, cv2.LINE_AA)
    positions = [(70, 42), (210, 91), (137, 24), (47, 110), (165, 145), (251, 35)]
    signals, without_icon, observations = {}, {}, []
    for i, (x, y) in enumerate(positions):
        background = cv2.resize(rng.integers(40, 160, (9, 14), np.uint8), (416, 240))
        frame = background.copy()
        for design, destination in ((mark, without_icon), (mark | icon, signals)):
            part = frame[y:y+60, x:x+114]
            source = background[y:y+60, x:x+114]
            part[:] = np.rint(source + (245-source.astype(np.float32))*design/330).astype(np.uint8)
            destination[i] = highpass(frame, 2.2)
        observations.append(Observation(i, (x+2, y+36, 97, 20)))
    layer = build_layer(signals, observations)
    assert layer is not None
    trace = lambda *_args, **_details: None
    expanded = complete_layer(layer, signals, trace)
    assert expanded is not None
    for item, (x, y, _) in expanded.matches.items():
        origin_x, origin_y = positions[item.frame]
        mask = patch(expanded.motif.mask, (origin_x-x, origin_y-y, 114, 60))
        assert np.mean(mask[icon > 60]) > .85
        a, b, score = locations(expanded.motif, -signals[item.frame])[0]
        assert abs(a-x) <= 1 and abs(b-y) <= 1 and score < -.64
        assert agrees_in_halves(expanded.motif, -signals[item.frame], a, b, -1)
    assert covered_fragment(layer, expanded, signals)
    missing = {**signals, len(positions)-1: without_icon[len(positions)-1]}
    assert complete_layer(layer, missing, trace) is None

    # Preserve the outer bounds but leave a disconnected part inside those bounds unmasked.
    edges = np.zeros_like(mark)
    cv2.circle(edges, (13, 8), 6, 235, 2, cv2.LINE_AA)
    cv2.rectangle(edges, (85, 36), (104, 48), 235, 2, cv2.LINE_AA)
    signal = highpass(edges, 2.2)
    weights = np.minimum(np.abs(signal)/12, 1).astype(np.float32)
    incomplete = Motif(signal, weights, (weights > .25).astype(np.uint8))
    matches, sources = {}, {}
    for i, (x, y) in enumerate(positions):
        frame = np.full((240, 416), 90, np.uint8)
        frame[y:y+60, x:x+114] += np.maximum(edges, icon)//2
        sources[i] = highpass(frame, 2.2)
        matches[Observation(i, (x, y, 114, 60))] = (x, y, .9)
    repaired = complete_layer(Layer(incomplete, .6, matches), sources, trace)
    assert repaired is not None
    for item, (x, y, _) in repaired.matches.items():
        origin_x, origin_y = positions[item.frame]
        mask = patch(repaired.motif.mask, (origin_x-x, origin_y-y, 114, 60))
        assert np.mean(mask[icon > 60]) > .85
    print('Partial watermark discovery recovered separate parts inside and outside the bounds using held-out evidence.')


def test_sampled_layer_completion():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif, patch as image_patch
    from wm_layers import Layer, Observation, complete_sampled_layer

    rng = np.random.default_rng(99)
    count, width, height = 36, 176, 128
    video = SimpleNamespace(count=count, width=width, height=height, planes=lambda image: (image,))
    mark = rng.normal(0, 33, (18, 38)).astype(np.float32)
    fragment = rng.choice([-27., 27.], (10, 9)).astype(np.float32)
    core = Motif(mark, np.ones_like(mark), np.ones(mark.shape, np.uint8))
    frames, anchors, signals = [], {}, {}
    for i in range(count):
        x, y = 35+(11*i)%38, 26+(7*i)%28
        image = rng.normal(0, 2, (height, width)).astype(np.float32)
        image[y:y+18, x:x+38] += mark
        image[y+4:y+14, x-15:x-6] += fragment
        if i in (2, 7, 16):
            image[y+4:y+14, x+47:x+56] += fragment
        frames.append(image)
        if i in (0, 10, 27):
            anchors[Observation(i, (x, y, 38, 18))] = x, y, .95
            signals[i] = image
    layer = Layer(core, .64, anchors)
    with (patch('wm_layers.read_frames', return_value=enumerate(frames)),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image)):
        expanded = complete_sampled_layer(video, layer, signals, False, lambda *_a, **_kw: None)
    assert expanded is not None and len(expanded.matches) == count
    anchor = next(iter(anchors))
    dx, dy = np.subtract(anchors[anchor][:2], expanded.matches[anchor][:2])
    assert np.array_equal(expanded.motif.weights[dy:dy+18, dx:dx+38], core.weights)
    assert expanded.motif.weights.sum() == core.weights.sum()
    assert expanded.motif.width <= core.width+20
    assert expanded.motif.mask[dy+4:dy+14, dx-15:dx-6].all()
    assert not expanded.motif.mask[dy+4:dy+14, dx+47:dx+56].any()
    print('More source anchors corroborate a disconnected fragment while retaining the original tracking signal.')

    video.count, video.fps = 3600, 30
    offset = 115
    located, proposals = {}, []
    for item, (x, y, score) in anchors.items():
        frame = item.frame+offset
        located[Observation(frame, item.rect)] = x, y, score
        proposals.append(Observation(frame, (x-15, y, 53, 18)))
    item, (x, y, _) = next(iter(located.items()))
    proposals.append(Observation(item.frame, (x-15, y, 71, 18)))
    for i, image in enumerate(frames):
        if i not in (2, 7, 16):
            x, y = 35+(11*i)%38, 26+(7*i)%28
            image[y+4:y+14, x+47:x+56] += fragment
    blank = np.zeros((height, width), np.float32)
    remote = {}

    def sampled(_video, start=0, stop=None, cuda=False, *, step=1):
        for i in range(start, video.count if stop is None else stop, step):
            yield i, frames[i-offset] if offset <= i < offset+len(frames) else remote.get(i, blank)

    short = Layer(core, .64, located)
    short_signals = {i+offset: data for i, data in signals.items() if i in (0, 10, 27)}
    trace = lambda *_a, **_kw: None
    with (patch('wm_layers.read_frames', side_effect=sampled),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image)):
        assert complete_sampled_layer(video, short, dict(short_signals), False, trace) is None
        restored = complete_sampled_layer(video, short, short_signals, False, trace, observations=proposals)
        other_scene = blank.copy()
        other_scene[35:53, 55:93] = mark
        remote[3*(video.count//48)] = other_scene
        assert complete_sampled_layer(video, short, dict(short_signals), False, trace, observations=proposals) is None
    assert restored is not None and len(restored.matches) >= 9
    item, (x, y, _) = next(iter(located.items()))
    dx, dy = np.subtract((x, y), restored.matches[item][:2])
    mask = image_patch(restored.motif.mask, (dx-15, dy, 71, 18))
    assert mask[4:14, :9].all() and not mask[4:14, 62:71].any()
    assert np.array_equal(restored.motif.weights[dy:dy+18, dx:dx+38], core.weights)
    print('Short watermark windows use repeated source proposals and exclude an unsupported nearby pattern.')

    video.count = count
    visible = {3, 6, 9, 12, 17, 24, 26, 31}
    images, located, signals = [], {}, {}
    for i in range(count):
        x, y = (35 if i in visible else 4), 26+(7*i)%28
        image = rng.normal(0, 2, (height, width)).astype(np.float32)
        image[y:y+18, x:x+38] += mark
        if i in visible:
            image[y+4:y+14, x-15:x-6] += fragment
        images.append(image)
        if i in (0, 10, 27):
            located[Observation(i, (x, y, 38, 18))] = x, y, .95
            signals[i] = image
    moving = Layer(core, .64, located)
    with (patch('wm_layers.read_frames', return_value=enumerate(images)),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image)):
        extended = complete_sampled_layer(video, moving, signals, False, trace)
    assert extended is not None
    item, (x, y, _) = next(iter(located.items()))
    dx, dy = np.subtract((x, y), extended.matches[item][:2])
    assert extended.motif.mask[dy+4:dy+14, dx-15:dx-6].all()
    print('An edge fragment needs multiple in-frame training and held-out observations.')

    color_frames, color_signals = [], {}
    for i in range(count):
        x, y = 35+(11*i)%38, 26+(7*i)%28
        luma, u, v = [rng.normal(0, 2, (height, width)).astype(np.float32) for _ in range(3)]
        luma[y:y+18, x:x+38] += mark
        v[y+4:y+14, x-15:x-6] += fragment
        if i in (2, 7, 16):
            v[y+4:y+14, x+47:x+56] += fragment
        color_frames.append((luma, u, v))
        if i in (0, 10, 27):
            color_signals[i] = luma
    video.planes = lambda frame: frame
    with (patch('wm_layers.read_frames', return_value=enumerate(color_frames)),
          patch('wm_layers.get_frame', side_effect=lambda _, i, __: color_frames[i]),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image),
          patch('wm_layers.color_patch', side_effect=lambda _, planes, *rect: image_patch(planes[2]-planes[1], rect))):
        color = complete_sampled_layer(video, layer, color_signals, False, trace)
    assert color is not None
    anchor = next(iter(anchors))
    dx, dy = np.subtract(anchors[anchor][:2], color.matches[anchor][:2])
    assert color.motif.mask[dy+4:dy+14, dx-15:dx-6].all()
    assert not color.motif.mask[dy+4:dy+14, dx+47:dx+56].any()
    assert color.motif.weights.sum() == core.weights.sum()


def test_translucent_mask_completion():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, Observation, complete_translucent_mask

    video = Video(Path('translucent-mask-test.mp4'), 128, 72, Fraction(25), 8, 0,
                  '1:1', 'bt709', 'bt709', 'bt709')
    mask = np.zeros((24, 54), np.uint8)
    mask[4:20, 5:11] = 1
    mask[4:20, 45:51] = 1
    missing = np.zeros_like(mask)
    missing[6:18, 25:28] = 1
    ink = (mask|missing).astype(np.float32)*.64
    signal = mask.astype(np.float32)*15
    motif = Motif(signal, mask.astype(np.float32), mask)
    x, y = 38, 24
    frames, matches = [], {}
    levels = (32, 52, 74, 96, 120, 156, 184, 227)
    for i, level in enumerate(levels):
        frame = np.full(video.width*video.height*3//2, level, np.uint8)
        roi = video.planes(frame)[0][y:y+24, x:x+54]
        roi[:] = np.rint(level+(202-level)*ink)
        frames.append(frame)
        matches[Observation(i, (x, y, 54, 24))] = (x, y, .9)
    layer = Layer(motif, .6, matches)
    trace = lambda *_args, **_details: None
    with patch('wm_layers.get_frame', side_effect=lambda _, i, __: frames[i]):
        expanded = complete_translucent_mask(video, layer, False, trace)
        assert expanded is not None and expanded.motif.mask[missing > 0].all()
        assert np.array_equal(expanded.motif.weights, motif.weights)
        for i in (2, 7):
            frames[i].fill(levels[i])
        assert complete_translucent_mask(video, layer, False, trace) is None
    print('Held-out background changes corroborate missing translucent strokes before enlarging the mask.')


def test_chroma_mask_completion():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_discovery import Motif, patch as image_patch
    from wm_layers import Layer, Observation, complete_chroma_mask

    video = Video(Path('chroma-mask-test.mp4'), 196, 120, Fraction(25), 6, 0, '1:1', 'bt709', 'bt709', 'bt709')
    x, y, width, height = 80, 50, 48, 22
    rng = np.random.default_rng(185)
    detail = rng.normal(0, 20, (height, width)).astype(np.float32)
    mask = np.ones((height, width), np.uint8)
    layer = Layer(Motif(detail, mask.astype(np.float32), mask), .62, {})
    frames, signals = [], {}
    for i in range(video.count):
        raw = np.full(video.width*video.height*3//2, 128, np.uint8)
        _, u, v = video.planes(raw)
        u.fill(112+3*i)
        v.fill(126+i)
        cv2.circle(u, ((x-13)//2, (y+10)//2), 4, 205, -1)
        cv2.circle(v, ((x-13)//2, (y+10)//2), 4, 44, -1)
        if i not in (0, 3):
            cv2.circle(u, ((x+width+10)//2, (y+10)//2), 4, 42, -1)
        frames.append(raw)
        signal = rng.normal(0, 2, (video.height, video.width)).astype(np.float32)
        signal[y:y+height, x:x+width] += detail
        signals[i] = signal
        layer.matches[Observation(i, (x, y, width, height))] = (x, y, .95)
    trace = lambda *_a, **_kw: None
    with patch('wm_layers.get_frame', side_effect=lambda _, i, __: frames[i]):
        expanded = complete_chroma_mask(video, layer, signals, False, trace)
        assert expanded is not None and len(expanded.matches) == len(layer.matches)
        u, v, _ = next(iter(expanded.matches.values()))
        assert np.mean(image_patch(expanded.motif.mask, (x-17-u, y+5-v, 9, 9))) > .80
        assert not image_patch(expanded.motif.mask, (x+width+6-u, y+5-v, 9, 9)).any()
        assert expanded.motif.weights.sum() == layer.motif.weights.sum()
        for i in (0, 3):
            _, u, v = video.planes(frames[i])
            u.fill(128)
            v.fill(128)
        assert complete_chroma_mask(video, layer, signals, False, trace) is None
    print('A chroma-only icon survives two held-out frames without widening luminance matching.')


def test_layer_alignment():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_layers import (Observation, absence_identity, build_layer, discover_layers, load_layer_evidence, near,
                           save_layer_evidence, visible_box)

    rng = np.random.default_rng(2026)
    graphic = rng.normal(0, 42, (19, 32)).astype(np.float32)
    positions = [(11, 19), (25, 40), (39, 62), (19, 75), (40, 29), (28, 60)]
    signals, proposals = {}, []
    for i, (x, y) in enumerate(positions):
        signal = rng.normal(0, 2, (115, 152)).astype(np.float32)
        signal[y:y+19, x:x+32] += graphic
        signals[i] = signal
        proposals.append(Observation(i, (x-3, y-4, 88, 27)))
    layer = build_layer(signals, proposals)
    assert layer is not None and len(layer.matches) == len(positions)
    for item, (u, v, score) in layer.matches.items():
        x, y, w, h = visible_box(layer.motif, u, v)
        left, top = positions[item.frame]
        assert x <= left and y <= top and x+w >= left+32 and y+h >= top+19 and score > .95
        assert near(layer.motif, signals[item.frame], item.rect) is None
    layer.label = 'verified design'
    variant = rng.normal(0, 45, (15, 26)).astype(np.float32)
    fresh, rows = {}, {}
    for i, x in enumerate((95, 100, 105)):
        image = rng.normal(0, 2, (115, 152)).astype(np.float32)
        image[12:27, x:x+26] += variant
        fresh[i] = image
        rows[i] = [{'rect': [x, 12, x+26, 27]}]
    video = SimpleNamespace(width=152, height=115, count=6, planes=lambda raw: (raw,))
    with (patch('wm_layers.get_frame', side_effect=lambda _v, i, _cuda: fresh[i]),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image),
          patch('wm_layers.review_full_layer', side_effect=lambda _v, found, *_args, **_options: (found, 'confirmed'))):
        both = discover_layers(video, rows, None, False, lambda *_a, **_kw: None, retained=[layer])
    assert len(both) == 2 and both[0].motif is layer.motif and both[0].matches == layer.matches
    assert both[1].motif.bounds[2] < layer.motif.bounds[2]
    for item, (x, y, _) in both[1].matches.items():
        a, b, w, h = visible_box(both[1].motif, x, y)
        u, v, width, height = item.rect
        assert a <= u and b <= v and a+w >= u+width and b+h >= v+height
    both[1].label = 'verified variant'
    with tempfile.TemporaryDirectory(prefix='motif-evidence-') as folder:
        path = Path(folder)/'confirmed-motifs.npz'
        provenance = {'source_sha256': 'source', 'review_method': 'test'}
        save_layer_evidence(path, both[:1], provenance)
        save_layer_evidence(path, both, provenance)
        reloaded, method = load_layer_evidence(path, video, 'source')
        with np.testing.assert_raises(ValueError):
            load_layer_evidence(path, video, 'different source')
        assert method == provenance and list(map(absence_identity, reloaded)) == list(map(absence_identity, both))
        assert [x.matches for x in reloaded] == [x.matches for x in both]
        resumed = discover_layers(video, {}, None, False, lambda *_a, **_kw: None, retained=reloaded)
        assert [x.label for x in resumed] == ['verified design', 'verified variant']
    print('The independent frame verified a tight motif offset from the center of wide proposals.')


def test_layer_pixel_seed():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_layers import build_layer, discover_layers, proposals_for, recover_layer, visible_box

    rng = np.random.default_rng(162)
    shape = (108, 152)
    graphic = rng.normal(0, 32, (20, 32)).astype(np.float32)
    positions = {0: (12, 20), 30: (78, 50), 90: (34, 63), 120: (89, 12)}
    signals = {i: rng.normal(0, 2, shape).astype(np.float32) for i in range(0, 121, 30)}
    for i, (x, y) in positions.items():
        signals[i][y:y+20, x:x+32] += graphic
    rows = dict.fromkeys(signals, ())
    rows[0] = [{'rect': [12, 20, 44, 40]}]
    rows[30] = [{'rect': [13, 75, 41, 93]}]
    video = SimpleNamespace(width=shape[1], height=shape[0], count=121, fps=30, planes=lambda raw: (raw,))
    observations = proposals_for(video, rows)
    assert build_layer(signals, observations) is None
    events = []
    with (patch('wm_layers.get_frame', side_effect=lambda _v, i, _cuda: signals[i]),
          patch('wm_layers.highpass', side_effect=lambda image, _s: image),
          patch('wm_layers.review_full_layer', side_effect=lambda _v, found, *_args, **_options: (found, 'confirmed')) as review):
        layers = discover_layers(video, rows, None, False, lambda event, **data: events.append((event, data)))
    assert len(layers) == 1 and review.call_count == 1
    assert any(event == 'layers.pixel_seed_recovered' for event, _ in events)
    assert set(positions) == {item.frame for item in layers[0].matches}
    for item, (u, v, _) in layers[0].matches.items():
        left, top, width, height = visible_box(layers[0].motif, u, v)
        x, y = positions[item.frame]
        assert left <= x and top <= y and left+width >= x+32 and top+height >= y+20
    stationary_scene = dict.fromkeys(signals, signals[0])
    assert recover_layer(video, rows, observations, stationary_scene, False, lambda *_a, **_kw: None) is None

    dense = list(range(0, 121, 4))
    dense_signals = {}
    dense_rows = dict.fromkeys(dense, ())
    seed_frame = 60
    assert seed_frame not in [dense[i] for i in np.linspace(0, len(dense)-1, 12, dtype=int)]
    for frame in dense:
        image = rng.normal(0, 2, shape).astype(np.float32)
        x, y = map(int, rng.integers(12, 76, size=2))
        image[y:y+20, x:x+32] += graphic
        dense_signals[frame] = image
        if frame == seed_frame:
            dense_rows[frame] = [{'rect': [x, y, x+32, y+20]}]
    recovered = recover_layer(video, dense_rows, proposals_for(video, dense_rows), dense_signals, False,
                              lambda *_a, **_kw: None)
    assert recovered is not None and any(item.frame == seed_frame for item in recovered.matches)
    print('One usable model box leads to independent, moving source matches and still requires overlay classification.')


def test_prior_absence_checkpoint_scope():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import local_tracks
    from wm_layers import uncovered_absence

    video = SimpleNamespace(count=20)
    layer = SimpleNamespace(label='test overlay')
    intervals = [np.arange(3, 6)]
    calls = []
    trace = lambda *_args, **_kwargs: None

    def confirm(_layer, frames):
        calls.append(frames)
        return dict.fromkeys(frames, 'absent')

    def scan(_video, _layers, _cuda, _trace, *, absence_proofs, save_proofs, **_options):
        proofs = absence_proofs.setdefault('same-motif', {})
        assert not uncovered_absence(layer, intervals, [.2]*video.count, confirm, trace, 0, set(),
                                     proofs=proofs, save_proofs=save_proofs)
        return ['verified track']

    with tempfile.TemporaryDirectory(prefix='wm-absence-checkpoint-') as root:
        prior, fresh = Path(root)/'prior', Path(root)/'fresh'
        prior.mkdir()
        fresh.mkdir()
        config = {'discovery_local': {'endpoint': 'local', 'model': 'fixture'}, 'model': 'fixture',
                  'reasoning_effort': 'xhigh', 'absence_source_sha256': 'same source',
                  'absence_proof_file': str(prior/'absence-evidence.json')}
        with (patch('wm_layers.discover_layers', return_value=[layer]),
              patch('wm_layers.scan_layers', side_effect=scan)):
            local_tracks(video, {}, config, False, trace)
            imported = {**config, 'absence_proof_file': str(fresh/'absence-evidence.json'),
                        'prior_layer_checkpoints': [str(prior)]}
            local_tracks(video, {}, imported, False, trace)
            assert calls == [[3, 4, 5]]
            local_tracks(video, {}, {**imported, 'reasoning_effort': 'high'}, False, trace)
            assert calls == [[3, 4, 5], [3, 4, 5]]
    print('An unchanged absence proof crosses batch directories and a changed review method rechecks it.')


def test_motif_checkpoint_partial_reuse():
    from types import SimpleNamespace

    from wm_discovery import Motif
    from wm_layers import Layer, Observation, load_layer_evidence, save_layer_evidence

    video = SimpleNamespace(width=32, height=24, count=8)
    matches = {Observation(i, (4, 6, 12, 8)): (4, 6, .9) for i in (1, 3, 5)}
    layers = []
    for width in (12, 80):
        pixels = np.ones((8, width), np.float32)
        layers.append(Layer(Motif(pixels, pixels, pixels.astype(np.uint8)), .7, matches, 'known overlay'))
    text = layers[0].motif
    padding = ((2, 1), (11, 11))
    padded = Motif(*(np.pad(field, padding) for field in (text.signal, text.weights, text.mask)))
    shifted = {item: (x-11, y-2, score) for item, (x, y, score) in matches.items()}
    layers.append(Layer(padded, .7, shifted, 'clipped canvas', ((1, 4, 6, 12, 8),), (text, 11, 2)))
    with tempfile.TemporaryDirectory(prefix='wm-motif-checkpoint-') as root:
        checkpoint = Path(root)/'motifs.npz'
        scope = {'source_sha256': 'source'}
        save_layer_evidence(checkpoint, layers, scope)
        rejected = []
        kept, provenance = load_layer_evidence(checkpoint, video, 'source', on_rejected=lambda **row: rejected.append(row))
        assert len(kept) == 2 and provenance == scope and rejected[0]['layer'] == 2
        assert kept[1].matches == matches and np.array_equal(kept[1].motif.mask, text.mask)
        assert kept[1].identifier_support[1:] == (0, 0) and kept[1].reviewed_regions == layers[2].reviewed_regions
        with np.testing.assert_raises(ValueError):
            load_layer_evidence(checkpoint, video, 'changed', on_rejected=lambda **_row: None)
    print('A bad checkpoint layer does not discard other verified motifs for the exact original.')


def test_complementary_motif_checkpoints():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import local_tracks, motif_review_method
    from wm_discovery import Motif
    from wm_layers import Layer, Observation, load_layer_evidence, save_layer_evidence

    video = SimpleNamespace(width=48, height=32, count=120)
    backend = {'model': 'fixture-local', 'endpoint': 'http://127.0.0.1:18082/v1/chat/completions'}
    config = {'model': 'fixture-remote', 'reasoning_effort': 'xhigh', 'discovery_local': backend,
              'absence_source_sha256': 'original', 'required_overlay_regions': []}
    scope = {'source_sha256': 'original', 'local_endpoint': backend['endpoint'],
             'local_launcher': dict.fromkeys(('executable', 'weights', 'projector', 'device')),
             'review_method': motif_review_method(config)}

    def layer(label, frames, shape):
        image = np.ones(shape, np.float32)
        motif = Motif(image, image, image.astype(np.uint8))
        rect = 4, 6, motif.width, motif.height
        matches = {Observation(i, rect): (4, 6, .9) for i in frames}
        return Layer(motif, .7, matches, label, tuple((i, *rect) for i in frames[:2]))

    first = layer('approved emblem first.example.com', (3, 12, 24), (8, 12))
    repeated = layer('Approved Emblem first.example.com', (30, 42, 54), (8, 12))
    second = layer('independent graphic second.example.com', (66, 78, 90), (9, 16))
    variant = layer('different first.example.com graphic', (9, 21, 33), (8, 20))
    with tempfile.TemporaryDirectory(prefix='wm-complementary-motifs-') as root:
        own = Path(root)/'working'
        peer = Path(root)/'peer'
        stale = Path(root)/'stale'
        for directory in (own, peer, stale):
            directory.mkdir()
        path = own/'confirmed-motifs.npz'
        save_layer_evidence(path, [first], scope)
        save_layer_evidence(peer/path.name, [repeated, second, variant], scope)
        save_layer_evidence(stale/path.name, [layer('stale', (1, 3, 5), (8, 10))],
                            {**scope, 'review_method': {'rules_sha256': 'other'}})
        config.update(absence_proof_file=str(own/'absence-evidence.json'), prior_layer_checkpoints=[peer, stale])
        with patch('wm_layers.scan_layers', return_value=[]):
            _, labels = local_tracks(video, {}, config, False, lambda *_a, **_kw: None, persist_layers=True)
        assert labels == [first.label, second.label]
        saved, saved_scope = load_layer_evidence(path, video, 'original')
        assert saved_scope == scope and len(saved) == 2 and len(saved[0].matches) == 6
        assert set(map(tuple, config['required_overlay_regions'])) == set(
            first.reviewed_regions+repeated.reviewed_regions+second.reviewed_regions+variant.reviewed_regions)
    print('Compatible checkpoints add confirmed motifs and anchors while repeated templates share one placement search.')


def test_local_library_without_detector():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import source_layer_candidates

    video = SimpleNamespace(count=120, planes=lambda raw: (raw,))
    config = {'learned_marks': 'library', 'local_proposal_frames': 0}
    expected = {10: [{'rect': [4, 6, 24, 26], 'provider': 'verified-library'}]}

    def known(_video, samples, library, identity, _trace):
        assert list(samples) == [(0, ('raw0',)), (10, ('raw1',))]
        assert library == Path('library') and identity == 'source'
        return expected

    with (patch('wm_batch.read_frames', return_value=iter([(0, 'raw0'), (10, 'raw1')])) as decode,
          patch('wm_batch.remembered_proposals', side_effect=known)):
        result = source_layer_candidates(video, None, None, config, False, lambda *_a, **_kw: None, 'source', [])
    decode.assert_called_once_with(video, cuda=False, step=10)
    assert result == expected
    print('Local discovery can reuse verified motifs without loading the optional object detector.')


def test_local_boundary_proposals():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import source_layer_candidates

    video = SimpleNamespace(count=120, fps=30, width=160, height=90)
    config = {'learned_marks': 'library', 'local_proposal_frames': 6, 'discovery_local': {'model': 'fixture'}}
    calls = []

    def locate(_video, frames, _cuda, _backend, _label, _trace):
        calls.append(frames)
        if not visible:
            return {'decision': 'review'}
        return {'decision': 'track', 'label': 'logo',
                'observations': [{'frame': i, 'rect': [100, 120, 260, 280]} for i in frames[:3]]}

    with (patch('wm_batch.remembered_proposals', return_value={}),
          patch('wm_batch.native_discovery', side_effect=locate)):
        visible = False
        source_layer_candidates(video, None, None, config, False, lambda *_a, **_kw: None, 'source', [])
        assert len(calls) == 1
        calls.clear()
        visible = True
        proposals = source_layer_candidates(video, None, None, config, False, lambda *_a, **_kw: None, 'source', [])
    assert len(calls) == 2 and set(calls[0]).isdisjoint(calls[1])
    assert {30, 60, 75, 90, 105}.issubset(calls[0])
    assert {22, 44, 75, 97}.issubset(set(calls[0]+calls[1]))
    assert all(proposals[i] for i in calls[1][:3])
    print('Repeated local candidates trigger fresh samples near both ends without revisiting empty samples.')


def test_residual_proposal_bounds():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import preview_residual_boxes, refine_layer_candidates

    mark = np.ones((10, 20), np.uint8)
    motif = SimpleNamespace(mask=mark, bounds=(0, 0, 20, 10))
    location = (40, 25, 20, 10)
    region = (10, 10, 100, 60)
    residue = [520, 260, 640, 400]
    filled = [310, 260, 490, 400]
    scene = [0, 0, 1000, 1000]
    video = SimpleNamespace(width=120, height=80)
    with (patch('wm_batch.filled_mask_evidence', return_value=('preview', b'image')) as preview,
          patch('wm_batch.locate_named', return_value=[filled, scene, residue])):
        result = preview_residual_boxes(video, 3, region, location, motif, {}, False, lambda *_a, **_kw: None)
    assert result == [residue]
    assert preview.call_args.args[4][0][0] is mark and preview.call_args.args[4][0][1:] == (40, 25)
    video = SimpleNamespace(width=120, height=80, count=4, fps=30)
    gap = SimpleNamespace(frames=[3], hints={3: location}, visible_frames=[], partial_mask=True,
                          needs_classification=False, layer=SimpleNamespace(motif=motif, label='logo'))
    with (patch('wm_batch.frame_evidence', return_value=('source', b'image')),
          patch('wm_batch.locate_named', return_value=[[850, 850, 1000, 1000]]),
          patch('wm_batch.preview_residual_boxes', return_value=[[160, 60, 400, 260]])):
        rows = refine_layer_candidates(video, {}, gap, {}, False, lambda *_a, **_kw: None)[3]
    assert rows[0]['provider'] == 'qwen-visible-remainder'
    assert rows[0]['rect'][3] < location[1]
    assert all(row['rect'][0] < location[0] for row in rows)
    print('Preview proposals keep visible residuals outside the filled pixels for independent source verification.')


def test_dense_replan_sampling():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import refine_layer_candidates

    video = SimpleNamespace(count=100, fps=30, width=160, height=90)
    gap = SimpleNamespace(frames=list(range(20, 61)), hints={}, visible_frames=[40],
                          layer=SimpleNamespace(label='earlier logo'), partial_mask=False, needs_classification=False)
    events = []

    def trace(event, **data):
        events.append((event, data))

    def discover(_video, frames, _cuda, _backend, label, _trace, *, tiled):
        assert not label and tiled
        return {'decision': 'track', 'label': 'fresh visual candidate',
                'observations': [{'frame': i, 'rect': [100, 120, 260, 280]} for i in frames]}

    with (patch('wm_batch.frame_evidence', return_value=('frame', b'image')),
          patch('wm_batch.locate_named', return_value=[]),
          patch('wm_batch.native_tile_boxes', return_value=[]),
          patch('wm_batch.native_discovery', side_effect=discover)):
        proposals = refine_layer_candidates(video, {}, gap, {}, False, trace)
    coarse = next(row['sampled'] for event, row in events if event == 'layers.visible_probes_replanned')
    dense = next(row['frames'] for event, row in events if event == 'layers.dense_replan_proposals')
    assert dense and len(dense) <= 7 and set(dense).isdisjoint(coarse) and set(proposals) == set(dense)
    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation

    signal = np.full((10, 16), 12, np.float32)
    motif = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
    retained = Layer(motif, .7, {Observation(30, (8, 8, 16, 10)): (8, 8, .9)}, 'logo')
    gap = LayerNeedsEvidence(retained, [40], 'remaining copy', visible_frames=[40],
                             hidden_masks={40: [(motif.mask, 8, 8)]})
    with (patch('wm_batch.frame_evidence', return_value=('source', b'image')),
          patch('wm_batch.filled_mask_evidence', return_value=('filled', b'preview')) as filled,
          patch('wm_batch.locate_reference_overlay', return_value=[[50, 90, 150, 200], [500, 400, 650, 600]]),
          patch('wm_batch.locate_named', return_value=[]), patch('wm_batch.native_tile_boxes', return_value=[])):
        proposals = refine_layer_candidates(video, {}, gap, {}, False, trace)
    assert filled.call_args.args[1] == 40 and filled.call_args.args[4][0][0] is motif.mask
    assert proposals == {40: [{'rect': [80, 36, 104, 54], 'provider': 'qwen-visible-remainder'}]}
    other = np.ones((22, 34), np.uint8)
    gap.hidden_masks[40] = [(other, 2, 3)]
    with (patch('wm_batch.frame_evidence', return_value=('source', b'image')),
          patch('wm_batch.filled_mask_evidence', return_value=('filled', b'preview')),
          patch('wm_batch.locate_reference_overlay', return_value=[]),
          patch('wm_batch.locate_named', side_effect=lambda _backend, _label, image, _timeout, **_kw:
                [[138, 80, 213, 200], [500, 400, 650, 600]] if image[0] == 'filled' else []),
          patch('wm_batch.native_tile_boxes', return_value=[])):
        proposals = refine_layer_candidates(video, {}, gap, {}, False, trace)
    assert proposals == {40: [{'rect': [80, 36, 104, 54], 'provider': 'qwen-visible-remainder'}]}
    print('Sparse uncertain motion prompts a bounded, independent dense sample around the reported evidence.')


def test_mask_replan_after_coverage_gap():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import local_tracks
    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation

    video = SimpleNamespace(count=100, width=160, height=90)
    signal = np.ones((12, 20), np.float32)
    motif = Motif(signal, signal, signal.astype(np.uint8))
    matches = {Observation(i, (20, 25, 20, 12)): (20, 25, .8) for i in (24, 26, 28)}
    earlier = Layer(motif, .55, matches, 'earlier overlay')
    new = Layer(motif, .55, matches, 'newly found overlay')
    missing = LayerNeedsEvidence(earlier, [24, 26, 28], 'tracking gap', visible_frames=[26])
    partial = LayerNeedsEvidence(new, [24, 26, 28], 'partial mask', {24: (20, 25, 20, 12)},
                                 partial_mask=True, confirmed=(earlier,))
    coarse = {i: [{'rect': [20, 25, 40, 37]}] for i in missing.frames}
    extension = {24: [{'rect': [12, 25, 40, 37]}]}
    config = {'discovery_local': {}}
    trace = lambda *_args, **_kw: None
    for updated in (extension, coarse):
        with (patch('wm_layers.discover_layers', side_effect=[[earlier], partial, [earlier, new]]) as discover,
              patch('wm_layers.scan_layers', side_effect=[missing, ['track']]) as scan,
              patch('wm_batch.refine_layer_candidates', side_effect=[coarse, updated])):
            if updated is extension:
                tracks, labels = local_tracks(video, {}, config, False, trace)
                assert tracks == ['track'] and labels == [earlier.label, new.label]
                assert discover.call_count == 3 and scan.call_count == 2
                assert discover.call_args.args[1][24] == extension[24]+coarse[24]
                assert discover.call_args.kwargs['retained'] == (earlier,)
            else:
                try:
                    local_tracks(video, {}, config, False, trace)
                except LayerNeedsEvidence as error:
                    assert error is partial
                else:
                    raise AssertionError('Repeated proposals must leave the incomplete mask pending.')
                assert discover.call_count == 2 and scan.call_count == 1
    with (patch('wm_layers.discover_layers', side_effect=[partial, partial]) as discover,
          patch('wm_batch.refine_layer_candidates', return_value=coarse) as refine):
        try:
            local_tracks(video, {}, config, False, trace)
        except LayerNeedsEvidence as error:
            assert error is partial
        else:
            raise AssertionError('An unchanged incomplete mask must stay pending.')
        assert discover.call_count == 2 and refine.call_count == 1
    other_partial = LayerNeedsEvidence(new, [24, 26, 28], 'remaining mask', {24: (12, 25, 28, 12)},
                                       partial_mask=True, confirmed=(earlier,))
    extra = {26: [{'rect': [12, 25, 40, 37]}]}
    with (patch('wm_layers.discover_layers', side_effect=[partial, other_partial, [earlier], [earlier, new]]) as discover,
          patch('wm_layers.scan_layers', side_effect=[missing, ['track']]) as scan,
          patch('wm_batch.refine_layer_candidates', side_effect=[extension, extra, coarse])):
        tracks, labels = local_tracks(video, {}, config, False, trace)
        assert tracks == ['track'] and labels == [earlier.label, new.label] and discover.call_count == 4
        assert [call.kwargs['replan_on_visible'] for call in scan.call_args_list] == [True, False]
    for partial_mask, limit in ((True, 3), (False, 2)):
        gaps = [LayerNeedsEvidence(new, [24], 'new evidence', {24: (20+i, 25, 20, 12)},
                                   partial_mask=partial_mask) for i in range(limit+1)]

        def refine(_video, _candidates, gap, *_args, **_kwargs):
            x, y, width, height = gap.hints[24]
            return {24: [{'rect': [x, y, x+width, y+height]}]}

        with (patch('wm_layers.discover_layers', side_effect=[*gaps[:limit], [earlier]]),
              patch('wm_layers.scan_layers', return_value=['track']),
              patch('wm_batch.refine_layer_candidates', side_effect=refine) as update):
            assert local_tracks(video, {}, config, False, trace)[0] == ['track']
            assert update.call_count == limit
        with (patch('wm_layers.discover_layers', side_effect=gaps),
              patch('wm_batch.refine_layer_candidates', side_effect=refine) as update):
            with np.testing.assert_raises(LayerNeedsEvidence) as pending:
                local_tracks(video, {}, config, False, trace)
            assert pending.exception is gaps[-1] and update.call_count == limit
    print('Bounded replanning preserves earlier layers and stops unchanged proposals.')


def test_layer_review_anchors():
    from wm_discovery import Motif
    from wm_layers import Layer, Observation, review_anchors

    rng = np.random.default_rng(170)
    repeated = rng.normal(0, 6, (70, 110)).astype(np.float32)
    different = rng.normal(0, 6, repeated.shape).astype(np.float32)
    graphic = repeated[16:36, 28:68].copy()
    motif = Motif(graphic, np.ones_like(graphic), np.ones(graphic.shape, np.uint8))
    strengths = [.92, .98, .60, .96, .91, .90]
    matches = {Observation(i, (28, 16, 40, 20)): (28, 16, strength) for i, strength in enumerate(strengths)}
    signals = dict.fromkeys(range(len(matches)), repeated)
    signals[2] = different
    layer = Layer(motif, .55, matches)
    anchors, correlation, baseline = review_anchors(layer, signals)
    assert [item.frame for item, _ in anchors] == [1, 3] and correlation == baseline > .99
    signals[4] = different
    anchors, correlation, baseline = review_anchors(layer, signals)
    assert [item.frame for item, _ in anchors] == [1, 4] and abs(correlation) < .1 and baseline > .99
    print('Visual review selects independent backgrounds only among strong matches.')


def test_local_layer_expansion_review():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from wm_layers import Layer, confirm_layer, review_full_layer

    core = Layer(None, .6, {})
    expanded = Layer(SimpleNamespace(bounds=(0, 0, 40, 16)), .6, {})
    initial_evidence = [(0, (), (20, 24, 32, 16), None), (10, (), (20, 24, 32, 16), None)]
    expanded_evidence = [(0, (), (12, 24, 40, 16), None), (10, (), (12, 24, 40, 16), None)]
    local_partial = [{'decision': 'watermark', 'fit': 'partial'}]*2
    local_uncertain = [{'decision': 'uncertain'}]*2
    reviews = [(initial_evidence, local_partial, .2), (expanded_evidence, local_uncertain, .2)]
    fallback = Mock(return_value={'decision': 'watermark', 'fit': 'complete', 'label': 'test logo'})
    trace = lambda *_args, **_details: None
    with (patch('wm_layers.layer_review', side_effect=reviews),
          patch('wm_layers.complete_layer', return_value=expanded)):
        candidate, status = review_full_layer(None, core, {}, None, False, trace, fallback)
    assert candidate is expanded and status == 'confirmed'
    assert candidate.reviewed_regions == tuple((i, *box) for i, _, box, _ in expanded_evidence)
    fallback.assert_called_once_with(expanded_evidence)
    assert not core.reviewed_regions

    wider = Layer(SimpleNamespace(bounds=(0, 0, 51, 16)), .6, {})
    full_evidence = [(0, (), (1, 24, 51, 16), None), (10, (), (1, 24, 51, 16), None)]
    reviews.append((full_evidence, local_uncertain, .2))
    fallback.reset_mock()
    fallback.side_effect = [{'decision': 'watermark', 'fit': 'partial', 'label': 'test logo'}, fallback.return_value]
    with (patch('wm_layers.layer_review', side_effect=reviews),
          patch('wm_layers.complete_layer', side_effect=[expanded, wider]) as complete):
        candidate, status = review_full_layer(None, core, {}, None, False, trace, fallback)
    assert candidate is wider and status == 'confirmed'
    assert [call.args[0] for call in complete.call_args_list] == [core, expanded]
    assert candidate.reviewed_regions == tuple((i, *box) for i, _, box, _ in expanded_evidence+full_evidence)

    fallback.side_effect = None
    fallback.reset_mock()
    same_account = [{'decision': 'watermark', 'fit': 'complete', 'label': name}
                    for name in ('平台账号：123456789', 'creator ID 123456789')]
    assert confirm_layer(None, core, {}, None, False, trace, fallback,
                         review=(initial_evidence, same_account, .2)) == 'confirmed'
    fallback.assert_not_called()
    same_account[1]['label'] = 'scene number 123456789'
    confirm_layer(None, core, {}, None, False, trace, fallback, review=(initial_evidence, same_account, .2))
    fallback.assert_called_once_with(initial_evidence)
    core = Layer(SimpleNamespace(bounds=(2, 3, 32, 16), mask=np.ones((22, 38), np.uint8)), .6, {})
    preview = {'decision': 'none', 'label': ''}
    fallback.reset_mock()
    with (patch('wm_layers.layer_review', return_value=(initial_evidence, local_partial, .2)),
          patch('wm_layers.filled_mask_evidence', return_value=('preview', b'pixels')) as fill,
          patch('wm_layers.preview_residue', return_value=preview),
          patch('wm_layers.expand_reviewed_layer') as expand):
        candidate, status = review_full_layer(object(), core, {}, None, False, trace, fallback)
        assert candidate is core and status == 'confirmed'
        expand.assert_not_called()
        assert len(fill.call_args_list) == 2 and fill.call_args.args[4][0][1:] == (18, 21)
    fallback.assert_called_once_with(initial_evidence)

    fallback.reset_mock()
    expanded = Layer(SimpleNamespace(bounds=(0, 0, 40, 16)), .6, {})
    with (patch('wm_layers.layer_review', side_effect=reviews[:2]),
          patch('wm_layers.preview_suggests_complete_mask', return_value=True),
          patch('wm_layers.complete_layer', return_value=expanded) as complete):
        candidate, status = review_full_layer(object(), core, {}, None, False, trace, fallback)
        assert candidate is expanded and status == 'confirmed'
        complete.assert_called_once()
    fallback.assert_called_once_with(expanded_evidence)

    fallback.reset_mock()
    fallback.return_value = {'decision': 'watermark', 'fit': 'partial', 'label': 'test logo'}
    with (patch('wm_layers.layer_review', return_value=(initial_evidence, local_partial, .2)),
          patch('wm_layers.filled_mask_evidence', return_value=('preview', b'pixels')),
          patch('wm_layers.preview_residue', return_value=preview),
          patch('wm_layers.expand_reviewed_layer', return_value=(core, 'partial')) as expand):
        candidate, status = review_full_layer(object(), core, {}, None, False, trace, fallback)
        assert candidate is core and status == 'partial'
        expand.assert_called_once()
    fallback.assert_called_once_with(initial_evidence)
    print('A clean local preview checks for repeated source pixels before choosing which mask to review.')


def test_registered_identifier_proposals():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from wm_layers import Layer, Observation, registered_identifier_proposals

    video = SimpleNamespace(width=128, height=96, planes=lambda raw: (raw,))
    core = SimpleNamespace(width=32, height=24, bounds=(3, 2, 27, 18))
    layer = Layer(core, .7, {})
    frames = (2, 8, 21, 38, 45, 61)
    sampled = {i: (Observation(i, (x, 30, 32, 24)), (x, 30, .9))
               for i, x in zip(frames, (21, 28, 34, 22, 25, 27), strict=True)}
    trace = Mock()
    reader = Mock(return_value=[('example.com', [27, 29, 51, 38])])
    with (patch('wm_layers.get_frame', return_value=np.full((96, 128), 90, np.uint8)) as decode,
          patch('wm_layers.locate_source_overlays', return_value=[('example.com', [10, 20, 900, 800])]) as visual):
        proposed = registered_identifier_proposals(video, layer, sampled, {}, True, trace, reader)
        assert proposed['example.com'][2] == [{'rect': [28, 39, 52, 48]}]
        assert set(proposed['example.com']) == set(frames)
        assert [call.args[1] for call in decode.call_args_list] == [2, 21, 45, 8, 38, 61]
        assert visual.call_count == 2
        visual.reset_mock()
        reader.side_effect = [[('example.com', [27, 29, 51, 38])], [('other.net', [27, 29, 51, 38])]]
        assert not registered_identifier_proposals(video, layer, sampled, {}, True, trace, reader)
        visual.assert_not_called()
    print('Registered source text needs matching locations in two groups before it can propose native boxes.')


def test_weak_match_replan():
    from functools import partial
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import confirm_fading_frames, confirm_layer_match, confirm_short_frames
    from wm_layers import Layer, Observation, review_short_weak_gaps, review_weak_matches

    video = SimpleNamespace(width=144, height=96)
    motif = SimpleNamespace(bounds=(0, 0, 40, 16), mask=np.ones((16, 40), np.uint8))
    layer = Layer(motif, .7, {}, label='play badge at example.com')
    required = []
    config = {'discovery_local': {}, 'required_overlay_regions': required, 'timeout_seconds': 60}
    check = partial(confirm_layer_match, video, config=config, cuda=False, trace=lambda *_a, **_kw: None)
    points = [(20, 24, .65), (52, 24, .66)]
    evidence = [([], [point], point[2]) for point in points]
    answers = [{'decision': 'matches', 'fit': 'complete', 'label': label} for label in
               ('different.example', 'www.example.com')]
    with (patch('wm_batch.frame_evidence', return_value=('frame', b'image')),
          patch('wm_batch.ask_local', side_effect=answers)):
        accepted, replans = review_weak_matches(layer, evidence, set(), set(), 12, check)
    assert accepted == 1 and replans == {0: points[0]}
    assert evidence[0][0] == [] and required == [[1, 52, 24, 40, 16]]
    reference = (20, 24, .95)
    short = Layer(motif, .7, {Observation(8, (20, 24, 40, 16)): reference}, layer.label)
    faint = (18, 25, .44)
    evidence = [([], points.copy(), .66), ([], [faint], .44), ([], points.copy(), .66), ([], [faint], .44)]
    alternatives = {}

    def screen(_layer, frame, point):
        if point == points[1]:
            return 'replan' if frame == 0 else 'unverified'
        return 'matches'

    def confirm(_layer, probes, anchor):
        assert anchor == (8, reference)
        expected = [(0, points[0]), (1, faint), (2, points[0]), (3, faint)]
        assert probes in (expected[:3], expected[3:])
        return {frame for frame, _ in probes if frame != 3}

    recovered, pending = review_short_weak_gaps(short, evidence, set(), set(), alternatives, screen, confirm,
                                                lambda *_a, **_kw: None)
    assert recovered == {0, 1, 2} and pending == {2} and alternatives == {0: points[1]}
    assert [hits for hits, _, _ in evidence] == [[points[0]], [faint], [points[0]], []]
    secondary = (86, 24, .66)
    overlapping = (21, 24, .66)
    evidence = [([reference], [point], .95) for point in (secondary, overlapping)]
    screened = []

    def screen_secondary(_layer, frame, point):
        screened.append((frame, point))
        return 'matches'

    def confirm_secondary(_layer, probes, anchor):
        assert probes == [(0, secondary)] and anchor == (8, reference)
        return {0}

    proofs = {}
    recovered, pending = review_short_weak_gaps(short, evidence, set(), set(), {}, screen_secondary,
                                                confirm_secondary, lambda *_a, **_kw: None,
                                                secondary=True, proofs=proofs)
    assert screened == [(0, secondary)] and recovered == {0} and not pending
    assert evidence[0][0] == [reference, secondary] and evidence[1][1] == [overlapping]
    assert proofs == {'short_secondary:0:86:24:8:20:24': [0]}
    trace = lambda *_a, **_kw: None
    same = {'decision': 'same', 'fit': 'complete', 'reason': 'Matching artwork'}
    with (patch('wm_batch.frame_evidence', return_value=('frame', b'image')),
          patch('wm_batch.ask_local', return_value=same),
          patch('wm_batch.ask_with_timeout_recovery', side_effect=[{'decision': 'matches'},
                {'decision': 'review', 'checks': [{'frame': 2, 'fit': 'complete', 'identity': 'same'},
                                                  {'frame': 3, 'fit': 'complete', 'identity': 'different'}]}])):
        probes = [(2, points[1]), (3, points[1])]
        anchor = (0, points[0])
        assert not confirm_fading_frames(video, layer, probes, anchor, config, False, trace)
        assert len(required) == 1
        assert confirm_fading_frames(video, layer, probes, anchor, config, False, trace) == {2}
    assert required[-1] == [2, 52, 24, 40, 16]
    different = {'decision': 'different', 'fit': 'complete'}
    cloud = {'decision': 'matches', 'checks': [{'frame': 4, 'fit': 'complete', 'identity': 'same'}]}
    with (patch('wm_batch.frame_evidence', return_value=('frame', b'image')),
          patch('wm_batch.ask_local', side_effect=[different, {'decision': 'same'}, different,
                                                 {'decision': 'different'}]) as local,
          patch('wm_batch.ask_with_timeout_recovery', return_value=cloud)):
        assert confirm_fading_frames(video, layer, [(4, points[1]), (5, points[1])], anchor,
                                     config, False, trace) == {4}
        assert all('reference' in local.call_args_list[i].kwargs for i in (1, 3))
    assert required[-1] == [4, 52, 24, 40, 16]
    probes = [(4, points[0]), (5, points[1])]
    anchor = (0, points[0])

    def timeout_then_split(_video, _layer, group, _anchor, _config, _cuda, log):
        if len(group) > 1:
            log('layers.fading_match_review', answer={'decision': 'review', 'reason': 'Grok timed out during the API visual request.'})
            return set()
        return {4} if group[0][0] == 4 else set()

    with patch('wm_batch.confirm_fading_frames', side_effect=timeout_then_split) as grouped:
        assert confirm_short_frames(video, layer, probes, anchor, config, False, trace) == {4}
        assert [call.args[2] for call in grouped.call_args_list] == [probes, probes[:1], probes[1:]]

    def uncertain(_video, _layer, _group, _anchor, _config, _cuda, log):
        log('layers.fading_match_review', answer={'decision': 'review', 'reason': 'faint',
                                                  'checks': [{'frame': 4, 'identity': 'uncertain'}]})
        return set()

    with patch('wm_batch.confirm_fading_frames', side_effect=uncertain) as grouped:
        assert not confirm_short_frames(video, layer, probes, anchor, config, False, trace)
        grouped.assert_called_once()
    print('A contradictory identifier triggers a new proposal while a verified match extends the track.')


def test_alternate_identifier_review():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import confirm_alternate_identifiers, read_native_identifier
    from wm_layers import Layer, Observation

    layer = Layer(None, .7, {Observation(0, (0, 0, 16, 10)): (0, 0, .95)}, 'brand at example.ai')
    matches = {i: ((0, 0, .6), {'example.com'}) for i in range(1, 6)}
    matches[6] = ((0, 0, .6), {'@example'})
    hide = {i: [(None, 0, 0)] for i in matches}
    different = {'decision': 'variant', 'reference_identifier': 'example.ai', 'candidate_identifier': 'example.com'}
    local = [different, different, different, {**different, 'candidate_identifier': 'different.example'},
             {**different, 'decision': 'same', 'candidate_identifier': 'example.ai'},
             {**different, 'candidate_identifier': '@example'}]

    def check(frame, remaining, **override):
        return {**different, 'frame': frame, 'identity': 'variant', 'remaining': remaining, **override}

    cloud = [{'checks': [check(1, 'absent'), check(2, 'present')]},
             {'checks': [check(3, 'absent', candidate_identifier='different.example')]}]
    trace = lambda *_args, **_kw: None
    config = {'discovery_local': {}, 'timeout_seconds': 75}
    with (patch('wm_batch.layer_match_evidence', return_value=(0, (), (), ('raw', b'raw'))),
          patch('wm_batch.ask_local', side_effect=local),
          patch('wm_batch.filled_mask_evidence', return_value=('full', b'full')) as preview,
          patch('wm_batch.ask_with_timeout_recovery', side_effect=cloud)):
        assert confirm_alternate_identifiers(None, layer, matches, config, False, trace, hide=hide) == {1}
    assert [call.args[1] for call in preview.call_args_list] == [1, 2, 3]
    video = SimpleNamespace(width=80, height=60)
    footprint = (6, 8, 30, 20)
    ocr = [('example.ai', [8, 10, 23, 18]), ('example.ai', [8, 11, 24, 18])]
    native = [('example.ai', [50, 100, 900, 900])]
    with (patch('wm_batch.frame_evidence', return_value=('crop', b'crop')),
          patch('wm_batch.locate_source_overlays', return_value=native) as local_read):
        assert read_native_identifier(video, 0, footprint, {'example.ai'}, {}, False, trace,
                                      lambda *_: ocr) == 'example.ai'
        assert local_read.call_args.kwargs['read_labels'] is True
        assert read_native_identifier(video, 0, footprint, {'example.ai'}, {}, False, trace,
                                      lambda *_: ocr[:1]) is None
        local_read.return_value = [('example.com', native[0][1])]
        assert read_native_identifier(video, 0, footprint, {'example.ai'}, {}, False, trace,
                                      lambda *_: ocr) is None
    with (patch('wm_batch.layer_match_evidence', return_value=(0, (), footprint, ('raw', b'raw'))),
          patch('wm_batch.ask_local', return_value={'decision': 'uncertain'}),
          patch('wm_batch.read_native_identifier', side_effect=['example.ai', 'example.com']),
          patch('wm_batch.filled_mask_evidence', return_value=('full', b'full')),
          patch('wm_batch.ask_with_timeout_recovery', return_value={'checks': [check(1, 'absent')]})):
        assert confirm_alternate_identifiers(video, layer, {1: matches[1]}, config, False, trace,
                                             hide=hide, read_text=lambda *_: ocr) == {1}
    print('Alternate identifiers need independent exact readings and a clean remaining frame.')


def test_layer_temporal_support():
    from wm_layers import temporal_support

    weak = [(42, 50, .58), (44, 51, .57), (45, 51, .54)]
    evidence = [([(40, 49, .85)], [], .85),
                *[([], [point, (170, 26, .56)], point[2]) for point in weak],
                ([(47, 52, .86)], [], .86)]
    assert temporal_support(evidence, set(), 3) == set(enumerate(weak, 1))
    assert not temporal_support(evidence, {2}, 3)
    assert not temporal_support(evidence, set(), 2)
    evidence[2] = ([], [], .32)
    assert not temporal_support(evidence, set(), 3)
    print('Weak matches need independent evidence on both sides and never cross a cut or an empty frame.')


def test_clipped_layer_motion():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, recover_edge_matches, review_anchored_matches
    from wm_media import clipped_plane

    rng = np.random.default_rng(318)
    detail = rng.normal(0, 9, (28, 28)).astype(np.float32)
    mask = np.ones(detail.shape, np.uint8)
    layer = Layer(Motif(detail, mask.astype(np.float32), mask), .73, {})
    video = SimpleNamespace(width=120, height=90, planes=lambda raw: (raw,))
    for axis, reverse in ((0, False), (1, True)):
        images, evidence = [], []
        for i in range(28):
            point = [38, 38]
            point[axis] = 23-2*i
            image = rng.normal(0, .8, (video.height, video.width)).astype(np.float32)
            if i != 20:
                visible, region = clipped_plane(image, (*point, layer.motif.width, layer.motif.height))
                if i not in (18, 19):
                    visible += detail[region]
                else:
                    dimension = 1-axis
                    split = visible.shape[dimension]//2
                    halves = zip(np.split(visible, [split], axis=dimension),
                                 np.split(detail[region], [split], axis=dimension), strict=True)
                    for noise, target in halves:
                        power = np.sum(target*target)
                        noise -= np.sum(noise*target)/power*target
                        noise *= np.sqrt((1-.66**2)*power/np.sum(noise*noise))
                        noise += .66*target
            images.append(image)
            evidence.append(([(*point, .98)] if i <= 13 else [], [], .98 if i <= 13 else .15))
        cuts = {22}
        expected = set(range(14, 18))
        weak = {18, 19}
        if reverse:
            images.reverse()
            evidence.reverse()
            cuts = {len(images)-22}
            expected = {len(images)-1-i for i in expected}
            weak = {len(images)-1-i for i in weak}

        def read(_video, start, stop, _cuda):
            return enumerate(images[start:stop], start)

        initial = {i for i, (hits, _, _) in enumerate(evidence) if hits}
        hints = {}
        with (patch('wm_layers.read_frames', side_effect=read),
              patch('wm_layers.highpass', side_effect=lambda image, _: image)):
            count = recover_edge_matches(video, layer, evidence, cuts, (video.width, video.height), False,
                                         lambda *_, **__: None, weak_matches=hints)
        recovered = {i for i, (hits, _, _) in enumerate(evidence) if hits}-initial
        assert recovered == expected and count == len(expected)
        assert set(hints) == weak
        first = min(weak)

        def review(_layer, probes, anchor):
            assert anchor[0] in initial
            assert {frame for frame, _ in probes} == weak
            return {first}

        probes = [(frame, point, anchor) for frame, (point, anchor) in hints.items()]
        checked = review_anchored_matches(layer, probes, evidence, review, lambda *_, **__: None, 'clipped')
        assert checked == {first} and evidence[first][0] and not evidence[max(weak)][0]
    print('Clipped motion verifies visible pixels in both directions and stops at a cut.')


def test_regional_match_and_residual():
    from wm_discovery import Motif, response
    from wm_discovery_render import projected_residual
    from wm_gpu_search import search_input
    from wm_layers import locations, regional_match

    rng = np.random.default_rng(184)
    detail = rng.normal(0, 10, (20, 48)).astype(np.float32)
    motif = Motif(detail, np.ones_like(detail), np.ones(detail.shape, np.uint8))
    origin = 35, 24
    source = np.zeros((75, 130), np.float32)
    source[24:44, 35:83] = detail
    source[24:44, 59:83] *= -1
    found = regional_match(motif, source, (34, 25))
    assert found is not None and found[:2] == origin and found[2] > .99
    half_restored = source.copy()
    half_restored[24:44, 59:83] = 0
    assert regional_match(motif, half_restored, origin) is None
    assert projected_residual(motif, source, half_restored, *origin, regional=True) >= .99
    assert projected_residual(motif, source, np.zeros_like(source), *origin, regional=True) == 0
    border = np.zeros_like(source)
    border[24:44, :5] = detail[:, -5:]
    assert projected_residual(motif, border, border, -43, 24, regional=True) >= .99
    assert projected_residual(motif, border, np.zeros_like(border), -43, 24, regional=True) == 0
    score_map = response(*search_input(motif, source))
    exact = locations(motif, source)
    assert np.allclose(locations(motif, source, score_map=score_map), exact, atol=.0001)
    assert locations(motif, source, score_map=np.ones_like(score_map)) == exact
    print('Local tracking and residual review use separate evidence on both sides of a contrast change.')


def test_moving_regional_scan():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, Observation, scan_layers
    from wm_track import highpass

    rng = np.random.default_rng(1402)
    field = rng.integers(-24, 25, (24, 50), dtype=np.int16)
    shape = field.shape
    positions = [(18+3*i, 24+i) for i in range(14)]
    video = Video(Path('moving-contrast-test.mp4'), 160, 96, Fraction(25), len(positions), 0,
                  '1:1', 'bt709', 'bt709', 'bt709')
    frames = []
    for i, (x, y) in enumerate(positions):
        frame = np.full(video.width*video.height*3//2, 128, np.uint8)
        detail = field.copy()
        if i >= 4:
            detail[:, shape[1]//2:] *= -1
        video.planes(frame)[0][y:y+shape[0], x:x+shape[1]] = 128+detail
        frames.append(frame)
    x, y = positions[0]
    signal = highpass(video.planes(frames[0])[0], 2.2)[y:y+shape[0], x:x+shape[1]].copy()
    motif = Motif(signal, np.ones_like(signal), np.ones(shape, np.uint8))
    matches = {Observation(i, (x, y, shape[1], shape[0])): (x, y, .98)
               for i, (x, y) in enumerate(positions[:3])}
    with patch('wm_layers.read_frames', return_value=enumerate(frames)):
        tracks = scan_layers(video, [Layer(motif, .73, matches)], False, lambda *_args, **_details: None)
    assert len(tracks) == 1 and tracks[0]['present'].all()
    assert np.abs(tracks[0]['positions']-positions).max() <= 1
    print('Moving watermark follows independently verified pixels when local contrast inverts.')


def test_variant_track_coverage():
    from fractions import Fraction
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, scan_layers

    def motif(w, h):
        mask = np.ones((h, w), np.uint8)
        values = np.full(mask.shape, 8, np.float32)
        return Motif(values, np.ones_like(values), mask)

    a, b = motif(16, 10), motif(22, 14)
    video = SimpleNamespace(count=12, fps=Fraction(30), width=100, height=60, planes=lambda frame: (frame,))
    samples = [(i, np.full((60, 100), i, np.uint8)) for i in range(video.count)]
    layers = [Layer(a, .73, {}, 'first'), Layer(b, .73, {}, 'second')]
    extra = False
    similar_variant = False
    simultaneous = False
    partial_replacement = False
    faint_variant = False

    def peaks(mark, frame):
        i = frame[0, 0]
        if mark is a:
            if i < 6:
                return [(20, 20, .92), (72, 20, .20)]
            if simultaneous:
                return [(65, 20, .93), (19, 20, .67), (60, 40, .10)]
            if similar_variant:
                score = {8: .42, 9: .375}.get(i, .41)
                return [(19, 20, score), (72, 20, .20)]
            if partial_replacement:
                return [(19, 20, .45 if faint_variant else .62), (72, 20, .58 if extra and i == 8 else .20)]
            return [(20, 20, .62), (72, 20, .58 if extra and i == 8 else .20), (60, 40, .10)]
        return [(20, 20, .94 if i >= 6 else .20), (72, 20, .12)]

    def same_shape(mark, frame, *_args):
        return not (similar_variant and mark is a and frame[0, 0] >= 6)

    def outside_mask(layer, probes, *, hide):
        assert set(hide) == set(probes)
        if layer is layers[1]:
            assert all(i < 6 for i in probes)
            assert all(mask is a.mask and (x, y) == (20, 20) for ((mask, x, y),) in hide.values())
            return dict.fromkeys(probes, 'absent')
        assert layer is layers[0]
        for covered in hide.values():
            (mask, x, y), = covered
            assert mask is b.mask and (x, y) == (20, 20)
        return {i: 'present' if extra and i == 8 else 'absent' for i in probes}

    with (patch('wm_layers.read_frames', return_value=samples), patch('wm_layers.highpass', side_effect=lambda data, _: data),
          patch('wm_layers.locations', side_effect=peaks), patch('wm_layers.agrees_in_halves', side_effect=same_shape),
          patch('wm_layers.regional_match', return_value=None)):
        options = {'confirm_absence': lambda layer, probes: dict.fromkeys(probes, 'absent') if layer.motif is b else {}}
        trace = lambda *_args, **_kw: None
        tracks = scan_layers(video, layers, False, trace, **options)
        assert [int(track['present'].sum()) for track in tracks] == [6, 6]
        extra = True
        try:
            scan_layers(video, layers, False, trace, **options)
        except LayerNeedsEvidence as error:
            assert error.frames == [8]
        else:
            raise AssertionError('A separate possible watermark was covered by an unrelated mask.')
        similar_variant = True
        options.update(confirm_borderline=lambda *_a: {}, confirm_masked=outside_mask)
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, layers, False, trace, **options)
        assert pending.exception.visible_frames == [8]
        layers[0].label, layers[1].label = 'first example.ai', 'second example.com'

        def exact_identifier(layer, cases, *, hide):
            assert layer is layers[0] and cases == {8: ((19, 20, .42), {'example.com'})}
            assert hide[8][0][0] is b.mask
            return {8}

        tracks = scan_layers(video, layers, False, trace, confirm_identifiers=exact_identifier, **options)
        assert [int(track['present'].sum()) for track in tracks] == [6, 6]
        extra = False
        tracks = scan_layers(video, layers, False, trace, **options)
        assert [int(track['present'].sum()) for track in tracks] == [6, 6]

        similar_variant, partial_replacement = False, True
        tracks = scan_layers(video, layers, False, trace, **options)
        assert [int(track['present'].sum()) for track in tracks] == [6, 6]
        extra = True
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, layers, False, trace, **options)
        assert pending.exception.frames == [8]
        extra = False
        faint_variant = True

        def faint_identifiers(layer, cases, *, hide):
            assert layer is layers[0] and set(cases) == set(hide)
            assert all(point == (19, 20, .45) and identifiers == {'example.com'} for point, identifiers in cases.values())
            return set(cases)

        def reject_untyped(layer, probes, *, hide):
            if layer is layers[1]:
                return outside_mask(layer, probes, hide=hide)
            raise AssertionError('A faint overlapping variant needs exact identifier evidence.')

        options['confirm_masked'] = reject_untyped
        tracks = scan_layers(video, layers, False, trace, confirm_identifiers=faint_identifiers, **options)
        assert [int(track['present'].sum()) for track in tracks] == [6, 6]
        faint_variant = False

        def check_extra(layer, probes, *, hide):
            if layer is layers[1]:
                return outside_mask(layer, probes, hide=hide)
            for covered in hide.values():
                assert len(covered) == 2 and covered[0][0] is b.mask and covered[1][0] is a.mask
            return {i: 'present' if extra and i == 8 else 'absent' for i in probes}

        similar_variant, simultaneous = False, True
        options['confirm_masked'] = check_extra
        tracks = scan_layers(video, layers, False, trace, **options)
        assert [int(track['present'].sum()) for track in tracks] == [12, 6]
        extra = True
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, layers, False, trace, **options)
        assert pending.exception.frames == [8]
    print('Variant tracking reconciled a covered candidate and retained an independent unmasked candidate.')


def test_fading_match_coverage():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation, scan_layers

    def motif(shape):
        mask = np.ones(shape, np.uint8)
        return Motif(mask.astype(np.float32)*12, mask.astype(np.float32), mask)

    a, b = motif((10, 18)), motif((14, 26))
    layer = Layer(b, .73, {Observation(i, (20, 20, 26, 14)): (20, 20, .93) for i in range(4, 7)}, 'second')
    layers = [Layer(a, .73, {}, 'first'), layer]
    video = SimpleNamespace(count=14, fps=30, width=100, height=60, planes=lambda frame: (frame,))
    frames = [(i, np.full((60, 100), i, np.uint8)) for i in range(video.count)]
    inspected = []

    def locations(mark, frame):
        i = frame[0, 0]
        if mark is a:
            score = .90 if i < 4 else .41 if i < 10 else .20
        else:
            score = .95 if 4 <= i < 7 else .84 if 7 <= i < 10 else .20
        return [(20, 20, score), (62, 30, .12)]

    def halves(mark, frame, *_args):
        i = frame[0, 0]
        return i < 4 if mark is a else 4 <= i < 7

    def verify(target, probes, anchor):
        assert target is layer and all(point[:2] == (20, 20) for _, point in probes)
        assert anchor == (6, (20, 20, .95))
        inspected.extend(frame for frame, _ in probes)
        return {frame for frame, _ in probes}

    trace = lambda *_a, **_kw: None
    absent = lambda _layer, probes, **_kw: dict.fromkeys(probes, 'absent')
    options = {'confirm_absence': absent, 'confirm_borderline': lambda *_a: {}, 'confirm_masked': absent}
    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.locations', side_effect=locations), patch('wm_layers.agrees_in_halves', side_effect=halves),
          patch('wm_layers.regional_match', return_value=None), patch('wm_layers.translucent_reference', return_value=None)):
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, layers, False, trace, **options)
        assert pending.exception.frames == [7, 8, 9]
        tracks = scan_layers(video, layers, False, trace, confirm_fade=verify, **options)
        assert inspected == [7, 8, 9]
        assert np.flatnonzero(tracks[0]['present']).tolist() == [0, 1, 2, 3]
        assert np.flatnonzero(tracks[1]['present']).tolist() == list(range(4, 10))
    print('Two-model fade checks extend an anchored track before testing other overlay masks.')


def test_global_chroma_gap():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation, global_chroma_point, scan_layers

    signal = np.full((12, 20), 12, np.float32)
    motif = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
    layer = Layer(motif, .73, {Observation(i, (20, 14, 20, 12)): (20, 14, .9) for i in range(3)}, 'logo')
    video = SimpleNamespace(count=6, fps=30, width=90, height=60, planes=lambda raw: (raw,))
    frames = [(i, np.full((60, 90), i, np.uint8)) for i in range(video.count)]
    primary, distractor = (50, 30, .91), (7, 8, .80)
    with patch('wm_layers.chroma_signals', return_value=(None, None)):
        with patch('wm_layers.locations', side_effect=[[primary, distractor], [(50, 31, .86), (8, 8, .52)]]):
            assert global_chroma_point([motif, motif], ()) == (50, 30, .86)
        with patch('wm_layers.locations', side_effect=[[primary, distractor], [(50, 31, .86), (8, 8, .82)]]):
            assert global_chroma_point([motif, motif], ()) is None

    def remains(_layer, probes, *, hide):
        assert probes == [3] and hide[3][0][0] is motif.mask and hide[3][0][1:] == (50, 30)
        return {3: outcome}

    trace = lambda *_a, **_kw: None
    visible = lambda _layer, probes: dict.fromkeys(probes, 'present')
    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.get_frame', side_effect=lambda _video, i, _cuda: frames[i][1]),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame.astype(np.float32)),
          patch('wm_layers.locations', side_effect=lambda _motif, frame: [(20, 14, .31 if frame[0, 0] == 3 else .92)]),
          patch('wm_layers.agrees_in_halves', return_value=True),
          patch('wm_layers.color_reference', return_value=[motif, motif]),
          patch('wm_layers.chroma_signals', return_value=()),
          patch('wm_layers.chroma_position', return_value=None),
          patch('wm_layers.regional_match', return_value=None),
          patch('wm_layers.translucent_reference', return_value=None),
          patch('wm_layers.global_chroma_point', return_value=(50, 30, .86))):
        settings = {'confirm_borderline': visible, 'confirm_masked': remains,
                    'screen_short': lambda *_a: 'matches', 'confirm_fade': lambda *_a: {3}}
        outcome = 'present'
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, [layer], False, trace, **settings)
        assert pending.exception.frames == [3]
        assert pending.exception.hidden_masks[3][0][0] is motif.mask
        outcome = 'absent'
        tracks = scan_layers(video, [layer], False, trace, **settings)
    assert tracks[0]['present'].all() and tracks[0]['positions'][3].tolist() == [50, 30]
    print('A two-channel proposal needs independent confirmation and an empty remainder before the gap clears.')


def test_identifier_gate():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import unverified_identifier_frames

    signal = np.random.default_rng(49).choice([-12., 12.], (10, 18)).astype(np.float32)
    text = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
    first, missing, extra = np.zeros((3, 55, 80), np.float32)
    first[12:22, 15:33] = signal
    missing[12:22, 15:24] = signal[:, :9]
    extra[12:22, 15:33] = signal
    frames = list(enumerate([first, missing, extra]))
    claims = [[(10, 8, .9)], [(10, 8, .9)], [(10, 8, .9), (40, 30, .8)]]
    video = SimpleNamespace(planes=lambda image: (image,))
    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.highpass', side_effect=lambda image, _: image)):
        rejected = unverified_identifier_frames(video, (text, 5, 4), claims, False, lambda *_a, **_kw: None)
    assert rejected == [1, 2]
    print('Identifier mask coverage rejects a missing text half or an unverified second placement.')


def test_dense_copy_tracking():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import MAX_LAYERS, Motif, discovery_tracks, pack_tracks
    from wm_layers import Layer, LayerNeedsEvidence, Observation, scan_layers

    signal = np.random.default_rng(59).choice([-12., 12.], (16, 26)).astype(np.float32)
    mark = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
    video = SimpleNamespace(count=4, fps=30, width=45*(MAX_LAYERS+2), height=120, planes=lambda pixels: (pixels,))
    points = [(12+45*i, 20+32*(i % 2)) for i in range(MAX_LAYERS+1)]
    x, y = points[0]
    layer = Layer(mark, .69, {Observation(i, (x, y, 26, 16)): (x, y, 1.) for i in range(3)}, 'confirmed logo')

    def scan(count):
        image = np.zeros((video.height, video.width), np.float32)
        for a, b in points[:count]:
            image[b:b+16, a:a+26] = signal
        with (patch('wm_layers.read_frames', return_value=list(enumerate([image]*video.count))),
              patch('wm_layers.highpass', side_effect=lambda pixels, _: pixels),
              patch('wm_layers.translucent_reference', return_value=None)):
            return scan_layers(video, [layer], False, lambda *_a, **_kw: None)

    tracks = discovery_tracks(pack_tracks(scan(5)))
    assert len(tracks) == 5 and all(track['present'].all() for track in tracks)
    assert {tuple(track['positions'][0]) for track in tracks} == set(points[:5])
    assert len(scan(MAX_LAYERS)) == MAX_LAYERS
    with np.testing.assert_raises(LayerNeedsEvidence) as pending:
        scan(MAX_LAYERS+1)
    assert pending.exception.frames == list(range(video.count))
    second = np.random.default_rng(97).choice([-12., 12.], signal.shape).astype(np.float32)
    other = Layer(Motif(second, mark.weights, mark.mask), .69,
                  {Observation(i, (x, y, 26, 16)): (x, y, 1.) for i in range(4, 7)}, 'another confirmed logo')
    first_frames, other_frames = [], []
    for pattern, frames in ((signal, first_frames), (second, other_frames)):
        image = np.zeros((video.height, video.width), np.float32)
        for a, b in points[:4]:
            image[b:b+16, a:a+26] = pattern
        frames.extend([image]*4)
    video.count = 8
    with (patch('wm_layers.read_frames', return_value=list(enumerate(first_frames+other_frames))),
          patch('wm_layers.highpass', side_effect=lambda pixels, _: pixels),
          patch('wm_layers.translucent_reference', return_value=None)):
        tracks = scan_layers(video, [layer, other], False, lambda *_a, **_kw: None,
                             confirm_absence=lambda _, frames: dict.fromkeys(frames, 'absent'))
    stored = discovery_tracks(pack_tracks(tracks))
    assert len(stored) == 8 and np.all(np.sum([track['present'] for track in stored], axis=0) == 4)
    print('Sixty independently matched copies remain separate and an overflow frame stays pending.')


def test_retained_layer_capacity():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import MAX_LAYERS, MAX_TEMPLATES, Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation, discover_layers

    signal = np.full((16, 24), 12, np.float32)
    motif = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
    retained = [Layer(motif, .72, {Observation(0, (10, 10, 24, 16)): (10, 10, .95)}, str(i))
                for i in range(MAX_TEMPLATES)]
    frame = Observation(10, (50, 40, 24, 16))
    added = Layer(motif, .72, {frame: (50, 40, .9)}, 'another proposal')
    rows = {10: [{'rect': [50, 40, 74, 56]}]}
    video = SimpleNamespace(count=30, width=100, height=80, planes=lambda pixels: (pixels,))
    trace = lambda *_a, **_kw: None
    with (patch('wm_layers.get_frame', return_value=np.zeros((80, 100), np.uint8)),
          patch('wm_layers.build_layer', side_effect=[added, added]),
          patch('wm_layers.covered_fragment', side_effect=[added.matches, *([None]*MAX_TEMPLATES)]),
          patch('wm_layers.review_full_layer', return_value=(added, 'confirmed')) as review):
        result = discover_layers(video, rows, {}, False, trace, retained=retained)
        assert len(result) == MAX_TEMPLATES and frame in result[0].matches and review.call_count == 0
        with np.testing.assert_raises_regex(LayerNeedsEvidence, 'exceed the repair layer limit'):
            discover_layers(video, rows, {}, False, trace, retained=retained)
        assert review.call_count == 1
    with (patch('wm_layers.get_frame', return_value=np.zeros((80, 100), np.uint8)),
          patch('wm_layers.build_layer', side_effect=[added, None]),
          patch('wm_layers.covered_fragment', return_value=None),
          patch('wm_layers.review_full_layer', return_value=(added, 'confirmed'))):
        result = discover_layers(video, rows, {}, False, trace, retained=retained[:MAX_LAYERS])
        assert len(result) == MAX_LAYERS+1 and result[-1] is added
    proposals = {i: [{'rect': [50, 40, 74, 56]}] for i in range(4)}
    hypotheses = [Layer(motif, .72, {Observation(i, (50, 40, 24, 16)): (50, 40, .90)}, str(i))
                  for i in proposals]
    classified = list(zip(hypotheses, ('scene', 'scene', 'confirmed', 'confirmed'), strict=True))
    with (patch('wm_layers.MAX_TEMPLATES', 2),
          patch('wm_layers.get_frame', return_value=np.zeros((80, 100), np.uint8)),
          patch('wm_layers.build_layer', side_effect=hypotheses),
          patch('wm_layers.covered_fragment', return_value=None),
          patch('wm_layers.review_full_layer', side_effect=classified)):
        assert discover_layers(video, proposals, {}, False, trace) == hypotheses[2:]
    print('A full set of retained templates can still absorb verified fragments and flags a new overlay.')


def test_verified_stationary_mask_review():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, Observation, scan_layers

    def motif(value):
        image = np.full((12, 16), value, np.float32)
        return Motif(image, np.ones_like(image), np.ones(image.shape, np.uint8))

    fixed, second = motif(12), motif(18)
    layers = [Layer(mark, .7, {Observation(i, (*xy, 16, 12)): (*xy, .95) for i in range(3)}, label)
              for mark, xy, label in [(fixed, (4, 8), 'fixed'), (second, (55, 29), 'second')]]
    video = SimpleNamespace(count=12, fps=30, width=90, height=60, planes=lambda frame: (frame,))
    frames = [(i, np.full((60, 90), i, np.uint8)) for i in range(video.count)]
    inspected = []

    def locate(mark, frame):
        if mark is fixed:
            return [(4, 8, .95)]
        return [(55, 29, .95 if frame[0, 0] < 3 else .22)]

    def hidden_absence(target, probes, *, hide):
        assert target is layers[1] and set(hide) == set(probes)
        for mask, x, y in (hide[frame][0] for frame in probes):
            assert mask is fixed.mask and (x, y) == (4, 8)
        inspected.extend(probes)
        return dict.fromkeys(probes, 'absent')

    trace = lambda *_a, **_kw: None
    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.locations', side_effect=locate),
          patch('wm_layers.agrees_in_halves', return_value=True),
          patch('wm_layers.regional_match', return_value=None),
          patch('wm_layers.translucent_reference', return_value=None)):
        tracks = scan_layers(video, layers, False, trace, confirm_masked=hidden_absence)
    assert inspected and all(i >= 3 for i in inspected)
    assert tracks[0]['present'].all() and np.flatnonzero(tracks[1]['present']).tolist() == [0, 1, 2]
    print('A fully verified stationary edit is hidden while checking another layer outside its pixels.')


def test_variant_mask_review():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_layers import Layer, LayerNeedsEvidence, Observation, scan_layers

    def layer(label, width, origin):
        signal = np.full((12, width), 12, np.float32)
        motif = Motif(signal, np.ones_like(signal), np.ones(signal.shape, np.uint8))
        return Layer(motif, .7, {Observation(i, (*origin, width, 12)): (*origin, .95) for i in range(3)}, label)

    small = layer('small logo example.com', 16, (4, 8))
    large = layer('large logo example.net', 28, (42, 29))
    video = SimpleNamespace(count=20, fps=30, width=100, height=60, planes=lambda frame: (frame,))
    frames = [(i, np.full((60, 100), i, np.uint8)) for i in range(video.count)]

    def locate(mark, frame):
        i = int(frame[0, 0])
        if mark is small.motif:
            return [(4, 8, .95 if i < 4 else .33 if i == 12 else .22)]
        return [(42+i, 29, .95 if 12 <= i <= 14 else .22)]

    def review(target, probes, *, hide):
        assert target is small and probes == [12, 9] and set(hide) == {12}
        mask, x, y = hide[12][0]
        assert mask is large.motif.mask and (x, y) == (54, 29)
        return {12: verdict, 9: 'absent'}

    trace = lambda *_a, **_kw: None
    options = {'confirm_absence': lambda _, probes: dict.fromkeys(probes, 'absent'), 'confirm_masked': review}
    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.locations', side_effect=locate),
          patch('wm_layers.agrees_in_halves', return_value=True),
          patch('wm_layers.regional_match', return_value=None),
          patch('wm_layers.translucent_reference', return_value=None)):
        verdict = 'present'
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, [small, large], False, trace, **options)
        assert pending.exception.visible_frames == [12]
        mask, x, y = pending.exception.hidden_masks[12][0]
        assert mask is large.motif.mask and (x, y) == (54, 29)
        verdict = 'absent'
        tracks = scan_layers(video, [small, large], False, trace, **options)
    assert np.flatnonzero(tracks[0]['present']).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(tracks[1]['present']).tolist() == [12, 13, 14]
    print('Absence probes hide only independently located alternate variants and retain visible residuals.')


def test_absence_budget():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import review_layer_visibility
    from wm_discovery import Motif, absence_probes, verify_presence
    from wm_layers import Layer, LayerNeedsEvidence, Observation, absence_intervals, scan_layers, uncovered_absence

    mask = np.ones((10, 16), np.uint8)
    values = mask.astype(np.float32)*8
    layer = Layer(Motif(values, np.ones_like(values), mask), .73, {Observation(0, (20, 20, 16, 10)): (20, 20, .92)})
    video = SimpleNamespace(count=70, width=100, height=60, fps=30, planes=lambda raw: (raw,))
    low, strict = absence_intervals([1, 2, 3, 4, 6, 7], [5], set())
    assert [row.tolist() for row in low] == [[1, 2, 3, 4]] and [row.tolist() for row in strict] == [[5, 6, 7]]
    frames = [(i, np.full((60, 100), i, np.uint8)) for i in range(video.count)]
    inspected = []
    reused = []
    proofs = {}
    locations_cache = {}

    def trace(event, **data):
        if event == 'layers.absence_reused':
            reused.append(data['frames'])

    def confirm(_layer, probes):
        inspected.append(probes)
        return {i: 'present' if i == 22 else 'absent' for i in probes}

    def locations(_motif, frame):
        i = frame[0, 0]
        score = .92 if i%5 == 0 else .37 if i == 22 else .2
        return [(20, 20, score)]

    with (patch('wm_layers.read_frames', return_value=frames),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.agrees_in_halves', return_value=True),
          patch('wm_layers.locations', side_effect=locations) as locate,
          patch('wm_layers.regional_match', return_value=None)):
        try:
            scan_layers(video, [layer], False, trace, confirm_absence=confirm, absence_proofs=proofs,
                        location_cache=locations_cache)
        except LayerNeedsEvidence as error:
            assert error.frames == [21, 22, 23, 24]
            assert error.visible_frames == [22]
        else:
            raise AssertionError('An unchecked absence interval was accepted.')
        assert inspected == [[3, 8, 13, 18], [22, 28, 33, 38], [43, 48, 53, 58], [63, 68]]
        inspected.clear()
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, [layer], False, trace, confirm_absence=confirm, absence_proofs=proofs,
                        location_cache=locations_cache)
        assert pending.exception.frames == [21, 22, 23, 24]
        assert pending.exception.visible_frames == [22]
        assert inspected == [[22]] and reused == [52]
        assert locate.call_count == video.count

        inspected.clear()
        different = Layer(layer.motif, layer.floor, layer.matches, label='Another graphic')
        with np.testing.assert_raises(LayerNeedsEvidence):
            scan_layers(video, [different], False, trace, confirm_absence=confirm, absence_proofs=proofs)
        assert len(inspected) == 4 and reused == [52]

        inspected.clear()

        def still_present(_layer, probes):
            inspected.append(probes)
            return dict.fromkeys(probes, 'present')

        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, [layer], False, trace, confirm_absence=still_present, replan_on_visible=True)
        assert inspected == [[3, 8, 13, 18]] and pending.exception.visible_frames == inspected[0]
        assert pending.exception.frames == [i for i in range(video.count) if i % 5]

        inspected.clear()

        def all_clear(_layer, probes):
            inspected.append(probes)
            return dict.fromkeys(probes, 'absent')

        tracks = scan_layers(video, [layer], False, trace, confirm_absence=all_clear)
        assert inspected == [[3, 8, 13, 18], [22, 28, 33, 38], [43, 48, 53, 58], [63, 68]]
        assert tracks[0]['verified'].all()

        second = Layer(Motif(values.copy(), np.ones_like(values), mask.copy()), .73,
                       {Observation(0, (60, 20, 16, 10)): (60, 20, .94)}, label='independent overlay')
        locate.side_effect = lambda mark, frame: [(60, 20, .94)] if mark is second.motif else locations(mark, frame)
        checked = []

        def coverage(event, **data):
            if event == 'layers.coverage':
                checked.append((data['label'], data['unresolved']))

        def uncertain_at_one_probe(_layer, probes):
            return {i: 'uncertain' if i == 22 else 'absent' for i in probes}

        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(video, [layer, second], False, coverage, confirm_absence=uncertain_at_one_probe,
                        absence_proofs={})
        assert pending.exception.layer is layer and checked == [(layer.label, 4), (second.label, 0)]

    intervals = [np.arange(i, i+4) for i in range(1, 70, 5)]
    inspected.clear()

    def stalled(_layer, probes):
        inspected.append(probes)
        return dict.fromkeys(probes, 'uncertain')

    waiting = uncovered_absence(layer, intervals, [.22]*video.count, stalled, trace, 0, set())
    assert len(inspected) == 4 and waiting == [i for interval in intervals for i in interval]

    long = [np.arange(1, 16), np.arange(20, 36)]
    strengths = [.22]*video.count
    first, second = [absence_probes([run], strengths) for run in long]
    inspected.clear()
    visible = set()
    old_proofs = {'0:1:16': first[:1]}

    def mixed_long(_layer, probes):
        inspected.append(probes)
        return {i: 'present' if i == first[1] else 'absent' for i in probes}

    pending = uncovered_absence(layer, long, strengths, mixed_long, trace, 0, visible, proofs=old_proofs)
    assert inspected == [second+first] and pending == long[0].tolist() and visible == {first[1]}
    assert old_proofs['0:20:36'] == second
    short = SimpleNamespace(count=10, fps=30, width=video.width, height=video.height, planes=video.planes)
    absent = lambda _layer, probes: dict.fromkeys(probes, 'absent')
    with (patch('wm_layers.read_frames', return_value=frames[:10]),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.agrees_in_halves', side_effect=lambda _m, frame, *_a: not 0 < frame[0, 0] < 5),
          patch('wm_layers.locations', side_effect=lambda m, frame: [(20, 20, .41)] if 0 < frame[0, 0] < 5
                else locations(m, frame)), patch('wm_layers.regional_match', return_value=None)):
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(short, [layer], False, trace, confirm_absence=absent, confirm_borderline=lambda *_a: {})
        assert pending.exception.frames == [1, 2, 3, 4]
        verified = scan_layers(short, [layer], False, trace, confirm_absence=absent,
                               confirm_borderline=lambda _layer, probes: absent(_layer, probes) if probes == [3, 1] else {})
        assert np.flatnonzero(verified[0]['present']).tolist() == [0, 5]

    with (patch('wm_layers.read_frames', return_value=frames[:10]),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.agrees_in_halves', side_effect=lambda _m, frame, *_a: not 0 < frame[0, 0] < 5),
          patch('wm_layers.locations', side_effect=lambda m, frame: [(20, 20, .51)] if 0 < frame[0, 0] < 5
                else locations(m, frame)), patch('wm_layers.regional_match', return_value=None)):
        def review_incoherent(_layer, probes):
            assert probes == [3, 1]
            return {3: 'present', 1: 'absent'}

        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(short, [layer], False, trace, confirm_absence=absent, confirm_borderline=review_incoherent)
        assert pending.exception.frames == [1, 2, 3, 4]
        assert pending.exception.visible_frames == [3] and 3 in pending.exception.hints
        verified = scan_layers(short, [layer], False, trace, confirm_absence=absent,
                               confirm_borderline=lambda _layer, probes: absent(_layer, probes) if probes == [3, 1] else {})
        assert np.flatnonzero(verified[0]['present']).tolist() == [0, 5]

    with (patch('wm_layers.read_frames', return_value=frames[:10]),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.agrees_in_halves', return_value=True),
          patch('wm_layers.locations', side_effect=lambda m, frame: [(20, 20, .41)] if 0 < frame[0, 0] < 5
                else locations(m, frame)), patch('wm_layers.regional_match', return_value=None)):
        def check_exact(_layer, probes):
            assert probes == [1, 2, 3, 4]
            return {i: 'present' if i == 3 else 'absent' for i in probes}

        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(short, [layer], False, trace, confirm_absence=absent, confirm_borderline=check_exact)
        assert pending.exception.frames == pending.exception.visible_frames == [3]

    mixed = SimpleNamespace(count=9, fps=30, width=video.width, height=video.height, planes=video.planes)
    checked = []

    def check_mixed(_layer, probes):
        checked.append(probes)
        return {i: 'present' if i == 7 else 'absent' for i in probes}

    with (patch('wm_layers.read_frames', return_value=frames[:9]),
          patch('wm_layers.highpass', side_effect=lambda frame, _: frame),
          patch('wm_layers.agrees_in_halves', side_effect=lambda _m, frame, *_a: frame[0, 0] not in (2, 7)),
          patch('wm_layers.locations', side_effect=lambda m, frame: [(20, 20, .445)] if frame[0, 0] in (2, 7)
                else locations(m, frame)), patch('wm_layers.regional_match', return_value=None)):
        with np.testing.assert_raises(LayerNeedsEvidence) as pending:
            scan_layers(mixed, [layer], False, trace, confirm_absence=absent, confirm_borderline=check_mixed)
        assert checked == [[2, 4], [6, 7, 8]]
        assert pending.exception.frames == [6, 7, 8] and pending.exception.visible_frames == [7]
        verified = scan_layers(mixed, [layer], False, trace, confirm_absence=absent, confirm_borderline=absent)
        assert np.flatnonzero(verified[0]['present']).tolist() == [0, 5]
    with (patch('wm_director.wm_grounding.frame_evidence', return_value=('frame', b'image')) as image,
          patch('wm_director.ask_with_timeout_recovery', return_value={
              'decision': 'absent', 'checks': [{'frame': 22, 'decision': 'absent'}]}) as ask):
        config = {'timeout_seconds': 60, 'discovery_local': {'model': 'fixture'}}
        clipped = Layer(layer.motif, layer.floor, {**layer.matches, Observation(1, (-12, 20, 16, 10)): (-12, 20, .99)})
        assert not review_layer_visibility(video, clipped, [22, 28], config, False, trace)
        assert image.call_args_list[0].kwargs['region'] == [20, 20, 16, 10]
        ask.return_value['checks'].append({'frame': 28, 'decision': 'present'})
        assert not review_layer_visibility(video, layer, [22, 28], config, False, trace)
        ask.return_value['decision'] = 'present'
        assert review_layer_visibility(video, layer, [22, 28], config, False, trace) == {22: 'absent', 28: 'present'}
        ask.return_value['decision'] = 'absent'
        ask.return_value['checks'][1]['decision'] = 'absent'
        assert review_layer_visibility(video, layer, [22, 28], config, False, trace) == {22: 'absent', 28: 'absent'}
        ask.reset_mock()
        with patch('wm_director.wm_grounding.ask_local', return_value={'decision': 'present'}) as local:
            assert review_layer_visibility(video, layer, [22, 28], config, False, trace,
                                           require_local=True) == {22: 'present', 28: 'present'}
            ask.assert_not_called()
            local.return_value['decision'] = 'absent'
            assert review_layer_visibility(video, layer, [22, 28], config, False, trace,
                                           require_local=True) == {22: 'absent', 28: 'absent'}
            assert local.call_args.kwargs['reference'] == ('Reference crop with the target graphic', b'image')
            ask.reset_mock()
            ask.side_effect = [
                {'decision': 'uncertain', 'checks': [{'frame': 22, 'decision': 'uncertain'},
                                                    {'frame': 28, 'decision': 'absent'}]},
                {'decision': 'absent', 'checks': [{'frame': 22, 'decision': 'absent'}]},
            ]
            with patch('wm_director.wm_grounding.frame_tiles',
                       return_value=[((0, 0, 20, 20), ('detail crop', b'tile'))]) as tiles:
                assert review_layer_visibility(video, layer, [22, 28], config, False, trace,
                                               require_local=True) == {22: 'absent', 28: 'absent'}
            tiles.assert_called_once_with(video, 22, False, max_side=640)
            assert ask.call_count == 2 and ('detail crop', b'tile') in ask.call_args.args[2]
            for tile_decision in ('present', 'absent'):
                local.side_effect = [{'decision': 'absent'}, {'decision': 'absent'}, {'decision': tile_decision}]
                ask.side_effect = [
                    {'decision': 'present', 'checks': [{'frame': 22, 'decision': 'present'},
                                                      {'frame': 28, 'decision': 'absent'}]},
                    {'decision': 'absent', 'checks': [{'frame': 22, 'decision': 'absent'}]},
                ]
                with patch('wm_director.wm_grounding.frame_tiles',
                           return_value=[((0, 0, 20, 20), ('detail crop', b'tile'))]):
                    assert review_layer_visibility(video, layer, [22, 28], config, False, trace,
                                                   require_local=True) == {22: tile_decision, 28: 'absent'}
                assert local.call_args.args[2] == ('detail crop', b'tile')
    track = {'signal': values, 'weights': np.ones_like(values), 'mask': mask,
             'positions': np.tile([20, 20], (70, 1)), 'scores': np.full(70, .92), 'verified': np.ones(70, bool)}
    track['verified'][21:25], track['scores'][21:25], track['scores'][22] = False, .2, .37
    verify_presence(video, track, False, trace)
    assert not track['verified'][22] and 'present' not in track
    verify_presence(video, track, False, trace, confirm_absence=lambda probes: probes == [22])
    assert track['verified'].all() and np.flatnonzero(~track['present']).tolist() == list(range(21, 25))
    print('Absence uses bounded groups of high-risk probes and keeps any rejected or unchecked frames pending.')


def test_masked_proof_resume():
    from types import SimpleNamespace

    from wm_layers import masked_proof_keys, review_overlapping_matches, uncovered_absence

    layer = SimpleNamespace(label='repeated overlay')
    mask = np.ones((8, 8), np.uint8)
    point = (8, 12, .65)
    candidates = {i: ([point], [(mask, 8, 12)]) for i in range(25)}
    proofs, inspected = {}, []
    trace = lambda *_a, **_kw: None

    def confirm(_layer, probes, *, hide):
        assert set(probes) == set(hide)
        inspected.extend(probes)
        return {i: 'present' if i == 0 else 'absent' for i in probes}

    for expected in (11, 22, 24):
        evidence = [([], [point], .65) for _ in candidates]
        removed = review_overlapping_matches(layer, candidates, evidence, set(), confirm, trace, proofs=proofs)
        assert removed == expected and sum(bool(row[1]) for row in evidence) == 25-expected
    assert inspected.count(0) == 3 and all(inspected.count(i) == 1 for i in range(1, 25))

    inspected.clear()
    candidates[1] = ([point], [(mask, 9, 12)])
    review_overlapping_matches(layer, candidates, evidence, set(), confirm, trace, proofs=proofs)
    assert inspected == [0, 1]

    run = [np.array([1, 2], dtype=int)]
    absent = lambda _layer, frames: dict.fromkeys(frames, 'absent')
    hidden = masked_proof_keys({i: rows for i, (_, rows) in candidates.items()})
    settings = {'borderline': True, 'proofs': proofs, 'hidden_keys': hidden}
    assert not uncovered_absence(layer, run, [.4]*25, absent, trace, 0, set(), **settings)
    assert not uncovered_absence(layer, run, [.4]*25, None, trace, 0, set(), **settings)
    assert uncovered_absence(layer, run, [.4]*25, None, trace, 0, set(), **{**settings, 'hidden_keys': None}) == [1, 2]
    settings['hidden_keys'] = {**hidden, 1: hidden[2]}
    assert uncovered_absence(layer, run, [.4]*25, None, trace, 0, set(), **settings) == [1, 2]
    print('Exact masked proof resumes on new frames and invalidates when a mask moves.')


def test_layer_fragment_coverage():
    from wm_discovery import Motif
    from wm_layers import Layer, Observation, covered_fragment

    rng = np.random.default_rng(1607)
    field = rng.normal(0, 12, (16, 26)).astype(np.float32)
    support = np.ones(field.shape, np.uint8)
    fragment = Motif(field, np.ones_like(field), support)
    sample = np.zeros((55, 60), np.float32)
    sample[19:35, 12:38] = field
    observation = Observation(0, (12, 19, 26, 16))
    candidate = Layer(fragment, .73, {observation: (12, 19, .98)})
    assert covered_fragment(candidate, Layer(fragment, .73, {}), {0: sample}) is not None
    incomplete = support.copy()
    incomplete[4:6, 8:10] = 0
    assert np.sum(field[incomplete == 0]**2) < .04*np.sum(field**2)
    owner = Layer(Motif(field, np.ones_like(field), incomplete), .73, {})
    assert covered_fragment(candidate, owner, {0: sample}) is None
    print('A fragment with uncovered watermark pixels stays independent despite strong image similarity.')


def test_matching_templates_keep_both_masks():
    from types import SimpleNamespace

    from wm_discovery import Motif
    from wm_layers import Layer, Observation, merge_nearby_templates

    shape = 32, 42
    field = np.random.default_rng(10).choice([-21., 21.], shape).astype(np.float32)
    left = np.zeros(shape, np.uint8)
    left[6:-6, 6:-6] = 1
    right = left.copy()
    left[5, 14] = right[-6, 20] = 1
    weights = np.ones(shape, np.float32)
    video = SimpleNamespace(width=132, height=84)
    signals = {}

    def overlay(mask, frames, label):
        matches = {}
        for i in frames:
            x, y = 13+i*3, 18+i
            image = np.zeros((video.height, video.width), np.float32)
            image[y:y+shape[0], x:x+shape[1]] = field
            signals[i] = image
            item = Observation(i, (x, y, shape[1], shape[0]))
            matches[item] = (x, y, 1.)
        return Layer(Motif(field, weights, mask), .68, matches, label,
                     tuple((item.frame, *item.rect) for item in matches))

    prior = overlay(left, (0, 1, 2), 'Logo example.com')
    incoming = overlay(right, (3, 4, 5), 'Badge example.com')
    merged = merge_nearby_templates(incoming, prior, signals, video, False)
    assert merged is not None
    assert np.array_equal(merged.motif.mask, left | right)
    assert merged.matches.keys() == prior.matches.keys() | incoming.matches.keys()
    assert merged.reviewed_regions == prior.reviewed_regions + incoming.reviewed_regions
    incoming.label = 'Badge example.net'
    assert merge_nearby_templates(incoming, prior, signals, video, False) is None
    prior.label = incoming.label = 'Logo account ID: 1234567B9'
    assert merge_nearby_templates(incoming, prior, signals, video, False) is None
    prior.label = 'Logo example.com'
    right[0, 14] = 1
    incoming = overlay(right, (3, 4, 5), 'Badge example.com')
    assert merge_nearby_templates(incoming, prior, signals, video, False) is None
    print('An aligned same-identity template retains both masks and every reviewed region.')


def test_clipped_render_windows():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_discovery import import_discovery, pack_tracks
    from wm_discovery_render import put_mask, restore_frames
    from wm_layers import source_sized_tracks
    from wm_media import digest

    with tempfile.TemporaryDirectory(prefix='wm-clipped-windows-') as name:
        directory = Path(name)
        source = directory/'source.mp4'
        source.write_bytes(b'exact fixture')
        video = Video(source, 160, 96, Fraction(30), 5, 0, '1:1', 'bt709', 'bt709', 'bt709')
        mask = np.zeros((100, 164), np.uint8)
        mask[:10, :12] = 1
        mask[35:45, 152:] = 1
        mask[90:, 74:86] = 1
        signal = mask.astype(np.float32)
        original = {'mask': mask, 'signal': signal, 'weights': signal, 'scores': np.ones(video.count, np.float32),
                    'verified': np.ones(video.count, bool), 'cuts': np.array([], np.int32),
                    'positions': np.array([[0, 0], [-4, 0], [0, -4], [-4, -4], [0, 0]], np.int32),
                    'present': np.array([True, True, True, True, False])}
        layers = source_sized_tracks(video, original)
        assert len(layers) == 4 and all(layer['mask'].shape == (96, 160) for layer in layers)
        assert np.array_equal(sum(layer['present'].astype(int) for layer in layers), original['present'])
        expected = np.zeros((video.count, video.height, video.width), np.uint8)
        for i in range(video.count):
            actual = np.zeros_like(expected[i])
            if original['present'][i]:
                put_mask(expected[i], mask, *original['positions'][i])
            for layer in layers:
                if layer['present'][i]:
                    put_mask(actual, layer['mask'], *layer['positions'][i])
            assert np.array_equal(actual, expected[i])

        track = pack_tracks(layers)
        asset, profile, prepared = (directory/file for file in ('track.npz', 'profile.json', 'prepared.npz'))
        np.savez(asset, **track, source_sha256=digest(source))
        profile.write_text('{}', encoding='utf-8')
        import_discovery(video, {'discovery': {'track': str(asset), 'sha256': digest(asset), 'layer_count': len(layers)}},
                         profile, prepared, False)
        raw = np.full(video.width*video.height*3//2, 128, np.uint8)
        with (patch('wm_discovery_render.read_frames', side_effect=lambda _, start, stop, __:
                    ((i, raw) for i in range(start, stop))), patch('wm_discovery_render.ProPainter') as model):
            model.return_value.predict.side_effect = lambda rgb, masks: np.full_like(rgb, 215)
            output = list(restore_frames(video, track, 0, video.count, False, .5, 6))
        for i, pixels in output:
            luma = video.planes(pixels)[0]
            assert np.array_equal(luma != 128, expected[i].astype(bool))
    print('Verified edge-clipped placements become source-sized render tracks without losing visible mask pixels.')


def test_discovery_composite():
    from fractions import Fraction
    from unittest.mock import patch

    from wm_director import overlay_coverage_findings, uncovered_track_regions
    from wm_discovery import pack_tracks
    from wm_discovery_render import consensus_layers_for_area, context_groups, restore_frames, verify_mask_area

    video = Video(Path('self-test.mp4'), 240, 128, Fraction(30), 3, 0, '1:1', 'bt709', 'bt709', 'bt709')
    original = np.full(video.width*video.height*3//2, 128, np.uint8)
    mask = np.ones((14, 18), np.uint8)
    common = {'mask': mask, 'weights': mask.astype(np.float32), 'signal': mask.astype(np.float32),
              'scores': np.ones(3, np.float32), 'verified': np.ones(3, bool), 'cuts': np.array([1, 2], np.int32)}
    layers = [{**common, 'positions': np.array([[14, 12], [14, 12], [0, 0]], np.int32),
               'present': np.array([True, True, False])},
              {**common, 'positions': np.array([[193, 85], [24, 17], [0, 0]], np.int32),
               'present': np.array([True, True, False])}]
    for x, y in ((70, 30), (123, 62), (167, 95)):
        layers.append({**common, 'positions': np.array([[x, y], [x+4, y-3], [0, 0]], np.int32),
                       'present': np.array([True, True, False])})

    def read(_, start, stop, __):
        return ((i, original) for i in range(start, stop))

    with (patch('wm_discovery_render.read_frames', read), patch('wm_discovery_render.ProPainter') as model):
        model.return_value.predict.side_effect = lambda rgb, masks: np.full_like(rgb, 215)
        results = list(restore_frames(video, pack_tracks(layers), 0, 3, False, .5, 6))
        assert model.call_count == 1 and model.return_value.predict.call_count >= 2
    for i, frame in results:
        luma = video.planes(frame)[0]
        edited = np.zeros(luma.shape, bool)
        for layer in layers:
            if layer['present'][i]:
                x, y = layer['positions'][i]
                edited[y:y+14, x:x+18] = True
                assert luma[y+7, x+9] != 128
        assert np.all(luma[~edited] == 128)
        if i == 2:
            assert np.array_equal(frame, original)
    assert np.all(original == 128)
    larger = np.ones((64, 64), np.uint8)
    points = np.array([[10, 10], [20, 12], [10, 10]], np.int32)
    alternate = np.array([[20, 12], [20, 12], [20, 12]], np.int32)
    edits = [(larger, points, np.array([True, False, True])),
             (larger, alternate, np.array([False, True, True]))]
    verify_mask_area(video, edits, 0, 3)
    alternate[2] = 140, 60
    verify_mask_area(video, edits, 0, 2)
    with np.testing.assert_raises_regex(ValueError, 'too much of frame 2'):
        verify_mask_area(video, edits, 0, 3)
    raw = [{**common, 'mask': item[0], 'positions': item[1], 'present': item[2]} for item in edits]
    with patch('wm_discovery_render.layer_masks', return_value=[(layer, None, mask) for layer in raw]) as derive:
        assert not consensus_layers_for_area(video, pack_tracks(layers), False, lambda *_a, **_kw: None)
        derive.assert_not_called()
        tightened = consensus_layers_for_area(video, pack_tracks(raw), False, lambda *_a, **_kw: None)
        assert tightened
        assert uncovered_track_regions(video, tightened, [[2, 10, 10, 64, 64]])
        assert derive.call_args.args[-2:] == (True, None)
    with patch('wm_discovery_render.layer_masks', return_value=[(layer, None, larger) for layer in raw]):
        with np.testing.assert_raises_regex(ValueError, 'too much of frame 2'):
            consensus_layers_for_area(video, pack_tracks(raw), False, lambda *_a, **_kw: None)
    alternate[2] = 225, 120
    verify_mask_area(video, edits, 0, 3)
    jumps = [(mask, np.array([[14, 12], [193, 85]], np.int32), np.ones(2, bool))]
    joined = context_groups(video, jumps)
    separate = context_groups(video, jumps, separate_jumps=True)
    assert len(joined) == 1 and len(separate) == 2
    sizes = lambda groups: sum((right-left)*(bottom-top) for (left, top, right, bottom), _ in groups)
    assert sizes(separate) < .5*sizes(joined)
    required = [[0, *map(int, layer['positions'][0]), 18, 14] for layer in layers]
    with tempfile.TemporaryDirectory(prefix='wm-coverage-test-') as folder:
        track = Path(folder)/'track.npz'
        np.savez(track, **pack_tracks(layers))
        assert not overlay_coverage_findings(video, {'discovery': True}, track, required)
        assert overlay_coverage_findings(video, {'discovery': True}, track, [[2, 14, 12, 18, 14]])
        np.savez(track, **layers[0])
        assert overlay_coverage_findings(video, {'discovery': True}, track, required)
    print('Discovery composite retained disjoint and overlapping masks and preserved untouched frames.')


def test_discovery_context_review():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_director import review_discovery

    source = SimpleNamespace(width=120, height=80, count=8)
    candidate = object()
    log = []

    def sheets(video, frames, width, cuda, regions, only_regions, **kwargs):
        assert video is candidate
        return [('restored target', b'crop')] if only_regions else [('first', b'1'), ('next', b'2')]

    def judge(config, prompt, images, decisions, trace):
        label = images[0][0]
        log.append(label)
        if label == 'first':
            return {'decision': 'review', 'reason': 'first concern', 'frame': 4, 'area': 'target',
                    'rect': [166, 250, 420, 400]}
        if label == 'next':
            return {'decision': 'review', 'reason': 'outside the target', 'frame': 5, 'area': 'elsewhere'}
        return {'decision': 'accept', 'reason': 'clear'}

    with tempfile.TemporaryDirectory(prefix='wm-review-test-') as root:
        attempt = Path(root)
        np.savez(attempt/'track.npz', positions=np.tile([20, 20], (8, 1)),
                 mask=np.ones((12, 30), np.uint8), present=np.ones(8, bool))
        args = (source, candidate, attempt, None, {'discovery': {'label': ''}},
                {'timeout_seconds': 30, 'sample_width': 640, 'discovery_local': {'model': 'fixture'}}, False,
                lambda *_args, **_kw: None, {'review_regions': [], 'review_only_regions': False}, lambda: None,
                [4, 5], {'discovery_worst_frames': [4]}, (), 0)
        with (patch('wm_director.comparison_evidence', return_value=[('paired', b'pair')]),
              patch('wm_director.evidence', side_effect=sheets),
              patch('wm_director.ask', side_effect=judge),
              patch('wm_director.wm_grounding.ask_local', return_value={'decision': 'accept', 'reason': 'clear'}) as local):
            verdict = review_discovery(*args)
        assert local.call_args.args[2] == ('restored target', b'crop')
        assert verdict == {'decision': 'review', 'reason': 'outside the target'}
        assert log == ['paired', 'first', 'restored target', 'next']
    print('Discovery review rechecked a previously sampled crop and retained a separate whole-frame concern.')


def test_pending_context_resume():
    from unittest.mock import Mock, patch

    from wm_director import review_comparisons, review_visual
    from wm_grok import SourceReviewDeferred

    config = {'batch_routing': True, 'model': 'fixture', 'reasoning_effort': 'xhigh'}
    issue = {'decision': 'review', 'reason': 'possible remaining overlay', 'frame': 10, 'area': 'elsewhere'}
    accepted = {'decision': 'accept', 'reason': 'the different evidence is clear'}
    trace = Mock()
    for review in (review_visual, review_comparisons):
        pending, persist = {}, Mock()

        def call(image, check):
            return review(config, 'context', [('frame 10', image)], trace, review_issue=check,
                          pending=pending, persist=persist)

        with patch('wm_director.ask', return_value=issue) as request:
            with np.testing.assert_raises(SourceReviewDeferred):
                call(b'original pixels', Mock(side_effect=SourceReviewDeferred))
            assert request.call_count == 1 and list(pending.values()) == [issue]
            assert persist.call_count == 1
        with patch('wm_director.ask', return_value=accepted) as request:
            assert call(b'changed pixels', Mock(side_effect=AssertionError))['decision'] == 'accept'
            assert request.call_count == 1 and list(pending.values()) == [issue]
        with patch('wm_director.ask', side_effect=AssertionError):
            check = Mock(return_value=issue)
            assert call(b'original pixels', check)['decision'] == 'review'
            check.assert_called_once_with(issue, 0)
        assert not pending and persist.call_count == 2
    print('Interrupted output concerns resume only with identical visual evidence and still run the follow-up check.')


def test_multiple_context_watermark_hints():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import feedback_hints
    from wm_director import review_discovery

    source = SimpleNamespace(width=120, height=80, count=8)
    first = {'decision': 'review', 'reason': 'scene damage needs review'}
    candidate = {'frame': 5, 'finding': 'watermark', 'label': 'other logo', 'rect': [700, 600, 850, 850]}
    second = {'decision': 'review', 'reason': 'separate concerns', 'findings': [
        {**candidate, 'frame': 100}, {**candidate, 'finding': 'damage'},
        {**candidate, 'rect': [0, 0, 1000, 1000]}, candidate]}
    verdicts = {'pair': {'decision': 'accept', 'reason': 'source crop passed'}, 'first': first, 'next': second}

    def judge(_config, _prompt, images, _decisions, _trace):
        return verdicts[images[0][0]]

    with tempfile.TemporaryDirectory(prefix='wm-multiple-context-') as root:
        attempt = Path(root)
        np.savez(attempt/'track.npz', positions=np.tile([20, 20], (8, 1)),
                 mask=np.ones((12, 30), np.uint8), present=np.ones(8, bool))
        record = {'review_regions': [], 'review_only_regions': False, 'split_context_review': True,
                  'pending_context': {'unsplit review': first}}
        config = {'timeout_seconds': 30, 'sample_width': 640, 'batch_routing': True,
                  'model': 'fixture', 'reasoning_effort': 'high'}
        with (patch('wm_director.comparison_evidence', return_value=[('pair', b'pair')]),
              patch('wm_director.evidence', return_value=[('first', b'1'), ('next', b'2')]),
              patch('wm_director.ask', side_effect=judge)):
            verdict = review_discovery(source, object(), attempt, None, {'discovery': {'label': ''}}, config,
                                       False, lambda *_a, **_k: None, record, lambda: None,
                                       [3, 4, 5], {'discovery_worst_frames': [3]}, (), 0)
        assert verdict == first
        assert not record['pending_context']
        hint = {'frame': 5, 'rect': [84, 48, 18, 20], 'label': 'other logo'}
        assert feedback_hints(record) == [hint]
        primary = {'frame': 4, 'rect': [10, 20, 18, 30], 'label': 'primary logo'}
        record.update(decision={'watermark_hint': primary}, context_watermark_hints=[primary, hint, {**hint, 'frame': 6}])
        assert feedback_hints(record) == [primary, hint]
    print('Separate output concerns stay pending and supply bounded, frame-validated repair proposals.')


def test_mislocalized_target_recheck():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_director import review_discovery
    from workflow import write_object

    source = SimpleNamespace(width=120, height=80, count=8)
    candidate = object()
    log = []
    missed = False
    context_result = None
    wide_clean = False

    def sheets(_source, _candidate, regions, width, cuda, **_options):
        name = 'original-pair' if width == 384 else 'exact-frame' if regions[0][1:] == [0, 0, 120, 80] else 'exact-crop'
        return [(name, b'pair')]

    def full(_candidate, frames, *_args, **_kwargs):
        return [('wide' if len(frames) > 1 else 'remaining', b'frame')]

    def judge(config, prompt, images, decisions, trace):
        label = images[0][0]
        log.append(label)
        if label == 'original-pair' and context_result:
            return {'decision': 'review', 'reason': 'the original overlay is opaque', 'issue': 'context'}
        if label == 'exact-crop' and context_result:
            assert [name for name, _ in images] == ['exact-crop', 'exact-frame']
            return {'decision': context_result, 'reason': 'checked both matching scales'}
        if label == 'wide' and wide_clean:
            return {'decision': 'accept', 'reason': 'whole frame appears clean'}
        if label == 'wide':
            return {'decision': 'review', 'reason': 'possible source mosaic', 'frame': 4,
                    'area': 'elsewhere', 'rect': [880, 600, 960, 850]}
        if label == 'remaining':
            return {'decision': 'review', 'reason': 'a separate concern'}
        if label == 'exact-frame' and missed:
            return {'decision': 'review', 'reason': 'a remaining distribution mark',
                    'finding': 'watermark', 'label': 'channel mark', 'rect': [880, 600, 960, 850]}
        return {'decision': 'accept', 'reason': 'source details preserved'}

    with tempfile.TemporaryDirectory(prefix='wm-wide-recheck-') as root:
        attempt = Path(root)
        np.savez(attempt/'track.npz', positions=np.tile([20, 20], (8, 1)),
                 mask=np.ones((12, 30), np.uint8), present=np.ones(8, bool))
        args = (source, candidate, attempt, None, {'discovery': {'label': ''}},
                {'timeout_seconds': 30, 'sample_width': 640, 'batch_routing': True,
                 'model': 'fixture', 'reasoning_effort': 'high'}, False,
                lambda *_args, **_kwargs: None, {'review_regions': [], 'review_only_regions': False}, lambda: None,
                [4, 5], {'discovery_worst_frames': [4]}, (), 0)
        with (patch('wm_director.comparison_evidence', side_effect=sheets),
              patch('wm_director.evidence', side_effect=full),
              patch('wm_director.ask', side_effect=judge),
              patch('wm_director.reported_region_delta', return_value=[{'p99': 1, 'maximum': 4}]) as measure):
            verdict = review_discovery(*args)
            assert verdict['reason'] == 'a separate concern'
            assert log == ['original-pair', 'wide', 'exact-frame', 'exact-crop', 'remaining']
            log.clear()
            measure.return_value = [{'p99': 6, 'maximum': 10}]
            verdict = review_discovery(*args)
            assert verdict['reason'] == 'possible source mosaic' and log == ['wide']
            log.clear()
            missed = True
            measure.return_value = [{'p99': 1, 'maximum': 4}]
            verdict = review_discovery(*args)
            assert verdict['decision'] == 'review' and log == ['wide', 'exact-frame']
            assert verdict['watermark_hint'] == {'frame': 4, 'rect': [104, 48, 12, 20], 'label': 'channel mark'}
            log.clear()
            args[8]['approved_comparisons'].clear()
            context_result = 'review'
            assert review_discovery(*args)['decision'] == 'review'
            assert log == ['original-pair', 'exact-crop']
            log.clear()
            context_result = 'accept'
            assert review_discovery(*args)['watermark_hint']['frame'] == 4
            assert log == ['original-pair', 'exact-crop', 'wide', 'exact-frame']
            context_result, wide_clean = None, True
            args[5]['local_layers'] = True
            hint = {'frame': 5, 'rect': [104, 48, 12, 20], 'label': 'channel mark'}
            with (patch('wm_director.local_untracked_watermark', return_value=hint) as sweep,
                  patch('wm_discovery_render.edge_residue', return_value={'eligible': False}) as edge):
                verdict = review_discovery(*args)
                assert verdict['decision'] == 'review' and verdict['watermark_hint'] == hint
                sweep.reset_mock()
                edge.return_value = {'eligible': True}
                assert review_discovery(*args)['decision'] == 'review'
                sweep.assert_not_called()
            recipe = attempt/'edge-retry.json'
            write_object(recipe, {'final_render': ['--model-scale', '1', '--raft-iterations', '12']})
            args[8]['approved_comparisons'].clear()
            action = {'kind': 'discovery-edge', 'frame': 4, 'defect': 'residue'}
            with patch('wm_director.ask', return_value={'decision': 'repair', 'reason': 'residue', 'action': action}):
                edge_args = (*args[:3], recipe, *args[4:-1], 1)
                assert review_discovery(*edge_args)['repair'] == {**action, 'layer': 0}
                action['frame'] = 5
                assert review_discovery(*edge_args)['decision'] == 'review'
    print('A misplaced target report requires pixel agreement, matching full-frame and crop review, and review of other frames.')


def test_feedback_track_join():
    from wm_batch import join_feedback_tracks
    from wm_discovery import LUMA_EXTENT_FIELDS, discovery_tracks

    def layer(x):
        return {'positions': np.tile([x, 2], (4, 1)).astype(np.int32), 'scores': np.ones(4, np.float32),
                'verified': np.ones(4, bool), 'present': np.ones(4, bool), 'cuts': np.array([2], np.int32),
                'mask': np.ones((8, 8), np.uint8), 'signal': np.ones((8, 8), np.float32),
                'weights': np.ones((8, 8), np.float32)}

    retained, added = layer(0), layer(12)
    retained['extent_signal'] = np.ones((2, 8, 8), np.float32)
    retained['extent_frames'] = np.array([0, 2, 3], np.int32)
    for key in LUMA_EXTENT_FIELDS:
        retained[key] = np.asarray(1)
    combined, first = join_feedback_tracks(retained, [added], 4)
    assert first == 1 and len(discovery_tracks(combined)) == 2
    assert np.array_equal(combined['positions'], retained['positions'])
    assert np.array_equal(discovery_tracks(combined)[1]['positions'], added['positions'])
    for field in ('extent_signal', 'extent_frames', *LUMA_EXTENT_FIELDS):
        assert np.array_equal(combined[field], retained[field])
    padded = layer(4)
    padded['mask'] = np.pad(padded['mask'], ((0, 0), (12, 0)))
    padded['signal'] = np.pad(padded['signal'], ((0, 0), (12, 0)))
    padded['weights'] = np.pad(padded['weights'], ((0, 0), (12, 0)))
    combined, first = join_feedback_tracks(retained, [padded], 4)
    assert first == 1 and np.array_equal(discovery_tracks(combined)[1]['mask'], padded['mask'])
    try:
        join_feedback_tracks(retained, [layer(5)], 4)
    except ValueError as error:
        assert 'overlaps a retained layer' in str(error)
    else:
        raise AssertionError('An overlapping recovery must require a joint plan.')
    print('A newly verified disjoint layer can be added without changing an earlier track.')


def test_feedback_observed_locations():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import feedback_proposals

    video = SimpleNamespace(count=240, width=480, height=854, fps=30)
    hint = {'frame': 180, 'rect': [103, 187, 193, 73], 'label': 'example mark',
            'observations': [[20, -3, 515, 105, 41], [180, 103, 187, 193, 73]]}
    events = []
    trace = lambda event, **data: events.append((event, data))
    with (patch('wm_batch.frame_evidence', return_value=('source', b'crop')),
          patch('wm_batch.ask_local', return_value={'decision': 'watermark', 'label': hint['label']}),
          patch('wm_batch.locate_named', return_value=[])):
        proposals = feedback_proposals(video, hint, [], {'discovery_local': {}}, False, trace)
    assert proposals[20] == [{'rect': [0, 515, 102, 556], 'provider': 'review-pixels'}]
    assert proposals[180] == [{'rect': [103, 187, 296, 260], 'provider': 'review-pixels'}]
    searches = {data['frame']: data['region'] for event, data in events if event == 'feedback.grounded' and data['region']}
    assert searches[20][1] > searches[180][1] + searches[180][3]

    hint = {'frame': 180, 'rect': [16, 600, 85, 33], 'label': 'example mark'}

    def source(_video, frame, _cuda, *, region, **_options):
        return (frame, region), b'pixels'

    for seed in (180, 170):
        events.clear()

        def locate(_backend, label, image, *_args, **_kwargs):
            return [[700, 800, 850, 860]] if image[0] == (seed, None) and label is None else []

        with (patch('wm_batch.frame_evidence', side_effect=source),
              patch('wm_batch.ask_local', return_value={'decision': 'uncertain'}),
              patch('wm_batch.locate_named', side_effect=locate)):
            proposals = feedback_proposals(video, hint, [], {'discovery_local': {}}, False, trace)
        assert proposals[seed] == [{'rect': [336, 683, 408, 734], 'provider': 'review-qwen'}]
        assert (proposals[180] == []) == (seed != 180)
        recovered = [data for event, data in events if event == 'feedback.grounding_recovered']
        assert recovered == [{'reported_frame': 180, 'anchor_frame': seed, 'region': None}]
    print('Feedback preserves verified pixel boxes and recovers out-of-crop candidates from the source.')


def test_independent_comparisons():
    from threading import Barrier
    from unittest.mock import patch

    from wm_director import review_comparisons

    barrier = Barrier(2)
    images = [('first', b'a'), ('second', b'b'), ('later', b'c')]
    defect = {'decision': 'review', 'reason': 'A visible watermark remnant.'}
    issues = []

    def judge(config, prompt, displayed, decisions, trace):
        assert prompt == 'paired review'
        assert len(displayed) == 1 and displayed[0] in images[:2]
        barrier.wait(timeout=3)
        return defect if displayed[0] == images[1] else {'decision': 'accept', 'reason': 'clear'}

    def flag(response, index):
        issues.append(index)
        return response

    with patch('wm_director.ask', side_effect=judge):
        result = review_comparisons({'batch_routing': True}, 'paired review', images,
                                    lambda *_args, **_kwargs: None, flag)
    assert result == defect and issues == [1]
    def mixed(_config, _prompt, displayed, _decisions, _trace):
        if displayed[0] == images[0]:
            raise RuntimeError('fixture outage')
        return defect

    with patch('wm_director.ask', side_effect=mixed):
        result = review_comparisons({'batch_routing': True}, 'paired review', images,
                                    lambda *_args, **_kwargs: None, flag)
    assert result == defect and issues == [1, 1]
    actionable = {'decision': 'repair', 'reason': 'A different crop has a repairable remnant.',
                  'repair': {'kind': 'discovery-luma', 'frame': 8, 'layer': 0, 'defect': 'residue'}}
    pending = []
    with patch('wm_director.ask', side_effect=lambda _config, _prompt, displayed, *_args:
               defect if displayed[0] == images[0] else actionable):
        result = review_comparisons({'batch_routing': True}, 'paired review', images[:2],
                                    lambda *_args, **_kwargs: None, flag, issues=pending)
    assert result == actionable and pending == [{'comparison': 0, 'verdict': defect},
                                                 {'comparison': 1, 'verdict': actionable}]
    print('Independent paired reviews run together and preserve the correct failing crop.')


def test_comparison_proof_resume():
    from unittest.mock import Mock, patch

    from wm_director import review_comparisons
    from wm_grok import SourceReviewDeferred, source_review_budget

    config = {'batch_routing': True, 'model': 'fixture', 'reasoning_effort': 'high'}
    images = [('first', b'a'), ('second', b'b')]
    approved, trace, persist = {}, Mock(), Mock()
    passed = {'decision': 'accept', 'reason': 'exact crop verified'}

    def interrupted(_config, _prompt, displayed, _decisions, _trace):
        if displayed[0] == images[0]:
            raise RuntimeError('fixture outage')
        return passed

    with patch('wm_director.ask', side_effect=interrupted):
        try:
            review_comparisons(config, 'pair', images, trace, None, approved=approved, persist=persist)
        except RuntimeError:
            pass
        else:
            raise AssertionError('An interrupted crop was accepted.')
    assert len(approved) == 1 and persist.call_count == 1
    with patch('wm_director.ask', return_value=passed) as ask:
        result = review_comparisons(config, 'pair', images, trace, None, approved=approved)
        assert result['decision'] == 'accept' and ask.call_count == 1
        assert ask.call_args.args[2] == [images[0]]
        result = review_comparisons(config, 'updated pair', images, trace, None, approved=approved)
        assert result['decision'] == 'accept' and ask.call_count == 3
    config.update(timeout_seconds=75)
    with patch('wm_grok.ask_api', return_value=passed) as provider:
        with source_review_budget(0), np.testing.assert_raises(SourceReviewDeferred):
            review_comparisons(config, 'pair', images+[('later', b'c')], trace, None, approved=approved)
        assert len(approved) == 4 and provider.call_count == 0
    print('A retry reuses only previously approved images with the same prompt and model settings.')


def test_grok_api_transport():
    from base64 import b64decode
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from io import BytesIO
    import json
    from threading import Thread
    from time import monotonic, sleep
    from unittest.mock import Mock, patch

    from PIL import Image
    from wm_batch import transient_grok_error
    from wm_director import ask, ask_with_timeout_recovery, load_settings
    from wm_grok import API_KEY_ENV, DEFAULT_BASE_URL, PromptTimeout, TransientAPIError

    verdict = {'decision': 'accept', 'reason': 'Both images are clear.'}

    def completion(content):
        return {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'fixture',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': content}}]}

    secret = 'fixture-gateway-key'
    success = completion(json.dumps(verdict))
    error = {'error': {'message': secret}}
    replies = iter(((200, success, False), (200, completion('invalid JSON'), False), (200, success, False),
                    (401, error, False), (429, error, False), (503, error, False), (None, None, False),
                    (200, success, True)))
    requests = []

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, self.headers.get('Authorization'), request))
            status, body, trickle = next(replies)
            if status is None:
                self.close_connection = True
                return
            payload = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                if trickle:
                    for offset in range(0, len(payload), 16):
                        self.wfile.write(payload[offset:offset+16])
                        self.wfile.flush()
                        sleep(.09)
                else:
                    self.wfile.write(payload)
            except OSError:
                pass

    images = []
    for label, color in (('frame 8', 'blue'), ('frame 20', 'yellow')):
        with BytesIO() as buffer:
            Image.new('RGB', (800, 400), color).save(buffer, 'JPEG')
            images.append((label, buffer.getvalue()))
    trace = Mock()
    decisions = ('accept', 'review')
    defaults = load_settings(Path(__file__).parent/'profiles'/'grok-director.json')
    assert defaults['grok_api_base_url'] == DEFAULT_BASE_URL
    with ThreadingHTTPServer(('127.0.0.1', 0), Gateway) as gateway:
        thread = Thread(target=gateway.serve_forever, daemon=True)
        thread.start()
        config = {**defaults, 'grok_api_base_url': f'http://127.0.0.1:{gateway.server_port}/v1',
                  'model': 'fixture', 'timeout_seconds': 5}
        try:
            with patch.dict('os.environ', {API_KEY_ENV: secret}):
                assert ask(config, 'inspect images', images, decisions, trace) == verdict
                assert ask_with_timeout_recovery(config, 'inspect images', images, decisions, trace) == verdict
                for expected in (RuntimeError, TransientAPIError, TransientAPIError, TransientAPIError):
                    with np.testing.assert_raises(expected):
                        ask(config, 'inspect images', images, decisions, trace)
                started = monotonic()
                with np.testing.assert_raises(PromptTimeout):
                    ask({**config, 'timeout_seconds': .4}, 'inspect images', images, decisions, trace)
                assert monotonic()-started < 1.2
        finally:
            gateway.shutdown()
            thread.join()

    assert len(requests) == 8
    assert all(path == '/v1/chat/completions' and auth == f'Bearer {secret}' for path, auth, _ in requests)
    first = requests[0][2]
    assert first['model'] == 'fixture' and first['reasoning_effort'] == 'xhigh'
    assert first['response_format'] == {'type': 'json_object'} and 'tools' not in first
    content = first['messages'][1]['content']
    assert [part['type'] for part in content] == ['text', 'text', 'image_url', 'text', 'image_url']
    assert content[0]['text'] == 'inspect images'
    for (label, jpeg), frame_label, picture in zip(images, content[1::2], content[2::2], strict=True):
        assert frame_label['text'] == label
        uri = picture['image_url']['url']
        assert uri.startswith('data:image/jpeg;base64,') and b64decode(uri.split(',', 1)[1]) == jpeg
    compact = requests[2][2]['messages'][1]['content'][2]['image_url']['url']
    with Image.open(BytesIO(b64decode(compact.split(',', 1)[1]))) as picture:
        assert max(picture.size) == 640
    failures = [entry.kwargs for entry in trace.call_args_list if entry.args[0] == 'grok.api.failed']
    assert [event['status_code'] for event in failures] == [200, 401, 429, 503, None, None]
    assert [transient_grok_error(event['error']) for event in failures] == [True, False, True, True, True, True]
    assert secret not in str(trace.call_args_list)
    print('The Grok API preserves labeled images, rejects invalid replies and enforces a total request deadline.')


def test_grok_prompt_recovery():
    from io import BytesIO
    from unittest.mock import Mock, patch

    from PIL import Image
    from wm_director import ask_with_timeout_recovery, review_comparisons, review_visual
    from wm_grok import InvalidResponse, PromptTimeout, ToolAttempt

    with BytesIO() as buffer:
        Image.new('RGB', (800, 400)).save(buffer, 'JPEG')
        data = buffer.getvalue()
    images = [('paired details', data), ('full restored frame', data)]
    trace = Mock()
    verdict = {'decision': 'review', 'reason': 'The context is still unclear.'}
    timeout = PromptTimeout('Grok timed out during the API visual request.')
    for review, maximum_side in [(False, 640), (True, 720)]:
        with patch('wm_director.ask', side_effect=[timeout, verdict]) as ask:
            if review:
                result = review_visual({}, 'same prompt', images, trace)
            else:
                result = ask_with_timeout_recovery({}, 'same prompt', images, ('accept', 'review'), trace)
        assert result == verdict and ask.call_count == 2
        first, second = [call.args for call in ask.call_args_list]
        assert first[1] == second[1] == 'same prompt'
        assert [label for label, _ in second[2]] == [label for label, _ in images]
        for _, image in second[2]:
            with Image.open(BytesIO(image)) as picture:
                assert max(picture.size) == maximum_side
    tool_attempt = ToolAttempt('Grok attempted a tool call in a visual-only request.')
    with patch('wm_director.ask', side_effect=[tool_attempt, verdict]) as ask:
        assert review_visual({}, 'same prompt', images, trace) == verdict
    retry = ask.call_args.args
    assert retry[1].startswith('same prompt') and 'without tools' in retry[1]
    assert [label for label, _ in retry[2]] == [label for label, _ in images]
    with patch('wm_director.ask', side_effect=[InvalidResponse('Malformed response'), verdict]) as ask:
        assert review_visual({}, 'same prompt', images, trace) == verdict
    retry = ask.call_args.args
    assert retry[1].startswith('same prompt') and 'exactly one JSON object' in retry[1]
    assert [label for label, _ in retry[2]] == [label for label, _ in images]
    approved = {}
    config = {'batch_routing': True, 'model': 'fixture', 'reasoning_effort': 'high'}
    with patch('wm_director.ask', side_effect=[timeout, {'decision': 'accept', 'reason': 'retry clear'}]):
        verdict = review_comparisons(config, 'same prompt', images[:1], trace, None, approved=approved)
    assert verdict['decision'] == 'accept' and not approved
    print('A Grok timeout or malformed response retries with the same evidence and preserves the actual verdict.')


def test_shared_grok_slots():
    from contextlib import ExitStack
    from unittest.mock import patch

    from wm_grok import (PromptTimeout, SourceReviewDeferred, ask, rendered_candidate_review, source_review_budget,
                         visual_slot)

    with tempfile.TemporaryDirectory(prefix='wm-grok-slots-') as path:
        root = Path(path)
        with ExitStack() as stack:
            for _ in range(3):
                stack.enter_context(visual_slot(None, root=root))
            with np.testing.assert_raises(PromptTimeout):
                with visual_slot(None, root=root, wait_seconds=0):
                    raise AssertionError('A fourth shared Grok request started.')
            with source_review_budget(300, max_slot_wait=0), np.testing.assert_raises(SourceReviewDeferred):
                with visual_slot(None, root=root):
                    raise AssertionError('The first turn did not yield to another source.')
        with visual_slot(None, root=root, wait_seconds=0):
            pass
        with source_review_budget(300, max_slot_wait=0), visual_slot(None, root=root):
            pass
    clock, events = [0.], []
    durations = iter((160., 145., 120., 5.))

    def trace(event, **fields):
        events.append((event, fields))

    args = ('http://127.0.0.1:8317/v1', 'model', 'high', 75, 'prompt', [], trace)
    with tempfile.TemporaryDirectory(prefix='wm-review-budget-') as path:
        def provider(*_args):
            clock[0] += 350.
            with visual_slot(trace, root=Path(path)):
                clock[0] += next(durations)
                return {'decision': 'accept'}

        with (patch('wm_grok.ask_api', side_effect=provider) as request,
              patch('wm_grok.monotonic', side_effect=lambda: clock[0])):
            with source_review_budget(300):
                assert ask(*args)['decision'] == ask(*args)['decision'] == 'accept'
                with np.testing.assert_raises(SourceReviewDeferred):
                    ask(*args)
                with rendered_candidate_review(True):
                    assert ask(*args)['decision'] == 'accept'
                    with np.testing.assert_raises(SourceReviewDeferred):
                        ask(*args)
            assert ask(*args)['decision'] == 'accept' and request.call_count == 4
    billed = [fields for event, fields in events if event == 'grok.source_review_charged']
    assert [(x['total_seconds'], x['limit'], x['rendered_candidate']) for x in billed] == [
        (160., 300, False), (305., 300, False), (425., 420, True)]
    print('Three Grok requests share an operating-system limit and release it on exit.')


def test_detail_retry_mask():
    from types import SimpleNamespace

    from wm_director import repair_recipe
    from workflow import read_object, write_object

    with tempfile.TemporaryDirectory(prefix='wm-detail-test-') as root:
        directory = Path(root)
        profile = directory/'profile.json'
        recipe = directory/'recipe.json'
        write_object(profile, {'discovery': {'track': 'track.npz'}})
        write_object(recipe, {'profile': profile.name})
        np.savez(directory/'track.npz', mask=np.ones((10, 10), np.uint8))
        source = SimpleNamespace(count=2)
        for defect, tighten in [(None, False), ('residue', False), ('damage', True)]:
            action = {'kind': 'discovery-detail', 'frame': 0, 'defect': defect}
            updated, focus = repair_recipe(source, directory, 2, recipe, directory, [], [0], action, 18)
            options = read_object(updated)
            assert focus == (0,) and '--model-scale' in options['final_render']
            assert read_object(Path(options['profile']))['discovery']['consensus_mask'] is tighten
    print('A detail retry retains confirmed coverage for residue and narrows the mask for scene damage.')


def test_luma_extent_phase():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import LUMA_EXTENT_FIELDS
    from wm_discovery_render import luma_extent_proofs, luma_extent_retry

    rng = np.random.default_rng(428)
    count = 48
    source = SimpleNamespace(count=count, planes=lambda image: (image,))
    candidate = SimpleNamespace(planes=source.planes)
    cleaned = SimpleNamespace(planes=source.planes)
    clean = [rng.normal(0, 1, (108, 152)).astype(np.float32) for _ in range(count)]
    raw = [image.copy() for image in clean]
    fragment = rng.choice([-22., 22.], (9, 14)).astype(np.float32)
    for i in set(range(40))-{20}:
        raw[i][40:49, 60:74] += fragment
    mask = np.zeros((56, 84), np.uint8)
    mask[4:45, 4:34] = 1
    positions = np.tile(np.asarray([16, 16], np.int32), (count, 1))
    positions[40:, 0] += 22
    track = {'positions': positions, 'mask': mask, 'verified': np.ones(count, bool),
             'cuts': np.asarray([10, 21, 31, 40], np.int32)}
    events = []
    trace = lambda event, **data: events.append((event, data))
    with (patch('wm_discovery_render.get_frame', side_effect=lambda video, i, _cuda: (clean if video is cleaned else raw)[i]),
          patch('wm_discovery_render.read_frames', side_effect=lambda _video, **_kw: iter(enumerate(raw))),
          patch('wm_discovery_render.highpass', side_effect=lambda image, _s: image),
          patch('wm_discovery_render.visual_signatures', return_value=(None, mask))):
        repair = luma_extent_retry(source, candidate, track, {}, 0, 3, False, trace)
        assert luma_extent_retry(source, cleaned, track, {}, 0, 3, False, trace) is None
    extent, = luma_extent_proofs(source, repair)
    index, extra, present = extent.layer, extent.mask, extent.present
    assert index == 0 and present.sum() == 39 and not present[20] and not present[40:].any()
    assert np.array_equal(repair['mask'], mask) and not (extra & mask).any()
    assert np.all(extra[24:33, 44:58]) and 3 in repair['luma_frames']
    assert events[0][1]['eligible'] and not events[-1][1]['eligible']
    invalid = {**repair, 'luma_present': np.ones(count, bool)}
    with np.testing.assert_raises(ValueError):
        luma_extent_proofs(source, invalid)

    both = [image.copy() for image in raw]
    once = [image.copy() for image in clean]
    for i in set(range(40))-{9, 31}:
        both[i][55:64, 75:89] += fragment*1.2
        once[i][55:64, 75:89] += fragment*1.2
    with (patch('wm_discovery_render.get_frame', side_effect=lambda video, i, _cuda: (both if video is source else once)[i]),
          patch('wm_discovery_render.read_frames', side_effect=lambda _video, **_kw: iter(enumerate(both))),
          patch('wm_discovery_render.highpass', side_effect=lambda image, _s: image),
          patch('wm_discovery_render.visual_signatures', return_value=(None, mask))):
        twice = luma_extent_retry(source, candidate, repair, {}, 0, 6, False, trace)
    previous, additional = luma_extent_proofs(source, twice)
    assert all(np.array_equal(repair[key], twice[key]) for key in LUMA_EXTENT_FIELDS)
    assert not (previous.mask & additional.mask).any() and additional.present.sum() == 38
    assert not additional.present[[9, 31]].any() and previous.present[9]
    print('Independent residue repairs preserve earlier masks and only affect their own matched source frames.')


def test_registered_extent_geometry():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery_render import luma_extent_proofs, refresh_luma_visibility

    video = SimpleNamespace(count=12, width=120, height=80)
    positions = np.column_stack((np.arange(12, dtype=np.int32)+20, np.full(12, 15, np.int32)))
    parent = np.ones(12, bool)
    parent[-1] = False
    checked = np.asarray([0, 3, 7, 10], np.int32)
    present = np.isin(np.arange(12), checked)
    signal = np.full((12, 28), 5., np.float32)
    track = {'positions': positions, 'verified': np.ones(12, bool), 'present': parent,
             'mask': np.ones((18, 20), np.uint8),
             'luma_layer': np.asarray(0, np.int32), 'luma_offset': np.asarray([-2, 13], np.int32),
             'luma_mask': np.ones(signal.shape, np.uint8), 'luma_signal': signal,
             'luma_weights': np.ones_like(signal), 'luma_present': present, 'luma_frames': checked}
    extent, = luma_extent_proofs(video, track)
    assert extent.registered and np.array_equal(extent.positions, positions+[-2, 13])
    with patch('wm_discovery_render.read_frames') as decode:
        refresh_luma_visibility(video, track, True)
        decode.assert_not_called()
    assert np.array_equal(track['luma_present'], present)
    invalid = {'luma_offset': np.array([120, 0], np.int32), 'luma_present': np.ones(12, bool)}
    for key, value in invalid.items():
        with np.testing.assert_raises(ValueError):
            luma_extent_proofs(video, {**track, key: value})
    print('Registered extent positions follow their verified layer without editing outside confirmed frames.')


def test_luma_reference_conditioning():
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_discovery import Motif
    from wm_discovery_render import (LumaExtent, extend_faint_runs, faint_luma_match,
                                     luma_reference_masks, luma_response)

    rng = np.random.default_rng(427)
    signal = rng.normal(0, 24, (8, 12)).astype(np.float32)
    weights = np.ones_like(signal)
    mask = np.ones(signal.shape, np.uint8)
    assert faint_luma_match(signal*.30, signal, weights, *luma_response(signal*.30, signal, weights))
    unrelated = signal*.30
    unrelated[:, 6:] *= -1
    assert not faint_luma_match(unrelated, signal, weights, *luma_response(unrelated, signal, weights))
    present = np.array([False, False, True, False, False, False, True, False])
    faint = np.array([True, True, False, True, True, False, False, True])
    assert np.flatnonzero(extend_faint_runs(present, faint, [2, 4])).tolist() == [2, 3, 6, 7]

    video = SimpleNamespace(planes=lambda image: (image,))
    positions = np.array([[4, 5], [4, 5], [4, 5], [8, 5]], np.int32)
    edited = np.array([True, False, False, False])
    extent = LumaExtent(0, positions, mask, signal, weights, edited, np.array([0]))
    buffer = [(i, np.zeros((24, 32), np.float32)) for i in range(4)]
    for i in (0, 1, 3):
        x, y = positions[i]
        buffer[i][1][y:y+8, x:x+12] = signal*(1 if i == 0 else .30)
    masks = np.zeros((4, 24, 32), np.uint8)
    masks[0, 5:13, 4:16] = mask
    with patch('wm_discovery_render.highpass', side_effect=lambda image, _s: image):
        context = luma_reference_masks(video, buffer, masks, [(mask, positions, edited)],
                                       (0, 0), {id(mask): (extent, np.ones(4, bool))})
    assert np.array_equal(context[0], masks[0]) and context[1, 5:13, 4:16].all()
    assert not masks[1:].any() and not context[2:].any()

    parent_signal = rng.normal(0, 24, signal.shape).astype(np.float32)
    parent = Motif(parent_signal, weights, mask.copy())
    parent_positions = np.column_stack((np.arange(5, dtype=np.int32)+8, np.full(5, 8, np.int32)))
    positions = parent_positions + np.asarray([0, 12], np.int32)
    active = np.asarray([False, False, True, False, False])
    known = np.asarray([True, True, True, False, False])
    extent = LumaExtent(0, positions, mask, signal, weights, active, np.asarray([2]), True)
    frames = [(i, np.zeros((48, 48), np.float32)) for i in range(5)]
    for i in (0, 2, 3):
        x, y = parent_positions[i]
        frames[i][1][y:y+8, x:x+12] = parent_signal
        x, y = positions[i]
        frames[i][1][y:y+8, x:x+12] = signal
    write_masks = np.zeros((5, 48, 48), np.uint8)
    x, y = positions[2]
    write_masks[2, y:y+8, x:x+12] = mask
    with patch('wm_discovery_render.highpass', side_effect=lambda image, _s: image):
        context = luma_reference_masks(video, frames, write_masks, [(mask, positions, active)], (0, 0),
                                       {id(mask): (extent, known)}, parents={0: (parent, parent_positions)})
    assert np.flatnonzero(context.any(axis=(1, 2))).tolist() == [0, 2, 3]
    assert np.flatnonzero(write_masks.any(axis=(1, 2))).tolist() == [2]
    assert context[0].sum() == context[3].sum() == 2*mask.sum()
    print('Faint matches stay in the same scene, and reference conditioning never changes the write mask.')


def test_chroma_extent_recipe():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from wm_director import repair_recipe
    from wm_media import digest
    from workflow import read_object, write_object

    with tempfile.TemporaryDirectory(prefix='wm-chroma-extent-') as root:
        directory = Path(root)
        recipe, profile = directory/'recipe.json', directory/'profile.json'
        write_object(profile, {'discovery': {'track': 'track.npz'}})
        write_object(recipe, {'profile': profile.name})
        np.savez(directory/'track.npz', mask=np.ones((10, 10), np.uint8))
        extension = {'mask': np.ones((12, 14), np.uint8), 'extent_frames': np.arange(6, dtype=np.int32)}
        video = SimpleNamespace(count=8)
        action = {'kind': 'discovery-detail', 'frame': 4, 'defect': 'residue'}
        with (patch('wm_director.Video.open', return_value=video),
              patch('wm_discovery_render.broad_chroma_extent_retry', return_value=None) as broad,
              patch('wm_discovery_render.chroma_extent_retry', return_value=extension) as retry):
            next_recipe, focus = repair_recipe(video, directory, 2, recipe, directory, [], [4], action, 18,
                                               comparisons=[(4, 0)], trace=Mock())
            assert focus == (4,) and broad.call_count == retry.call_count == 1
            options = read_object(next_recipe)
            settings = read_object(Path(options['profile']))['discovery']
            assert settings['chroma_extent_repair'] == {'frame': 4, 'layer': 0}
            assert settings['sha256'] == digest(Path(settings['track']))
            assert '--model-scale' not in options.get('final_render', [])
    print('A displayed residue can trigger a separately verified chroma mask before a full re-render.')


def test_repeated_edge_residue():
    from wm_discovery_render import repeated_edge_residue, restoration_mask

    mask = np.zeros((30, 40), np.uint8)
    mask[4:-4, 4:-4] = 1
    edge = (mask > 0) & (cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) < 2)
    source = mask.astype(np.float32)*12
    retained = source*.07
    retained[edge] = source[edge]*.48
    clean = np.zeros_like(source)
    training = [source.copy() for _ in range(12)]
    supported = [(i, source, retained) for i in range(5)]
    identified = repeated_edge_residue(mask, training, supported)
    assert identified['eligible'] and not identified['cleared']
    verified = repeated_edge_residue(mask, training, [(i, source, clean) for i in range(5)])
    assert verified['cleared'] and not verified['eligible']
    assert not repeated_edge_residue(mask, training, [supported[0], *[(i, source, clean) for i in range(1, 5)]])['eligible']
    assert not repeated_edge_residue(mask, training, [(i, source, source*.45) for i in range(5)])['eligible']
    updated = restoration_mask({}, None, mask, edge=True)
    assert updated[3, 5] == 1 and updated[2, 5] == 0 and updated[3, 3] == 0
    assert mask[3, 5] == 0
    print('An edge correction needs a clean core and a repeated rim defect in independent frames.')


def test_candidate_review_resume():
    from threading import BoundedSemaphore
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import resume_candidate_review
    from wm_director import recheck_job, review_method
    from workflow import write_object

    with tempfile.TemporaryDirectory(prefix='wm-review-resume-') as root:
        directory = Path(root)
        source = directory/'source.mp4'
        report_path = directory/'pass-01/local-layers/director.json'
        attempt = report_path.parent/'attempt-01'
        attempt.mkdir(parents=True)
        write_object(attempt/'run.json', {'status': 'complete'})
        config = {'model': 'fixture', 'reasoning_effort': 'low', 'max_url_p95': .08}
        last = {'directory': str(attempt), 'local_findings': [],
                'decision': {'decision': 'review', 'reason': 'earlier evidence was unclear'}}
        report = {'source': str(source), 'status': 'needs_review', 'attempts': [last]}
        write_object(report_path, report)
        trace = lambda *_args, **_kwargs: None
        with (patch('wm_batch.recheck_job', return_value={**report, 'status': 'complete'}) as recheck,
              patch('wm_batch.execute_plan', side_effect=AssertionError('Render was repeated.'))):
            assert resume_candidate_review(source, directory, config, False, [], trace)['status'] == 'complete'
            last['review_method'] = review_method(config)
            write_object(report_path, report)
            assert resume_candidate_review(source, directory, config, False, [], trace) is None
        recheck.assert_called_once_with(source, report_path.parent, config, False, render_gate=None)
        report.update(status='failed', error='Grok attempted a tool call in a visual-only session.')
        write_object(report_path, report)
        with patch('wm_batch.recheck_job', return_value={**report, 'status': 'complete'}) as recheck:
            assert resume_candidate_review(source, directory, config, False, [], trace)['status'] == 'complete'
        recheck.assert_called_once_with(source, report_path.parent, config, False, render_gate=None)
        report.pop('error')
        report['status'] = 'needs_review'

        job = report_path.parent
        recipe, detail = job/'recipe-01.json', job/'recipe-02.json'
        write_object(recipe, {})
        write_object(detail, {})
        write_object(attempt/'run.json', {'status': 'complete', 'stages': [], 'inputs': {}})
        last.update(recipe=str(recipe), frames=[2], review_regions=[], comparison_regions=[[2, 0, 0, 8, 8]],
                    comparison_layers=[0])
        write_object(report_path, report)
        config.update(max_repairs=1, repair_radius_frames=18, review_samples=4)
        calls = []

        def review(_source, candidate, _recipe, _config, _cuda, _trace, record, _persist, **options):
            calls.append((candidate.name, options['repairs_left'], options['focus_frames']))
            record['local_findings'] = []
            if candidate == attempt:
                return {'decision': 'repair', 'repair': {'kind': 'discovery-detail', 'frame': 2}}
            return {'decision': 'accept'}

        with (patch('wm_director.Video.open', return_value=SimpleNamespace(path=source)),
              patch('wm_director.build_plan', return_value=[]), patch('wm_director.input_files', return_value=[]),
              patch('wm_director.review_candidate', side_effect=review),
              patch('wm_director.repair_recipe', return_value=(detail, (2,))) as repair,
              patch('wm_director.execute_plan') as execute, patch('wm_director.learn_from_release'),
              patch('wm_director.publish_result', side_effect=lambda _d, _a, row, _t: row.update(status='complete'))):
            assert recheck_job(source, job, config, False, render_gate=BoundedSemaphore(1))['status'] == 'complete'
        assert calls == [('attempt-01', 1, ()), ('attempt-02', 0, (2,))]
        assert repair.call_args.args[4] == attempt and repair.call_args.args[5] == []
        assert execute.call_count == 1 and execute.call_args.args[0] == job/'attempt-02'
    print('Resuming reused a verified candidate and only rendered the newly scheduled repair.')


def test_batch_prior_evidence():
    import json
    from threading import BoundedSemaphore
    from types import SimpleNamespace
    from unittest.mock import patch

    from wm_batch import completed_route, prior_visual_history, run_batch, run_source
    from wm_media import digest
    from workflow import read_object, write_object

    with tempfile.TemporaryDirectory(prefix='wm-batch-evidence-') as directory:
        base = Path(directory)
        source, root, prior = base/'input', base/'batch', base/'prior'
        source.mkdir()
        original = source/'source.mp4'
        original.write_bytes(b'source')
        item = {'index': 1, 'source': str(original), 'source_sha256': digest(original)}
        route = root/'jobs/001/pass-01/local-layers'
        attempt = route/'attempt-01'
        attempt.mkdir(parents=True)
        output = route/'result.mp4'
        candidate = attempt/'result.mp4'
        for path in (output, candidate):
            path.write_bytes(b'candidate')
        track, profile, recipe = attempt/'track.npz', attempt/'profile.json', attempt/'recipe.json'
        np.savez(track, mask=np.ones((12, 12), np.uint8), positions=np.array([[6, 6]]))
        write_object(profile, {'discovery': {'track': str(track)}})
        write_object(recipe, {'profile': profile.name})
        write_object(attempt/'verification.json', {'full_decode_ok': True})
        write_object(attempt/'run.json', {'status': 'complete', 'inputs': {
            str(path): digest(path) for path in (original, recipe, profile)}, 'stages': [
            {'name': 'scan', 'artifacts': {str(track): digest(track)}},
            {'name': 'render', 'artifacts': {str(candidate): digest(candidate)}}]})
        write_object(route/'director.json', {'source': str(original), 'status': 'complete', 'result': str(output),
            'attempts': [{'directory': str(attempt), 'recipe': str(recipe),
                          'decision': {'decision': 'accept', 'reason': 'clear'}, 'local_findings': []}]})
        write_object(root/'manifest.json', {'sources': [item]})
        write_object(root/'batch.json', {'jobs': [{**item, 'status': 'complete', 'result': str(output),
                                                 'result_sha256': digest(output)}]})
        prior_job = prior/'jobs/008'
        prior_job.mkdir(parents=True)
        prior_source = prior/'other-name.mp4'
        write_object(prior/'manifest.json', {'sources': [{**item, 'index': 8, 'source': str(prior_source)}]})
        write_object(prior_job/'overlay-evidence.json', {'source_sha256': item['source_sha256'],
                                                       'regions': [[0, 40, 20, 10, 10]]})
        history_report = prior_job/'pass-01/adaptive/director.json'
        history_report.parent.mkdir(parents=True)
        write_object(history_report, {'source': str(prior_source), 'planning': {'rounds': [
            {'frames': [0], 'decision': {'decision': 'track', 'label': 'fixture',
                                        'observations': [{'frame': 0, 'rect': [125, 500, 250, 667]}]}}]}})
        with patch('wm_batch.Video.open', return_value=SimpleNamespace(width=80, height=60, count=1)):
            assert completed_route(item, root, [[0, 6, 6, 10, 10]]) is not None
            config = {'model': 'test', 'reasoning_effort': 'xhigh'}
            assert run_batch(source, root, config, None, False, 1, 1, True, False, prior_runs=[prior]) == 2
        job = read_object(root/'batch.json')['jobs'][0]
        assert job['status'] == 'needs_review' and job.get('result') is None
        assert output.read_bytes() == b'candidate'

        class OCR:
            def crop(self, *_args):
                return []

        reader = OCR()

        def plan(_video, _directory, _identity, candidates, *_args, **options):
            assert candidates[0][0] == {'rect': [40, 20, 50, 30], 'provider': 'confirmed-source'}
            assert [row['provider'] for row in candidates[0]] == ['confirmed-source', 'prior-visual']
            assert options['read_text'] == reader.crop
            return recipe

        def review(_source, _directory, name, _recipe, settings, *_args, **_options):
            json.dumps(settings)
            if name == 'adaptive':
                settings['required_overlay_regions'].append([0, 55, 20, 10, 10])
            return {'status': 'needs_review', 'attempts': []}

        video = SimpleNamespace(width=80, height=60, count=1, check_timestamps=lambda: None)
        local = SimpleNamespace(sample=lambda *_args, **_opts: ([], {}), recipe=lambda *_args: None)
        with (patch('wm_batch.Video.open', return_value=video), patch('wm_batch.color_retry_recipe', return_value=None),
              patch('wm_batch.remembered_proposals', return_value={}),
              patch('wm_batch.local_layer_recipe', side_effect=plan), patch('wm_batch.run_route', side_effect=review)):
            config.update(local_layers=True, learned_marks=str(base/'marks'), discovery_local={'model': 'local'})
            history = prior_visual_history([prior])[item['source_sha256']]
            result = run_source(item, root, config, local, False, BoundedSemaphore(1), reader, history=history)
            assert result['status'] == 'needs_review'
        evidence = read_object(root/'jobs/001/overlay-evidence.json')
        assert evidence['regions'] == [[0, 40, 20, 10, 10], [0, 55, 20, 10, 10]]
    print('A resumed batch withheld an incomplete candidate, reused identical-source observations, and saved new evidence.')


def test_failed_render_reuse():
    import json
    from unittest.mock import patch

    from types import SimpleNamespace

    from wm_batch import interrupted_recipe, recorded_context_hints
    from wm_media import digest, tools_directory
    from workflow import write_object

    with tempfile.TemporaryDirectory(prefix='wm-failed-render-') as directory:
        base = Path(directory).resolve()
        tools, code = base/'.watermark-tools', base/'runs'/'fixed-code'
        tools.mkdir()
        code.mkdir(parents=True)
        with patch.dict('os.environ', {'WATERMARK_TOOLS_DIR': ''}):
            assert tools_directory(code) == tools
        with patch.dict('os.environ', {'WATERMARK_TOOLS_DIR': str(base/'custom')}):
            assert tools_directory(code) == base/'custom'

        source = base/'source.mp4'
        source.write_bytes(b'exact input')
        other_name = base/'duplicate.mp4'
        other_name.write_bytes(source.read_bytes())
        job = base/'job'
        attempt = job/'pass-01'/'local-layers'/'attempt-01'
        attempt.mkdir(parents=True)
        profile, recipe = job/'profile.json', job/'recipe.json'
        write_object(profile, {'discovery': {'track': 'local'}})
        write_object(recipe, {'profile': profile.name})
        report_path = attempt.parent/'director.json'
        report = {'source': str(source), 'status': 'failed', 'attempts': [
            {'directory': str(attempt), 'recipe': str(recipe)}]}
        write_object(report_path, report)
        state = {'inputs': {str(path): digest(path) for path in (source, profile, recipe)},
                 'stages': [{'name': 'scan', 'status': 'complete'}, {'name': 'render', 'status': 'failed'}]}
        write_object(attempt/'run.json', state)
        identity = digest(source)
        assert interrupted_recipe(source, job, identity) == (recipe, attempt)
        assert interrupted_recipe(other_name, base/'new', identity, [(job, str(source))]) == (recipe, None)
        write_object(report_path, {**report, 'status': 'running'})
        from workflow import lock_job
        with lock_job(attempt):
            assert interrupted_recipe(other_name, base/'new', identity, [(job, str(source))]) is None
        assert interrupted_recipe(other_name, base/'new', identity, [(job, str(source))]) == (recipe, None)
        write_object(report_path, report)
        state['stages'][1]['status'] = 'complete'
        state['stages'].append({'name': 'audit', 'status': 'failed'})
        write_object(attempt/'run.json', state)
        assert interrupted_recipe(source, job, identity) is None
        state['stages'][1]['status'] = 'failed'
        write_object(attempt/'run.json', state)
        write_object(profile, {'discovery': {'track': 'changed'}})
        assert interrupted_recipe(source, job, identity) is None
        result = attempt/'result.mp4'
        result.write_bytes(b'completed video')
        review = {'decision': 'review', 'reason': 'another mark remains'}
        record = {**report['attempts'][0], 'frames': [7], 'decision': review}
        final = {**report, 'status': 'needs_review', 'attempts': [record]}
        state.update(status='complete', stages=[{'name': 'render', 'status': 'complete',
                                                 'artifacts': {str(result): digest(result)}}])
        write_object(attempt/'run.json', state)
        finding = {'frame': 7, 'finding': 'watermark', 'label': 'other', 'rect': [0, 0, 200, 200]}
        event = {'event': 'grok.decision', 'phase': 'review', 'attempt': 1,
                 'response': {**review, 'findings': [finding]}}
        (attempt.parent/'events.jsonl').write_text(json.dumps(event)+'\n', encoding='utf-8')
        with patch('wm_batch.Video.open', return_value=SimpleNamespace(width=120, height=100)):
            assert recorded_context_hints(source, report_path, final, record) == [
                {'frame': 7, 'label': 'other', 'rect': [0, 0, 24, 20]}]
            result.write_bytes(b'different output')
            assert not recorded_context_hints(source, report_path, final, record)
    print('A frozen workflow finds its model assets and retries only the matching prepared render.')


def test_batch_learning_release_integrity():
    from wm_batch import learn_from_batches
    from workflow import fingerprints, write_object

    with tempfile.TemporaryDirectory(prefix='wm-prior-learning-') as root:
        root = Path(root)
        source, profile, recipe = (root/name for name in ('source.mp4', 'profile.json', 'recipe.json'))
        source.write_bytes(b'original')
        write_object(profile, {'discovery': {'layer_count': 1}})
        write_object(recipe, {'profile': 'profile.json'})
        inputs = fingerprints([source, profile, recipe])
        item = {'index': 1, 'source': str(source), 'source_sha256': inputs[str(source)]}
        batch = root/'batch'
        route = batch/'jobs/001/pass-01/local-layers'
        attempt = route/'attempt-01'
        attempt.mkdir(parents=True)
        candidate = attempt/'result.mp4'
        candidate.write_bytes(b'processed')
        output = route/'result.mp4'
        output.write_bytes(candidate.read_bytes())
        audit, verify, track = (attempt/name for name in ('watermark-audit.json', 'verification.json', 'track.npz'))
        write_object(audit, {'discovery_passed': True, 'discovery_verified_frames': 3})
        write_object(verify, dict(frames=3, same_video_timing_and_size=True,
                                  same_audio_payload_and_timing=True, full_decode_ok=True))
        signal = np.ones((8, 8), np.float32)
        np.savez(track, source_sha256=item['source_sha256'], signal=signal, weights=signal, mask=signal.astype(np.uint8))
        stages = [{'name': name, 'args': [name, str(profile)], 'status': 'complete',
                   'outputs': [str(path)], 'artifacts': fingerprints([path])}
                  for name, path in [('scan', track), ('render', candidate), ('verify', verify), ('audit', audit)]]
        write_object(attempt/'run.json', dict(status='complete', inputs=inputs, stages=stages))
        accepted = {'decision': 'accept', 'reason': 'clear'}
        write_object(route/'director.json', dict(status='complete', source=str(source), result=str(output),
                     attempts=[dict(directory=str(attempt), recipe=str(recipe), local_findings=[],
                                    source_comparison=accepted, decision=accepted)]))
        write_object(batch/'manifest.json', {'sources': [item]})
        write_object(batch/'batch.json', {'jobs': [{**item, 'status': 'complete',
                      'result_sha256': fingerprints([output])[str(output)]}]})
        events = []
        trace = lambda event, **data: events.append((event, data))
        library = root/'library'
        learn_from_batches([batch], library, trace)
        learn_from_batches([batch], library, trace)
        assert len(list(library.glob('*.npz'))) == 1
        assert [data['new'] for event, data in events if event == 'batch.learned_prototype'] == [True, False]
        write_object(audit, {'discovery_passed': True, 'discovery_verified_frames': 99})
        rejected = root/'rejected'
        learn_from_batches([batch], rejected, trace)
        assert not rejected.exists() and events[-1][0] == 'batch.learning_rejected'
    print('Prior releases enroll only unchanged verified tracks and never duplicate a stored prototype.')


def test_stationary_white():
    from wm_white_matte import FixedWhiteMatte, WHITE

    height, width = 60, 96
    rect = [12, 8, 56, 34]
    x, y, w, h = rect
    alpha = np.zeros((h, w), np.float32)
    alpha[6:25, 8:45] = .075
    alpha[8:22, 13:39] = .142
    matte = FixedWhiteMatte([{'rect': rect, 'ranges': [[1, 2]]}], [alpha], 3)
    clean = tuple(np.full((height//s, width//s), value, np.uint8)
                  for s, value in zip((1, 2, 2), (91, 111, 155), strict=True))
    marked = tuple(plane.copy() for plane in clean)
    for plane, s, white in zip(marked, (1, 2, 2), WHITE, strict=True):
        patch = plane[y//s:(y+h)//s, x//s:(x+w)//s]
        opacity = cv2.resize(alpha, patch.shape[::-1], interpolation=cv2.INTER_AREA)
        patch[:] = np.rint(patch*(1-opacity) + white*opacity)
    inactive = tuple(plane.copy() for plane in marked)
    matte.restore(0, inactive)
    if any(not np.array_equal(a, b) for a, b in zip(marked, inactive, strict=True)):
        raise ValueError('Stationary white repair altered a frame outside its active range.')
    matte.restore(1, marked)
    if any(np.abs(a.astype(np.int16)-b).max() > 1 for a, b in zip(clean, marked, strict=True)):
        raise ValueError('Stationary white repair did not recover the original YUV within rounding error.')
    print('Stationary white inverse compositing and inactive frames passed.')


def test_white_alignment():
    from wm_white_matte import WhiteEdgeAlignment

    alpha = np.zeros((72, 96), np.float32)
    cv2.circle(alpha, (42, 35), 25, .25, -1)
    cv2.circle(alpha, (35, 31), 8, 0, -1)
    align = WhiteEdgeAlignment(alpha)
    yy, xx = np.mgrid[:160, :180]
    luma = (76 + xx*.12 + yy*.04).astype(np.float32)
    rect = (38, 36, 96, 72)
    expected = np.float32([1, -2])
    x, y, w, h = rect
    x += int(expected[0])
    y += int(expected[1])
    patch = luma[y:y+h, x:x+w]
    patch[:] = patch*(1-alpha) + 235*alpha
    result = align.align(np.rint(luma).astype(np.uint8), rect, np.zeros(2, np.float32))
    if result is None or np.linalg.norm(result-expected) > .35:
        raise ValueError('White edge alignment did not recover the known displacement.')
    print('White edge alignment recovered the known displacement.')


def test_corner_grid():
    from wm_corner_grid import CornerGrid

    height, width = 144, 224
    yy, xx = np.mgrid[:height, :width]
    phase_x, phase_y = xx*(2*np.pi/19.2), yy*(2*np.pi/19.2)
    pattern = 5*np.sin(phase_x) + 6*np.cos(phase_y) + 4*np.cos(phase_y-phase_x)
    known = np.full((height, width), 128, np.uint8)
    marked = np.rint(known+pattern).astype(np.uint8)
    clean_c = np.full((height//2, width//2), 128, np.uint8)
    marked_c = np.rint(clean_c+.4*pattern[::2, ::2]).astype(np.uint8)
    blend = np.ones((24, height, width), np.float32)
    blend[:, 106:125, 73:149] = 0
    examples = tuple(np.stack([a]*12+[b]*12) for a, b in
                     ((known, marked), (clean_c, marked_c), (clean_c, marked_c)))
    grid = CornerGrid((height, width), (320, 576))
    grid.calibrate(examples, blend, np.arange(24) >= 12)
    grid.finish_calibration()
    samples = tuple(p[12:19] for p in examples)
    restored = grid.restore(samples, blend[12:19], np.zeros(7, np.int32))
    body = np.s_[:, 24:100, 28:190]
    error = lambda p: np.abs(p[body].astype(np.int16)-128).mean()
    if error(restored[0]) > .6*error(samples[0]):
        raise ValueError('The GPU corner filter did not suppress the synthetic stationary grid.')
    masked = blend[0] == 0
    shared = cv2.resize(masked.astype(np.float32), clean_c.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    for before, after, protect in zip(samples, restored, (masked, shared, shared), strict=True):
        if not np.array_equal(before[:, protect], after[:, protect]):
            raise ValueError('Corner grid filtering changed protected YUV samples.')
    print('GPU corner filtering reduced the synthetic grid and preserved protected YUV samples.')


def test_encoder_colors(video, root):
    from dataclasses import replace

    from wm_media import encode, get_frame

    # Constant luma with contrasting chroma exposes unintended matrix conversion.
    frame = np.full(video.width*video.height*3//2, 128, np.uint8)
    _, u, v = video.planes(frame)
    u[:, :u.shape[1]//3] = 96
    v[:, -v.shape[1]//3:] = 160
    source = root / 'color-source.y4m'
    header = f'YUV4MPEG2 W{video.width} H{video.height} F30:1 Ip A1:1 C420jpeg\nFRAME\n'
    source.write_bytes(header.encode('ascii') + frame.tobytes())
    video = replace(video, path=source, count=1)
    result = root / 'color-test.mp4'
    with encode(video, result, 0, 1, 14) as stream:
        stream.write(frame)
    decoded = get_frame(Video.open(result), 0, cuda=False)
    errors = [np.abs(a.astype(np.int16)-b).mean()
              for a, b in zip(video.planes(frame), video.planes(decoded), strict=True)]
    if max(errors) > .6:
        raise ValueError(f'GPU encoding changed the YUV color samples: {errors}.')
    print('GPU encode preserved the synthetic BT.709 color samples.')


def test_mux_offset(video, root):
    from dataclasses import replace

    from wm_media import encode, executable

    audio = root/'offset-audio.m4a'
    subprocess.run([executable('ffmpeg'), '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    'sine=frequency=440:sample_rate=48000:duration=0.6', '-c:a', 'aac', str(audio)], check=True)
    source = replace(video, path=audio, count=16, offset=.021333)
    result = root/'offset-video.mp4'
    pixels = np.full(source.width*source.height*3//2, 128, np.uint8)
    with encode(source, result, 0, source.count, 18) as stream:
        for frame in range(source.count):
            source.planes(pixels)[0][:] = 68+frame
            stream.write(pixels)
    decoded = Video.open(result)
    decoded.check_timestamps()
    if decoded.count != source.count or decoded.offset != source.offset:
        raise ValueError('MP4 B-frame mux did not preserve the small positive video offset.')
    print('GPU MP4 mux retained the 21.333 ms video offset at 48 kHz audio.')


def test_aligned_edge():
    from wm_edge_retouch import fit_rising_edge, repair_along_edge

    yy, xx = np.mgrid[:130, :350]
    edge = 34 + 1.75*yy
    light = np.clip((xx-edge+4)/8, 0, 1)
    clean = np.rint(63 + 148*light + yy/26).astype(np.uint8)
    support = np.zeros_like(clean)
    cv2.putText(support, 'QA', (98, 79), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 1, 5)
    marked = (clean + 11*support).astype(np.uint8)
    curve = fit_rising_edge(marked, 14, 114)
    if curve is None or np.max(np.abs(np.polyval(curve, (20, 65, 105))-(34+1.75*np.array([20, 65, 105])))) > 4:
        raise ValueError('Diagonal retouch did not locate the synthetic edge.')
    error_before = np.abs(marked.astype(np.int16)-clean).sum()
    repair_along_edge(marked, marked.copy(), support, 0, 0, curve, (18, 98), 1)
    error_after = np.abs(marked.astype(np.int16)-clean).sum()
    if error_after >= 0.65*error_before or not np.array_equal(marked[:18], clean[:18]):
        raise ValueError(f'Aligned-edge retouch did not meet the synthetic target: {error_before} -> {error_after}.')
    print(f'Aligned-edge synthetic error {error_before} -> {error_after}.')


def test_stripe_retouch():
    from wm_retouch import retouch_stripe

    yy, xx = np.mgrid[:100, :240]
    clean = np.where(xx < 38 + yy, 43, 202 + xx % 13).astype(np.uint8)
    support = ((yy > 30) & (yy < 70) & (xx > 35) & (xx < 200)).astype(np.uint8)
    light = (np.minimum(clean[18], clean[82]) > 170)[None].astype(np.float32)
    noise = ((support > 0) & (light > 0)) * 9
    marked = (clean + noise).astype(np.uint8)
    before = marked.copy()
    retouch_stripe(marked, support, 18, 82, light, 0)
    if (np.abs(marked.astype(np.int16)-clean).sum() > 0.35 * noise.sum()
            or not np.array_equal(marked[:, light[0] == 0], before[:, light[0] == 0])):
        raise ValueError('Stripe retouch crossed a protected dark edge or failed to remove a bright ghost.')
    print('Bright stripe restoration and protected dark edge passed.')


def test_caption_references():
    from wm_caption import CaptionReferences

    clean = np.full((50, 124), 120, np.uint8)
    cv2.putText(clean, 'QA', (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 24, 7)
    cv2.putText(clean, 'QA', (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 225, 2)
    glyph = (clean == 24) | (clean == 225)
    support = np.zeros_like(glyph)
    support[:, 32:69] = True
    tested = support & glyph
    source = clean.copy()
    source[tested] = np.clip(clean[tested].astype(np.int16) - 14, 0, 255).astype(np.uint8)
    inconsistent = np.roll(clean, 3, axis=1)
    valid = np.ones_like(glyph)
    CaptionReferences.restore_luma(source, [inconsistent, clean, clean], [valid]*3, support)
    repaired = (source == clean) & tested
    if (repaired.sum() < 0.8 * tested.sum()
            or not np.array_equal(source[~support], clean[~support])):
        raise ValueError('Caption consensus failed to restore verified strokes inside the mask.')
    print('Caption consensus and untouched pixels passed.')


def test_caption_context():
    from wm_caption import CaptionReferences

    clean = np.full((64, 144), 121, np.uint8)
    cv2.putText(clean, 'QA', (34, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 25, 7)
    cv2.putText(clean, 'QA', (34, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 224, 2)
    validity = [np.ones_like(clean, dtype=bool)] * 4
    donors = [clean] * 3 + [np.roll(clean, 2, axis=1)]
    model = CaptionReferences.consensus(donors, validity, details=True)
    support = np.zeros_like(clean, dtype=bool)
    support[17:51, 36:75] = True
    affected = support & (clean == 121)
    current = clean.copy()
    current[affected] += 11
    baseline = current.copy()
    CaptionReferences.restore_context(current, baseline, donors, validity, support, model)
    if not np.any((current == clean) & affected) or np.any(current[~affected] != baseline[~affected]):
        raise ValueError('Caption context failed to recover stable background or touched protected pixels.')
    u = np.full((32, 72), 128, np.uint8)
    v = u.copy()
    colors = [(donor, u, v) for donor in donors]
    tinted = [clean.copy(), u.copy(), v.copy()]
    tinted[1][:] = 139
    tinted[2][:] = 118
    CaptionReferences.restore_chroma(tinted, colors, validity, support, model)
    if np.all(tinted[1] == 139) or not np.array_equal(tinted[0], clean) or np.any(tinted[1][:, :16] != 139):
        raise ValueError('Caption chroma recovery failed to preserve out-of-mask samples.')
    print('Caption context and neutral stroke chroma passed.')


def test_caption_patch_guard():
    from wm_caption_inpaint import protected_composite, subtitle_guard

    y = np.full((48, 100), 113, np.uint8)
    cv2.putText(y, 'QA', (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 24, 6)
    cv2.putText(y, 'QA', (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 225, 2)
    u = np.full((24, 50), 126, np.uint8)
    v = np.full_like(u, 131)
    guard = subtitle_guard(y, 900, (900, 948))
    support = np.zeros_like(guard)
    support[:, 10:79] = True
    before = [p.copy() for p in (y, u, v)]
    proposal = np.full((48, 100, 3), (150, 92, 41), np.uint8)
    count = protected_composite((y, u, v), proposal, support, guard, 'bt709', 0.85)
    chroma_guard = cv2.resize(guard.astype(np.float32), (50, 24), interpolation=cv2.INTER_AREA) > 0
    if (not count or not np.array_equal(y[guard | ~support], before[0][guard | ~support])
            or not np.array_equal(u[chroma_guard], before[1][chroma_guard])
            or not np.array_equal(v[chroma_guard], before[2][chroma_guard])
            or not np.any(y[support & ~guard] != before[0][support & ~guard])):
        raise ValueError('Subtitle-protected GPU compositing changed a protected sample or skipped the background.')
    print('Subtitle-protected GPU compositing passed.')


def test_caption_layer():
    from fractions import Fraction

    from wm_caption_inpaint import CaptionPatch
    from wm_caption_layers import estimate_layer
    from wm_media import rgb_to_yuv

    width, height = 100, 48
    opacity = np.zeros((height, width), np.uint8)
    white = np.zeros_like(opacity)
    for plane, thickness in ((opacity, 7), (white, 2)):
        cv2.putText(plane, 'QA', (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness, cv2.LINE_AA)
    opacity, white = opacity.astype(np.float32)/255, white.astype(np.float32)/255
    fill = 16*opacity + 219*white
    reference = (np.full_like(opacity, 136), np.full((height//2, width//2), 103, np.float32),
                 np.full((height//2, width//2), 154, np.float32))
    coverage = cv2.resize(opacity, reference[1].shape[::-1], interpolation=cv2.INTER_AREA)
    caption = (reference[0]*(1-opacity)+fill,
               128+(reference[1]-128)*(1-coverage), 128+(reference[2]-128)*(1-coverage))
    alpha, premultiplied = estimate_layer(caption, reference, np.ones_like(white, dtype=bool))
    prediction = np.full((height, width, 3), (161, 106, 78), np.uint8)
    target_y = rgb_to_yuv(prediction, 'bt709')[0]
    expected = target_y*(1-opacity)+fill
    support = np.zeros_like(opacity, dtype=bool)
    support[4:44, 7:79] = True
    video = Video(Path('self-test.mp4'), width, height, Fraction(30), 3, 0, '1:1', 'bt709', 'bt709', 'bt709')
    track = dict(source_sha256='layer-test', profile_sha256='layer-test')
    with tempfile.TemporaryDirectory(prefix='watermark-layer-') as root:
        cache = Path(root) / 'layer.npz'
        np.savez(cache, region=[0, 0, width, height], frames=[0, 3], layer_frames=[1, 2], band=[0, height],
                 diameter=13, strength=1, fade=1, prediction=np.stack([prediction]*3), masks=np.stack([support]*3),
                 layer_alpha=alpha, layer_fill=premultiplied, top_weight=np.ones((height, 1), np.float32), **track)
        patch = CaptionPatch(video, track, cache)
        for i in range(video.count):
            frame = np.full(width*height*3//2, 130, np.uint8)
            patch.apply(i, frame)
            y, u, v = video.planes(frame)
            inside = opacity == 1
            reference_y = expected if i == 1 else target_y
            error = np.abs(y[inside].astype(np.float32)-reference_y[inside]).mean()
            chroma_outside = cv2.resize(support.astype(np.float32), u.shape[::-1], interpolation=cv2.INTER_AREA) == 0
            if error > 2 or np.any(y[~support] != 130) or np.any(u[chroma_outside] != 130) or np.any(v[chroma_outside] != 130):
                raise ValueError('Caption layer timing, foreground recovery or patch boundaries failed.')
    print('Caption layer timing, foreground recovery and unchanged outside samples passed.')


def test_capsule_rim():
    width, height = 200, 80
    rim = CapsuleRim(width, height, (5, 5, 5, 5))
    shift = np.float32([0.5, -0.5])
    _, band, _ = rim.samples(*shift, 1)
    _, _, template_fraction = rim.samples(0, 0, 1)
    pulse = np.exp(-0.5 * ((template_fraction * 16 - 8) / 1.1) ** 2)
    d = -0.4 * pulse
    k = 0.28 * pulse
    mx, my = np.meshgrid(np.arange(width, dtype=np.float32)-shift[0],
                         np.arange(height, dtype=np.float32)-shift[1])
    reference = np.full((height, width), 120, np.uint8)
    reference[:, 88:96] = 25
    reference[:, 96:104] = 230
    observed = reference.astype(np.float32) + cv2.remap(d, mx, my, cv2.INTER_LINEAR) * (reference.astype(np.float32)-128)
    observed += 128 * cv2.remap(k, mx, my, cv2.INTER_LINEAR)
    observed = np.rint(observed).astype(np.uint8)
    aligned = rim.align(observed, np.zeros(2, np.float32), (d, k))
    if np.max(np.abs(aligned - shift)) > 0.25:
        raise ValueError(f'Rim phase alignment missed the synthetic shift: {aligned}.')
    # Only the clean background is contaminated in this synthetic ringing case.
    noise = np.rint(13 * band).astype(np.uint8)
    noise[:, 88:104] = 0
    marked = reference + noise
    rim.restore([marked], shift)
    if (np.abs(marked.astype(np.int16)-reference).mean() > noise.mean() * 0.4
            or not np.array_equal(marked[:, 88:104], reference[:, 88:104])
            or not np.array_equal(marked[band == 0], reference[band == 0])):
        raise ValueError('Capsule-rim repair did not preserve the synthetic caption or outside pixels.')
    print('Capsule-rim phase alignment and synthetic caption preservation passed.')
