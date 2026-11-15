import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
import framemend_pipeline as fp  # noqa: E402


FAKE_WM_AUTO = """
import json, sys
from pathlib import Path
src, job = Path(sys.argv[1]), Path(sys.argv[2])
job.mkdir(parents=True)
mode = Path(__file__).with_name("mode").read_text().strip()
(job / "plan-01").mkdir()
(job / "plan-01" / "big.bin").write_bytes(b"x" * 10)
if mode == "complete":
    (job / "result.mp4").write_bytes(b"clean-" + src.read_bytes())
    json.dump({"status": "complete", "result": str(job / "result.mp4")}, open(job / "auto.json", "w"))
else:
    json.dump({"status": mode, "reason": "test"}, open(job / "auto.json", "w"))
"""


class PipelineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(fp, "needs_normalizing", return_value=(False, "30"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.downloads = base / "crawler/lizhi/downloads"
        self.ready = base / "produced/jobs/lizhi/ready-downloads"
        self.root = base / "framemend"
        self.app = base / "app"
        self.app.mkdir()
        (self.app / "wm_auto.py").write_text(FAKE_WM_AUTO)
        (self.app / "mode").write_text("complete")
        self.key = base / "produced.key"
        self.key.write_text("secret-key\n")
        self.cfg = fp.Config({
            "root": str(self.root), "app": str(self.app), "python": sys.executable,
            "sources": {"lizhi": str(self.downloads), "chiguo": str(base / "crawler/chiguo/downloads")},
            "enabled_sources": ["lizhi"], "min_size": 10, "settle_seconds": 0,
            "api_url": "", "api_key_file": str(self.key), "tool": "framemend@abc1234",
        })
        self.hot = self.make_drama("530_Hot", "530", hot={"field": "x", "value": 200, "threshold": 100})
        self.cold = self.make_drama("531_Cold", "531")

    def tearDown(self):
        self.tmp.cleanup()

    def make_drama(self, name, drama_id, hot=None, episodes=(1, 2)):
        drama = self.downloads / "ai-duanju" / name
        for number in episodes:
            episode = drama / "episodes" / f"{number:02d}"
            episode.mkdir(parents=True)
            (episode / "video.mp4").write_bytes(b"original-video-%d" % number)
            (episode / "metadata.json").write_text(json.dumps({"episode_number": number}))
            view = self.ready / "ai-duanju" / name / "episodes" / f"{number:02d}"
            view.mkdir(parents=True)
            os.link(episode / "video.mp4", view / "video.mp4")
        (drama / "metadata.json").write_text(json.dumps({"drama_id": drama_id}))
        if hot is not None:
            (drama / ".dewatermark").write_text(json.dumps(hot))
        return drama

    def worker_args(self, name, drama_id, number, work=None):
        metadata = Path(self.tmp.name) / f"meta-{drama_id}-{number}.json"
        metadata.write_text(json.dumps({"sourceIdentity": f"lizhi:{drama_id}:{number}"}))
        work = work or Path(self.tmp.name) / f"work-{drama_id}-{number}"
        view = self.ready / "ai-duanju" / name / "episodes" / f"{number:02d}" / "video.mp4"
        return ["run", "--api-url", "https://api.example/api", "--source", str(view), "--metadata", str(metadata),
                "--work-dir", str(work), "--engine", "auto", "--prepared-fd", "3"]

    def complete_state(self, drama_id="530", number=1):
        output = self.cfg.output_path("lizhi", drama_id, number)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clean")
        fp.write_json(self.cfg.state_path("lizhi", drama_id, number),
                      {"source": "lizhi", "drama": drama_id, "episode": number, "status": fp.COMPLETE,
                       "output": str(output), "sha256": fp.sha256(output), "tool": "framemend@abc1234"})
        return output

    # ------------------------------------------------------------------ guard

    def test_cold_drama_passes_through_unchanged(self):
        args = self.worker_args("531_Cold", "531", 1)
        reason, out, binding = fp.decide(self.cfg, args)
        self.assertEqual(out, args)
        self.assertIsNone(binding)

    def test_hot_episode_is_held_until_processed(self):
        reason, out, _ = fp.decide(self.cfg, self.worker_args("530_Hot", "530", 1))
        self.assertIsNone(out)
        self.assertIn("等待", reason)

    def test_review_episode_stays_held(self):
        fp.write_json(self.cfg.state_path("lizhi", "530", 1), {"status": fp.REVIEW})
        _, out, _ = fp.decide(self.cfg, self.worker_args("530_Hot", "530", 1))
        self.assertIsNone(out)

    def test_complete_episode_swaps_source_and_binds(self):
        output = self.complete_state()
        args = self.worker_args("530_Hot", "530", 1)
        _, out, binding = fp.decide(self.cfg, args)
        self.assertEqual(fp.argument(out, "--source"), str(output))
        self.assertEqual(fp.argument(out, "--prepared-fd"), "3")
        self.assertEqual(len(out), len(args))
        self.assertEqual(binding["sha256"], fp.sha256(output))

    def test_started_original_session_is_kept(self):
        self.complete_state()
        work = Path(self.tmp.name) / "work-started"
        fp.write_json(work / "abc" / "state.json", {"source": "/some/ready/video.mp4"})
        args = self.worker_args("530_Hot", "530", 1, work=work)
        _, out, binding = fp.decide(self.cfg, args)
        self.assertEqual(out, args)
        self.assertIsNone(binding)

    def test_no_watermark_and_released_submit_original(self):
        for status in (fp.NO_WATERMARK, fp.RELEASED):
            fp.write_json(self.cfg.state_path("lizhi", "530", 2), {"status": status})
            args = self.worker_args("530_Hot", "530", 2)
            self.assertEqual(fp.decide(self.cfg, args)[1], args)

    def test_disabled_source_passes_through(self):
        self.cfg.enabled = set()
        args = self.worker_args("530_Hot", "530", 1)
        self.assertEqual(fp.decide(self.cfg, args)[1], args)

    def test_guard_defers_with_75_and_execs_real_worker(self):
        real = Path(self.tmp.name) / "real-worker"
        real.write_text("#!/bin/sh\nexit 0\n")
        real.chmod(0o755)
        with patch.dict(os.environ, {"FRAMEMEND_REAL_WORKER": str(real)}):
            self.assertEqual(fp.guard(self.cfg, self.worker_args("530_Hot", "530", 1)), fp.DEFER)
            output = self.complete_state()
            with patch.object(fp.os, "execve") as execve:
                fp.guard(self.cfg, self.worker_args("530_Hot", "530", 1))
            self.assertEqual(fp.argument(execve.call_args[0][1], "--source"), str(output))
            self.assertTrue(self.cfg.binding_path("lizhi", "530", 1).is_file())

    # ------------------------------------------------------------------ queue

    def test_normalize_skips_conforming_input_and_rewrites_vfr(self):
        with patch.object(fp, "needs_normalizing", return_value=(False, "30")):
            video = Path(self.tmp.name) / "a.mp4"
            self.assertEqual(fp.normalize(self.cfg, video, Path(self.tmp.name) / "job"), video)
        with patch.object(fp, "needs_normalizing", return_value=(True, "30/1")), \
                patch.object(fp.subprocess, "run") as run:
            out = fp.normalize(self.cfg, video, Path(self.tmp.name) / "job")
        command = run.call_args[0][0]
        self.assertEqual(command[command.index("-r") + 1], "30/1")
        self.assertEqual(out, Path(self.tmp.name) / "job-input.mp4")

    def test_queue_processes_hottest_episode_first_and_keeps_output(self):
        with patch.object(fp, "output_matches", return_value=None):
            self.assertTrue(fp.queue_once(self.cfg))
        state = fp.read_json(self.cfg.state_path("lizhi", "530", 1))
        self.assertEqual(state["status"], fp.COMPLETE)
        self.assertTrue(Path(state["output"]).read_bytes().startswith(b"clean-"))
        job = self.cfg.job_dir("lizhi", "530", 1) / "attempt-01"
        self.assertEqual(sorted(p.name for p in job.iterdir()), ["auto.json"])
        self.assertFalse(fp.read_json(self.cfg.state_path("lizhi", "531", 1)))

    def test_failure_retries_once_then_waits_for_review(self):
        (self.app / "mode").write_text("needs_review")
        fp.queue_once(self.cfg)
        self.assertEqual(fp.read_json(self.cfg.state_path("lizhi", "530", 1))["status"], fp.PENDING)
        fp.queue_once(self.cfg)
        state = fp.read_json(self.cfg.state_path("lizhi", "530", 1))
        self.assertEqual(state["status"], fp.REVIEW)
        self.assertEqual(len(state["attempts"]), 2)
        self.assertEqual(fp.status(self.cfg)["review"], ["lizhi:530:1"])

    def test_release_lets_held_episodes_go_as_original(self):
        self.assertEqual(fp.release(self.cfg, "lizhi", "530", None), 2)
        args = self.worker_args("530_Hot", "530", 2)
        self.assertEqual(fp.decide(self.cfg, args)[1], args)

    def test_paused_queue_does_nothing(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pause").write_text("x")
        self.assertFalse(fp.queue_once(self.cfg))

    # ------------------------------------------------------------------ marking

    def test_mark_posts_once_after_publication(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, self.headers["Authorization"], body))
                code = 422 if "/99/" in self.path else 200
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"dewatermarked": code == 200}).encode())

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.cfg.api_url = f"http://127.0.0.1:{server.server_port}/api"

        output = self.complete_state()
        args = self.worker_args("530_Hot", "530", 1)
        _, out, binding = fp.decide(self.cfg, args)
        fp.write_json(self.cfg.binding_path("lizhi", "530", 1), binding)
        self.assertEqual(fp.mark_pending(self.cfg)["waiting"], 1)  # not published yet
        digest = fp.sha256(output)
        fp.write_json(Path(binding["work_dir"]) / "h" / "state.json", {
            "source": str(output),
            "receipt": {"id": 41, "videoId": 7, "status": "published", "recipe": {"sourceSha256": digest}}})
        self.assertEqual(fp.mark_pending(self.cfg)["marked"], 1)
        self.assertEqual(fp.mark_pending(self.cfg)["marked"], 0)  # idempotent locally
        self.assertEqual(calls, [("/api/v1/produced-submissions/41/dewatermark", "ApiKey secret-key",
                                  {"tool": "framemend@abc1234", "sourceSha256": digest})])
        saved = fp.read_json(self.cfg.binding_path("lizhi", "530", 1))
        self.assertEqual((saved["submission_id"], saved["video_id"]), (41, 7))

        # A rejected mark is recorded as final and not retried forever.
        second = self.complete_state(number=2)
        _, _, binding2 = fp.decide(self.cfg, self.worker_args("530_Hot", "530", 2))
        fp.write_json(self.cfg.binding_path("lizhi", "530", 2), binding2)
        fp.write_json(Path(binding2["work_dir"]) / "h" / "state.json", {
            "source": str(second), "receipt": {"id": 99, "videoId": 8, "status": "published"}})
        self.assertEqual(fp.mark_pending(self.cfg)["failed"], 1)
        self.assertIn("422", fp.read_json(self.cfg.binding_path("lizhi", "530", 2))["mark_error_final"])
        self.assertEqual(fp.mark_pending(self.cfg)["failed"], 0)

    def test_cleanup_drops_outputs_after_keep_window(self):
        output = self.complete_state()
        fp.write_json(self.cfg.binding_path("lizhi", "530", 1),
                      {"output": str(output), "marked": fp.now() - 3 * 86400})
        self.assertEqual(fp.cleanup(self.cfg), 1)
        self.assertFalse(output.exists())

    def test_ready_view_maps_to_download(self):
        view = self.ready / "ai-duanju" / "530_Hot" / "episodes" / "01" / "video.mp4"
        self.assertEqual(fp.original_for(self.cfg, "lizhi", view),
                         self.downloads / "ai-duanju" / "530_Hot" / "episodes" / "01" / "video.mp4")


if __name__ == "__main__":
    unittest.main()
