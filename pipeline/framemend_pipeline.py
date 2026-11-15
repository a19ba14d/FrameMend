#!/usr/bin/env python3
"""FrameMend production hookup for the 02 crawler / produced-submit pipeline.

dupeguard writes ``.dewatermark`` into the directory of every hot drama that has not been submitted yet.
This module:

* ``queue``  - loops over those dramas, runs ``wm_auto.py`` on each episode (hottest first) and keeps the
  clean output under ``<root>/out``;
* ``guard``  - is installed as ``PRODUCED_WORKER``; it holds hot episodes (exit 75, produced-submit retries
  the same arguments next round) until FrameMend finished them, then hands the clean file to the real worker;
* ``mark``   - after the worker published a clean episode, tells MediaCMS the video was dewatermarked
  (``POST /v1/produced-submissions/{id}/dewatermark``);
* ``status`` / ``release`` / ``retry`` / ``pause`` / ``resume`` / ``cleanup`` for operators.

Nothing here edits crawler downloads or produced-submit state; per-episode state lives under ``<root>/state``.
Only the standard library is used so the guard starts fast inside produced-submit's loop.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CONFIG = "/srv/framemend/config.json"
MARKER = ".dewatermark"
DEFER = 75  # EX_TEMPFAIL: produced-submit logs it and retries with the same arguments next loop

# Episode states written by the queue.
PENDING, COMPLETE, NO_WATERMARK, REVIEW, RELEASED = "pending", "complete", "no_watermark", "review", "released"
FINAL = (COMPLETE, NO_WATERMARK, RELEASED)


# --------------------------------------------------------------------------------------------- helpers

def now() -> float:
    return time.time()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str, **fields: Any) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "msg": message, **fields}
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def safe_part(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return cleaned.strip(".") or "_"


class Config:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.root = Path(data.get("root", "/srv/framemend"))
        self.app = Path(data.get("app", str(self.root / "app")))
        self.python = data.get("python", sys.executable)
        # source -> crawler downloads root
        self.sources: dict[str, Path] = {k: Path(v) for k, v in (data.get("sources") or {}).items()}
        # sources whose hot dramas are processed and held; guard passes every other source through untouched
        self.enabled = set(data.get("enabled_sources") or [])
        self.min_size = int(data.get("min_size", 300_000))
        self.settle_seconds = int(data.get("settle_seconds", 600))
        self.timeout = int(data.get("episode_timeout", 4 * 3600))
        self.retry_dilation = int(data.get("retry_dilation", 8))
        self.api_url = str(data.get("api_url", "")).rstrip("/")
        self.api_key_file = Path(data.get("api_key_file", "/srv/produced/produced.key"))
        version = (self.app / "VERSION").read_text().strip() if (self.app / "VERSION").is_file() else ""
        self.tool = str(data.get("tool") or ("framemend@" + version if version else "framemend"))[:64]
        self.keep_days = float(data.get("keep_days", 2))
        self.ffprobe = str(data.get("ffprobe") or shutil.which("ffprobe") or "ffprobe")

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        location = Path(path or os.environ.get("FRAMEMEND_CONFIG") or DEFAULT_CONFIG)
        data = read_json(location)
        if not data:
            raise SystemExit(f"FrameMend pipeline config missing or invalid: {location}")
        return cls(data)

    def state_path(self, source: str, drama: str, episode: int) -> Path:
        return self.root / "state" / safe_part(source) / safe_part(drama) / f"{episode:04d}.json"

    def job_dir(self, source: str, drama: str, episode: int) -> Path:
        return self.root / "jobs" / safe_part(source) / safe_part(drama) / f"{episode:04d}"

    def output_path(self, source: str, drama: str, episode: int) -> Path:
        return self.root / "out" / safe_part(source) / safe_part(drama) / f"{episode:04d}.mp4"

    def binding_path(self, source: str, drama: str, episode: int) -> Path:
        return self.root / "bindings" / safe_part(source) / safe_part(drama) / f"{episode:04d}.json"

    @property
    def paused(self) -> bool:
        return (self.root / "pause").exists()


# --------------------------------------------------------------------------------------------- catalogue

def drama_id_of(drama_dir: Path) -> str:
    meta = read_json(drama_dir / "metadata.json")
    return str(meta.get("drama_id") or drama_dir.name.split("_", 1)[0])


def episode_number(episode_dir: Path) -> int | None:
    meta = read_json(episode_dir / "metadata.json")
    value = meta.get("episode_number")
    try:
        return int(value if value is not None else episode_dir.name)
    except (TypeError, ValueError):
        return None


def hot_dramas(cfg: Config, source: str) -> Iterator[tuple[Path, str, float]]:
    """Yield (drama_dir, drama_id, priority) for every marked drama of a source."""
    root = cfg.sources.get(source)
    if not root or not root.is_dir():
        return
    for channel in root.iterdir():
        if not channel.is_dir() or channel.name.startswith("."):
            continue
        for drama in channel.iterdir():
            marker = drama / MARKER
            if not marker.is_file():
                continue
            info = read_json(marker)
            try:
                value, threshold = float(info.get("value")), float(info.get("threshold"))
                priority = value / threshold if threshold > 0 else 1.0
            except (TypeError, ValueError):
                priority = 1.0
            yield drama, drama_id_of(drama), priority


def episodes(cfg: Config, drama: Path) -> Iterator[tuple[int, Path]]:
    folder = drama / "episodes"
    if not folder.is_dir():
        return
    for entry in sorted(folder.iterdir()):
        video = entry / "video.mp4"
        number = episode_number(entry)
        if number is None or (entry / ".released").exists():
            continue
        try:
            stat = video.stat()
        except OSError:
            continue
        if stat.st_size < cfg.min_size or now() - stat.st_mtime < cfg.settle_seconds:
            continue
        yield number, video


def candidates(cfg: Config) -> list[tuple[float, str, str, int, Path]]:
    """Hot episodes still waiting for FrameMend, hottest drama first, episodes in order."""
    found = []
    for source in sorted(cfg.enabled):
        for drama, drama_id, priority in hot_dramas(cfg, source):
            for number, video in episodes(cfg, drama):
                state = read_json(cfg.state_path(source, drama_id, number))
                if state.get("status") in FINAL or state.get("status") == REVIEW:
                    continue
                found.append((priority, source, drama_id, number, video))
    found.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return found


# --------------------------------------------------------------------------------------------- queue

def probe(path: Path, tool: str = "ffprobe") -> dict[str, Any]:
    try:
        out = subprocess.run(
            [tool, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    kinds = [s.get("codec_type") for s in data.get("streams", [])]
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration": duration, "video": kinds.count("video"), "audio": kinds.count("audio")}


def output_matches(original: Path, result: Path, tool: str = "ffprobe") -> str | None:
    """Return a reason when the clean file cannot stand in for the original."""
    a, b = probe(original, tool), probe(result, tool)
    if not b or not b.get("video"):
        return "成品无法解析或没有画面"
    if a.get("audio") and not b.get("audio"):
        return "成品丢了音轨"
    if a.get("duration") and abs(a["duration"] - b["duration"]) > max(0.5, a["duration"] * 0.01):
        return f"成品时长不一致 {a['duration']:.2f}s → {b['duration']:.2f}s"
    return None


def run_framemend(cfg: Config, video: Path, job: Path, dilation: int | None) -> dict[str, Any]:
    command = [cfg.python, str(cfg.app / "wm_auto.py"), str(video), str(job)]
    if dilation:
        command += ["--dilation", str(dilation)]
    job.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    try:
        with (job.parent / f"{job.name}.log").open("ab") as handle:
            completed = subprocess.run(command, cwd=cfg.app, stdout=handle, stderr=subprocess.STDOUT,
                                       timeout=cfg.timeout)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = -1
    report = read_json(job / "auto.json")
    report.setdefault("status", "failed")
    report["exit_code"] = code
    report["wall_seconds"] = round(now() - started, 1)
    return report


def trim_job(job: Path) -> None:
    """Keep the small reports, drop the large intermediates once the result has been copied out."""
    if not job.is_dir():
        return
    for entry in job.iterdir():
        if entry.name in ("auto.json", "events.jsonl", "batch.json"):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def process_episode(cfg: Config, source: str, drama: str, number: int, video: Path) -> dict[str, Any]:
    state_path = cfg.state_path(source, drama, number)
    state = read_json(state_path) or {"source": source, "drama": drama, "episode": number, "attempts": []}
    attempt = len(state.get("attempts", [])) + 1
    job = cfg.job_dir(source, drama, number) / f"attempt-{attempt:02d}"
    if job.exists():
        shutil.rmtree(job)
    state.update(status=PENDING, original=str(video), running=attempt, updated=now())
    write_json(state_path, state)
    report = run_framemend(cfg, video, job, cfg.retry_dilation if attempt > 1 else None)
    status = report["status"]
    summary = {"attempt": attempt, "status": status, "reason": report.get("reason") or report.get("error"),
               "seconds": report.get("wall_seconds"), "exit_code": report.get("exit_code")}
    state.pop("running", None)
    if status == "complete":
        result = Path(report.get("result") or job / "result.mp4").resolve()
        problem = output_matches(video, result, cfg.ffprobe) if result.is_file() else "找不到成品文件"
        if problem:
            status = summary["status"] = "failed"
            summary["reason"] = problem
        else:
            output = cfg.output_path(source, drama, number)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                os.link(result, temporary)
            except OSError:
                shutil.copy2(result, temporary)
            os.replace(temporary, output)
            state.update(status=COMPLETE, output=str(output), sha256=sha256(output),
                         size=output.stat().st_size, tool=cfg.tool, completed=now())
            trim_job(job)
    if status == "no_watermark":
        state.update(status=NO_WATERMARK)
        trim_job(job)
    elif status != "complete":
        # One retry with a wider mask, then wait for a person (the episode stays held).
        state.update(status=PENDING if attempt < 2 else REVIEW)
    state.setdefault("attempts", []).append(summary)
    state["updated"] = now()
    write_json(state_path, state)
    log("episode processed", source=source, drama=drama, episode=number, **summary, state=state["status"])
    return state


def queue_once(cfg: Config) -> bool:
    """Process at most one episode. Returns True when something was processed."""
    if cfg.paused:
        return False
    for _priority, source, drama, number, video in candidates(cfg):
        with locked(cfg.root / "locks" / f"{safe_part(source)}-{safe_part(drama)}-{number:04d}.lock") as ok:
            if not ok:
                continue
            state = read_json(cfg.state_path(source, drama, number))
            if state.get("status") in FINAL or state.get("status") == REVIEW:
                continue
            process_episode(cfg, source, drama, number, video)
            return True
    return False


class locked:
    def __init__(self, path: Path, blocking: bool = False):
        self.path, self.blocking, self.handle = path, blocking, None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
        except OSError:
            return False
        return True

    def __exit__(self, *_exc: Any) -> None:
        if self.handle:
            self.handle.close()


def queue_loop(cfg_path: str | None, once: bool) -> int:
    with locked(Config.load(cfg_path).root / "queue.lock") as ok:
        if not ok:
            log("another queue is running")
            return 0
        last_mark = 0.0
        while True:
            cfg = Config.load(cfg_path)  # pick up enabled_sources / pause edits without a restart
            if now() - last_mark > 300:
                try:
                    mark_pending(cfg)
                    cleanup(cfg)
                except Exception as error:  # marking must never stop the GPU queue
                    log("mark/cleanup failed", error=str(error))
                last_mark = now()
            worked = queue_once(cfg)
            if once:
                return 0
            if not worked:
                time.sleep(60)


# --------------------------------------------------------------------------------------------- guard

def argument(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def replace_argument(args: list[str], name: str, new: str) -> list[str]:
    out = list(args)
    for index, value in enumerate(out):
        if value == name and index + 1 < len(out):
            out[index + 1] = new
            return out
        if value.startswith(name + "="):
            out[index] = f"{name}={new}"
            return out
    raise ValueError(f"missing {name}")


def original_for(cfg: Config, source: str, view: Path) -> Path:
    """Map a produced-submit ready-view path back to the crawler download it was linked from."""
    root = cfg.sources[source]
    if view.is_relative_to(root):
        return view
    parts = view.parts
    if "ready-downloads" in parts:
        index = len(parts) - 1 - parts[::-1].index("ready-downloads")
        return root.joinpath(*parts[index + 1:])
    return view


def worker_states(work_dir: Path) -> list[dict[str, Any]]:
    paths = [work_dir / "state.json", *sorted(work_dir.glob("*/state.json"))] if work_dir.is_dir() else []
    return [state for state in (read_json(p) for p in paths) if state]


def decide(cfg: Config, args: list[str]) -> tuple[str, list[str] | None, dict[str, Any] | None]:
    """Return (reason, worker args or None to hold, binding to record or None)."""
    source_arg, metadata_arg, work_arg = (argument(args, n) for n in ("--source", "--metadata", "--work-dir"))
    if not (source_arg and metadata_arg and work_arg):
        return "参数不完整，原样投稿", args, None
    identity = str(read_json(Path(metadata_arg)).get("sourceIdentity") or "")
    parts = identity.split(":")
    if len(parts) != 3 or parts[0] not in cfg.enabled or parts[0] not in cfg.sources:
        return "来源未启用去水印", args, None
    source, drama, episode_text = parts
    try:
        number = int(episode_text)
    except ValueError:
        return "集号无效，原样投稿", args, None
    original = original_for(cfg, source, Path(source_arg))
    drama_dir = original.parent.parent.parent
    if not (drama_dir / MARKER).is_file():
        return "不是高热剧", args, None
    state = read_json(cfg.state_path(source, drama, number))
    output = state.get("output")
    # A worker session that already started keeps its source: the worker refuses a different file.
    for session in worker_states(Path(work_arg)):
        started = str(session.get("source") or "")
        if output and started == output:
            break
        if started:
            return "投稿会话已用原片开始，保持原片", args, None
    status = state.get("status")
    if status == COMPLETE and output and Path(output).is_file():
        binding = {"source": source, "drama": drama, "episode": number, "work_dir": work_arg,
                   "output": output, "sha256": state.get("sha256"), "tool": state.get("tool") or cfg.tool,
                   "identity": identity, "bound": now()}
        return "使用去水印成品", replace_argument(args, "--source", output), binding
    if status in (NO_WATERMARK, RELEASED):
        return ("未发现水印，投原片" if status == NO_WATERMARK else "人工放行，投原片"), args, None
    if status == REVIEW:
        return "去水印待人工确认", None, None
    return "等待去水印", None, None


def guard(cfg: Config, args: list[str]) -> int:
    real = os.environ.get("FRAMEMEND_REAL_WORKER", "")
    if not real or not os.access(real, os.X_OK) or Path(real).name.startswith("framemend"):
        print("FrameMend guard: FRAMEMEND_REAL_WORKER 未设置或不可执行", file=sys.stderr)
        return 78
    try:
        reason, worker_args, binding = decide(cfg, args)
    except Exception as error:  # never let a guard bug publish a hot episode with its watermark
        print(f"FrameMend guard 出错，暂缓：{error}", file=sys.stderr)
        return DEFER
    if worker_args is None:
        print(f"FrameMend：{reason}", file=sys.stderr)
        return DEFER
    if binding:
        path = cfg.binding_path(binding["source"], binding["drama"], binding["episode"])
        previous = read_json(path)
        if previous.get("marked"):
            binding.update(marked=previous["marked"], submission_id=previous.get("submission_id"))
        write_json(path, {**previous, **binding})
    os.execve(real, [real, *worker_args], os.environ.copy())
    return 0


# --------------------------------------------------------------------------------------------- marking

def api_key(cfg: Config) -> str:
    return cfg.api_key_file.read_text(encoding="utf-8").strip()


def post_mark(cfg: Config, submission_id: int, tool: str, digest: str) -> tuple[int, dict[str, Any]]:
    body = json.dumps({"tool": tool, "sourceSha256": digest}).encode()
    request = urllib.request.Request(
        f"{cfg.api_url}/v1/produced-submissions/{submission_id}/dewatermark", data=body, method="POST",
        headers={"Authorization": "ApiKey " + api_key(cfg), "Content-Type": "application/json",
                 "User-Agent": "framemend-pipeline/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read(1 << 20) or b"{}")
    except urllib.error.HTTPError as error:
        with error:
            try:
                payload = json.loads(error.read(1 << 16) or b"{}")
            except ValueError:
                payload = {}
        return error.code, payload if isinstance(payload, dict) else {}


def published_receipt(binding: dict[str, Any]) -> dict[str, Any] | None:
    for session in worker_states(Path(binding.get("work_dir", ""))):
        if session.get("source") != binding.get("output"):
            continue
        receipt = session.get("receipt") or {}
        if receipt.get("status") == "published" and int(receipt.get("id") or 0) > 0:
            return receipt
    return None


def bindings(cfg: Config) -> Iterator[tuple[Path, dict[str, Any]]]:
    root = cfg.root / "bindings"
    if root.is_dir():
        for path in sorted(root.glob("*/*/*.json")):
            data = read_json(path)
            if data:
                yield path, data


def mark_pending(cfg: Config) -> dict[str, int]:
    counts = {"marked": 0, "waiting": 0, "failed": 0}
    if not cfg.api_url:
        return counts
    for path, binding in bindings(cfg):
        if binding.get("marked") or binding.get("mark_error_final"):
            continue
        receipt = published_receipt(binding)
        if not receipt:
            counts["waiting"] += 1
            continue
        submission = int(receipt["id"])
        recipe_sha = str((receipt.get("recipe") or {}).get("sourceSha256") or "")
        if binding.get("sha256") and recipe_sha and recipe_sha != binding["sha256"]:
            binding.update(mark_error_final="回执里的源摘要和成品不一致", submission_id=submission)
            write_json(path, binding)
            counts["failed"] += 1
            continue
        try:
            code, payload = post_mark(cfg, submission, binding.get("tool") or cfg.tool,
                                      binding.get("sha256") or recipe_sha)
        except (OSError, ValueError) as error:
            binding.update(mark_error=str(error), mark_tried=now())
            write_json(path, binding)
            counts["failed"] += 1
            continue
        if code == 200:
            binding.update(marked=now(), submission_id=submission, video_id=receipt.get("videoId"))
            binding.pop("mark_error", None)
            counts["marked"] += 1
        else:
            binding.update(mark_error=f"HTTP {code} {payload.get('detail') or payload.get('title') or ''}".strip(),
                           mark_tried=now(), submission_id=submission)
            if code in (400, 403, 404, 422):
                binding["mark_error_final"] = binding["mark_error"]
            counts["failed"] += 1
        write_json(path, binding)
        log("mark", source=binding.get("source"), drama=binding.get("drama"), episode=binding.get("episode"),
            submission=submission, http=code)
    return counts


# --------------------------------------------------------------------------------------------- operations

def all_states(cfg: Config) -> Iterator[tuple[Path, dict[str, Any]]]:
    root = cfg.root / "state"
    if root.is_dir():
        for path in sorted(root.glob("*/*/*.json")):
            data = read_json(path)
            if data:
                yield path, data


def cleanup(cfg: Config) -> int:
    """Drop clean outputs a while after MediaCMS marked them; the worker never needs them again."""
    removed = 0
    for _path, binding in bindings(cfg):
        marked = binding.get("marked")
        output = Path(binding.get("output") or "")
        if marked and now() - float(marked) > cfg.keep_days * 86400 and output.is_file():
            output.unlink()
            removed += 1
    return removed


def status(cfg: Config) -> dict[str, Any]:
    per: dict[str, dict[str, int]] = {}
    seconds: list[float] = []
    review = []
    for _path, state in all_states(cfg):
        bucket = per.setdefault(state.get("source", "?"), {})
        key = state.get("status", "?")
        bucket[key] = bucket.get(key, 0) + 1
        for attempt in state.get("attempts", []):
            if attempt.get("status") == "complete" and attempt.get("seconds"):
                seconds.append(float(attempt["seconds"]))
        if key == REVIEW:
            review.append(f"{state.get('source')}:{state.get('drama')}:{state.get('episode')}")
    waiting = candidates(cfg)
    average = sum(seconds) / len(seconds) if seconds else None
    marks = {"marked": 0, "waiting": 0, "error": 0}
    for _path, binding in bindings(cfg):
        if binding.get("marked"):
            marks["marked"] += 1
        elif binding.get("mark_error_final") or binding.get("mark_error"):
            marks["error"] += 1
        else:
            marks["waiting"] += 1
    return {
        "paused": cfg.paused,
        "enabled_sources": sorted(cfg.enabled),
        "states": per,
        "queued": len(waiting),
        "queued_by_source": {s: sum(1 for c in waiting if c[1] == s) for s in sorted(cfg.enabled)},
        "average_seconds": round(average, 1) if average else None,
        "eta_hours": round(len(waiting) * average / 3600, 1) if average else None,
        "review": review[:50],
        "marks": marks,
    }


def release(cfg: Config, source: str, drama: str, episode: int | None) -> int:
    """Let held episodes go out as originals (operator decision), including ones never processed."""
    changed = 0
    for drama_dir, drama_id, _priority in hot_dramas(cfg, source):
        if drama_id != drama:
            continue
        folder = drama_dir / "episodes"
        for entry in sorted(folder.iterdir()) if folder.is_dir() else []:
            number = episode_number(entry)
            if number is None or (episode is not None and number != episode):
                continue
            path = cfg.state_path(source, drama, number)
            state = read_json(path) or {"source": source, "drama": drama, "episode": number, "attempts": []}
            if state.get("status") in (COMPLETE, NO_WATERMARK, RELEASED):
                continue
            state.update(status=RELEASED, updated=now(), operator_change=RELEASED)
            write_json(path, state)
            changed += 1
    return changed


def set_state(cfg: Config, source: str, drama: str, episode: int | None, status_value: str,
              only: tuple[str, ...]) -> int:
    changed = 0
    for path, state in all_states(cfg):
        if state.get("source") != source or str(state.get("drama")) != drama:
            continue
        if episode is not None and int(state.get("episode", -1)) != episode:
            continue
        if state.get("status") not in only:
            continue
        state.update(status=status_value, updated=now(), operator_change=status_value)
        if status_value == PENDING:
            state["attempts"] = []
        write_json(path, state)
        changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["guard"]:  # worker arguments are opaque: do not let argparse touch them
        return guard(Config.load(), raw[1:])
    parser = argparse.ArgumentParser(description="FrameMend 去水印接入 02 投稿链路")
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)
    queue = sub.add_parser("queue", help="常驻：按热度逐集去水印")
    queue.add_argument("--once", action="store_true")
    sub.add_parser("mark", help="给已发布的去水印成品在中台打标记")
    sub.add_parser("status")
    sub.add_parser("cleanup")
    sub.add_parser("pause")
    sub.add_parser("resume")
    for name in ("release", "retry"):
        command = sub.add_parser(name, help="release：待人工的集改投原片；retry：待人工的集重新排队")
        command.add_argument("source")
        command.add_argument("drama")
        command.add_argument("episode", nargs="?", type=int)
    args = parser.parse_args(raw)
    if args.command == "queue":
        return queue_loop(args.config, args.once)
    cfg = Config.load(args.config)
    if args.command == "mark":
        result: Any = mark_pending(cfg)
    elif args.command == "status":
        result = status(cfg)
    elif args.command == "cleanup":
        result = {"removed": cleanup(cfg)}
    elif args.command == "pause":
        (cfg.root / "pause").write_text(time.strftime("%F %T") + "\n")
        result = {"paused": True}
    elif args.command == "resume":
        (cfg.root / "pause").unlink(missing_ok=True)
        result = {"paused": False}
    elif args.command == "release":
        result = {"released": release(cfg, args.source, args.drama, args.episode)}
    else:
        result = {"requeued": set_state(cfg, args.source, args.drama, args.episode, PENDING, (REVIEW,))}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
