"""Per-source resource limits and reusable analysis shared by all routing strategies."""

from contextlib import contextmanager
from collections import OrderedDict
from contextvars import ContextVar
from time import monotonic
from threading import RLock


class WorkflowLimit(RuntimeError):
    pass


class SourceRuntime:
    def __init__(self, config):
        self.started = monotonic()
        self.deadline = self.started + config.get('source_budget_seconds', 1800)
        self.planning_remaining = min(float(config.get('planning_budget_seconds', 600)),
                                      config.get('source_budget_seconds', 1800)/2)
        self.coverage_budget = config.get('coverage_budget_seconds', 180)
        self.coverage_deadline = None
        self.phase = 'planning'
        self.checked = self.started
        self.locations = {}
        self.pixel_passes = {}
        self.gaps = {}
        self.stalled = False
        self.overlap_frames = set()
        self.frames = OrderedDict()
        self.frame_lock = RLock()
        self.frame_bytes = 0
        self.frame_cache_hits = 0
        self.save_review_checkpoint = None
        self.review_recipe = None
        self.review_checkpoints = 0

    def remaining(self):
        now = monotonic()
        if self.phase == 'planning':
            self.planning_remaining -= now-self.checked
        self.checked = now
        left = self.deadline-now
        if left <= 0:
            raise WorkflowLimit('The video reached its total processing time budget.')
        if self.phase == 'planning':
            if self.planning_remaining <= 0:
                raise WorkflowLimit('Watermark planning reached its time budget without a usable recipe.')
            left = min(left, self.planning_remaining)
            if self.coverage_deadline is not None:
                if self.coverage_deadline <= now:
                    raise WorkflowLimit('Coverage review reached its time budget; confirmed placements were retained for review.')
                left = min(left, self.coverage_deadline-now)
        return left

    def start_coverage_review(self):
        if self.coverage_deadline is None:
            self.coverage_deadline = monotonic()+self.coverage_budget

    def gap_progress(self, key, frames):
        frames = frozenset(map(int, frames))
        prior = self.gaps.get(key)
        self.gaps[key] = frames
        # New candidate boxes alone are not progress on a previously seen gap.
        if prior is not None and not frames < prior:
            self.stalled = True
            return False
        return True


_runtime = ContextVar('watermark_source_runtime', default=None)


def current_runtime():
    return _runtime.get()


def check_budget():
    state = _runtime.get()
    return state.remaining() if state is not None else None


def bounded_timeout(seconds):
    remaining = check_budget()
    return seconds if remaining is None else min(seconds, remaining)


@contextmanager
def processing_slot(gate):
    if gate is None:
        yield
        return
    if not gate.acquire(timeout=bounded_timeout(86400)):
        raise WorkflowLimit('The video exhausted its time budget waiting for a processing slot.')
    try:
        check_budget()
        yield
    finally:
        gate.release()


@contextmanager
def source_runtime(config):
    if _runtime.get() is not None:
        yield _runtime.get()
        return
    state = SourceRuntime(config)
    token = _runtime.set(state)
    try:
        yield state
    finally:
        _runtime.reset(token)


@contextmanager
def processing_phase(*, allow_planning_exhausted=False):
    state = _runtime.get()
    if state is None:
        yield
        return
    if not allow_planning_exhausted:
        state.remaining()
    else:
        now = monotonic()
        if state.phase == 'planning':
            state.planning_remaining -= now-state.checked
        state.checked = now
    previous = state.phase
    state.phase = 'processing'
    try:
        state.remaining()
        yield
    finally:
        # Charge elapsed processing only against the total budget.
        state.checked = monotonic()
        state.phase = previous
