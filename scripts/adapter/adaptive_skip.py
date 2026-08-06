"""
adaptive_skip.py — Time-budget aware skip-rate controller.

Why this exists:
  The original CLI scripts used fixed skip rates (green=10, red+active=1,
  red alone=3, yellow=3). That worked on a GPU because the GPU did the
  heavy lifting and the CPU mostly waited. On a 4-core / 8-thread CPU
  (i5-10300H + UHD 630) the same rules are dangerous: forcing skip=1 in
  red+active can be *unpayable* if the model takes > 33 ms / frame, and
  the pipeline falls progressively behind the video (the "spiral of
  latency" we want to avoid).

  This controller measures the actual cost of the last N inference
  batches and decides a skip rate that keeps the pipeline within the
  time budget of one video frame.

Usage:
    ctrl = AdaptiveSkipController(target_fps_video=30.0, window=10)

    # ... after each _process_batch:
    ctrl.record(batch_elapsed_ms, batch_size)

    # ... before deciding whether to process a frame:
    skip = ctrl.suggest_skip(semaphore_state, tracker.active_count)
    if frame_index % skip != 0:
        continue
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class AdaptiveSkipController:
    """Decides how many frames to skip based on measured inference cost.

    The controller keeps a small ring of per-frame inference times and
    computes a `ratio` = actual cost / frame budget. The skip rate is
    then derived from the ratio, modulated by the current traffic-light
    state and the number of infractors currently being tracked.

    Semaphore state semantics:
      "green"  — idle, ok to skip aggressively
      "yellow" — pre-alert, keep some resolution
      "red"    — must catch the runner; only skip when the budget forces
                 it, and never by more than ratio (we always want to
                 keep up with the video, never to fall behind).
    """

    MIN_SKIP = 1
    MAX_SKIP = 12

    # Slightly more conservative in red+active: never skip more than 3
    # even if the system is pathologically slow. Better to lose some
    # frames than to miss the runner.
    RED_ACTIVE_CAP = 3

    # In green we want to skip a lot to save CPU for when it matters.
    GREEN_MIN = 8

    # In yellow / red-alone we want to keep some resolution.
    IDLE_MIN = 2
    IDLE_MAX = 5

    def __init__(self, target_fps_video: float = 30.0, window: int = 10) -> None:
        if target_fps_video <= 0:
            raise ValueError(f"target_fps_video must be > 0, got {target_fps_video}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")

        self.target_fps_video = target_fps_video
        self.frame_budget_ms = 1000.0 / target_fps_video
        self._recent_per_frame_ms: Deque[float] = deque(maxlen=window)
        # Stats for introspection / logging.
        self._last_avg_ms: float = self.frame_budget_ms
        self._last_ratio: int = 1

    # ── configuration ───────────────────────────────────────────────

    def set_target_fps(self, fps: float) -> None:
        """Update the frame budget when the actual video FPS is known."""
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.target_fps_video = fps
        self.frame_budget_ms = 1000.0 / fps

    # ── measurement ─────────────────────────────────────────────────

    def record(self, batch_time_ms: float, batch_size: int) -> None:
        """Record the wall time of a single batch.

        The controller divides by `batch_size` to get a per-frame
        average, which is the meaningful unit for the time budget.
        A batch of 0 frames is ignored.
        """
        if batch_size <= 0 or batch_time_ms < 0:
            return
        per_frame_ms = batch_time_ms / batch_size
        self._recent_per_frame_ms.append(per_frame_ms)

    def reset(self) -> None:
        """Clear the measurement window (e.g. after a long pause)."""
        self._recent_per_frame_ms.clear()
        self._last_avg_ms = self.frame_budget_ms
        self._last_ratio = 1

    # ── decision ────────────────────────────────────────────────────

    def _current_ratio(self) -> int:
        """How many video frames one inference actually costs, rounded.

        If we have no measurements yet, assume the system is keeping
        up (ratio = 1) so we start with the most sensitive config and
        only relax it once reality proves we need to.
        """
        if not self._recent_per_frame_ms:
            return 1
        avg = sum(self._recent_per_frame_ms) / len(self._recent_per_frame_ms)
        self._last_avg_ms = avg
        # round() gives the closest int; max(1, ...) guards against
        # the optimistic case where we measured slightly under budget.
        ratio = max(1, round(avg / self.frame_budget_ms))
        self._last_ratio = ratio
        return ratio

    def suggest_skip(self, semaphore_state: str, active_count: int) -> int:
        """Return the skip rate to apply for the next frame.

        Args:
            semaphore_state: "green" | "yellow" | "red".
            active_count: number of infractors currently being tracked.

        Returns:
            An int >= 1 representing "process 1 of every N frames".
        """
        ratio = self._current_ratio()

        if semaphore_state == "green":
            # Idle: skip aggressively. Always at least 8 so we don't
            # burn CPU when nothing useful is happening. The `* 2`
            # doubles the ratio's contribution so the skip grows
            # quickly when the system is under stress.
            return min(self.MAX_SKIP, max(self.GREEN_MIN, ratio * 2))

        if semaphore_state == "red" and active_count > 0:
            # Critical: we have a live infractor. If we can keep up
            # (ratio <= 1) we process every frame. If not, we cap the
            # skip at RED_ACTIVE_CAP=3 because losing more than 2/3 of
            # the red frames is too risky for detection.
            if ratio <= 1:
                return self.MIN_SKIP
            return min(self.RED_ACTIVE_CAP, ratio)

        # yellow OR red alone (no active infractor yet): keep moderate
        # resolution so we can pick up a runner entering the polygon.
        return min(self.IDLE_MAX, max(self.IDLE_MIN, ratio))

    # ── read-back for logging / profiler ────────────────────────────

    @property
    def frame_budget_ms(self) -> float:
        return self._frame_budget_ms

    @frame_budget_ms.setter
    def frame_budget_ms(self, value: float) -> None:
        self._frame_budget_ms = value

    @property
    def last_avg_ms(self) -> float:
        return self._last_avg_ms

    @property
    def last_ratio(self) -> int:
        return self._last_ratio

    @property
    def sample_count(self) -> int:
        return len(self._recent_per_frame_ms)

    def __repr__(self) -> str:
        return (
            f"AdaptiveSkipController(target={self.target_fps_video:.1f}fps, "
            f"budget={self.frame_budget_ms:.1f}ms, "
            f"avg={self._last_avg_ms:.1f}ms, "
            f"ratio={self._last_ratio}, "
            f"samples={self.sample_count})"
        )
