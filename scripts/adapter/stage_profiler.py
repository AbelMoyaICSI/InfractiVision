"""
stage_profiler.py — Granular per-stage timing for the CLI pipeline.

Why a custom profiler:
  The original CLI scripts only measured `t_model_elapsed` and a global
  `t_proc_elapsed`. That is enough to know "how fast was the run?" but
  not enough to know "where did the time go?" — and on a 4-core CPU
  without a discrete GPU, the answer is often surprising (preprocess,
  NMS, decode can each be a non-trivial slice).

This profiler:
  - Uses time.perf_counter() for monotonic, sub-microsecond resolution.
  - Distinguishes one-shot stages (model_load, model_warmup) from
    recurring ones (decode, inference, tracker) in the report.
  - Returns a string from report() (caller decides where to log) and
    a plain dict from as_dict() (caller can serialize to JSON).

The profiler is NOT thread-safe. It assumes the pipeline is single-
threaded (which it is today: the only background thread, once we add
the FrameReader, only reads frames and never touches the profiler).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class StageProfiler:
    """Accumulates wall-clock time per named stage."""

    def __init__(self) -> None:
        self._totals_ms: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        # Track which stages have been called more than once — these
        # are "recurring" and their per-call avg is meaningful.
        self._max_count_seen: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that times the wrapped block into `name`."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._totals_ms[name] += elapsed_ms
            self._counts[name] += 1
            if self._counts[name] > self._max_count_seen[name]:
                self._max_count_seen[name] = self._counts[name]

    # ── read-back ────────────────────────────────────────────────────

    @property
    def total_ms(self) -> float:
        return sum(self._totals_ms.values())

    def stages(self) -> list[str]:
        """Return stage names in insertion order (preserved by dict)."""
        return list(self._totals_ms.keys())

    def as_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the profiler state."""
        total = self.total_ms
        out: dict = {
            "total_ms": round(total, 3),
            "stages": [],
        }
        for name, ms in self._totals_ms.items():
            n = self._counts[name]
            out["stages"].append({
                "name": name,
                "total_ms": round(ms, 3),
                "count": n,
                "avg_ms": round(ms / n, 3) if n else 0.0,
                "pct": round(100.0 * ms / total, 2) if total else 0.0,
                "recurring": n > 1,
            })
        # Sort stages by total time descending — easier to read in JSON.
        out["stages"].sort(key=lambda s: -s["total_ms"])
        return out

    def report(self) -> str:
        """Return a human-readable table. Caller decides where to print it."""
        total = self.total_ms
        if total == 0:
            return "(StageProfiler: no stages recorded yet)"

        # Sort stages by total time desc, but keep model_load and
        # model_warmup on top in their original order (one-shot stages
        # get a special header so the percentages are honest about it).
        rows = sorted(self._totals_ms.items(), key=lambda kv: -kv[1])

        lines = [
            f"{'stage':<22s} {'total_ms':>10s}  {'pct':>6s}  "
            f"{'avg_ms':>8s}  {'n':>4s}  type",
            "-" * 70,
        ]
        for name, ms in rows:
            n = self._counts[name]
            pct = 100.0 * ms / total
            avg = ms / n if n else 0.0
            kind = "recurring" if n > 1 else "one-shot"
            lines.append(
                f"{name:<22s} {ms:>10.1f}  {pct:>5.1f}%  "
                f"{avg:>8.2f}  {n:>4d}  {kind}"
            )
        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<22s} {total:>10.1f}  100.0%")
        return "\n".join(lines)
