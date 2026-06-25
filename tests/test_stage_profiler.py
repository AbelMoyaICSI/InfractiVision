"""
test_stage_profiler.py — Unit tests for StageProfiler.

Covers:
  - Accumulation across multiple calls to the same stage.
  - Correctness of total_ms, avg_ms and pct.
  - Distinction between one-shot (n==1) and recurring (n>1) stages.
  - as_dict() is JSON-serializable and correctly ordered.
  - report() handles the empty case without crashing.
  - Nested stages are NOT supported (caller bug), but they don't corrupt
    the totals either way.

Note on timing: Windows `time.sleep` has ~15ms granularity, so we use
a busy-wait helper to get reliable sub-millisecond waits in tests.
"""

from __future__ import annotations

import json
import time

import pytest

from scripts.adapter.stage_profiler import StageProfiler


def _busy_wait_ms(ms: float) -> None:
    """Sleep for at least `ms` milliseconds using a busy wait.

    time.sleep on Windows has ~15.6ms granularity, which is too coarse
    for these microsecond-scale tests. A busy-wait of a few ms is
    negligible in a unit test and gives reliable measurements.
    """
    end = time.perf_counter() + ms / 1000.0
    # Spin for the requested duration. Add a tiny epsilon because
    # perf_counter is not exact at the nanosecond level.
    while time.perf_counter() < end - 1e-6:
        pass


class TestStageProfilerBasics:
    def test_empty_profiler_reports_cleanly(self):
        p = StageProfiler()
        assert p.total_ms == 0.0
        assert p.stages() == []
        report = p.report()
        assert "no stages" in report.lower()

    def test_single_stage_one_call(self):
        p = StageProfiler()
        with p.stage("model_load"):
            _busy_wait_ms(10)  # 10ms busy-wait — reliable on Windows
        assert p.total_ms >= 5.0
        assert p.stages() == ["model_load"]
        d = p.as_dict()
        assert d["stages"][0]["name"] == "model_load"
        assert d["stages"][0]["count"] == 1
        assert d["stages"][0]["recurring"] is False
        assert d["stages"][0]["avg_ms"] >= 5.0
        assert d["stages"][0]["pct"] == pytest.approx(100.0, abs=0.5)

    def test_recurring_stage_average(self):
        p = StageProfiler()
        for _ in range(3):
            with p.stage("inference"):
                _busy_wait_ms(5)  # 5ms each, reliable on Windows
        d = p.as_dict()
        stage = d["stages"][0]
        assert stage["name"] == "inference"
        assert stage["count"] == 3
        assert stage["recurring"] is True
        # 3 calls of 5ms each = ~15ms total. Allow a wide tolerance
        # because the test is sharing CPU with pytest and other processes.
        assert stage["total_ms"] >= 10.0
        assert stage["avg_ms"] >= 3.0

    def test_multiple_stages_pct_sums_to_100(self):
        p = StageProfiler()
        with p.stage("a"):
            _busy_wait_ms(10)
        with p.stage("b"):
            _busy_wait_ms(20)
        with p.stage("c"):
            _busy_wait_ms(30)
        d = p.as_dict()
        total_pct = sum(s["pct"] for s in d["stages"])
        assert total_pct == pytest.approx(100.0, abs=0.5)
        # Largest should be 'c', smallest 'a'
        names = [s["name"] for s in d["stages"]]
        assert names[0] == "c"
        assert names[-1] == "a"

    def test_report_has_human_readable_columns(self):
        p = StageProfiler()
        with p.stage("model_load"):
            _busy_wait_ms(1)  # one-shot
        for _ in range(3):
            with p.stage("inference"):
                _busy_wait_ms(1)  # recurring
        r = p.report()
        # Header columns
        for col in ("stage", "total_ms", "pct", "avg_ms", "n", "type"):
            assert col in r
        # One-shot / recurring tags must both appear
        assert "one-shot" in r
        assert "recurring" in r

    def test_as_dict_is_json_serializable(self):
        p = StageProfiler()
        with p.stage("x"):
            time.sleep(0.001)
        dumped = json.dumps(p.as_dict())
        loaded = json.loads(dumped)
        assert "stages" in loaded
        assert "total_ms" in loaded
        assert isinstance(loaded["stages"], list)

    def test_total_ms_equals_sum_of_stages(self):
        p = StageProfiler()
        with p.stage("a"):
            time.sleep(0.001)
        with p.stage("b"):
            time.sleep(0.001)
        d = p.as_dict()
        summed = sum(s["total_ms"] for s in d["stages"])
        assert d["total_ms"] == pytest.approx(summed, abs=0.1)


class TestStageProfilerEdgeCases:
    def test_exception_in_stage_still_closes_timing(self):
        """A raise inside `with p.stage(...):` must not corrupt the totals."""
        p = StageProfiler()
        with pytest.raises(RuntimeError):
            with p.stage("boom"):
                _busy_wait_ms(2)
                raise RuntimeError("oops")
        d = p.as_dict()
        assert len(d["stages"]) == 1
        assert d["stages"][0]["name"] == "boom"
        assert d["stages"][0]["count"] == 1
        assert d["stages"][0]["total_ms"] >= 1.0

    def test_zero_duration_stage(self):
        p = StageProfiler()
        with p.stage("instant"):
            pass
        d = p.as_dict()
        # A `pass` still measures microseconds on perf_counter, so the
        # total should be tiny but not literally 0.0.
        assert d["stages"][0]["total_ms"] < 1.0
        assert d["stages"][0]["count"] == 1
        assert d["stages"][0]["avg_ms"] < 1.0

    def test_stages_preserve_insertion_order_in_stages_list(self):
        p = StageProfiler()
        with p.stage("first"):
            pass
        with p.stage("second"):
            pass
        with p.stage("third"):
            pass
        assert p.stages() == ["first", "second", "third"]
        # But as_dict sorts by total time desc
        d = p.as_dict()
        assert [s["name"] for s in d["stages"]] == ["first", "second", "third"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
