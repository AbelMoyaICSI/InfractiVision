"""
test_adaptive_skip.py — Unit tests for AdaptiveSkipController.

The expected behavior is documented in the original analysis:
  - red+active:  1 if budget OK, else min(3, ratio)
  - green:       min(12, max(8, ratio*2))
  - yellow / red-alone:  min(5, max(2, ratio))

These tests pin the policy so a future tweak (or a bug) cannot silently
change the contract of the controller.
"""

from __future__ import annotations

import pytest

from scripts.adapter.adaptive_skip import AdaptiveSkipController


# ── Construction ──────────────────────────────────────────────────────


class TestConstruction:
    def test_default_30fps_budget(self):
        c = AdaptiveSkipController()
        assert c.target_fps_video == 30.0
        assert c.frame_budget_ms == pytest.approx(33.333, abs=0.01)
        assert c.sample_count == 0

    def test_custom_fps(self):
        c = AdaptiveSkipController(target_fps_video=60.0)
        assert c.target_fps_video == 60.0
        assert c.frame_budget_ms == pytest.approx(16.666, abs=0.01)

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError):
            AdaptiveSkipController(target_fps_video=0)
        with pytest.raises(ValueError):
            AdaptiveSkipController(target_fps_video=-1)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            AdaptiveSkipController(window=0)

    def test_set_target_fps_updates_budget(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        c.set_target_fps(60.0)
        assert c.target_fps_video == 60.0
        assert c.frame_budget_ms == pytest.approx(16.666, abs=0.01)

    def test_set_target_fps_rejects_invalid(self):
        c = AdaptiveSkipController()
        with pytest.raises(ValueError):
            c.set_target_fps(0)


# ── Cold-start behavior (no measurements yet) ─────────────────────────


class TestColdStart:
    """With no measurements, the controller should assume the best case
    (ratio=1) so the pipeline starts with maximum sensitivity and only
    relaxes once reality proves it needs to."""

    def test_green_with_no_data_returns_min_green(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        # ratio=1 -> max(8, 2) = 8
        assert c.suggest_skip("green", active_count=0) == 8

    def test_red_active_with_no_data_returns_one(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        assert c.suggest_skip("red", active_count=2) == 1

    def test_yellow_with_no_data_returns_min_idle(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        # ratio=1 -> max(2, 1) = 2
        assert c.suggest_skip("yellow", active_count=0) == 2

    def test_red_alone_with_no_data_returns_min_idle(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        assert c.suggest_skip("red", active_count=0) == 2


# ── Recording measurements ────────────────────────────────────────────


class TestRecording:
    def test_record_ignores_zero_or_negative(self):
        c = AdaptiveSkipController()
        c.record(0, 0)
        c.record(-5, 1)
        c.record(10, 0)
        assert c.sample_count == 0

    def test_record_divides_by_batch_size(self):
        """A 100ms batch of 4 frames contributes 25ms / frame, not 100ms."""
        c = AdaptiveSkipController(target_fps_video=30.0)  # budget 33.3ms
        c.record(batch_time_ms=100.0, batch_size=4)
        assert c.sample_count == 1
        # 25ms < 33.3ms -> ratio = 1
        assert c.suggest_skip("red", active_count=1) == 1

    def test_record_appends_to_window(self):
        c = AdaptiveSkipController(target_fps_video=30.0, window=3)
        c.record(50, 1)
        c.record(50, 1)
        c.record(50, 1)
        assert c.sample_count == 3
        # Push one more — oldest should be evicted, count stays at 3.
        c.record(50, 1)
        assert c.sample_count == 3

    def test_reset_clears_window(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        c.record(80, 1)
        c.record(80, 1)
        assert c.sample_count == 2
        c.reset()
        assert c.sample_count == 0
        # After reset, controller should behave like cold start.
        assert c.suggest_skip("red", active_count=1) == 1


# ── Policy: red+active (the critical case) ────────────────────────────


class TestRedActivePolicy:
    """The whole point of the controller: red+active should NEVER spend
    more than 1/3 of its frames idle when the system is slow, and never
    skip at all when the system is keeping up."""

    def _ctrl_with_avg_per_frame(self, avg_ms: float, fps: float = 30.0):
        c = AdaptiveSkipController(target_fps_video=fps, window=5)
        # Fill the window so the average is exactly avg_ms
        for _ in range(5):
            c.record(avg_ms, 1)
        return c

    def test_budget_met_returns_one(self):
        c = self._ctrl_with_avg_per_frame(20.0)  # 20ms < 33.3ms budget
        assert c.suggest_skip("red", active_count=1) == 1

    def test_2x_budget_returns_2(self):
        c = self._ctrl_with_avg_per_frame(66.6)  # ~2x budget
        assert c.suggest_skip("red", active_count=1) == 2

    def test_3x_budget_returns_3(self):
        c = self._ctrl_with_avg_per_frame(100.0)  # 3x budget
        assert c.suggest_skip("red", active_count=1) == 3

    def test_caps_at_3_even_when_pathologically_slow(self):
        c = self._ctrl_with_avg_per_frame(500.0)  # 15x budget
        # min(3, 15) = 3 — we never want to skip more than 2/3 of red frames
        assert c.suggest_skip("red", active_count=1) == 3

    def test_active_count_zero_falls_through_to_idle(self):
        """No active infractor -> same policy as yellow."""
        c = self._ctrl_with_avg_per_frame(50.0)  # ratio=2
        # idle: min(5, max(2, 2)) = 2
        assert c.suggest_skip("red", active_count=0) == 2


# ── Policy: green (aggressive skip) ───────────────────────────────────


class TestGreenPolicy:
    def _ctrl_with_avg_per_frame(self, avg_ms: float):
        c = AdaptiveSkipController(target_fps_video=30.0, window=5)
        for _ in range(5):
            c.record(avg_ms, 1)
        return c

    def test_fast_system_uses_min_green(self):
        c = self._ctrl_with_avg_per_frame(20.0)  # ratio=1
        # max(8, 2) = 8
        assert c.suggest_skip("green", active_count=0) == 8

    def test_2x_budget_still_uses_min_green(self):
        c = self._ctrl_with_avg_per_frame(66.6)  # ratio=2
        # max(8, 4) = 8
        assert c.suggest_skip("green", active_count=0) == 8

    def test_4x_budget_uses_8_still(self):
        c = self._ctrl_with_avg_per_frame(133.0)  # ratio=4
        # max(8, 8) = 8
        assert c.suggest_skip("green", active_count=0) == 8

    def test_5x_budget_uses_10(self):
        c = self._ctrl_with_avg_per_frame(166.5)  # ratio=5
        # max(8, 10) = 10
        assert c.suggest_skip("green", active_count=0) == 10

    def test_extremely_slow_caps_at_max(self):
        c = self._ctrl_with_avg_per_frame(1000.0)  # ratio~30
        # min(12, 60) = 12
        assert c.suggest_skip("green", active_count=0) == 12


# ── Policy: yellow / red-alone (moderate) ─────────────────────────────


class TestIdlePolicy:
    def _ctrl_with_avg_per_frame(self, avg_ms: float):
        c = AdaptiveSkipController(target_fps_video=30.0, window=5)
        for _ in range(5):
            c.record(avg_ms, 1)
        return c

    def test_fast_system_uses_min_idle(self):
        c = self._ctrl_with_avg_per_frame(20.0)  # ratio=1
        # max(2, 1) = 2
        assert c.suggest_skip("yellow", active_count=0) == 2

    def test_2x_budget_uses_2(self):
        c = self._ctrl_with_avg_per_frame(66.6)  # ratio=2
        assert c.suggest_skip("yellow", active_count=0) == 2

    def test_3x_budget_uses_3(self):
        c = self._ctrl_with_avg_per_frame(100.0)  # ratio=3
        assert c.suggest_skip("yellow", active_count=0) == 3

    def test_5x_budget_uses_5(self):
        c = self._ctrl_with_avg_per_frame(166.5)  # ratio=5
        assert c.suggest_skip("yellow", active_count=0) == 5

    def test_extremely_slow_caps_at_idle_max(self):
        c = self._ctrl_with_avg_per_frame(1000.0)  # ratio huge
        # min(5, large) = 5
        assert c.suggest_skip("yellow", active_count=0) == 5

    def test_red_alone_uses_same_policy_as_yellow(self):
        c = self._ctrl_with_avg_per_frame(100.0)  # ratio=3
        assert c.suggest_skip("red", active_count=0) == \
               c.suggest_skip("yellow", active_count=0)


# ── Introspection ─────────────────────────────────────────────────────


class TestIntrospection:
    def test_last_avg_and_ratio_updated(self):
        c = AdaptiveSkipController(target_fps_video=30.0)
        for _ in range(5):
            c.record(60.0, 1)
        c.suggest_skip("green", 0)
        assert c.last_avg_ms == pytest.approx(60.0)
        assert c.last_ratio == 2  # 60 / 33.3 ~= 1.8 -> rounds to 2

    def test_repr_is_informative(self):
        c = AdaptiveSkipController(target_fps_video=30.0, window=4)
        c.record(40, 1)
        c.suggest_skip("green", 0)
        r = repr(c)
        assert "30.0fps" in r
        assert "budget" in r
        assert "ratio" in r
        assert "samples" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
