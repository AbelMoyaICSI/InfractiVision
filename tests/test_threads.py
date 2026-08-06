"""
test_threads.py — Unit tests for configure_thread_budget().

Covers:
  - Returns a dict with the expected keys.
  - Idempotent: calling twice does not raise.
  - Honors custom values for cv_threads / torch_threads.
  - Does not raise when the environment is minimal (e.g. cv2 missing).
"""

from __future__ import annotations

import os

import pytest


class TestConfigureThreadBudget:
    def test_returns_dict_with_expected_keys(self):
        from scripts.adapter.threads import configure_thread_budget

        result = configure_thread_budget()
        assert isinstance(result, dict)
        # OpenCV is required by the scripts that import this module, so
        # it should always be present in the test env.
        assert "cv_threads" in result
        assert "torch_threads" in result
        assert "omp_num_threads" in result
        assert "mkl_num_threads" in result

    def test_sets_cv_threads(self):
        from scripts.adapter.threads import configure_thread_budget

        configure_thread_budget(cv_threads=3)
        import cv2
        assert cv2.getNumThreads() == 3

    def test_sets_torch_threads(self):
        from scripts.adapter.threads import configure_thread_budget

        configure_thread_budget(torch_threads=5)
        import torch
        assert torch.get_num_threads() == 5

    def test_sets_blas_env_vars(self):
        from scripts.adapter.threads import configure_thread_budget

        configure_thread_budget(torch_threads=6)
        assert os.environ.get("OMP_NUM_THREADS") == "6"
        assert os.environ.get("MKL_NUM_THREADS") == "6"

    def test_is_idempotent(self):
        from scripts.adapter.threads import configure_thread_budget

        r1 = configure_thread_budget(cv_threads=2, torch_threads=4)
        r2 = configure_thread_budget(cv_threads=2, torch_threads=4)
        # Same logical result
        assert r1["cv_threads"] == r2["cv_threads"]
        assert r1["torch_threads"] == r2["torch_threads"]

    def test_custom_values_propagate(self):
        from scripts.adapter.threads import configure_thread_budget

        r = configure_thread_budget(cv_threads=1, torch_threads=2)
        assert r["cv_threads"] == 1
        assert r["torch_threads"] == 2

    def test_default_values_match_documented_budget(self):
        """The default 2/4 split is intentional for 4-core / 8-thread CPUs."""
        from scripts.adapter.threads import configure_thread_budget

        r = configure_thread_budget()
        assert r["cv_threads"] == 2
        assert r["torch_threads"] == 4
        assert r["omp_num_threads"] == "4"
        assert r["mkl_num_threads"] == "4"

    def test_does_not_set_interop_threads(self):
        """interop_threads is intentionally not exposed — see threads.py."""
        from scripts.adapter.threads import configure_thread_budget

        r = configure_thread_budget()
        assert "torch_interop_threads" not in r


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
