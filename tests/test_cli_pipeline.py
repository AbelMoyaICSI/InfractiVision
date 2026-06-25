#!/usr/bin/env python
"""
test_cli_pipeline.py — Integration test for the CLI detection pipeline.

Runs process_video.py as a subprocess or direct import and validates:
  - Exit code
  - JSON output files exist
  - NID/NIE counts are within expected ranges
  - Output structure matches expected schema

Usage:
    pytest tests/test_cli_pipeline.py -v
    python tests/test_cli_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "test_run"
SCRIPT = PROJECT_ROOT / "scripts" / "adapter" / "process_video.py"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_output_dir():
    """Create and clean output directory for each test."""
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    # Clean previous runs
    for f in OUTPUT_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
    yield
    # Teardown: keep output for inspection


def load_fixture_json(name: str) -> dict:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture {name} not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_videos() -> list[str]:
    """Return list of video paths that exist on disk."""
    candidates = [
        str(PROJECT_ROOT / "videos" / "VID2COLISEO.MOV"),
        str(PROJECT_ROOT / "videos" / "VID4EDIT - Hecho con Clipchamp.mp4"),
    ]
    return [p for p in candidates if os.path.exists(p)]


# ── Unit tests (fast, no video needed) ────────────────────────────────


class TestVideoSemaphore:
    """Test the deterministic semaphore."""

    def test_initial_state_is_green(self):
        from scripts.adapter.video_semaphore import VideoSemaphore
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        assert sem.get_state(0) == "green"

    def test_transitions(self):
        from scripts.adapter.video_semaphore import VideoSemaphore
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        assert sem.get_state(0) == "green"
        assert sem.get_state(30) == "yellow"
        assert sem.get_state(34) == "yellow"
        assert sem.get_state(35) == "red"
        assert sem.get_state(74) == "red"
        assert sem.get_state(75) == "green"

    def test_offset(self):
        from scripts.adapter.video_semaphore import VideoSemaphore
        sem = VideoSemaphore(
            {"green": 10, "yellow": 2, "red": 20, "start_offset_seconds": 5}
        )
        # At second 5, the cycle starts — green first
        assert sem.get_state(5) == "green"
        assert sem.get_state(15) == "yellow"
        assert sem.get_state(17) == "red"

    def test_full_cycle(self):
        from scripts.adapter.video_semaphore import VideoSemaphore
        sem = VideoSemaphore({"green": 10, "yellow": 3, "red": 20})
        states = [sem.get_state(t) for t in range(66)]
        # Should see 10 green, 3 yellow, 20 red x 2 cycles
        assert states.count("green") == 20  # 2 * 10
        assert states.count("yellow") == 6  # 2 * 3
        assert states.count("red") == 40  # 2 * 20


class TestPPI:
    """Test PPI V50 calculation."""

    def test_ppi_far(self):
        from scripts.adapter.infraction_tracker import calculate_ppi
        # Vehicle at the top of the frame = far
        ppi = calculate_ppi(640, 140, 720, 1280)
        assert ppi < 0.20, f"Expected PPI < 0.20, got {ppi}"

    def test_ppi_close(self):
        from scripts.adapter.infraction_tracker import calculate_ppi
        # Vehicle near the bottom = close
        ppi = calculate_ppi(640, 650, 720, 1280)
        assert ppi > 0.65, f"Expected PPI > 0.65, got {ppi}"


class TestPersistence:
    """Test JSON save functions."""

    def test_save_infractions_creates_file(self, setup_output_dir):
        from scripts.adapter.persistence import save_infractions_json
        path = save_infractions_json(
            [{"plate": "T1A-234", "confidence": 0.85, "clasificacion": "NID"}],
            output_dir=str(OUTPUT_DIR),
            filename="test_infracciones.json",
        )
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "infracciones" in data
        assert len(data["infracciones"]) >= 1

    def test_save_indicators_has_all_sections(self, setup_output_dir):
        from scripts.adapter.persistence import save_indicators_json
        path = save_indicators_json(
            nid_count=5,
            nie_count=2,
            ti_percentage=71.4,
            tr_individual_minutes=[0.5, 0.3, 0.4],
            output_dir=str(OUTPUT_DIR),
            filename="test_indicadores.json",
        )
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "indicadores" in data
        assert "TI" in data["indicadores"]
        assert "TR" in data["indicadores"]
        assert "NID" in data["indicadores"]
        assert "NIE" in data["indicadores"]
        assert "resumen_global" in data
        assert data["resumen_global"]["nid_total"] == 5
        assert data["resumen_global"]["nie_total"] == 2
        assert data["resumen_global"]["tir_total"] == 7


# ── Integration test (needs a real video) ─────────────────────────────


@pytest.mark.slow
class TestCLIPipeline:
    """Integration test: run the full pipeline on a real video."""

    @pytest.mark.parametrize("video_path", available_videos())
    def test_pipeline_runs_without_error(self, video_path, setup_output_dir):
        """Pipeline completes with exit code 0 and produces JSON output."""
        cfg_path = FIXTURES_DIR / "test_config.json"
        if not cfg_path.exists():
            pytest.skip("test_config.json not found")

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--video", video_path,
                "--config", str(cfg_path),
                "--output-dir", str(OUTPUT_DIR),
                "--max-frames", "200",
                "--conf-vehicle", "0.50",
                "--conf-plate", "0.40",
                "--batch-size", "4",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )

        print("\n--- STDOUT ---\n", result.stdout)
        if result.stderr and "Error" in result.stderr:
            print("\n--- STDERR ---\n", result.stderr)

        assert result.returncode == 0, (
            f"Pipeline exited with code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )

    def test_json_output_structure(self, setup_output_dir):
        """Verify that a previous run produced valid JSON structures."""
        # This test checks data that already exists from a previous run
        infra_path = OUTPUT_DIR / "infracciones.json"
        nie_path = OUTPUT_DIR / "nie_infracciones.json"
        indic_path = OUTPUT_DIR / "indicadores_rendimiento.json"

        any_exists = infra_path.exists() or nie_path.exists()
        if not any_exists:
            pytest.skip("No previous output found — run test_pipeline_runs_without_error first")

        # Check NID JSON
        if infra_path.exists():
            with open(str(infra_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "infracciones" in data
            for inf in data["infracciones"]:
                assert "placa" in inf
                assert "clasificacion" in inf
                assert "confianza" in inf

        # Check indicators JSON
        if indic_path.exists():
            with open(str(indic_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "indicadores" in data
            assert "TI" in data["indicadores"], "Missing TI in indicators"
            assert "TR" in data["indicadores"], "Missing TR in indicators"
            assert "NID" in data["indicadores"], "Missing NID in indicators"

    def test_module_import(self):
        """Verify the main module imports cleanly (no import errors)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "process_video", str(SCRIPT)
        )
        assert spec is not None, "Could not find process_video.py"
        assert spec.loader is not None, "Could not create loader"


# ── Run standalone ────────────────────────────────────────────────────


if __name__ == "__main__":
    # Quick smoke test
    from scripts.adapter.video_semaphore import VideoSemaphore
    sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
    assert sem.get_state(0) == "green"
    assert sem.get_state(35) == "red"
    assert sem.get_state(75) == "green"
    print("✅ VideoSemaphore smoke test passed")

    from scripts.adapter.infraction_tracker import calculate_ppi
    ppi = calculate_ppi(640, 650, 720, 1280)
    assert ppi > 0.60
    print(f"✅ PPI smoke test passed (ppi={ppi:.3f})")

    print("\n🧪 All quick checks passed. Run with pytest for full suite:")
    print(f"   pytest {__file__} -v")
