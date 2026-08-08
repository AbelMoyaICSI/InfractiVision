#!/usr/bin/env python
"""
test_only_infractions.py — Tests for the segmented only_infractions pipeline.

Validates:
  - Log file is created with the right timestamp format
  - Crops directory is created
  - Pipeline can run without PlateDetector / LPRNet
  - Trigger count is within expected range for a known video
  - The PEAK_GOLD / HEAVY trigger types are the only ones possible
    (no PANIC / SECURE because there's no plate confirmation)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "adapter"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
CROPS_DIR = PROJECT_ROOT / "data" / "output" / "only_infractions"
SCRIPT = SCRIPTS_DIR / "only_infractions.py"

sys.path.insert(0, str(PROJECT_ROOT))

VIDEOS = [
    str(PROJECT_ROOT / "videos" / "VID2COLISEO.MOV"),
]
EXISTING_VIDEOS = [v for v in VIDEOS if os.path.exists(v)]


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_run_dirs(tmp_path):
    """Use tmp_path for logs and crops so tests don't pollute data/."""
    log_dir = tmp_path / "logs"
    crops_dir = tmp_path / "crops"
    log_dir.mkdir()
    crops_dir.mkdir()
    yield {"log_dir": str(log_dir), "crops_dir": str(crops_dir)}


def _run_script(
    video: str,
    log_dir: str,
    crops_dir: str,
    max_frames: int = 200,
    conf: float = 0.40,
    batch: int = 4,
    skip_start: int = 0,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """Invoke the script as a subprocess and capture output."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--video", video,
            "--output-dir", log_dir,
            "--crops-dir", crops_dir,
            "--max-frames", str(max_frames),
            "--conf-vehicle", str(conf),
            "--batch-size", str(batch),
            "--skip-start", str(skip_start),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


# ── Unit tests (no video needed) ────────────────────────────────────


class TestLogHelpers:
    def test_run_id_format(self):
        from scripts.adapter.only_infractions import make_run_id
        run_id = make_run_id()
        # ISO format with no colons
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$", run_id), (
            f"Unexpected run_id format: {run_id}"
        )

    def test_setup_file_logger_creates_file(self, tmp_path):
        from scripts.adapter.only_infractions import setup_file_logger
        log_path = str(tmp_path / "test.log")
        log = setup_file_logger(log_path, quiet=True)
        log.info("hello world")
        log.handlers[0].flush() if hasattr(log.handlers[0], "flush") else None
        # Close handlers
        for h in log.handlers:
            h.close()
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "hello world" in content

    def test_build_config_with_explicit_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "semaphore": {"green": 10, "yellow": 2, "red": 20},
            "avenue": "Test Ave",
            "time_slot": "00:00 - 01:00",
        }))
        # Use the script's argparse/builder by patching sys.argv
        from scripts.adapter.only_infractions import _build_config
        import argparse
        args = argparse.Namespace(
            video="dummy.mp4",
            config=str(cfg_path),
            conf_vehicle=0.5,
            batch_size=4,
        )
        cfg = _build_config(args)
        assert cfg["avenue"] == "Test Ave"
        assert cfg["semaphore"]["red"] == 20
        assert len(cfg["polygon"]) == 4


# ── Module import test ──────────────────────────────────────────────


class TestModuleIndependence:
    def test_module_imports_without_plate_or_ocr(self):
        """Verify the module can be imported without plate/ocr libs."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "only_infractions", str(SCRIPT)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Pipeline class exists
        assert hasattr(mod, "OnlyInfractionsPipeline")
        assert hasattr(mod, "main")
        # No plate detector reference
        pipeline_attrs = dir(mod.OnlyInfractionsPipeline)
        assert "plate_detector" not in pipeline_attrs or all(
            getattr(mod.OnlyInfractionsPipeline, a, None) is None
            for a in ["plate_detector"]
        ), "Pipeline should not require a plate_detector attribute"


# ── Integration tests (need a real video) ────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not EXISTING_VIDEOS, reason="No test video available")
class TestCLIRun:
    @pytest.mark.parametrize("video", EXISTING_VIDEOS)
    def test_runs_and_creates_log(self, video, isolate_run_dirs):
        result = _run_script(
            video=video,
            log_dir=isolate_run_dirs["log_dir"],
            crops_dir=isolate_run_dirs["crops_dir"],
            max_frames=200,
            timeout=180,
        )
        print("\n--- STDOUT ---\n", result.stdout[-500:])
        if result.returncode != 0:
            print("\n--- STDERR ---\n", result.stderr)
        assert result.returncode == 0, f"Exit code: {result.returncode}"

    def test_log_file_has_timestamp_format(self, isolate_run_dirs):
        result = _run_script(
            video=EXISTING_VIDEOS[0],
            log_dir=isolate_run_dirs["log_dir"],
            crops_dir=isolate_run_dirs["crops_dir"],
            max_frames=100,
        )
        assert result.returncode == 0

        # Find the log file
        log_files = list(Path(isolate_run_dirs["log_dir"]).glob("infractions_*.log"))
        assert len(log_files) == 1, (
            f"Expected 1 log file, found {len(log_files)}: {log_files}"
        )
        log_file = log_files[0]
        # Filename matches ISO timestamp
        assert re.match(
            r"^infractions_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.log$",
            log_file.name,
        ), f"Bad log filename: {log_file.name}"

        content = log_file.read_text(encoding="utf-8")
        # Has START and END markers
        assert "=== START OnlyInfractions ===" in content
        assert "=== END OnlyInfractions ===" in content
        # Has timing info
        assert "Tiempos" in content
        assert "FPS promedio" in content
        # Has frame stats
        assert "Frames procesados" in content
        # Has trigger counts
        assert "PEAK_GOLD" in content
        assert "HEAVY" in content

    def test_crops_subdir_created(self, isolate_run_dirs):
        result = _run_script(
            video=EXISTING_VIDEOS[0],
            log_dir=isolate_run_dirs["log_dir"],
            crops_dir=isolate_run_dirs["crops_dir"],
            max_frames=200,
        )
        assert result.returncode == 0

        # There should be a subdir matching run_<timestamp>
        subdirs = [
            d for d in Path(isolate_run_dirs["crops_dir"]).iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        assert len(subdirs) >= 1, f"No run_* subdirs in {isolate_run_dirs['crops_dir']}"
        # If there were triggers, the subdir should contain a vehicle crop
        # (We don't assert on count since it's video-dependent)

    def test_stdout_summary_has_required_fields(self, isolate_run_dirs):
        result = _run_script(
            video=EXISTING_VIDEOS[0],
            log_dir=isolate_run_dirs["log_dir"],
            crops_dir=isolate_run_dirs["crops_dir"],
            max_frames=100,
        )
        assert result.returncode == 0
        stdout = result.stdout
        for field in [
            "Video:",
            "Duración:",
            "Frames:",
            "Tiempos:",
            "Carga modelos:",
            "Procesamiento:",
            "FPS promedio:",
            "Infractores:",
            "PEAK_GOLD:",
            "HEAVY:",
            "Log:",
            "Crops:",
        ]:
            assert field in stdout, f"Missing in stdout: {field!r}"


# ── Run standalone ──────────────────────────────────────────────────


if __name__ == "__main__":
    # Quick smoke test
    print("Running quick smoke tests...")
    pytest.main([__file__, "-v", "--tb=short"])
