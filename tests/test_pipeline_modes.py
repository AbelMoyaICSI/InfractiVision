"""
Tests para el flag --new en process_video.py y only_infractions.py.

Cubre:
- _legacy_skip_rate (reconstruido del doc REFACTOR_CPU_PIPELINE.md)
- argparse: --new default false
- argparse: --new flag enable
- CLIInfractionPipeline: use_new=False (skip legacy, batch=4, no profiler)
- CLIInfractionPipeline: use_new=True (skip controller, batch=2, profiler)
- OnlyInfractionsPipeline: use_new=False (no crop_writer)
- OnlyInfractionsPipeline: use_new=True (crop_writer executor)
"""

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapter.process_video import (  # noqa: E402
    _legacy_skip_rate,
    build_parser as build_process_parser,
)
from scripts.adapter.only_infractions import (  # noqa: E402
    build_parser as build_only_inf_parser,
)


# ── _legacy_skip_rate (reconstruido del doc) ──────────────────────────


class TestLegacySkipRate:
    """Skip policy pre-refactor, reconstruido de REFACTOR_CPU_PIPELINE.md sección 4."""

    def test_red_with_active_returns_1(self):
        # La "excepción" que el refactor removió: process every frame
        assert _legacy_skip_rate("red", active_count=1) == 1
        assert _legacy_skip_rate("red", active_count=2) == 1
        assert _legacy_skip_rate("red", active_count=5) == 1

    def test_green_returns_10(self):
        assert _legacy_skip_rate("green", active_count=0) == 10
        assert _legacy_skip_rate("green", active_count=1) == 10

    def test_red_alone_returns_3(self):
        assert _legacy_skip_rate("red", active_count=0) == 3

    def test_yellow_returns_3(self):
        assert _legacy_skip_rate("yellow", active_count=0) == 3
        assert _legacy_skip_rate("yellow", active_count=1) == 3

    def test_unknown_state_returns_3(self):
        # Comportamiento default para cualquier estado no reconocido
        assert _legacy_skip_rate("unknown", active_count=0) == 3
        assert _legacy_skip_rate("", active_count=0) == 3

    def test_safety_invariant_red_active_never_skips_above_1(self):
        """Safety invariant: red+active NUNCA debe tener skip > 1 con la policy legacy."""
        for active in range(0, 5):
            rate = _legacy_skip_rate("red", active)
            if active > 0:
                assert rate == 1, (
                    f"red+active({active}) debería ser 1, no {rate}"
                )


# ── argparse: --new flag ──────────────────────────────────────────────


class TestArgparseNewFlag:
    def test_process_video_new_default_false(self):
        parser = build_process_parser()
        args = parser.parse_args(["--video", "x.mp4"])
        assert args.new is False

    def test_process_video_new_flag_sets_true(self):
        parser = build_process_parser()
        args = parser.parse_args(["--video", "x.mp4", "--new"])
        assert args.new is True

    def test_only_infractions_new_default_false(self):
        parser = build_only_inf_parser()
        args = parser.parse_args(["--video", "x.mp4"])
        assert args.new is False

    def test_only_infractions_new_flag_sets_true(self):
        parser = build_only_inf_parser()
        args = parser.parse_args(["--video", "x.mp4", "--new"])
        assert args.new is True

    def test_process_video_video_required_without_new(self):
        # --video sigue siendo required en ambos modos
        parser = build_process_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_only_infractions_video_required_without_new(self):
        parser = build_only_inf_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_process_video_new_with_other_args(self):
        parser = build_process_parser()
        args = parser.parse_args(
            ["--video", "x.mp4", "--new", "--batch-size", "4", "--display"]
        )
        assert args.new is True
        assert args.batch_size == 4
        assert args.display is True


# ── CLIInfractionPipeline: use_new flag ──────────────────────────────


class TestProcessVideoPipeline:
    """Verifica que el pipeline respete el flag use_new sin cargar modelos reales."""

    def _build_config(self):
        return {
            "semaphore": {"green": 15, "yellow": 3, "red": 20},
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "avenue": "Test",
            "time_slot": "Test",
            "conf_vehicle": 0.5,
            "conf_plate": 0.4,
            "rectification": True,
        }

    def test_use_new_false_default_batch_size_is_4(self, tmp_path, monkeypatch):
        """Sin --new, el default de batch_size debe ser 4 (legacy)."""
        from scripts.adapter.process_video import CLIInfractionPipeline

        # Mock del detector para no cargar el modelo YOLO real
        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[] for _ in range(len(a[0]) if a else 0)]

        monkeypatch.setattr(
            "scripts.adapter.process_video.VehicleDetector", MockDetector
        )
        monkeypatch.setattr(
            "scripts.adapter.process_video.PlateDetector", MockDetector
        )

        cfg = self._build_config()
        pipeline = CLIInfractionPipeline(cfg, use_new=False)
        assert pipeline.batch_size == 4
        assert pipeline.use_new is False
        assert pipeline.skip_controller is None
        assert pipeline.profiler is None

    def test_use_new_true_default_batch_size_is_2(self, monkeypatch):
        """Con --new, el default de batch_size debe ser 2 (refactor)."""
        from scripts.adapter.process_video import CLIInfractionPipeline

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[] for _ in range(len(a[0]) if a else 0)]

        monkeypatch.setattr(
            "scripts.adapter.process_video.VehicleDetector", MockDetector
        )
        monkeypatch.setattr(
            "scripts.adapter.process_video.PlateDetector", MockDetector
        )

        cfg = self._build_config()
        pipeline = CLIInfractionPipeline(cfg, use_new=True)
        assert pipeline.batch_size == 2
        assert pipeline.use_new is True
        assert pipeline.skip_controller is not None
        assert pipeline.profiler is not None

    def test_explicit_batch_size_overrides_default_in_both_modes(self, monkeypatch):
        """--batch-size explícito gana sobre el default en ambos modos."""
        from scripts.adapter.process_video import CLIInfractionPipeline

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[]]

        monkeypatch.setattr(
            "scripts.adapter.process_video.VehicleDetector", MockDetector
        )
        monkeypatch.setattr(
            "scripts.adapter.process_video.PlateDetector", MockDetector
        )

        cfg = self._build_config()
        cfg["batch_size"] = 8

        p_legacy = CLIInfractionPipeline(cfg, use_new=False)
        assert p_legacy.batch_size == 8

        p_new = CLIInfractionPipeline(cfg, use_new=True)
        assert p_new.batch_size == 8


# ── OnlyInfractionsPipeline: use_new flag ────────────────────────────


class TestOnlyInfractionsPipeline:
    """Verifica que el pipeline respete el flag use_new, especialmente crop_writer."""

    def _build_config(self):
        return {
            "semaphore": {"green": 15, "yellow": 3, "red": 20},
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "conf_vehicle": 0.5,
        }

    def test_use_new_false_no_crop_writer(self, tmp_path, monkeypatch):
        """Sin --new, crop_writer debe ser None (legacy: imwrite síncrono)."""
        from scripts.adapter.only_infractions import OnlyInfractionsPipeline

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[]]

        monkeypatch.setattr(
            "scripts.adapter.only_infractions.VehicleDetector", MockDetector
        )

        import logging
        logger = logging.getLogger("test_oi")
        logger.setLevel(logging.WARNING)
        crops_dir = str(tmp_path / "crops")

        cfg = self._build_config()
        pipeline = OnlyInfractionsPipeline(
            cfg, logger, crops_dir, use_new=False
        )
        assert pipeline.use_new is False
        assert pipeline.crop_writer is None
        assert pipeline.batch_size == 4  # legacy default

    def test_use_new_true_has_crop_writer(self, tmp_path, monkeypatch):
        """Con --new, crop_writer debe ser un ThreadPoolExecutor."""
        from scripts.adapter.only_infractions import OnlyInfractionsPipeline
        from concurrent.futures import ThreadPoolExecutor

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[]]

        monkeypatch.setattr(
            "scripts.adapter.only_infractions.VehicleDetector", MockDetector
        )

        import logging
        logger = logging.getLogger("test_oi")
        logger.setLevel(logging.WARNING)
        crops_dir = str(tmp_path / "crops")

        cfg = self._build_config()
        pipeline = OnlyInfractionsPipeline(
            cfg, logger, crops_dir, use_new=True
        )
        assert pipeline.use_new is True
        assert isinstance(pipeline.crop_writer, ThreadPoolExecutor)
        assert pipeline.batch_size == 2  # new default
        # Clean up
        pipeline.crop_writer.shutdown(wait=True)

    def test_use_new_true_has_profiler_and_skip_controller(
        self, tmp_path, monkeypatch
    ):
        """Con --new, profiler y skip_controller deben existir."""
        from scripts.adapter.only_infractions import OnlyInfractionsPipeline

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[]]

        monkeypatch.setattr(
            "scripts.adapter.only_infractions.VehicleDetector", MockDetector
        )

        import logging
        logger = logging.getLogger("test_oi")
        logger.setLevel(logging.WARNING)
        crops_dir = str(tmp_path / "crops")

        cfg = self._build_config()
        pipeline = OnlyInfractionsPipeline(
            cfg, logger, crops_dir, use_new=True
        )
        assert pipeline.skip_controller is not None
        assert pipeline.profiler is not None
        pipeline.crop_writer.shutdown(wait=True)

    def test_use_new_false_no_profiler_no_skip_controller(
        self, tmp_path, monkeypatch
    ):
        """Sin --new, ni profiler ni skip_controller deben existir."""
        from scripts.adapter.only_infractions import OnlyInfractionsPipeline

        class MockDetector:
            def __init__(self, *a, **kw):
                pass
            def detect_batch(self, *a, **kw):
                return [[]]

        monkeypatch.setattr(
            "scripts.adapter.only_infractions.VehicleDetector", MockDetector
        )

        import logging
        logger = logging.getLogger("test_oi")
        logger.setLevel(logging.WARNING)
        crops_dir = str(tmp_path / "crops")

        cfg = self._build_config()
        pipeline = OnlyInfractionsPipeline(
            cfg, logger, crops_dir, use_new=False
        )
        assert pipeline.skip_controller is None
        assert pipeline.profiler is None


# ── Verificación de equivalencia: legacy skip = sección 4 del doc ────


class TestLegacySkipMatchesDoc:
    """Verifica que la tabla de la sección 4 del doc se reproduce exactamente."""

    def test_table_from_section_4(self):
        # Tabla de la sección 4 del doc REFACTOR_CPU_PIPELINE.md
        cases = [
            # (state, active, expected_skip, rationale)
            ("green", 0, 10, "Idle: ahorra CPU"),
            ("green", 1, 10, "Idle: ahorra CPU (active_count no afecta green)"),
            ("red", 0, 3, "Pre-alerta en rojo sin activos"),
            ("red", 1, 1, "Detectando infractor (excepción legacy)"),
            ("red", 5, 1, "Detectando infractor (excepción legacy)"),
            ("yellow", 0, 3, "Pre-alerta"),
            ("yellow", 2, 3, "Pre-alerta"),
        ]
        for state, active, expected, rationale in cases:
            actual = _legacy_skip_rate(state, active)
            assert actual == expected, (
                f"({state!r}, active={active}): esperado {expected}, "
                f"obtenido {actual}. Rationale: {rationale}"
            )
