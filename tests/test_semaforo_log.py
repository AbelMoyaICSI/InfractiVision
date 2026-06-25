"""
Tests para ``scripts/adapter/semaforo_log.py``.

Cubre:
- ``format_timestamp`` (segundos, minutos, horas, negativos)
- ``compute_transitions`` (ciclo normal, video corto, con offset, truncado)
- ``iterate_all_videos`` (lee el JSON real, filtra inválidos, errores)
- ``build_parser`` (defaults, --output-dir, --quiet)
- ``_run_batch_mode`` (escribe secciones, no llama a cv2)
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapter.semaforo_log import (  # noqa: E402
    VERDAD_JSON,
    _run_batch_mode,
    build_parser,
    compute_transitions,
    format_timestamp,
    iterate_all_videos,
)
from scripts.adapter.video_semaphore import VideoSemaphore  # noqa: E402


# ── format_timestamp ──────────────────────────────────────────────────


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00"

    def test_seconds_only(self):
        assert format_timestamp(5) == "00:05"
        assert format_timestamp(59) == "00:59"

    def test_minutes(self):
        assert format_timestamp(60) == "01:00"
        assert format_timestamp(75) == "01:15"
        assert format_timestamp(125) == "02:05"

    def test_hours(self):
        assert format_timestamp(3600) == "01:00:00"
        assert format_timestamp(3661) == "01:01:01"
        assert format_timestamp(7200) == "02:00:00"

    def test_negative_clamped_to_zero(self):
        assert format_timestamp(-5) == "00:00"
        assert format_timestamp(-999) == "00:00"


# ── compute_transitions ──────────────────────────────────────────────


class TestComputeTransitions:
    def test_30_5_40_two_full_cycles(self):
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        # 200s / 75s = 2.67 ciclos. n=0,1,2 → 9 transiciones
        ts = compute_transitions(sem, 200.0)
        assert len(ts) == 9
        assert ts[0] == (0.0, "green")
        assert ts[1] == (30.0, "yellow")
        assert ts[2] == (35.0, "red")
        assert ts[3] == (75.0, "green")
        assert ts[4] == (105.0, "yellow")
        assert ts[5] == (110.0, "red")
        assert ts[6] == (150.0, "green")
        assert ts[7] == (180.0, "yellow")
        assert ts[8] == (185.0, "red")

    def test_short_video_only_initial(self):
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        # 20s < ciclo 75s → solo t=0
        assert compute_transitions(sem, 20.0) == [(0.0, "green")]

    def test_zero_duration_returns_empty(self):
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        assert compute_transitions(sem, 0.0) == []
        assert compute_transitions(sem, -1.0) == []

    def test_with_offset_shifts_all_transitions(self):
        sem = VideoSemaphore(
            {"green": 10, "yellow": 2, "red": 20, "start_offset_seconds": 5}
        )
        # 50s, ciclo 32s, offset 5
        ts = compute_transitions(sem, 50.0)
        assert ts[0] == (5.0, "green")
        assert ts[-1] == (49.0, "red")
        assert len(ts) == 6

    def test_truncates_at_duration(self):
        sem = VideoSemaphore({"green": 30, "yellow": 5, "red": 40})
        # 100s / 75s = 1.33 ciclos
        ts = compute_transitions(sem, 100.0)
        assert ts == [
            (0.0, "green"),
            (30.0, "yellow"),
            (35.0, "red"),
            (75.0, "green"),
        ]


# ── iterate_all_videos ───────────────────────────────────────────────


class TestIterateAllVideos:
    def test_reads_real_json(self):
        """Lee el JSON real del repo; todas las entradas tienen g/y/r > 0."""
        videos = iterate_all_videos(VERDAD_JSON)
        assert len(videos) == 11
        for v in videos:
            assert "path_name" in v
            assert "time" in v
            assert v["green"] > 0
            assert v["yellow"] > 0
            assert v["red"] > 0

    def test_returns_known_video(self):
        videos = iterate_all_videos(VERDAD_JSON)
        coliseo = [v for v in videos if v["path_name"] == "VID2COLISEO.MOV"]
        assert len(coliseo) == 1
        assert coliseo[0]["green"] == 15
        assert coliseo[0]["yellow"] == 3
        assert coliseo[0]["red"] == 20
        assert coliseo[0]["time"] == 201

    def test_filters_zero_durations(self, tmp_path):
        """Entradas con g=y=r=0 se filtran."""
        cfg = tmp_path / "verdad.json"
        cfg.write_text(json.dumps({
            "videos_verdad": [
                {"path_name": "good.mp4", "time": 100,
                 "green": 10, "yellow": 3, "red": 15},
                {"path_name": "invalid.mp4", "time": 50,
                 "green": 0, "yellow": 0, "red": 0},
            ]
        }), encoding="utf-8")
        videos = iterate_all_videos(cfg)
        assert len(videos) == 1
        assert videos[0]["path_name"] == "good.mp4"

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            iterate_all_videos(tmp_path / "does_not_exist.json")

    def test_malformed_json_raises(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            iterate_all_videos(cfg)

    def test_missing_videos_verdad_key(self, tmp_path):
        """Si el JSON no tiene la key `videos_verdad`, retorna lista vacía."""
        cfg = tmp_path / "empty.json"
        cfg.write_text(json.dumps({"other_key": []}), encoding="utf-8")
        assert iterate_all_videos(cfg) == []


# ── build_parser ─────────────────────────────────────────────────────


class TestBuildParser:
    def test_no_args_works(self):
        """El parser no requiere args; corre con defaults."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.output_dir == str(VERDAD_JSON.parent.parent / "data" / "logs")
        assert args.quiet is False

    def test_output_dir_override(self):
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/foo"])
        assert args.output_dir == "/tmp/foo"

    def test_quiet_default_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.quiet is False

    def test_quiet_flag_sets_true(self):
        parser = build_parser()
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_help_no_required_args(self):
        """No debe haber ningún flag required (cualquier cosa puede ser default)."""
        parser = build_parser()
        # Si hay required, parse_args([]) falla. Aquí debe funcionar.
        args = parser.parse_args([])
        assert args is not None


# ── _run_batch_mode ──────────────────────────────────────────────────


class TestRunBatchMode:
    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger("test_batch")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        return logger

    def test_writes_sections_for_each_video(self, tmp_path, capsys):
        """Verifica que se emite un header `── [i/N] ──` por cada video."""
        log_path = tmp_path / "semaforo.log"
        logger = self._make_logger()
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        logger.addHandler(fh)

        rc = _run_batch_mode(logger, log_path)
        assert rc == 0

        fh.close()
        content = log_path.read_text(encoding="utf-8")

        # Header de batch completo
        assert "=== START Semaforo Log (batch completo) ===" in content
        # 11 secciones
        for i in range(1, 12):
            assert f"[{i}/11]" in content, f"Falta sección [{i}/11]"
        # Algunos videos conocidos
        assert "VID2COLISEO.MOV" in content
        assert "Av-Condorcanqui.mp4" in content
        # Resumen final
        assert "Total videos: 11" in content
        assert "=== END Semaforo Log ===" in content

    def test_no_cv2_imports_or_calls(self, tmp_path):
        """En batch mode, cv2 NO debe ser tocado (duración viene del JSON)."""
        log_path = tmp_path / "semaforo.log"
        logger = self._make_logger()
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        logger.addHandler(fh)

        with patch("cv2.VideoCapture") as mock_cap:
            _run_batch_mode(logger, log_path)
            mock_cap.assert_not_called()
        fh.close()

    def test_total_transitions_counted(self, tmp_path):
        """El contador global suma las transiciones de los 11 videos."""
        log_path = tmp_path / "semaforo.log"
        logger = self._make_logger()
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        logger.addHandler(fh)

        _run_batch_mode(logger, log_path)
        fh.close()

        content = log_path.read_text(encoding="utf-8")
        # Extraer el número de transiciones globales del resumen final
        import re
        m = re.search(r"Transiciones globales: (\d+)", content)
        assert m is not None
        total = int(m.group(1))
        # Cota inferior: cada video tiene al menos 1 transición
        assert total >= 11
        # Cota superior: cada video tiene <= 3 * (duration/ciclo) transiciones
        # 11 videos con duraciones entre 12 y 1097, ciclos entre 28 y 95
        # estimación: < 200 transiciones globales
        assert total < 500

    def test_error_on_missing_json(self, tmp_path, monkeypatch):
        """Si verdad.test.json no existe, retorna exit 1 + mensaje claro."""
        log_path = tmp_path / "semaforo.log"
        logger = self._make_logger()
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        logger.addHandler(fh)

        # Apuntar a un path inexistente
        from scripts.adapter import semaforo_log
        monkeypatch.setattr(semaforo_log, "VERDAD_JSON", tmp_path / "nope.json")

        rc = _run_batch_mode(logger, log_path)
        assert rc == 1
        fh.close()
        content = log_path.read_text(encoding="utf-8")
        assert "No se pudo leer" in content

    def test_error_on_malformed_json(self, tmp_path, monkeypatch):
        log_path = tmp_path / "semaforo.log"
        logger = self._make_logger()
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        logger.addHandler(fh)

        bad = tmp_path / "bad.json"
        bad.write_text("{ broken", encoding="utf-8")
        from scripts.adapter import semaforo_log
        monkeypatch.setattr(semaforo_log, "VERDAD_JSON", bad)

        rc = _run_batch_mode(logger, log_path)
        assert rc == 1


# ── Verificación de que la nueva implementación es batch-only ─────────


class TestBatchOnlyDesign:
    """Verifica que NO existen las features del modo single que se eliminaron."""

    def test_no_load_verdad_calibration_function(self):
        """La función single-video lookup fue eliminada."""
        from scripts.adapter import semaforo_log
        assert not hasattr(semaforo_log, "load_verdad_calibration")

    def test_no_resolve_semaphore_config_function(self):
        """La función de cascade fue eliminada."""
        from scripts.adapter import semaforo_log
        assert not hasattr(semaforo_log, "resolve_semaphore_config")

    def test_no_read_video_metadata_function(self):
        """cv2 ya no se usa."""
        from scripts.adapter import semaforo_log
        assert not hasattr(semaforo_log, "read_video_metadata")

    def test_no_video_arg_in_parser(self):
        """El flag --video fue eliminado."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--video", "x.mp4"])
