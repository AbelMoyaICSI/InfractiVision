"""BD como fuente única de configuración por video."""
from __future__ import annotations

import json
from pathlib import Path

from src.infrastructure.configuration.video_config_repository import (
    VideoConfigRepository,
)
from src.infrastructure.database.app_repository import AppRepository


def _repo(tmp_path: Path) -> AppRepository:
    repo = AppRepository(tmp_path / "test.sqlite")
    repo.clear_video_configs()
    return repo


def test_save_and_get_roundtrip(tmp_path):
    repo = _repo(tmp_path)
    repo.save_video_config(
        "a.mp4", avenue="Av X", green=10, yellow=3, red=15,
        time_slot="08:00 - 09:00", polygon=[[0, 0], [10, 0], [10, 10]],
        danger_zone_margin_pixels=90, pre_red_seconds=1.5, green_skip_rate=30,
    )
    row = repo.get_video_config("a.mp4")
    assert row["avenue"] == "Av X"
    assert (row["green"], row["yellow"], row["red"]) == (10.0, 3.0, 15.0)
    assert row["polygon"] == [[0, 0], [10, 0], [10, 10]]
    assert row["danger_zone_margin_pixels"] == 90.0
    assert row["pre_red_seconds"] == 1.5
    assert row["green_skip_rate"] == 30


def test_partial_upsert_preserves_other_columns(tmp_path):
    repo = _repo(tmp_path)
    repo.save_video_config("a.mp4", green=10, yellow=3, red=15,
                           polygon=[[0, 0], [1, 0], [1, 1]])
    repo.save_video_config("a.mp4", avenue="Av Y")
    row = repo.get_video_config("a.mp4")
    assert row["avenue"] == "Av Y"
    assert row["green"] == 10.0
    assert row["polygon"] == [[0, 0], [1, 0], [1, 1]]


def test_clear_preset_keeps_polygon_and_avenue(tmp_path):
    repo = _repo(tmp_path)
    repo.save_video_config("a.mp4", avenue="Av X", green=10, yellow=3,
                           red=15, polygon=[[0, 0], [1, 0], [1, 1]])
    assert repo.clear_video_preset("a.mp4") is True
    row = repo.get_video_config("a.mp4")
    assert row["green"] is None and row["polygon"] and row["avenue"] == "Av X"


def test_delete_and_clear_all(tmp_path):
    repo = _repo(tmp_path)
    repo.save_video_config("a.mp4", green=1, yellow=1, red=1)
    repo.save_video_config("b.mp4", green=1, yellow=1, red=1)
    assert repo.delete_video_config("a.mp4") is True
    assert repo.get_video_config("a.mp4") is None
    assert repo.delete_video_config("a.mp4") is False
    assert repo.clear_video_configs() == 1
    assert repo.all_video_configs() == {}


def test_import_legacy_json_once(tmp_path):
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    (root / "config" / "polygon_config.json").write_text(
        json.dumps({"b.mp4": [[1, 1], [2, 1], [2, 2]]}), encoding="utf-8")
    (root / "config" / "time_presets.json").write_text(
        json.dumps({"b.mp4": {"green": 5, "yellow": 2, "red": 7,
                              "pre_red_seconds": 1.5}}), encoding="utf-8")
    (root / "config" / "avenue_config.json").write_text(
        json.dumps({"b.mp4": "Av B"}), encoding="utf-8")
    repo = _repo(tmp_path)
    first = repo.import_legacy_configs(root)
    assert first["imported"] == 1
    row = repo.get_video_config("b.mp4")
    assert row["avenue"] == "Av B" and row["pre_red_seconds"] == 1.5
    # Sin force no pisa lo ya configurado en BD.
    repo.save_video_config("b.mp4", red=99)
    repo.import_legacy_configs(root, force=False)
    assert repo.get_video_config("b.mp4")["red"] == 99.0


def test_repository_reads_db_first(tmp_path):
    db = tmp_path / "repo.sqlite"
    repo = AppRepository(db)
    repo.clear_video_configs()
    repo.save_video_config("v.mp4", avenue="Av DB", green=12, yellow=3,
                           red=20, polygon=[[0, 0], [5, 0], [5, 5]],
                           green_skip_rate=42)
    config = VideoConfigRepository(tmp_path, db_path=db).get("v.mp4")
    assert config is not None
    assert (config.green, config.yellow, config.red) == (12.0, 3.0, 20.0)
    assert config.avenue == "Av DB" and config.green_skip_rate == 42
    assert config.polygon == ((0, 0), (5, 0), (5, 5))
    # Sin BD ni verdad para ese video: None (sin rama root/config).
    assert VideoConfigRepository(tmp_path, db_path=db).get("nadie.mp4") is None


def test_repository_applies_defaults_for_missing_extras(tmp_path):
    db = tmp_path / "repo2.sqlite"
    repo = AppRepository(db)
    repo.clear_video_configs()
    repo.save_video_config("w.mp4", green=10, yellow=2, red=10,
                           polygon=[[0, 0], [5, 0], [5, 5]])
    config = VideoConfigRepository(tmp_path, db_path=db).require("w.mp4")
    assert config.danger_zone_margin_pixels == 80.0
    assert config.pre_red_seconds == 0.5
    assert config.green_skip_rate == 60
