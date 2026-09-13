"""Single source of truth for video configuration used by GUI and CLI.

FUENTE UNICA: la tabla `video_configs` de SQLite (`AppRepository`).
Los 3 JSON legacy solo se importan una vez (flag en `meta`); después la BD
manda. `verdad.test.json` queda como fallback para corridas CLI/evaluación
sin BD configurada.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoConfig:
    video_name: str
    polygon: tuple[tuple[int, int], ...]
    green: float
    yellow: float
    red: float
    avenue: str = ""
    danger_zone_margin_pixels: float = 80.0
    pre_red_seconds: float = 0.5
    green_skip_rate: int = 60

    @property
    def semaphore(self) -> dict[str, float]:
        return {"green": self.green, "yellow": self.yellow, "red": self.red}


class VideoConfigRepository:
    """Lee la configuración por video desde la BD, con fallback a verdad."""

    def __init__(self, project_root: str | Path, db_path: str | Path | None = None):
        root = Path(project_root)
        self.root = root
        self.truth_path = root / "tests" / "verdad.test.json"
        self._db_path = db_path

    def _read(self, path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except (OSError, json.JSONDecodeError):
            return default

    def _from_db(self, video_name: str) -> dict | None:
        try:
            from src.infrastructure.database.app_repository import AppRepository

            repo = AppRepository(self._db_path) if self._db_path else AppRepository()
            # Import único JSON→BD (flag en meta): lo legacy aparece solo.
            try:
                repo.import_legacy_configs(project_root=self.root)
            except Exception:
                pass
            return repo.get_video_config(video_name)
        except Exception:
            return None

    @staticmethod
    def _from_row(row: dict, video_name: str) -> VideoConfig | None:
        polygon = row.get("polygon")
        if isinstance(polygon, str):
            try:
                polygon = json.loads(polygon)
            except (TypeError, json.JSONDecodeError):
                polygon = []
        green, yellow, red = row.get("green"), row.get("yellow"), row.get("red")
        if not polygon or green is None or yellow is None or red is None:
            return None
        try:
            points = tuple((int(point[0]), int(point[1])) for point in polygon)
            return VideoConfig(
                video_name=video_name,
                polygon=points,
                green=float(green),
                yellow=float(yellow),
                red=float(red),
                avenue=str(row.get("avenue") or ""),
                danger_zone_margin_pixels=float(row.get("danger_zone_margin_pixels") or 80),
                pre_red_seconds=float(row.get("pre_red_seconds") or 0.5),
                green_skip_rate=int(row.get("green_skip_rate") or 60),
            )
        except (TypeError, ValueError, IndexError):
            return None

    def get(self, video_name: str) -> VideoConfig | None:
        # 1) BD (fuente única, incluye import único de los JSON legacy).
        row = self._from_db(video_name)
        if row is not None:
            config = self._from_row(row, video_name)
            if config is not None:
                return config
        # 2) Fallback CLI/evaluación sin BD configurada.
        for entry in self._read(self.truth_path, {}).get("videos_verdad", []):
            if entry.get("path_name") == video_name:
                polygon = entry.get("polygon", [])
                preset = entry if all(k in entry for k in ("green", "yellow", "red")) else {}
                if not polygon or not preset:
                    break
                return self._from_row(
                    {
                        "polygon": polygon,
                        "green": preset.get("green"),
                        "yellow": preset.get("yellow"),
                        "red": preset.get("red"),
                        "avenue": entry.get("avenue", ""),
                        "danger_zone_margin_pixels": preset.get("danger_zone_margin_pixels", 80),
                        "pre_red_seconds": preset.get("pre_red_seconds", 0.5),
                        "green_skip_rate": preset.get("green_skip_rate", 60),
                    },
                    video_name,
                )
        return None

    def require(self, video_name: str) -> VideoConfig:
        config = self.get(video_name)
        if config is None:
            raise ValueError(f"Video sin configuración válida: {video_name}")
        return config
