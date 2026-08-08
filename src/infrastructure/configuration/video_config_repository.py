"""Single source of truth for video configuration used by GUI and CLI."""
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
    """Reads the files written by the existing configuration windows.

    `verdad.test.json` is only a fallback for CLI/evaluation runs. The GUI
    configuration files always take precedence.
    """

    def __init__(self, project_root: str | Path):
        root = Path(project_root)
        self.root = root
        self.config_dir = root / "config"
        self.truth_path = root / "tests" / "verdad.test.json"

    def _read(self, path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except (OSError, json.JSONDecodeError):
            return default

    def get(self, video_name: str) -> VideoConfig | None:
        polygon_cfg = self._read(self.config_dir / "polygon_config.json", {})
        preset_cfg = self._read(self.config_dir / "time_presets.json", {})
        avenue_cfg = self._read(self.config_dir / "avenue_config.json", {})

        polygon = polygon_cfg.get(video_name)
        preset = preset_cfg.get(video_name, {})
        avenue = avenue_cfg.get(video_name, "")

        if not polygon or not preset:
            for entry in self._read(self.truth_path, {}).get("videos_verdad", []):
                if entry.get("path_name") == video_name:
                    polygon = polygon or entry.get("polygon", [])
                    preset = preset or entry
                    avenue = avenue or entry.get("avenue", "")
                    break

        if not polygon or not all(key in preset for key in ("green", "yellow", "red")):
            return None

        try:
            points = tuple((int(point[0]), int(point[1])) for point in polygon)
            return VideoConfig(
                video_name=video_name,
                polygon=points,
                green=float(preset["green"]),
                yellow=float(preset["yellow"]),
                red=float(preset["red"]),
                avenue=str(avenue or ""),
                danger_zone_margin_pixels=float(preset.get("danger_zone_margin_pixels", 80)),
                pre_red_seconds=float(preset.get("pre_red_seconds", 0.5)),
                green_skip_rate=int(preset.get("green_skip_rate", 60)),
            )
        except (TypeError, ValueError, IndexError):
            return None

    def require(self, video_name: str) -> VideoConfig:
        config = self.get(video_name)
        if config is None:
            raise ValueError(f"Video sin configuración válida: {video_name}")
        return config
