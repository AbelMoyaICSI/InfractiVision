"""Configuración centralizada del sistema.

Único punto de lectura de variables de entorno y rutas. Los Casos de Uso
NO deben leer directamente del entorno: reciben un `Settings` por DI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.utils import resource_path
from src.core.utils.paths import user_data_path

try:
    from src.infrastructure.storage.model_downloader import get_model_path
except ImportError:
    # Fallback dev: usa resource_path si el downloader no esta disponible
    def get_model_path(filename: str, dest_dir=None) -> str:  # type: ignore[no-redef]
        return resource_path(f"models/{filename}")


@dataclass(frozen=True)
class ModelPaths:
    # Usa get_model_path para descarga selectiva (APPDATA/models en frozen).
    # Live = solo YOLOv8-vehiculos; yolo_plate se usa 1x por infractor en
    # post-proceso. La lectura la hace la API de Plate Recognizer.
    yolo_vehicle: str = field(default_factory=lambda: get_model_path("yolov8n.pt"))
    yolo_plate: str = field(default_factory=lambda: get_model_path("license_plate_detector.pt"))
    fsrcnn: str = field(default_factory=lambda: get_model_path("FSRCNN_x3.pb"))


@dataclass(frozen=True)
class DatabaseSettings:
    backend: str = field(default_factory=lambda: os.getenv("INFRACTI_DB_BACKEND", "sqlite"))
    sqlite_path: str = field(default_factory=lambda: user_data_path("data/infractions.sqlite"))
    mysql_url: str = field(default_factory=lambda: os.getenv("INFRACTI_MYSQL_URL", ""))


@dataclass(frozen=True)
class OCRSettings:
    backend: str = field(default_factory=lambda: os.getenv("INFRACTI_OCR_BACKEND", "lprnet"))
    min_confidence: float = 0.55
    regional_context: str = "Trujillo"


@dataclass(frozen=True)
class DetectionSettings:
    confidence_threshold: float = 0.30
    half_precision: bool = True


@dataclass(frozen=True)
class PerformanceSettings:
    """Perfil de rendimiento para no saturar la CPU (i3-9100F 4C/4T).

    Todo overridable por entorno para no recompilar:
      IV_TORCH_THREADS, IV_CV_THREADS, IV_SKIP_GREEN, IV_SKIP_RED,
      IV_IMGSZ (0=auto), IV_NIGHT_CHECK_INTERVAL, IV_ENABLE_RECTIFIER_LIVE,
      IV_DISPLAY_FPS, INFRACTI_TRACKER (centroid|deepsort).
    """

    profile: str = field(default_factory=lambda: os.getenv("IV_PERF", "i3_9100F"))
    torch_threads: int = field(default_factory=lambda: int(os.getenv("IV_TORCH_THREADS", "2")))
    cv_threads: int = field(default_factory=lambda: int(os.getenv("IV_CV_THREADS", "2")))
    skip_green: int = field(default_factory=lambda: int(os.getenv("IV_SKIP_GREEN", "5")))
    skip_red: int = field(default_factory=lambda: int(os.getenv("IV_SKIP_RED", "2")))
    imgsz_override: int = field(default_factory=lambda: int(os.getenv("IV_IMGSZ", "0")))
    night_check_interval: int = field(
        default_factory=lambda: int(os.getenv("IV_NIGHT_CHECK_INTERVAL", "15"))
    )
    enable_rectifier_live: bool = field(
        default_factory=lambda: os.getenv("IV_ENABLE_RECTIFIER_LIVE", "0") == "1"
    )
    display_fps: int = field(default_factory=lambda: int(os.getenv("IV_DISPLAY_FPS", "15")))
    tracker: str = field(default_factory=lambda: os.getenv("INFRACTI_TRACKER", "centroid"))


@dataclass(frozen=True)
class StoragePaths:
    videos: str = field(default_factory=lambda: user_data_path("data/videos"))
    images: str = field(default_factory=lambda: user_data_path("data/images"))
    evidences: str = field(default_factory=lambda: user_data_path("data/evidences"))


@dataclass(frozen=True)
class ConfigPaths:
    camera_config: str = field(default_factory=lambda: resource_path("config/camera_config.json"))
    zones: str = field(default_factory=lambda: resource_path("config/zones.json"))
    presets: str = field(default_factory=lambda: resource_path("config/time_presets.json"))


@dataclass(frozen=True)
class Settings:
    """Settings inmutables: viajan por DI a casos de uso e infraestructura."""
    models: ModelPaths = field(default_factory=ModelPaths)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    storage: StoragePaths = field(default_factory=StoragePaths)
    config_files: ConfigPaths = field(default_factory=ConfigPaths)

    def ensure_directories(self) -> None:
        for p in (self.storage.videos, self.storage.images, self.storage.evidences):
            Path(p).mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Factoría única. Cualquier ajuste por entorno se cablea aquí."""
    s = Settings()
    s.ensure_directories()
    return s
