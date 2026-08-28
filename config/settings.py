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
    # Usa get_model_path para descarga selectiva (APPDATA/models en frozen)
    yolo_vehicle: str = field(default_factory=lambda: get_model_path("yolov8n.pt"))
    yolo_plate: str = field(default_factory=lambda: get_model_path("license_plate_detector.pt"))
    lprnet_master: str = field(default_factory=lambda: get_model_path("LPRNet_Peru_MASTER_FINAL.pth"))
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
