"""Configuración centralizada del sistema.

Único punto de lectura de variables de entorno y rutas. Los Casos de Uso
NO deben leer directamente del entorno: reciben un `Settings` por DI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.utils import resource_path


@dataclass(frozen=True)
class ModelPaths:
    yolo_vehicle: str = field(default_factory=lambda: resource_path("models/yolov8n.pt"))
    yolo_plate: str = field(default_factory=lambda: resource_path("models/license_plate_detector.pt"))
    lprnet_master: str = field(default_factory=lambda: resource_path("models/LPRNet_Peru_MASTER_FINAL.pth"))
    fsrcnn: str = field(default_factory=lambda: resource_path("models/FSRCNN_x3.pb"))


@dataclass(frozen=True)
class DatabaseSettings:
    backend: str = field(default_factory=lambda: os.getenv("INFRACTI_DB_BACKEND", "sqlite"))
    sqlite_path: str = field(default_factory=lambda: resource_path("data/infractions.sqlite"))
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
    videos: str = field(default_factory=lambda: resource_path("data/videos"))
    images: str = field(default_factory=lambda: resource_path("data/images"))
    evidences: str = field(default_factory=lambda: resource_path("data/evidences"))


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
