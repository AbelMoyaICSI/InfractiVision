"""Adapter YOLO. Implementa `VehicleDetectorPort` reutilizando la lógica
algorítmica YA EXISTENTE en `src.core.detection.vehicle_detector.VehicleDetector`.

OBJETIVO: NO tocar pesos ni lógica del modelo, solo envolverlos.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.exceptions import DetectorError
from src.core.logger import get_logger
from src.domain.entities import BoundingBox, Vehicle
from src.domain.interfaces import VehicleDetectorPort

log = get_logger("infra.yolo")


class YoloVehicleDetector(VehicleDetectorPort):
    """Implementación concreta del puerto, usando YOLOv8 (Ultralytics)."""

    def __init__(self, model_path: str):
        # Importación perezosa para no penalizar el arranque del proceso
        # cuando este detector no se usa (Bean lifecycle).
        from src.core.detection.vehicle_detector import VehicleDetector

        try:
            self._detector = VehicleDetector(model_path=model_path)
        except Exception as e:
            raise DetectorError(f"No se pudo cargar YOLO desde {model_path}: {e}") from e
        log.info("YoloVehicleDetector listo (modelo=%s)", model_path)

    def detect(self, frame_bgr: np.ndarray) -> Sequence[Vehicle]:
        try:
            raw = self._detector.detect(frame_bgr, draw=False)
        except Exception as e:
            raise DetectorError(f"Falla durante detección YOLO: {e}") from e

        out: list[Vehicle] = []
        for x1, y1, x2, y2, cls_id in raw:
            out.append(
                Vehicle(
                    bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                    class_id=int(cls_id),
                    confidence=float(self._detector.conf_threshold),
                )
            )
        return out
