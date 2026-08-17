"""Caso de uso: detectar vehículos en un frame.

Orquesta el `VehicleDetectorPort`. NO conoce YOLO ni OpenCV directamente.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.logger import get_logger
from src.domain.entities import Vehicle
from src.domain.interfaces import VehicleDetectorPort

log = get_logger("usecase.detect_vehicle")


class DetectVehicleUseCase:
    def __init__(self, detector: VehicleDetectorPort):
        self._detector = detector

    def execute(self, frame_bgr: np.ndarray) -> Sequence[Vehicle]:
        vehicles = self._detector.detect(frame_bgr)
        log.debug("Vehículos detectados: %d", len(vehicles))
        return vehicles
