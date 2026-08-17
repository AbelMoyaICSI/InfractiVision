"""Puertos (Protocols) para detección. La infraestructura los implementa."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.domain.entities import BoundingBox, TrafficLightState, Vehicle


@runtime_checkable
class VehicleDetectorPort(Protocol):
    """Detector de vehículos sobre un frame BGR."""

    def detect(self, frame_bgr: np.ndarray) -> Sequence[Vehicle]: ...


@runtime_checkable
class PlateDetectorPort(Protocol):
    """Detector de la región de placa dentro de la bbox del vehículo."""

    def detect_plate(
        self, frame_bgr: np.ndarray, vehicle_bbox: BoundingBox
    ) -> BoundingBox | None: ...


@runtime_checkable
class TrafficLightDetectorPort(Protocol):
    """Detector del color actual del semáforo (vía visión o panel virtual)."""

    def current_state(self) -> TrafficLightState: ...
