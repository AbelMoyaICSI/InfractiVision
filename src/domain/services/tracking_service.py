"""Servicio de tracking: orquesta el TrackerPort mantieniendo
historial mínimo (posición previa por track_id) para la regla de cruce.

NO conoce DeepSORT: solo el TrackerPort.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.utils import Point
from src.domain.entities import Vehicle
from src.domain.interfaces import TrackerPort


class TrackingService:
    def __init__(self, tracker: TrackerPort):
        self._tracker = tracker
        self._previous_positions: dict[int, Point] = {}

    def track(
        self, frame_bgr: np.ndarray, detections: Sequence[Vehicle]
    ) -> Sequence[Vehicle]:
        return self._tracker.update(frame_bgr, detections)

    def previous_position(self, track_id: int) -> Point | None:
        return self._previous_positions.get(track_id)

    def remember_position(self, track_id: int, pos: Point) -> None:
        self._previous_positions[track_id] = pos

    def reset(self) -> None:
        self._previous_positions.clear()
        self._tracker.reset()
