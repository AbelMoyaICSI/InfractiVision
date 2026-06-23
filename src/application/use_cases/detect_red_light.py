"""Caso de uso: leer el estado actual del semáforo."""
from __future__ import annotations

from src.domain.entities import TrafficLightState
from src.domain.interfaces import TrafficLightDetectorPort


class DetectRedLightUseCase:
    def __init__(self, traffic_light_detector: TrafficLightDetectorPort):
        self._detector = traffic_light_detector

    def execute(self) -> TrafficLightState:
        return self._detector.current_state()

    def is_red(self) -> bool:
        return self.execute().is_red
