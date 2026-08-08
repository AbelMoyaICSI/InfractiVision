"""Entidad TrafficLight (semáforo). Sin Tk ni hilos: solo el estado."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.core.constants import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_YELLOW,
)


class TrafficLightState(str, Enum):
    GREEN = TRAFFIC_LIGHT_GREEN
    YELLOW = TRAFFIC_LIGHT_YELLOW
    RED = TRAFFIC_LIGHT_RED

    @property
    def is_red(self) -> bool:
        return self is TrafficLightState.RED


@dataclass(slots=True)
class TrafficLight:
    state: TrafficLightState = TrafficLightState.GREEN
    green_seconds: int = 12
    yellow_seconds: int = 2
    red_seconds: int = 10
    last_change_at: datetime = field(default_factory=datetime.now)

    def cycle_durations(self) -> dict[TrafficLightState, int]:
        return {
            TrafficLightState.GREEN: self.green_seconds,
            TrafficLightState.YELLOW: self.yellow_seconds,
            TrafficLightState.RED: self.red_seconds,
        }
