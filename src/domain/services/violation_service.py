"""Reglas de negocio de infracciones (sin I/O).

REGLA DE ORO: Una infracción de luz roja existe SI y SOLO SI
    1. el vehículo cruzó la línea de detención y
    2. el semáforo estaba en estado RED en el momento del cruce.
"""
from __future__ import annotations

from typing import Sequence

from src.core.utils import Point, segments_intersect
from src.domain.entities import TrafficLight, Vehicle


class ViolationService:
    """Servicio puro de dominio: stateless, no depende de nada externo."""

    def __init__(self, stop_line: tuple[Point, Point]):
        self._stop_line = stop_line  # (p1, p2) en coordenadas de imagen

    # ─── Cruce de línea ──────────────────────────────────────────────────
    def has_crossed_stop_line(
        self, prev_pos: Point, current_pos: Point
    ) -> bool:
        return segments_intersect(prev_pos, current_pos, *self._stop_line)

    # ─── Regla central ───────────────────────────────────────────────────
    def is_red_light_violation(
        self, vehicle: Vehicle, traffic_light: TrafficLight
    ) -> bool:
        return vehicle.has_crossed_line and traffic_light.state.is_red

    # ─── Filtrado batch ──────────────────────────────────────────────────
    def select_violators(
        self,
        vehicles: Sequence[Vehicle],
        traffic_light: TrafficLight,
    ) -> list[Vehicle]:
        if not traffic_light.state.is_red:
            return []
        return [v for v in vehicles if v.has_crossed_line]
