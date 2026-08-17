"""Entidad Vehicle. Es un dato puro: nada de OpenCV/YOLO aquí."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.constants import VEHICLE_CLASS_NAMES


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class Vehicle:
    """Vehículo detectado y (opcionalmente) trackeado.

    `track_id` se asigna después por el tracker (DeepSORT). Hasta entonces es None.
    """
    bbox: BoundingBox
    class_id: int
    confidence: float
    track_id: Optional[int] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    has_crossed_line: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        return VEHICLE_CLASS_NAMES.get(self.class_id, f"Vehiculo_{self.class_id}")
