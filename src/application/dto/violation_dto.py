"""DTOs para comunicación presentation <-> application.

Son @dataclass simples y serializables. NO contienen lógica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.domain.entities import Violation


@dataclass(frozen=True, slots=True)
class ViolationDTO:
    plate_text: str
    plate_confidence: float
    vehicle_class_id: int
    track_id: int
    occurred_at: datetime
    violation_type: str
    image_path: Optional[str]
    video_path: Optional[str]
    ticket_number: Optional[str]

    @classmethod
    def from_entity(cls, v: Violation) -> "ViolationDTO":
        ev = v.evidence
        return cls(
            plate_text=v.plate_text,
            plate_confidence=v.plate_confidence,
            vehicle_class_id=v.vehicle_class_id,
            track_id=v.track_id,
            occurred_at=v.occurred_at,
            violation_type=v.violation_type,
            image_path=ev.image_path if ev else None,
            video_path=ev.video_path if ev else None,
            ticket_number=v.ticket_number,
        )


@dataclass(frozen=True, slots=True)
class FrameProcessingResultDTO:
    """Resultado por frame, consumido por la GUI para refrescar la vista."""
    vehicle_count: int
    traffic_light_state: str
    new_violations: tuple[ViolationDTO, ...] = field(default_factory=tuple)
