"""Entidad Violation (infracción)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.core.constants import VIOLATION_RED_LIGHT


@dataclass(frozen=True, slots=True)
class ViolationEvidence:
    """Rutas a los archivos de evidencia (foto y video)."""
    image_path: str
    video_path: Optional[str] = None


@dataclass(slots=True)
class Violation:
    """Infracción de tránsito detectada por el sistema."""
    plate_text: str
    plate_confidence: float
    vehicle_class_id: int
    track_id: int
    occurred_at: datetime = field(default_factory=datetime.now)
    violation_type: str = VIOLATION_RED_LIGHT
    evidence: Optional[ViolationEvidence] = None
    ticket_number: Optional[str] = None
