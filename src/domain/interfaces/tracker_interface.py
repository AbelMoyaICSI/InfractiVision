"""Puerto de tracking (DeepSORT u otro)."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.domain.entities import Vehicle


@runtime_checkable
class TrackerPort(Protocol):
    def update(
        self, frame_bgr: np.ndarray, detections: Sequence[Vehicle]
    ) -> Sequence[Vehicle]:
        """Recibe detecciones del frame y devuelve los Vehicle con `track_id` asignado."""
        ...

    def reset(self) -> None: ...
