"""Puerto OCR. Implementaciones: LPRNet, EasyOCR, PaddleOCR."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OCRReaderPort(Protocol):
    def read_plate(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        """Devuelve (texto_formateado, confianza)."""
        ...
