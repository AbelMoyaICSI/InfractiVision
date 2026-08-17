"""Adapter EasyOCR (lazy import). Implementa `OCRReaderPort`."""
from __future__ import annotations

import numpy as np

from src.core.exceptions import OCRError
from src.core.logger import get_logger
from src.domain.interfaces import OCRReaderPort

log = get_logger("infra.ocr.easyocr")


class EasyOCRReader(OCRReaderPort):
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = True):
        try:
            import easyocr  # type: ignore

            self._reader = easyocr.Reader(list(languages), gpu=gpu)
        except Exception as e:
            raise OCRError(f"No se pudo inicializar EasyOCR: {e}") from e

    def read_plate(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        try:
            results = self._reader.readtext(plate_bgr, detail=1)
        except Exception as e:
            raise OCRError(f"Falla EasyOCR: {e}") from e
        if not results:
            return "", 0.0
        # Tomamos el resultado de mayor confianza
        best = max(results, key=lambda r: r[2])
        text = "".join(ch for ch in best[1].upper() if ch.isalnum() or ch == "-")
        return text, float(best[2])
