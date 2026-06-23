"""Adapter PaddleOCR (lazy import). Implementa `OCRReaderPort`."""
from __future__ import annotations

import numpy as np

from src.core.exceptions import OCRError
from src.core.logger import get_logger
from src.domain.interfaces import OCRReaderPort

log = get_logger("infra.ocr.paddle")


class PaddleOCRReader(OCRReaderPort):
    def __init__(self, lang: str = "en", use_gpu: bool = True):
        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        except Exception as e:
            raise OCRError(f"No se pudo inicializar PaddleOCR: {e}") from e

    def read_plate(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        try:
            result = self._ocr.ocr(plate_bgr, cls=True)
        except Exception as e:
            raise OCRError(f"Falla PaddleOCR: {e}") from e
        if not result or not result[0]:
            return "", 0.0
        best = max(result[0], key=lambda r: r[1][1])
        text = "".join(ch for ch in best[1][0].upper() if ch.isalnum() or ch == "-")
        return text, float(best[1][1])
