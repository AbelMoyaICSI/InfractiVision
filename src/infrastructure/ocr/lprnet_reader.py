"""Adapter del LPRNet master existente.

Reutiliza la función `recognize_plate` de `src.core.ocr.recognizer` SIN tocar
el modelo, los pesos ni el algoritmo. Solo traduce la salida al contrato
`OCRReaderPort: (texto, confianza)`.
"""
from __future__ import annotations

import numpy as np

from src.core.exceptions import OCRError
from src.core.logger import get_logger
from src.domain.interfaces import OCRReaderPort

log = get_logger("infra.ocr.lprnet")


class LPRNetReader(OCRReaderPort):
    def __init__(self, regional_context: str = "Trujillo"):
        self._regional_context = regional_context
        # Pre-cargamos el predictor (singleton) para evitar costo en el 1er frame.
        try:
            from src.core.ocr.recognizer import get_lprnet_predictor

            get_lprnet_predictor()
        except Exception as e:
            log.warning("No se pudo precargar LPRNet: %s", e)

    def read_plate(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        from src.core.ocr.recognizer import recognize_plate

        try:
            text, conf = recognize_plate(
                plate_bgr,
                regional_context=self._regional_context,
            )
        except Exception as e:
            raise OCRError(f"Falla LPRNet: {e}") from e
        return text or "", float(conf or 0.0)
