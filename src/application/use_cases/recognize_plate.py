"""Caso de uso: localizar la placa de un vehículo recortado (SOLO detección).

En vivo NO se lee texto: se detecta el bbox de la placa con YOLO y se
guarda en `vehicle.extras["plate_bbox"]`. La lectura OCR la hace la API de
Plate Recognizer al final, en `PlateReviewWindow` (un crop por evidencia).
LPRNet fue eliminado del proyecto.
"""
from __future__ import annotations

import numpy as np

from src.core.logger import get_logger
from src.domain.entities import Vehicle
from src.domain.interfaces import OCRReaderPort, PlateDetectorPort

log = get_logger("usecase.recognize_plate")


class RecognizePlateUseCase:
    def __init__(
        self,
        plate_detector: PlateDetectorPort,
        ocr_reader: OCRReaderPort | None = None,
        min_confidence: float = 0.55,
    ):
        self._plate_detector = plate_detector
        self._ocr = ocr_reader
        self._min_conf = min_confidence

    def execute(self, frame_bgr: np.ndarray, vehicle: Vehicle) -> Vehicle:
        plate_box = self._plate_detector.detect_plate(frame_bgr, vehicle.bbox)
        if plate_box is None:
            log.debug("Placa no encontrada para track_id=%s", vehicle.track_id)
            return vehicle

        plate_crop = frame_bgr[plate_box.y1:plate_box.y2, plate_box.x1:plate_box.x2]
        if plate_crop.size == 0:
            return vehicle

        # Solo-detección: se conserva el bbox para la evidencia; el texto lo
        # resuelve la API en la revisión final.
        vehicle.extras["plate_bbox"] = plate_box.as_tuple()
        log.debug("Placa localizada track=%s bbox=%s", vehicle.track_id, plate_box.as_tuple())
        return vehicle
