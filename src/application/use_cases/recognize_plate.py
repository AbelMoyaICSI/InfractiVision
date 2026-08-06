"""Caso de uso: reconocer la placa de un vehículo recortado.

Pipeline: PlateDetector (recorta region de placa) → OCRReader (lee texto).
"""
from __future__ import annotations

import numpy as np

from src.core.exceptions import OCRError
from src.core.logger import get_logger
from src.domain.entities import Vehicle
from src.domain.interfaces import OCRReaderPort, PlateDetectorPort

log = get_logger("usecase.recognize_plate")


class RecognizePlateUseCase:
    def __init__(
        self,
        plate_detector: PlateDetectorPort,
        ocr_reader: OCRReaderPort,
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

        try:
            text, conf = self._ocr.read_plate(plate_crop)
        except Exception as e:
            raise OCRError(str(e)) from e

        if conf >= self._min_conf and text:
            vehicle.plate_text = text
            vehicle.plate_confidence = conf
            log.info("Placa reconocida: %s (%.2f) track=%s", text, conf, vehicle.track_id)
        return vehicle
