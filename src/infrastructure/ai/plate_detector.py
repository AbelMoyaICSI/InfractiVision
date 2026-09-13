"""Adapter del detector de placas. Envuelve `PlateDetector` existente."""
from __future__ import annotations

import numpy as np

from src.core.exceptions import DetectorError
from src.core.logger import get_logger
from src.domain.entities import BoundingBox
from src.domain.interfaces import PlateDetectorPort

log = get_logger("infra.plate_detector")


class YoloPlateDetector(PlateDetectorPort):
    def __init__(self, model_path: str | None = None):
        from src.core.detection.plate_detector import PlateDetector

        try:
            self._detector = PlateDetector(model_path=model_path)
        except Exception as e:
            raise DetectorError(f"No se pudo cargar PlateDetector: {e}") from e
        log.info("YoloPlateDetector listo")

    def detect_plate(
        self, frame_bgr: np.ndarray, vehicle_bbox: BoundingBox
    ) -> BoundingBox | None:
        x1, y1, x2, y2 = vehicle_bbox.as_tuple()
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            detections = self._detector.detect(crop, conf=0.4, draw=False)
        except Exception as e:
            log.warning("Falla detect_plate: %s", e)
            return None

        if not detections:
            return None

        # `PlateDetector.detect` retorna lista de tuplas/listas. Tomamos el primero.
        first = detections[0]
        # Soportamos formato [x1,y1,x2,y2,...] devuelto por la implementación legada
        if hasattr(first, "__len__") and len(first) >= 4:
            px1, py1, px2, py2 = map(int, first[:4])
            return BoundingBox(
                x1=x1 + px1, y1=y1 + py1, x2=x1 + px2, y2=y1 + py2
            )
        return None

    def detect_batch(
        self, crops: list[np.ndarray], conf: float = 0.40
    ) -> list[BoundingBox | None]:
        """Una sola inferencia GPU para N crops (evita N launches CUDA).

        Los boxes se devuelven en coords locales del crop; el caller suma el
        offset del vehículo. Retorna None por crop sin placa.
        """
        if not crops:
            return []
        try:
            per_crop = self._detector.detect_batch_quadrants(
                list(crops), conf=conf, classes=[0]
            )
        except Exception as e:
            log.warning("Falla detect_batch: %s", e)
            return [None for _ in crops]
        out: list[BoundingBox | None] = []
        for dets in per_crop:
            if not dets:
                out.append(None)
                continue
            px1, py1, px2, py2 = map(int, dets[0][:4])
            out.append(BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2))
        return out
