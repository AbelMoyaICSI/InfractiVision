"""Extracción y persistencia de frames como evidencia (.jpg)."""
from __future__ import annotations

import os

import cv2  # type: ignore
import numpy as np

from src.core.exceptions import VideoSourceError
from src.domain.interfaces import FrameExtractorPort


class OpenCVFrameExtractor(FrameExtractorPort):
    def extract(self, frame_bgr: np.ndarray, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, frame_bgr):
            raise VideoSourceError(f"No se pudo escribir frame en {output_path}")
        return output_path
