"""Grabador de video de evidencia. Implementa `RecorderPort`."""
from __future__ import annotations

import os

import cv2  # type: ignore
import numpy as np

from src.core.exceptions import VideoSourceError
from src.core.logger import get_logger
from src.domain.interfaces import RecorderPort

log = get_logger("infra.video.recorder")


class OpenCVRecorder(RecorderPort):
    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._path: str = ""

    def start(self, output_path: str, fps: float, size: tuple[int, int]) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        if not writer.isOpened():
            raise VideoSourceError(f"No se pudo iniciar grabación en {output_path}")
        self._writer = writer
        self._path = output_path
        log.info("Grabando evidencia en %s (%.1f fps)", output_path, fps)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.write(frame_bgr)

    def stop(self) -> str:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self._path
