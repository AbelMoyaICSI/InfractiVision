"""Lector de video basado en OpenCV. Implementa `VideoSourcePort`."""
from __future__ import annotations

from typing import Iterator

import cv2  # type: ignore
import numpy as np

from src.core.exceptions import VideoSourceError
from src.core.logger import get_logger
from src.domain.interfaces import VideoSourcePort

log = get_logger("infra.video.reader")


class OpenCVVideoReader(VideoSourcePort):
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0

    def open(self, source: str | int) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise VideoSourceError(f"No se pudo abrir la fuente: {source}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        log.info("Video abierto (%s, %.1f fps)", source, self._fps)

    def read(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            return False, np.empty((0,))
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
