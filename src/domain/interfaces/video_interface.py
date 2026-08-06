"""Puertos de video: lectura, extracción de frames, grabación de evidencia."""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VideoSourcePort(Protocol):
    def open(self, source: str | int) -> None: ...
    def read(self) -> tuple[bool, np.ndarray]: ...
    def release(self) -> None: ...
    def __iter__(self) -> Iterator[np.ndarray]: ...

    @property
    def fps(self) -> float: ...

    @property
    def is_open(self) -> bool: ...


@runtime_checkable
class FrameExtractorPort(Protocol):
    def extract(self, frame_bgr: np.ndarray, output_path: str) -> str:
        """Guarda el frame y devuelve la ruta absoluta donde quedó."""
        ...


@runtime_checkable
class RecorderPort(Protocol):
    def start(self, output_path: str, fps: float, size: tuple[int, int]) -> None: ...
    def write(self, frame_bgr: np.ndarray) -> None: ...
    def stop(self) -> str: ...
