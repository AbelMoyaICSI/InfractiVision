"""Utilidades genéricas (paths, geometría) para la nueva arquitectura.

Este paquete reexporta:
    * `resource_path` — compatible con PyInstaller (definido aquí para que las
      capas nuevas no dependan de `src.path_helper`).
    * `segments_intersect`, `bbox_bottom_center`, `Point` — geometría 2D
      usada por la regla de cruce de línea en domain/services.

Los submódulos legacy (`paths`, `timestamp`) siguen disponibles como antes:
    from src.core.utils.timestamp import TimestampUpdater   # ← legacy OK
    from src.core.utils import resource_path                 # ← nuevo OK
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

# === Resolución de recursos (compatible con PyInstaller) ===
if hasattr(sys, "_MEIPASS"):
    _BASE = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    _BASE = Path(".").resolve()


def resource_path(relative_path: str | os.PathLike) -> str:
    """Ruta absoluta a un recurso, en desarrollo o dentro del .exe."""
    return str((_BASE / Path(relative_path)).resolve())


# === Geometría 2D para detección de cruce de línea ===
Point = Tuple[float, float]


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """¿Los segmentos p1-p2 y p3-p4 se cruzan? (algoritmo CCW clásico)."""

    def ccw(a: Point, b: Point, c: Point) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def bbox_bottom_center(bbox: tuple[int, int, int, int]) -> Point:
    """Centro inferior del bbox (mejor punto de contacto con el suelo)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


__all__ = [
    "resource_path",
    "segments_intersect",
    "bbox_bottom_center",
    "Point",
]
