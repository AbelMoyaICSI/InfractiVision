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

# RTX 5050 Blackwell sm_120 requiere CUDA 12.4+. Forzar arch list para
# instalaciones que recompilen kernels o carguen extensiones dinámicas.
# Debe setearse ANTES de importar torch (por eso va aquí, sin importar torch aún).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1;12.6;12.7;12.8;13.0")

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


# === Detección real de GPU (probe activo) — port de windows_machine_owner ===
# No se importa torch a nivel de módulo para no penalizar el arranque de la GUI.
# Cada llamada hace probe real (tensor + synchronize) para detectar sm_120 etc.
def _detect_cuda_compatibility() -> bool:
    """Verifica si la GPU CUDA es realmente usable (no solo reportada)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        device_index = 0
        try:
            device_index = torch.cuda.current_device()
        except Exception:
            device_index = 0
        props = torch.cuda.get_device_properties(device_index)
        if props is None:
            return False
        device = torch.device(f"cuda:{device_index}")
        tensor = torch.tensor([1.0], device=device)
        _ = tensor * 1.0
        torch.cuda.synchronize(device)
        print(f"[core.utils] CUDA usable detectada: {props.name} (cap={props.major}.{props.minor})")
        return True
    except Exception as e:
        # Silenciar "No module named torch" en CI sin GPU; log solo si torch existe
        try:
            import torch as _t

            if _t.cuda.is_available():
                print(f"[core.utils] CUDA no usable: {e}")
        except Exception:
            pass
        return False


def get_default_device():
    """Devuelve torch.device('cuda:0') si la GPU pasa el probe, else cpu. Lazy + cache."""
    global USE_CUDA
    # Si ya se evaluo, reutilizar sin re-probear
    if USE_CUDA is not None:
        try:
            import torch

            return torch.device("cuda:0") if USE_CUDA else torch.device("cpu")
        except ImportError:
            return None
    try:
        import torch

        USE_CUDA = _detect_cuda_compatibility()
        return torch.device("cuda:0") if USE_CUDA else torch.device("cpu")
    except ImportError:
        USE_CUDA = False
        return None
    except Exception:
        USE_CUDA = False
        try:
            import torch as _t

            return _t.device("cpu")
        except Exception:
            return None


# Lazy: no se evalua probe en import (evita 2s bloqueo antes de Tk).
# Se resuelve en primer get_default_device() y se cachea en USE_CUDA.
USE_CUDA: bool | None = None


__all__ = [
    "resource_path",
    "segments_intersect",
    "bbox_bottom_center",
    "Point",
    "get_default_device",
    "USE_CUDA",
]
