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

# Forzar compatibilidad con arquitecturas CUDA más nuevas como sm_120
# en instalaciones que puedan recompilar kernels o cargar extensiones dinámicamente.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0;12.1;12.6;12.7;12.8;13.0")

import torch

# Compatibilidad PyTorch 2.6+: permitir carga de modelos Ultralytics y objetos DetectionModel.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

try:
    from ultralytics.yolo.engine.model import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    # Si la versión de ultralytics o la API de PyTorch no lo permiten, ignorar.
    pass

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


def _detect_cuda_compatibility() -> bool:
    """Verifica si la GPU CUDA es realmente usable para el runtime actual."""
    import torch

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
        print(f"[core.utils] CUDA no usable: {e}")
        return False


def get_default_device() -> 'torch.device':
    """Devuelve el dispositivo predeterminado global: GPU usable o CPU.

    Ejecuta una comprobación activa de PyTorch en cada llamada para garantizar
    que el estado GPU se actualiza desde el primer momento y no depende de un
    valor estático obsoleto.
    """
    import torch
    global USE_CUDA

    USE_CUDA = _detect_cuda_compatibility()
    return torch.device("cuda:0") if USE_CUDA else torch.device("cpu")


USE_CUDA = _detect_cuda_compatibility()

__all__ = [
    "resource_path",
    "segments_intersect",
    "bbox_bottom_center",
    "Point",
    "get_default_device",
    "USE_CUDA",
]
