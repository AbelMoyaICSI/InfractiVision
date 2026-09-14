"""Guarda de carga de modelos de IA (multihilo).

torch + CUDA NO es seguro para cargar varios modelos a la vez
(torch.load + .to(cuda) + fuse) desde hilos distintos; puede provocar
SIGSEGV/Abort. Todos los constructores de detectores adquieren este lock
durante su inicialización pesada.

En vivo solo hay detección (YOLO vehículos + YOLO placas); la lectura OCR
la hace la API de Plate Recognizer en la revisión final.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

MODEL_LOAD_LOCK = threading.RLock()


def serialized(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Serializa una función (constructor) bajo el lock global de carga."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with MODEL_LOAD_LOCK:
            return fn(*args, **kwargs)

    return wrapper


def free_torch_memory() -> None:
    """Libera RAM/VRAM retenida por torch tras soltar un modelo.

    Idempotente y nunca lanza: hace `gc.collect()` + `cuda.empty_cache()`
    ( + `ipc_collect` si existe). Sin esto, al salir de Foto Rojo los pesos
    YOLO quedan huérfanos pero la VRAM sigue reservada por el caching
    allocator y cada re-entrada suma memoria.
    """
    try:
        import gc as _gc

        _gc.collect()
    except Exception:
        pass
    try:
        import torch as _torch

        try:
            if _torch.cuda.is_available():
                try:
                    _torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    _torch.cuda.ipc_collect()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


def release_detector(det: Any) -> None:
    """Libera un detector si expone `release()`; si no, suelta `.model`.

    Helper para no duplicar try/except en cada `shutdown`. Nunca lanza.
    """
    if det is None:
        return
    try:
        release = getattr(det, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
            return
    except Exception:
        pass
    try:
        model = getattr(det, "_detector", None)
        if model is not None:
            release_detector(model)
    except Exception:
        pass
    try:
        if hasattr(det, "model"):
            try:
                det.model = None
            except Exception:
                pass
    except Exception:
        pass
    try:
        free_torch_memory()
    except Exception:
        pass