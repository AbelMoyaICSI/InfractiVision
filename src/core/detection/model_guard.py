"""Guarda de carga de modelos de IA (multihilo).

torch 1.13 + CUDA NO es seguro para cargar varios modelos a la vez
(torch.load + .to(cuda) + fuse) desde hilos distintos; puede provocar
SIGSEGV/Abort. Todos los constructores de detectores adquieren este lock
durante su inicialización pesada.

Es un RLock porque los constructores se anidan: LPRNetPredictor crea un
PlateDetector internamente.
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