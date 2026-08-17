"""Compatibilidad torch/ultralytics compartida.

PyTorch 2.6+ cambió el default de `torch.load` a `weights_only=True`, y
ultralytics 8.x necesita `weights_only=False` para des-serializar sus modelos.
El parche se aplica UNA vez por proceso, justo antes de cargar cualquier modelo
YOLO, y es idempotente. Se mantiene en módulo propio para no duplicarlo en
`vehicle_detector`, `plate_detector` y `anpr`, y para permitir import lazy de
torch (el arranque de la GUI no paga el costo de importarlo).
"""
from __future__ import annotations

_patched = False


def ensure_torch_compat() -> None:
    global _patched
    if _patched:
        return
    import torch

    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    _patched = True