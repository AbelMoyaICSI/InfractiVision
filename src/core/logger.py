"""Logger único reutilizable. Formato consistente sin acoplar a 3rd-parties."""
from __future__ import annotations

import logging
import sys
from logging import Logger

_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s :: %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("infractivision")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    # En frozen (windowed, console=False) stdout es invisible: persistir a
    # %APPDATA%/InfractiVision/logs/infractivision.log (best-effort, sin romper).
    try:
        from pathlib import Path as _P

        from src.core.utils.paths import LOGS_DIR

        _P(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(_P(LOGS_DIR) / "infractivision.log"), encoding="utf-8")
        _fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(_fh)
    except Exception:
        pass
    root.propagate = False
    _configured = True


def get_logger(name: str) -> Logger:
    _configure_root()
    return logging.getLogger(f"infractivision.{name}")
