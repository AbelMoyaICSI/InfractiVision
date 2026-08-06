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
    root.propagate = False
    _configured = True


def get_logger(name: str) -> Logger:
    _configure_root()
    return logging.getLogger(f"infractivision.{name}")
