"""Helper multiplataforma para asignar el icono a una ventana Tk.

En Windows `iconbitmap` con `.ico` funciona perfecto; en Linux/macOS lanza
`TclError`. Este helper:

    1. Intenta `.ico` (rápido y nativo en Windows).
    2. Si falla, busca un `.png` equivalente y usa `iconphoto` (portable).
    3. Si todo falla, no hace nada (no es bloqueante).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from src.core.utils import resource_path


def set_window_icon(window, ico_relpath: str = "img/icon.ico",
                    png_relpath: Optional[str] = "img/InfractiVision-logo.png") -> bool:
    """Asigna el icono a `window` (Tk root, Toplevel, dialog).

    Devuelve True si pudo asignar algún icono, False si no.
    """
    # 1) .ico (Windows nativo)
    try:
        ico_path = resource_path(ico_relpath)
        if os.path.exists(ico_path) and sys.platform.startswith("win"):
            window.iconbitmap(ico_path)
            return True
    except Exception:
        pass

    # 2) .png portable (Linux/macOS y Windows fallback)
    if png_relpath:
        try:
            from tkinter import PhotoImage
            png_path = resource_path(png_relpath)
            if os.path.exists(png_path):
                img = PhotoImage(file=png_path)
                # Mantener referencia para evitar GC
                window._icon_photo_ref = img  # type: ignore[attr-defined]
                window.iconphoto(False, img)
                return True
        except Exception:
            pass

    return False


__all__ = ["set_window_icon"]
