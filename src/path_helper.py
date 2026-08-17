# src/path_helper.py
from __future__ import annotations
import os
import sys
from pathlib import Path

# Calculamos el BASE una sola vez
if hasattr(sys, "_MEIPASS"):
    # Ejecutable PyInstaller: los datos van empaquetados en esta carpeta temporal
    BASE = Path(sys._MEIPASS)
else:
    # Desarrollo: raíz del proyecto (directorio actual al ejecutar)
    BASE = Path(".").resolve()

def resource_path(relative_path: str | os.PathLike) -> str:
    """
    Devuelve una ruta absoluta a un recurso para leer/abrir archivos.
    Funciona en desarrollo y dentro del ejecutable PyInstaller.
    """
    return str((BASE / Path(relative_path)).resolve())
