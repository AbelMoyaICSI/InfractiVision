# src/core/utils/paths.py
"""Resolución multiplataforma de rutas de datos del usuario.

Cumple convenciones nativas en cada SO:
    Windows: %APPDATA%\\InfractiVision         (ej. C:\\Users\\X\\AppData\\Roaming)
    macOS:   ~/Library/Application Support/InfractiVision
    Linux:   $XDG_CONFIG_HOME/InfractiVision   (default: ~/.config/InfractiVision)
"""
from pathlib import Path
import sys, os, json, shutil


def resource_path(rel: str) -> str:
    # Carpeta temporal si es onefile, o carpeta del exe si es onedir
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(getattr(sys, "frozen", False) and sys.executable or __file__).resolve().parent
    return str(base / rel)


def user_data_path(rel: str) -> str:
    """Ruta persistente para datos ESCRIBIBLES del usuario.

    En frozen (onefile) `resource_path` apunta a `_MEIPASS` (extracción
    temporal que se borra al salir). Los datos que deben sobrevivir
    (videos, sqlite, evidencias) van a `APPDATA_DIR` cuando el exe está
    empaquetado; en desarrollo se resuelve contra el directorio actual
    (raíz del proyecto, igual que `src.core.utils.resource_path`).
    """
    if hasattr(sys, "_MEIPASS"):
        return str(APPDATA_DIR / rel)
    return str((Path(".").resolve() / rel).resolve())


def _user_data_root() -> Path:
    """Directorio raíz para datos del usuario, según la plataforma."""
    if sys.platform.startswith("win"):
        # Windows: %APPDATA% (Roaming) suele estar definido siempre.
        root = os.getenv("APPDATA")
        if root:
            return Path(root) / "InfractiVision"
        return Path.home() / "AppData" / "Roaming" / "InfractiVision"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "InfractiVision"
    # Linux / *BSD: estándar XDG.
    xdg = os.getenv("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / "InfractiVision"


# Dirs de datos del usuario (no requieren admin)
APPDATA_DIR = _user_data_root()
CONFIG_DIR  = APPDATA_DIR / "config"
OUTPUT_DIR  = APPDATA_DIR / "output"
LOGS_DIR    = APPDATA_DIR / "logs"
for d in (CONFIG_DIR, OUTPUT_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Archivos clave
ICONO_APP   = resource_path("img/icon.ico")
YOLO_MODEL  = resource_path("models/yolov8n.pt")  # ajusta el nombre si usas otro
SETTINGS_JSON = CONFIG_DIR / "settings.json"

# Copia inicial de configs por defecto (solo si no existen en APPDATA)
_DEFAULT_CONFIGS = [
    "avenue_config.json",
    "direction_config.json",
    "polygon_config.json",
    "speed_limit_config.json",
    "time_presets.json",
]
def load_settings() -> dict:
    if SETTINGS_JSON.exists():
        try:
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

