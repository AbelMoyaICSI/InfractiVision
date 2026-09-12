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


# Dirs de datos del usuario (no requieren admin) — lazy mkdir, no en import
APPDATA_DIR = _user_data_root()
CONFIG_DIR = APPDATA_DIR / "config"
OUTPUT_DIR = APPDATA_DIR / "output"
LOGS_DIR = APPDATA_DIR / "logs"

def ensure_user_dirs() -> None:
    """Crea dirs de usuario bajo demanda (llamar desde main/settings, no en import)."""
    for d in (CONFIG_DIR, OUTPUT_DIR, LOGS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    # Dirs escribibles que la GUI legacy usa vía writable_data_path
    for rel in ("data", "data/output", "data/output/placas", "data/output/autos",
                "data/output/official", "data/videos", "data/images",
                "data/evidences", "videos", "models"):
        try:
            (APPDATA_DIR / rel).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def writable_config_path(filename: str) -> str:
    """Ruta ESCRIBIBLE para configs (polígono/avenida/presets).

    En frozen (instalado) apunta a %APPDATA%/InfractiVision/config/<file>
    con seed desde el bundle (_MEIPASS/config) en el primer arranque.
    En desarrollo conserva el archivo del proyecto (config/<file>).
    """
    if not hasattr(sys, "_MEIPASS"):
        # Desarrollo: conservar el archivo del proyecto (raíz, no .venv).
        # paths.resource_path usa sys.executable/__file__ como base y en venv
        # apunta a .venv/Scripts o src/core/utils; usar el helper de raíz.
        try:
            from src.path_helper import resource_path as _dev_rp
        except ImportError:
            from src.core.utils import resource_path as _dev_rp  # type: ignore[no-redef]
        return _dev_rp(f"config/{filename}")
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    dest = CONFIG_DIR / filename
    if not dest.exists():
        try:
            seed = Path(resource_path(f"config/{filename}"))
            if seed.exists():
                shutil.copy2(seed, dest)
        except Exception:
            pass
    return str(dest)


def writable_data_path(rel: str) -> str:
    """Ruta ESCRIBIBLE para datos (data/*.json, data/output/...).

    En frozen va a %APPDATA%/InfractiVision/<rel> (crea padres).
    En desarrollo resuelve contra el proyecto, igual que user_data_path.
    """
    if hasattr(sys, "_MEIPASS"):
        p = APPDATA_DIR / rel
        try:
            parent = p.parent if "." in Path(rel).name else p
            parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(p)
    return user_data_path(rel)

# Archivos clave (SETTINGS_JSON usa CONFIG_DIR pero no crea dirs en import)
ICONO_APP = resource_path("img/icon.ico")
YOLO_MODEL = resource_path("models/yolov8n.pt")  # ajusta el nombre si usas otro
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

