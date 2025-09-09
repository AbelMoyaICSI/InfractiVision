# src/utils/paths.py
from pathlib import Path
import sys, os, json, shutil

def resource_path(rel: str) -> str:
    # Carpeta temporal si es onefile, o carpeta del exe si es onedir
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(getattr(sys, "frozen", False) and sys.executable or __file__).resolve().parent
    return str(base / rel)

# Dirs de datos del usuario (no requieren admin)
APPDATA_DIR = Path(os.getenv("APPDATA", str(Path.home() / "AppData/Roaming"))) / "InfractiVision"
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
def ensure_user_config():
    for name in _DEFAULT_CONFIGS:
        dst = CONFIG_DIR / name
        if not dst.exists():
            src = Path(resource_path(f"config/{name}"))
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    dst.write_text("{}", encoding="utf-8")
            else:
                dst.write_text("{}", encoding="utf-8")

# Helpers para cargar/guardar settings
def load_settings() -> dict:
    if SETTINGS_JSON.exists():
        try:
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_settings(cfg: dict) -> None:
    SETTINGS_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
