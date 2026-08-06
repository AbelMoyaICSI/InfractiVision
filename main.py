"""InfractiVision – Composition Root.

Punto de entrada único del sistema. Aquí (y SOLO aquí):
    1. Se carga configuración (config/settings.py).
    2. Se construye el contenedor de dependencias (DI).
    3. Se levanta la presentación (Tkinter GUI).

NO contiene lógica de negocio: todo se delega a Casos de Uso.
"""
from __future__ import annotations

import getpass
import json
import socket
import threading
import tkinter as tk
import uuid
from pathlib import Path

from src.composition_root import build_container
from src.core.logger import get_logger
from src.core.utils.icon import set_window_icon
from src.core.utils.paths import APPDATA_DIR

log = get_logger("main")


# ─── IDs de usuario / dispositivo (config persistente) ─────────────────────
def _config_file() -> Path:
    """Ruta del JSON de IDs en el directorio de datos del usuario."""
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    return APPDATA_DIR / "infractivision_config.json"


def _load_or_create_ids() -> dict:
    path = _config_file()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    ids = {
        "user_id": str(uuid.uuid4()),
        "device_id": str(uuid.uuid4()),
        "username": getpass.getuser(),
        "hostname": socket.gethostname(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    return ids


# ─── Precarga LPRNet (background) ──────────────────────────────────────────
def _preload_lprnet_in_background() -> None:
    def _job():
        try:
            from src.core.ocr.recognizer import get_lprnet_predictor
            get_lprnet_predictor()
            log.info("LPRNet Master Engine precargado")
        except Exception as e:
            log.warning("LPRNet preload falló: %s", e)

    threading.Thread(target=_job, daemon=True).start()


# ─── Bootstrap GUI ─────────────────────────────────────────────────────────
def main() -> None:
    ids = _load_or_create_ids()

    root = tk.Tk()
    root.title("InfractiVision")
    set_window_icon(root)
    root.geometry("1280x720")
    try:
        root.state("zoomed")
    except Exception:
        pass

    _preload_lprnet_in_background()

    # El proveedor de estado del semáforo se inyecta más tarde, cuando la GUI
    # crea su panel `Semaforo`. Por ahora, valor por defecto "green".
    traffic_light_state: dict[str, str] = {"value": "green"}

    container = build_container(
        traffic_light_state_provider=lambda: traffic_light_state["value"]
    )
    log.info("Container listo. Iniciando MainWindow.")

    # Presentación: usamos MainWindow que monta la GUI legacy (AppManager).
    from src.presentation.gui import MainWindow

    main_window = MainWindow(  # noqa: F841 (se mantiene viva por el mainloop)
        root=root,
        process_frame_uc=container.process_frame,
        user_id=ids["user_id"],
        device_id=ids["device_id"],
        traffic_light_state=traffic_light_state,
    )

    # Exponemos el container y el dict de estado para que las pantallas
    # internas (p. ej. el reproductor) puedan actualizar el estado del semáforo
    # llamando a `root.tk_infractivision["traffic_light_state"]["value"] = "red"`.
    root.tk_infractivision = {  # type: ignore[attr-defined]
        "container": container,
        "traffic_light_state": traffic_light_state,
    }

    root.mainloop()


if __name__ == "__main__":
    main()
