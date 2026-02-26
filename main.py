# main.py
import tkinter as tkcls
import uuid
import json
import os
import getpass
import socket
import sys
import threading
import time

from src.gui.app_manager import AppManager
from src.path_helper import resource_path  # <-- usamos el helper

# Precarga de LPRNet en background
def preload_lprnet_engine():
    """Precarga LPRNet en background para inicio más rápido"""
    try:
        print("⚡ Precargando LPRNet Master Engine en background...")
        from src.core.ocr.recognizer import get_lprnet_predictor
        get_lprnet_predictor()  # Esto inicializa el motor LPRNet y carga pesos
        print("✅ LPRNet Master Engine precargado exitosamente")
    except Exception as e:
        print(f"⚠️ Error precargando LPRNet: {e}")

def get_config_path() -> str:
    """
    Devuelve una ruta ESCRIBIBLE para la configuración del usuario.
    En Windows usa %APPDATA%\InfractiVision\infractivision_config.json,
    y si no existe APPDATA, cae a ~/.infractivision/infractivision_config.json
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        cfg_dir = os.path.join(appdata, "InfractiVision")
    else:
        # Fallback genérico
        cfg_dir = os.path.join(os.path.expanduser("~"), ".infractivision")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "infractivision_config.json")


CONFIG_PATH = get_config_path()


def load_ids():
    """
    Carga user_id, device_id, username y hostname desde CONFIG_PATH.
    Si no existe, los genera y los guarda.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    ids = {
        "user_id":   str(uuid.uuid4()),
        "device_id": str(uuid.uuid4()),
        "username":  getpass.getuser(),
        "hostname":  socket.gethostname()
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    return ids


def main():
    # Cargamos (o creamos) los IDs únicos
    ids = load_ids()
    user_id   = ids["user_id"]
    device_id = ids["device_id"]

    # Ventana principal
    root = tkcls.Tk()
    root.title("InfractiVision")
    try:
        # Ícono desde recursos empaquetados
        root.iconbitmap(resource_path("img/icon.ico"))
    except Exception:
        # No rompemos la app si el icono no está
        pass

    # Tamaño/estado
    root.geometry("1280x720")
    try:
        root.state("zoomed")  # Windows
    except Exception:
        pass

    # Iniciar precarga de LPRNet en background
    lpr_thread = threading.Thread(target=preload_lprnet_engine, daemon=True)
    lpr_thread.start()
    
    # Instanciar gestor de la app
    app = AppManager(root, user_id=user_id, device_id=device_id)

    root.mainloop()


if __name__ == "__main__":
    main()
