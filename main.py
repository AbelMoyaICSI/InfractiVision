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
import platform
import socket
import struct
import sys
import tkinter as tk
import traceback
import uuid
from pathlib import Path

from src.core.logger import get_logger
from src.core.utils.paths import APPDATA_DIR

# Windows cp1252 (Spanish) can't encode emoji/em-dash -> force UTF-8 for stdout/stderr
# to avoid UnicodeEncodeError in welcome_window and _show_startup_error.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

log = get_logger("main")


def _show_startup_error(title: str, message: str) -> None:
    """Muestra un error de arranque con GUI si es posible, o consola como fallback."""
    # Sanitize for cp1252 console: replace emojis/em-dash with ASCII to avoid encode error
    safe_title = title.replace("—", "-").replace("✅", "[OK]").replace("❌", "[ERROR]")
    safe_msg = message.replace("—", "-").replace("✅", "[OK]").replace("❌", "[ERROR]").replace("🖼️", "[IMG]")
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb

        _r = _tk.Tk()
        _r.withdraw()
        _r.attributes("-topmost", True)
        _mb.showerror(safe_title, safe_msg)
        _r.destroy()
    except Exception:
        pass
    # Siempre loguea a consola también (visible si console=True o en log file)
    # Use buffer with utf-8 to survive cp1252 stderr
    for t, m in [(title, message), (safe_title, safe_msg)]:
        try:
            print(f"[FATAL] {t}: {m}", file=sys.stderr)  # type: ignore[name-defined]
            break
        except UnicodeEncodeError:
            try:
                sys.stderr.buffer.write(f"[FATAL] {t}: {m}\n".encode("utf-8", errors="replace"))  # type: ignore[attr-defined]
                sys.stderr.flush()
                break
            except Exception:
                continue
        except Exception:
            pass


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


# ─── Bootstrap GUI ─────────────────────────────────────────────────────────
def main() -> None:
    # Lazy imports para arranque rápido: no pagan torch/cv2/ultralytics antes de Tk.
    import tkinter as tk

    from src.core.utils.icon import set_window_icon
    from src.core.utils.paths import ensure_user_dirs

    ensure_user_dirs()

    ids = _load_or_create_ids()

    root = tk.Tk()
    root.title("InfractiVision")
    set_window_icon(root)
    root.geometry("1280x720")
    try:
        root.state("zoomed")
    except Exception:
        pass

    # Descarga selectiva: modelos AI primero (requeridos para inferencia),
    # luego videos demo. Idempotente y no bloquea el arranque.
    def _ensure_assets() -> None:
        # 1) Modelos: solo required (yolov8n, license_plate_detector)
        try:
            from src.infrastructure.storage.model_downloader import ensure_models_async, missing_models

            pending = missing_models()
            if pending:
                log.info("Modelos faltantes detectados: %s — descargando en background", pending)

            def _on_models_done(result: dict):
                log.info("Modelos: %s (dest=%s)", result, result.get("dest_dir"))
                if result.get("failed"):
                    log.warning("Algunos modelos fallaron — la app usara fallbacks si existen")

            ensure_models_async(callback=_on_models_done)
        except Exception as e:
            log.warning("No se pudo iniciar descarga de modelos: %s", e)

        # 2) Videos demo (5 videos grandes, descarga lazy no critica)
        try:
            from src.infrastructure.storage.demo_video_downloader import ensure_demo_videos_async

            ensure_demo_videos_async()
        except Exception as e:
            log.warning("No se pudo iniciar descarga de videos demo: %s", e)

    root.after(500, _ensure_assets)

    # El proveedor de estado del semáforo se inyecta más tarde, cuando la GUI
    # crea su panel `Semaforo`. Por ahora, valor por defecto "green".
    traffic_light_state: dict[str, str] = {"value": "green"}

    try:
        from src.composition_root import build_container
    except ImportError as exc:
        # Caso crítico: DLL load failed while importing cv2. Antes mostraba
        # "Failed to execute script 'main'". ERROR 193 != Python 32-bit: este
        # build ya es 64-bit, es el cv2.pyd empaquetado corrupto/truncado
        # (upx/strip, antivirus o ZIP mal extraido) o falta de VC++ Redist.
        msg = str(exc)
        is_cv2_dll = "cv2" in msg or "DLL load failed" in msg or "Win32" in msg or "no es una aplicaci" in msg
        bits = struct.calcsize("P") * 8
        log.error("Fallo importando composition_root (cv2/DLL): %s", exc, exc_info=True)
        if is_cv2_dll:
            low = msg.lower()
            is_bad_exe = ("no es una aplicaci" in msg or "win32" in low or "error 193" in low or "%1" in msg)
            is_missing = ("126" in msg or "specified module" in low or "vcruntime" in low or "msvcp" in low or "no se encontr" in low)
            if is_bad_exe and not is_missing:
                causa = (
                    "El cv2.pyd del portable esta corrupto/truncado (ERROR 193). "
                    "NO es Python 32-bit: este build ya es 64-bit.\n"
                    "Pasos:\n"
                    "  1) Re-extrae el ZIP con 7-Zip en ruta corta (ej. C:\\IV\\), no en Downloads anidado.\n"
                    "  2) Pausa el antivirus o anade excepcion para InfractiVision (cuarentena deja el .pyd en 0 KB).\n"
                    "  3) Verifica _internal/cv2/cv2.pyd (>10MB). Si esta en 0 KB, la extraccion fallo: repite 1+2.\n"
                    "  4) Instala VC++ Redist x64 (el ZIP portable no lo auto-instala como el Setup):\n"
                    "     https://aka.ms/vs/17/release/vc_redist.x64.exe y reinicia.\n"
                    "  5) Si el ZIP es anterior al fix sin upx/strip, descarga el build nuevo."
                )
            elif is_missing:
                causa = (
                    "Falta Microsoft Visual C++ Redistributable 2015-2022 x64 (ERROR 126).\n"
                    "Instalalo: https://aka.ms/vs/17/release/vc_redist.x64.exe y reinicia.\n"
                    "El ZIP portable no lo auto-instala (solo lo hace el Setup Inno)."
                )
            else:
                causa = (
                    "Causas probables:\n"
                    "  - ZIP mal extraido o antivirus (cv2.pyd en 0 KB). Re-extrae con 7-Zip en C:\\IV\\.\n"
                    "  - Falta VC++ Redist x64: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    "  - Solo en dev: conflicto opencv-python vs opencv-python-headless:\n"
                    "    pip uninstall opencv-python-headless opencv-python -y && pip install --no-cache --force-reinstall opencv-python==4.9.0.80"
                )
            detail = (
                f"Error cargando OpenCV (cv2): {exc}\n\n"
                f"Detectado: Python {bits}-bit ({platform.architecture()[0]}) en {platform.machine()} - {sys.version.split()[0]}\n"
                f"{causa}\n"
                f"\nDetalle tecnico: {traceback.format_exc()[-1200:]}"
            )
            _show_startup_error("InfractiVision - Error de OpenCV (cv2)", detail)
            try:
                root.destroy()
            except Exception:
                pass
            sys.exit(1)
        # Otro ImportError no relacionado a cv2: re-lanzar con mensaje genérico
        _show_startup_error("InfractiVision - Error de arranque", f"No se pudo iniciar la aplicacion:\n{exc}\n\n{traceback.format_exc()[-1000:]}")
        raise

    try:
        container = build_container(
            traffic_light_state_provider=lambda: traffic_light_state["value"]
        )
    except ImportError as exc:
        msg = str(exc)
        is_cv2_dll = "cv2" in msg or "DLL load failed" in msg or "Win32" in msg or "no es una aplicaci" in msg
        bits = struct.calcsize("P") * 8
        log.error("Fallo en build_container: %s", exc, exc_info=True)
        if is_cv2_dll:
            detail = (
                f"Error inicializando OpenCV/cv2: {exc}\n\n"
                f"Python {bits}-bit - {sys.version.split()[0]}\n"
                "Si dice '%1 no es Win32 valida' (ERROR 193): ZIP mal extraido o antivirus "
                "(re-extrae con 7-Zip en C:\\IV\\, verifica _internal/cv2/cv2.pyd >10MB).\n"
                "Si falta DLL (ERROR 126): instala VC++ Redist x64: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"{traceback.format_exc()[-1000:]}"
            )
            _show_startup_error("InfractiVision - Error de OpenCV (cv2)", detail)
            try:
                root.destroy()
            except Exception:
                pass
            sys.exit(1)
        raise
    log.info("Container listo (modelos lazy). Iniciando MainWindow.")

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
    try:
        main()
    except ImportError as exc:
        # Fallback global para el caso "DLL load failed while importing cv2" antes de que Tk exista
        msg = str(exc)
        if "cv2" in msg or "DLL load failed" in msg or "Win32" in msg or "no es una aplicaci" in msg:
            import struct as _st
            bits = _st.calcsize("P") * 8
            detail = (
                f"Error cargando OpenCV (cv2): {exc}\n\n"
                f"Python {bits}-bit - {sys.version.split()[0] if 'sys' in dir() else ''}\n"
                "Si es ERROR 193 '%1 no es Win32 valida': cv2.pyd corrupto (ZIP mal extraido/antivirus). "
                "Re-extrae con 7-Zip en C:\\IV\\ y verifica _internal/cv2/cv2.pyd >10MB.\n"
                "Si es ERROR 126 o falta DLL: instala https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            )
            _show_startup_error("InfractiVision - Error de OpenCV (cv2)", detail)
            sys.exit(1)
        raise
    except Exception as exc:
        # Cualquier otra excepción no controlada: log + mensaje amigable antes de salir
        try:
            log.error("Fallo no controlado en main: %s", exc, exc_info=True)
        except Exception:
            pass
        _show_startup_error("InfractiVision - Error inesperado", f"{exc}\n\n{traceback.format_exc()[-1200:]}")
        raise
