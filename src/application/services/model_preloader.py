"""Precarga de modelos IA al entrar a Foto Rojo (SOLO detección).

En vivo solo hay detección: YOLO-vehículos + YOLO-placas. La lectura OCR
la hace la API de Plate Recognizer al final, en `PlateReviewWindow`.

La entrada a Foto Rojo muestra un spinner bloqueante una sola vez y el
`PreprocessingDialog` / `VideoPlayerOpenCV` reutilizan las instancias
inyectadas (fast-path) en vez de cargar durante el análisis del video.
"""
from __future__ import annotations

import time
from typing import Callable


def has_plate_token() -> bool:
    try:
        from src.infrastructure.ocr.cloud_plate_readers import has_plate_recognizer_token
        return bool(has_plate_recognizer_token())
    except Exception:
        return False


def preload_foto_rojo_models(
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Calienta YOLO-vehículos y YOLO-placas en serie.

    Serializado por `MODEL_LOAD_LOCK` (vía `@serialized` en cada
    constructor): nunca carga dos modelos en paralelo (evita SIGSEGV CUDA).
    No toca Tk: el llamador marshala `progress` al hilo principal.

    Retorna dict con instancias + flags:
      vehicle_detector, plate_detector, plate_token_ok, errors, elapsed_s
    """
    t0 = time.monotonic()
    errors: list[str] = []

    def _say(msg: str) -> None:
        try:
            if progress is not None:
                progress(msg)
        except Exception:
            pass

    vehicle_detector = None
    plate_detector = None
    warmup_s: dict[str, float] = {}

    # 1) Vehículos (yolov8n.pt)
    try:
        _say("Cargando detector de vehículos (YOLO)...")
        from src.core.detection.vehicle_detector import VehicleDetector
        from src.path_helper import resource_path
        vehicle_detector = VehicleDetector(model_path=resource_path("models/yolov8n.pt"))
    except Exception as e:
        errors.append(f"vehículos: {e}")

    # 1b) Warm-up: la PRIMERA inferencia paga fuse()+CUDA/cuDNN. Si no se hace
    # aquí (spinner), cae en amarillo tardío (el planner no detecta en verde)
    # y se ve como un "lagazo". Una pasada dummy lo deja compilado.
    if vehicle_detector is not None:
        try:
            _say("Calentando detector de vehículos...")
            import numpy as np
            size = int(getattr(vehicle_detector, "imgsz", 640) or 640)
            t1 = time.monotonic()
            vehicle_detector.detect(np.zeros((size, size, 3), dtype=np.uint8), conf=0.4)
            warmup_s["vehicle"] = round(time.monotonic() - t1, 2)
        except Exception as e:
            errors.append(f"warmup vehículos: {e}")

    # 2) Placas (license_plate_detector.pt)
    try:
        _say("Cargando detector de placas (YOLO)...")
        from src.core.detection.plate_detector import PlateDetector
        plate_detector = PlateDetector()
        if getattr(plate_detector, "model", None) is None:
            errors.append("placas: modelo no encontrado (license_plate_detector.pt)")
            plate_detector = None
    except Exception as e:
        errors.append(f"placas: {e}")
        plate_detector = None

    if plate_detector is not None:
        try:
            _say("Calentando detector de placas...")
            import numpy as np
            t1 = time.monotonic()
            try:
                plate_detector.detect(np.zeros((320, 320, 3), dtype=np.uint8), conf=0.4, draw=False)
            except TypeError:
                plate_detector.detect_plates(np.zeros((320, 320, 3), dtype=np.uint8), confidence=0.4)
            warmup_s["plate"] = round(time.monotonic() - t1, 2)
        except Exception as e:
            errors.append(f"warmup placas: {e}")

    # 3) Warm-up FSRCNN (si está disponible; solo mejora crops, no bloquea)
    try:
        from src.core.ocr.super_resolution import get_upscaler
        upscaler = get_upscaler()
        if getattr(upscaler, "model_loaded", False):
            _say("Calentando super-resolución...")
            import numpy as np
            t1 = time.monotonic()
            upscaler.upscale(np.zeros((40, 60, 3), dtype=np.uint8))
            warmup_s["fsrcnn"] = round(time.monotonic() - t1, 2)
    except Exception as e:
        errors.append(f"warmup FSRCNN: {e}")

    _say("Modelos listos.")
    return {
        "vehicle_detector": vehicle_detector,
        "plate_detector": plate_detector,
        "plate_token_ok": has_plate_token(),
        "errors": errors,
        "warmup_s": warmup_s,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
