"""CI smoke test: valida el stack de dependencias (requirements.txt) en
Windows y Linux. Lo ejecuta .github/workflows/deps.yml.

Verifica los pins críticos de compatibilidad y la ABI numpy<->torch que
rompió la inferencia (YOLO/OCR) cuando numpy subió a 2.x.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import struct
    import platform
    import importlib.metadata as _md
    bits = struct.calcsize("P") * 8
    print(f"arch   : {bits}-bit ({platform.architecture()[0]}, {platform.machine()})")
    if bits != 64:
        print(f"ERROR: Se requiere Python 64-bit (detectado {bits}-bit). Ese desajuste produce 'DLL load failed while importing cv2: %1 no es una aplicacion Win32 valida.'", file=sys.stderr)
        return 1
    # easy_ocr / paddle_ocr solo para tests — nunca en prod. headless contamina el build.
    try:
        _md.version("opencv-python-headless")
        print("ERROR: opencv-python-headless detectado. EasyOCR solo en tests, no en build prod. Desinstala headless.", file=sys.stderr)
        print("  pip uninstall opencv-python-headless opencv-python -y && pip install --no-cache --force-reinstall opencv-python==4.9.0.80", file=sys.stderr)
        return 1
    except _md.PackageNotFoundError:
        pass

    import numpy
    import cv2
    import torch

    print(f"python : {sys.version.split()[0]}")
    print(f"numpy  : {numpy.__version__}")
    print(f"opencv : {cv2.__version__}")
    print(f"torch  : {torch.__version__} (cuda: {torch.cuda.is_available()})")
    # Verificar que opencv realmente es win_amd64 / 64-bit (no win32 headless corrupto)
    try:
        print(f"cv2    : {cv2.__file__}")
    except Exception:
        pass

    # Pines críticos de compatibilidad (no mover: rompen la inferencia)
    assert numpy.__version__.startswith("1.26"), \
        f"numpy debe ser 1.26.x (compat torch 1.13): {numpy.__version__}"
    assert cv2.__version__.startswith("4.9"), \
        f"opencv debe ser 4.9.x (compat numpy 1.x): {cv2.__version__}"

    # ABI numpy<->torch (el bug de "Numpy is not available")
    t = torch.from_numpy(numpy.zeros((4, 4), dtype="float32"))
    assert t is not None, "torch.from_numpy falló (ABI numpy/torch rota)"

    # Imports de la app (cadena completa de la arquitectura)
    import tkinter  # en Linux requiere python3-tk (se instala en el workflow)
    import src.composition_root  # noqa: F401
    from src.infrastructure.database import app_repository  # noqa: F401
    from src.core.video import videoplayer_opencv  # noqa: F401
    import src.core.detection.torch_compat  # noqa: F401
    print("imports app OK")

    # YOLO quick check (descarga yolov8n.pt de la CDN si no está local)
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    r = model.predict(numpy.zeros((320, 320, 3), dtype="uint8"), verbose=False)
    assert len(r) == 1 and r[0].boxes is not None, "YOLO no produjo detecciones"
    print("YOLO predict OK")

    print("✅ CI SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())