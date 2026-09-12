"""Fase 0: bench baseline CPU vs GPU por etapa (sin romper nada).

Uso:
  .venv/bin/python scripts/bench_gpu_pipeline.py [--video videos/VID1EDIT*] [--frames 60]

Mide:
  - YOLO vehículos single vs batch
  - YOLO placas single
  - night-enhance (5 variantes) vs fast
  - _quality (Canny+Laplacian)
  - auto_rectifier encontrar_esquinas
  - imwrite sync vs async / VideoWriter
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gpu_info():
    try:
        import torch

        avail = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if avail else "CPU-only"
        ver = getattr(torch, "__version__", "?")
        cudaver = getattr(getattr(torch, "version", object()), "cuda", "?")
        return f"torch={ver} cuda_avail={avail} cuda={cudaver} dev={name}"
    except Exception as e:  # noqa: BLE001
        return f"torch no disponible: {e}"


def _pick_video(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    for cand in sorted((ROOT / "videos").glob("*.mp4")):
        return cand
    for cand in sorted((ROOT / "videos").glob("*.MOV")):
        return cand
    raise SystemExit("No hay videos en videos/")


def _grab_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"No se pudieron leer frames de {video}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()

    print(f"[bench] {_gpu_info()}")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[bench] GPU {props.name} CC={props.major}.{props.minor} VRAM={props.total_memory/1024**3:.1f}GB")
    except Exception:
        pass

    video = _pick_video(args.video)
    frames = _grab_frames(video, args.frames)
    print(f"[bench] video={video.name} frames={len(frames)} shape={frames[0].shape}")

    # --- YOLO vehículos ---
    from src.core.detection.vehicle_detector import VehicleDetector

    vd = VehicleDetector(str(ROOT / "models" / "yolov8n.pt"))
    print(f"[bench] vehicle device={vd.device} half={vd.half} imgsz={vd.imgsz} batch={vd.batch_size}")

    # warmup fuera del timing
    vd.detect(frames[0], conf=0.4)
    if torch.cuda.is_available() if 'torch' in sys.modules else False:
        try:
            import torch as _t

            _t.cuda.synchronize()
        except Exception:
            pass

    t0 = time.perf_counter()
    for f in frames[:8]:
        vd.detect(f, conf=0.4)
    single_ms = (time.perf_counter() - t0) / 8 * 1000

    batch_ms = float("nan")
    try:
        t0 = time.perf_counter()
        vd.detect_batch(frames[:8], conf=0.4)
        batch_ms = (time.perf_counter() - t0) / 8 * 1000
    except Exception as e:  # noqa: BLE001
        print(f"[bench] detect_batch falló: {e}")

    print(f"[bench] YOLO-veh single={single_ms:.0f}ms/frame batch={batch_ms:.0f}ms/frame speedup={single_ms/max(batch_ms,1e-6):.2f}x")

    # --- YOLO placas (1 cuadrante por frame) ---
    from src.core.detection.plate_detector import PlateDetector

    pd = PlateDetector()
    print(f"[bench] plate device={getattr(pd, 'device', '?')} half={getattr(pd, 'half', '?')}")
    quadrant = frames[0][frames[0].shape[0] // 2 :, :]
    pd.detect(quadrant, conf=0.4)
    t0 = time.perf_counter()
    for f in frames[:8]:
        q = f[f.shape[0] // 2 :, :]
        pd.detect(q, conf=0.4)
    print(f"[bench] YOLO-placa single={(time.perf_counter()-t0)/8*1000:.0f}ms/crop")

    # --- night enhance: full 5 variantes vs fast ---
    crop = quadrant
    t0 = time.perf_counter()
    pd._select_best_night_enhancement(crop)
    full_ms = (time.perf_counter() - t0) * 1000
    fast_ms = float("nan")
    if hasattr(pd, "_fast_night_enhance"):
        t0 = time.perf_counter()
        pd._fast_night_enhance(crop)
        fast_ms = (time.perf_counter() - t0) * 1000
    print(f"[bench] night full5={full_ms:.0f}ms fast={fast_ms:.0f}ms")

    # --- _quality ---
    from src.application.use_cases.process_violation_video import OfficialVideoProcessor

    t0 = time.perf_counter()
    for f in frames[:16]:
        OfficialVideoProcessor._quality(f)
    print(f"[bench] _quality={(time.perf_counter()-t0)/16*1000:.1f}ms/crop-fullframe")
    small = cv2.resize(frames[0], (320, 180))
    t0 = time.perf_counter()
    for _ in range(16):
        OfficialVideoProcessor._quality(small)
    print(f"[bench] _quality-320px={(time.perf_counter()-t0)/16*1000:.1f}ms")

    # --- auto_rectifier ---
    try:
        from src.core.processing.auto_rectifier import encontrar_esquinas

        roi = cv2.resize(quadrant, (200, 80))
        t0 = time.perf_counter()
        encontrar_esquinas(roi)
        print(f"[bench] rectifier-full={(time.perf_counter()-t0)*1000:.0f}ms/roi-200x80")
        if "fast" in encontrar_esquinas.__code__.co_varnames:
            t0 = time.perf_counter()
            encontrar_esquinas(roi, fast=True)
            print(f"[bench] rectifier-fast={(time.perf_counter()-t0)*1000:.0f}ms/roi-200x80")
    except Exception as e:  # noqa: BLE001
        print(f"[bench] rectifier N/A: {e}")

    # --- imwrite sync ---
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        for i, f in enumerate(frames[:8]):
            cv2.imwrite(f"{td}/{i}.jpg", f)
        print(f"[bench] imwrite-sync={(time.perf_counter()-t0)/8*1000:.0f}ms/frame")


if __name__ == "__main__":
    main()
