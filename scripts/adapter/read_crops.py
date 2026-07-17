#!/usr/bin/env python
"""
read_crops.py — Lee crops de placas y ejecuta OCR (LPRNet).

Usage:
    python scripts/adapter/read_crops.py --crops-dir data/output/crops/
    python scripts/adapter/read_crops.py --crops-dir data/output/crops/ --sharpen
    python scripts/adapter/read_crops.py --crops-dir data/output/crops/ --lprnet-weights v3 --sharpen
    python scripts/adapter/read_crops.py --crops-dir data/output/crops/ --lprnet-weights custom --custom-path models/mi_modelo.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ocr.recognizer import recognize_plate, calculate_siiv_confidence


def sharpen_image(img: np.ndarray, sigma: float = 1.5, amount: float = 1.2) -> np.ndarray:
    """Aplica unsharp masking para realzar bordes sin exagerar ruido."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


# Alias → archivo de pesos LPRNet
LPRNET_WEIGHTS = {
    "v4":     PROJECT_ROOT / "models" / "LPRNet_V4_CORREGIDO.pth",
    "v3":     PROJECT_ROOT / "models" / "LPRNet_V3_ESPECIALISTA.pth",
    "v2":     PROJECT_ROOT / "models" / "LPRNet_CONSENSO_V2.pth",
    "master": PROJECT_ROOT / "models" / "LPRNet_Peru_MASTER_FINAL.pth",
}


def resolve_lprnet_weights(args) -> str | None:
    """Resuelve el alias --lprnet-weights a la ruta del archivo .pth."""
    alias = args.lprnet_weights
    if alias is None:
        return None  # auto-resolve (default)
    if alias == "custom":
        custom = args.custom_path
        if custom is None:
            print("Error: --lprnet-weights custom requiere --custom-path")
            sys.exit(1)
        p = Path(custom)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"Error: modelo no encontrado: {p}")
            sys.exit(1)
        return str(p)
    return str(LPRNET_WEIGHTS[alias])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InfractiVision OCR Reader para crops de placas"
    )
    parser.add_argument(
        "--crops-dir",
        required=True,
        help="Directorio con crops de placa (*.jpg)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern para filtrar archivos (default: *.jpg)",
    )
    parser.add_argument(
        "--lprnet-weights",
        choices=list(LPRNET_WEIGHTS.keys()) + ["custom"],
        default=None,
        help="Peso LPRNet a usar: v4 (default), v3, v2, master, custom",
    )
    parser.add_argument(
        "--custom-path",
        default=None,
        help="Ruta a modelo .pth custom (solo con --lprnet-weights custom)",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Aplica unsharp masking antes del OCR para realzar bordes",
    )
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    if not crops_dir.exists():
        print(f"Directorio no encontrado: {crops_dir}")
        sys.exit(1)

    files = sorted(crops_dir.glob(args.pattern))
    if not files:
        print(f"No se encontraron archivos '{args.pattern}' en {crops_dir}")
        sys.exit(1)

    model_path = resolve_lprnet_weights(args)
    model_label = args.lprnet_weights or "auto"
    if model_path:
        model_label += f" ({Path(model_path).name})"

    print(f"\n=== InfractiVision OCR Reader ===")
    print(f"Directorio: {crops_dir}")
    print(f"Modelo:     {model_label}")
    print(f"Sharpen:    {'ON' if args.sharpen else 'OFF'}")
    print(f"Crops:      {len(files)} archivos\n")

    for f in files:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {f.name:40s} -> (no se pudo leer)")
            continue

        if args.sharpen:
            img = sharpen_image(img)

        text, conf = recognize_plate(img, autocrop=False, preprocessed=True, model_path=model_path)
        siiv_conf, _ = calculate_siiv_confidence(text, conf) if text else (0.0, {})

        display_text = text if text else "(vacio)"
        print(f"  {f.name:40s} -> {display_text:10s} ({siiv_conf * 100:.1f}%)")


if __name__ == "__main__":
    main()
