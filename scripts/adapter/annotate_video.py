#!/usr/bin/env python
"""CLI adapter for the official InfractiVision video processor.

The processing rules live in ``src/application/use_cases``. This module only
parses command-line arguments and wires the shared configuration/models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.process_violation_video import OfficialVideoProcessor
from src.infrastructure.configuration import VideoConfigRepository

VIDEOS_DIR = PROJECT_ROOT / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "video_track" / "output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InfractiVision - official video analysis")
    parser.add_argument("--video", help="Configured video name")
    parser.add_argument("--all", action="store_true", help="Process all configured videos")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--speed", type=int, default=None, help="Sobrescribe el skip del verde")
    parser.add_argument("--crops-only", action="store_true", help="Do not write annotated video")
    parser.add_argument("--save-crops", action="store_true", help="Save plate evidence")
    parser.add_argument("--best-only", action="store_true", help="Compatibility flag; official flow always keeps one best frame")
    parser.add_argument("--border", action="store_true", help="Legacy compatibility flag")
    parser.add_argument("--original-size", action="store_true", help="Keep original crop size")
    parser.add_argument("--stack", action="store_true", help="Deprecated; official flow does not use stacking")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.video) == bool(args.all):
        raise SystemExit("Se requiere exactamente uno de --video o --all")

    repository = VideoConfigRepository(PROJECT_ROOT)
    if args.all:
        entries = repository._read(repository.truth_path, {}).get("videos_verdad", [])
        names = [entry["path_name"] for entry in entries if repository.get(entry["path_name"])]
    else:
        names = [args.video]

    processor = OfficialVideoProcessor(PROJECT_ROOT)
    output_dir = Path(args.output)

    def report_callback(event):
        if event["type"] == "complete":
            payload = event["payload"]
            print(
                f"  Completado: {payload.get('infractor_count', 0)} infractores con placa | "
                f"{len(payload.get('evidence', []))} mejores crops"
            )

    for index, name in enumerate(names, 1):
        config = repository.require(name)
        if args.speed is not None:
            from dataclasses import replace
            config = replace(config, green_skip_rate=max(1, args.speed))
        video_path = VIDEOS_DIR / name
        print(f"[{index}/{len(names)}] Procesando {name}")
        print(f"  Verde: skip x{config.green_skip_rate} | Pre-rojo: {config.pre_red_seconds}s | Rojo: precisión completa")
        processor.process(
            video_path=video_path,
            config=config,
            output_dir=output_dir,
            conf=args.conf,
            save_video=not args.crops_only,
            save_crops=True,
            callback=report_callback,
        )
    print(f"Listo. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
