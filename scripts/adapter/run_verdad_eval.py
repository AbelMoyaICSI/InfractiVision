#!/usr/bin/env python
"""
run_verdad_eval.py — Run process_video.py on all videos from tests/verdad.test.json.

Saves best crops in original size (no 24x94 resize) for visual inspection.

Usage:
    python scripts/adapter/run_verdad_eval.py
    python scripts/adapter/run_verdad_eval.py --skip-video "Night Time"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERDAD_PATH = PROJECT_ROOT / "tests" / "verdad.test.json"
PROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "adapter" / "process_video.py"
VIDEOS_DIR = PROJECT_ROOT / "videos"


def safe_name(video_name: str) -> str:
    """Create a filesystem-safe directory name from a video name."""
    name = Path(video_name).stem
    # Replace chars that are problematic on Windows
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    # Collapse multiple underscores/spaces
    parts = [p for p in name.replace(" ", "_").split("_") if p]
    return "_".join(parts)


def build_per_video_config(video_entry: dict) -> dict:
    """Build the per-video config dict with polygon + semaphore."""
    return {
        "polygon": video_entry.get("polygon"),
        "semaphore": {
            "green": video_entry.get("green", 30),
            "yellow": video_entry.get("yellow", 5),
            "red": video_entry.get("red", 40),
        },
    }


def run_video(
    video_entry: dict,
    output_dir: Path,
    log_path: Path,
    extra_args: list[str],
) -> dict:
    """Run process_video.py for a single video. Returns parsed stdout metrics."""
    video_name = video_entry["path_name"]
    video_path = VIDEOS_DIR / video_name

    if not video_path.exists():
        print(f"  ⚠️  Video no encontrado: {video_path}")
        return {"error": f"not found: {video_path}"}

    # Write per-video config to a temp JSON file
    cfg = build_per_video_config(video_entry)
    cfg_path = output_dir / "_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROCESS_SCRIPT),
        "--video", str(video_path),
        "--new",
        "--save-crops-original",
        "--best-only",
        "--output-dir", str(output_dir),
        "--config", str(cfg_path),
    ] + extra_args

    print(f"  🚀 Ejecutando: {' '.join(cmd[-8:])}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )

    # Write combined output to log
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n")
        logf.write(f"VIDEO: {video_name}\n")
        logf.write(f"{'='*60}\n")
        if result.stdout:
            logf.write(result.stdout)
        if result.stderr:
            logf.write(f"\n--- STDERR ---\n{result.stderr}")

    # Print stdout to terminal
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"    {line}")

    if result.returncode != 0:
        print(f"  ❌ Error (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.splitlines()[-5:]:
                print(f"    {line}")

    # Parse metrics from stdout
    metrics = {"video": video_name, "returncode": result.returncode}
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if "Frames procesados:" in line:
            metrics["frames_line"] = line
        elif "NID:" in line and "nie" not in line.lower():
            metrics["nid_line"] = line
        elif "TI" in line and "%" in line:
            metrics["ti_line"] = line
        elif "Crops guardados" in line:
            metrics["crops_line"] = line

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run process_video.py on all verdad.test.json videos"
    )
    parser.add_argument(
        "--skip-video",
        action="append",
        default=[],
        help="Substring to skip videos (can be repeated)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help="Override processing speed (default: None = use --new default 60)",
    )
    args = parser.parse_args()

    # Load ground truth
    if not VERDAD_PATH.exists():
        print(f"❌ No se encontró {VERDAD_PATH}")
        sys.exit(1)

    verdad = json.loads(VERDAD_PATH.read_text(encoding="utf-8"))
    videos = verdad.get("videos_verdad", [])
    print(f"\n📋 Videos en verdad.test.json: {len(videos)}")

    # Create timestamped output root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_root = PROJECT_ROOT / "data" / "verdad_eval" / f"run_{ts}"
    eval_root.mkdir(parents=True, exist_ok=True)

    log_path = eval_root / f"_run_{ts}.log"
    log_path.write_text("", encoding="utf-8")  # clear

    print(f"📂 Output root: {eval_root}")
    print(f"📝 Log: {log_path}\n")

    # Process each video
    results = []
    for i, vid in enumerate(videos, 1):
        name = vid["path_name"]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] {name}")
        print(f"{'='*60}")

        # Skip check
        if any(skip.lower() in name.lower() for skip in args.skip_video):
            print("  ⏭️  Saltado (por --skip-video)")
            results.append({"video": name, "skipped": True})
            continue

        out_dir = eval_root / safe_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)

        extra_args = []
        if args.speed is not None:
            extra_args += ["--speed", str(args.speed)]

        metrics = run_video(vid, out_dir, log_path, extra_args)
        results.append(metrics)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESUMEN DE EVALUACIÓN")
    print(f"{'='*60}")
    for r in results:
        name = r.get("video", "?")
        if r.get("skipped"):
            print(f"  ⏭️  {name}: saltado")
        elif r.get("error"):
            print(f"  ❌ {name}: {r['error']}")
        else:
            crops = r.get("crops_line", "")
            print(f"  {name}")
            if crops:
                print(f"      {crops}")
    print(f"\n📂 Crops: {eval_root}")
    print(f"📝 Log:   {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
