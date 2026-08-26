#!/usr/bin/env python3
"""generate_demo_manifest.py — Regenera config/demo_videos.json.

Computa size + sha256 de los videos locales y fusiona las URLs de Firebase
(con token) ya presentes en el manifest existente para no perderlas.

Uso: python scripts/generate_demo_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "videos"
MANIFEST = ROOT / "config" / "demo_videos.json"
BUCKET = "infractivision-e8c03.firebasestorage.app"

# Los 5 videos demo con datos ya cargados en presets/infractions_preset.db
FILENAMES = [
    "Av-Condorcanqui.mp4",
    "VID1EDIT \u2010 Hecho con Clipchamp.mp4",
    "VID2COLISEO.MOV",
    "VID2EDIT \u2010 Hecho con Clipchamp.mp4",
    "VID4EDIT \u2010 Hecho con Clipchamp.mp4",
]


def _sha256_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def main() -> None:
    old: dict = {}
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old_by_name = {v["filename"]: v for v in old.get("videos", [])}

    videos = []
    for name in FILENAMES:
        src = VIDEOS / name
        if not src.exists():
            print(f"[skip] {name}: no existe en videos/")
            continue
        sha, size = _sha256_size(src)
        enc = urllib.parse.quote(name, safe="")
        prev = old_by_name.get(name, {})
        videos.append({
            "filename": name,
            "gcs_path": name,
            "size": size,
            "sha256": sha,
            "url": prev.get("url") or f"https://firebasestorage.googleapis.com/v0/b/{BUCKET}/o/{enc}?alt=media&token=PONER_TOKEN",
            "public_url": f"https://storage.googleapis.com/{BUCKET}/{enc}",
        })
        print(f"[ok] {name}  {size}  {sha[:16]}")

    MANIFEST.write_text(
        json.dumps({"version": 1, "base_dir": "videos", "videos": videos},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest regenerado: {MANIFEST} ({len(videos)} videos)")


if __name__ == "__main__":
    main()