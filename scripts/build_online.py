#!/usr/bin/env python3
"""
build_online.py — Build helper para instalador ONLINE
Uso:
  python scripts/build_online.py --variant cpu   # usa requirements-cpu.txt
  python scripts/build_online.py --variant cuda  # usa requirements.txt (+cu117)
  python scripts/build_online.py --variant all   # ambos (CI)
  python scripts/build_online.py --variant cpu --zip  # + zip para Releases
"""
from __future__ import annotations
import argparse, subprocess, sys, shutil, os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str], **kw):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)

def build_variant(variant: str, do_zip: bool):
    spec = ROOT / f"InfractiVision-{variant.upper()}.spec"
    if not spec.exists():
        spec = ROOT / "InfractiVision.spec"
    print(f"[build] variant={variant} spec={spec}")
    # Limpieza previa
    for d in ["build", "dist"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    # PyInstaller
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec)], cwd=str(ROOT))
    exe = ROOT / "dist" / ("InfractiVision.exe" if os.name == "nt" else "InfractiVision")
    if not exe.exists():
        # fallback onefile name
        candidates = list((ROOT / "dist").glob("InfractiVision*"))
        print(f"[build] candidates: {candidates}")
        if not candidates:
            raise SystemExit("Build falló: no se generó dist/InfractiVision")
    print(f"[build] OK variant={variant}")
    if do_zip:
        zip_name = ROOT / "dist" / f"InfractiVision-{variant}-{_platform_tag()}.zip"
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
            for p in (ROOT / "dist").rglob("*"):
                if p.is_file() and p.suffix != ".zip":
                    z.write(p, p.relative_to(ROOT / "dist"))
        print(f"[zip] {zip_name} ({zip_name.stat().st_size/1e6:.1f} MB)")
        return zip_name

def _platform_tag() -> str:
    import platform
    sys_tag = {"Windows": "Win", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), platform.system())
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"): arch = "x64"
    elif arch in ("aarch64", "arm64"): arch = "arm64"
    return f"{sys_tag}-{arch}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cpu","cuda","all"], default="cpu")
    ap.add_argument("--zip", action="store_true", help="empaquetar zip para Releases")
    args = ap.parse_args()
    variants = ["cpu","cuda"] if args.variant == "all" else [args.variant]
    for v in variants:
        build_variant(v, args.zip)

if __name__ == "__main__":
    main()
