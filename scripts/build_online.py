#!/usr/bin/env python3
"""
build_online.py — Build helper para instalador ONLINE (ONEDIR prioritario)
Uso:
  python scripts/build_online.py --variant cpu        # ONEDIR CPU (dist/InfractiVision/)
  python scripts/build_online.py --variant cuda       # ONEDIR CUDA (dist/InfractiVision/)
  python scripts/build_online.py --variant cpu --onefile  # ONEFILE legacy portable
  python scripts/build_online.py --variant all        # ambos ONEDIR (CI)
  python scripts/build_online.py --variant cpu --zip  # + zip para Releases
  Nota: Setup-Online ONEDIR arranca <1.5s (sin extraccion _MEIPASS).
"""
from __future__ import annotations
import argparse, subprocess, sys, shutil, os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str], **kw):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)

def build_variant(variant: str, do_zip: bool, onedir: bool = True):
    # Prioridad Setup-Online: ONEDIR (sin extraccion). ONEFILE solo si --onefile.
    suffix = "-ONEDIR" if onedir else ""
    spec = ROOT / f"InfractiVision{suffix}-{variant.upper()}.spec"
    if not spec.exists():
        spec = ROOT / f"InfractiVision-{variant.upper()}.spec"
    if not spec.exists():
        spec = ROOT / "InfractiVision.spec"
    print(f"[build] variant={variant} onedir={onedir} spec={spec}")
    # Limpieza previa
    for d in ["build", "dist"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    # PyInstaller
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec)], cwd=str(ROOT))
    if onedir:
        exe = ROOT / "dist" / "InfractiVision" / ("InfractiVision.exe" if os.name == "nt" else "InfractiVision")
        dist_dir = ROOT / "dist" / "InfractiVision"
    else:
        exe = ROOT / "dist" / ("InfractiVision.exe" if os.name == "nt" else "InfractiVision")
        dist_dir = ROOT / "dist"
    if not exe.exists():
        candidates = list((ROOT / "dist").glob("**/InfractiVision*"))
        print(f"[build] candidates: {candidates}")
        if not candidates:
            raise SystemExit("Build falló: no se generó dist/InfractiVision")
    print(f"[build] OK variant={variant} -> {exe} ({'ONEDIR' if onedir else 'ONEFILE'})")
    if onedir and dist_dir.exists():
        # ONEDIR: medir peso de carpeta (sin comprimir) para estimar arranque
        total = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
        print(f"[build] ONEDIR size: {total/1e6:.1f} MB en {dist_dir}")
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
    ap.add_argument("--variant", choices=["cpu", "cuda", "all"], default="cpu")
    ap.add_argument("--zip", action="store_true", help="empaquetar zip para Releases")
    ap.add_argument("--onefile", action="store_true", help="usar spec ONEFILE portable en vez de ONEDIR")
    ap.add_argument("--onedir", action="store_true", help="forzar ONEDIR (default para Setup-Online)")
    args = ap.parse_args()
    # Default: ONEDIR para Setup-Online (rapido). ONEFILE solo si --onefile.
    use_onedir = not args.onefile
    if args.onedir:
        use_onedir = True
    variants = ["cpu", "cuda"] if args.variant == "all" else [args.variant]
    for v in variants:
        build_variant(v, args.zip, onedir=use_onedir)

if __name__ == "__main__":
    main()
