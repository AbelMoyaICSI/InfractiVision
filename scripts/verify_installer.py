#!/usr/bin/env python3
"""verify_installer.py — Smoke tests offline para instaladores online (sin red)
Valida: specs saneados, stubs sintacticamente correctos, GPU detection logic, artefactos.
Uso: python scripts/verify_installer.py
"""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
FAILED = 0

def check(name: str, fn):
    global FAILED
    try:
        fn()
        print(f"✓ {name}")
    except Exception as e:
        FAILED += 1
        print(f"✗ {name}: {e}")

def test_spec_no_videos():
    text = (ROOT / "InfractiVision.spec").read_text()
    # No debe tener datas con videos/secrets (comentarios si permiten mencion)
    assert '("videos"' not in text and "('videos'" not in text, "spec no debe empaquetar videos como data"
    assert '("secrets"' not in text and "('secrets'" not in text, "spec no debe referenciar secrets como data"
    assert "data\", \"data\"" not in text and "('data'" not in text, "spec no debe empaquetar data/"
    # debe incluir modelos criticos
    assert "yolov8n.pt" in text and "license_plate_detector.pt" in text

def test_spec_compile():
    import py_compile
    for f in ["InfractiVision.spec","InfractiVision-CPU.spec","InfractiVision-CUDA.spec"]:
        py_compile.compile(str(ROOT/f), doraise=True)

def test_linux_sh_syntax():
    subprocess.run(["bash","-n", str(ROOT/"installer/linux/install.sh")], check=True)
    assert "has_nvidia_gpu" in (ROOT/"installer/linux/install.sh").read_text()

def test_win_iss_syntax():
    text = (ROOT/"installer/win/online.iss").read_text()
    assert "HasNvidiaGPU" in text and "GetArtifactURL" in text
    assert "C:\\Users\\Abel" not in text, "ISS aun tiene rutas hardcodeadas"
    assert "PrivilegesRequired=lowest" in text

def test_mac_sh_syntax():
    subprocess.run(["bash","-n", str(ROOT/"installer/mac/install.sh")], check=True)
    subprocess.run(["bash","-n", str(ROOT/"installer/mac/build-pkg.sh")], check=True)

def test_build_helper():
    subprocess.run([sys.executable, "scripts/build_online.py", "--help"], check=True, cwd=str(ROOT))

def test_workflow_exists():
    assert (ROOT/".github/workflows/release.yml").exists()
    txt = (ROOT/".github/workflows/release.yml").read_text()
    assert "build-artifacts" in txt and "Setup Online" in txt

def test_gpu_detection_unit():
    # Simula logica: si nvidia-smi existe -> cuda else cpu
    sh = (ROOT/"installer/linux/install.sh").read_text()
    assert 'resolve_variant' in sh and 'has_nvidia_gpu' in sh
    # Runtime fallback en python
    assert "cuda:0" in (ROOT/"src/core/ocr/lprnet_engine.py").read_text()

check("spec no videos/secrets", test_spec_no_videos)
check("spec compile", test_spec_compile)
check("linux sh syntax", test_linux_sh_syntax)
check("win iss content", test_win_iss_syntax)
check("mac sh syntax", test_mac_sh_syntax)
check("build_online.py --help", test_build_helper)
check("release.yml exists", test_workflow_exists)
check("gpu detection logic", test_gpu_detection_unit)

print("\n" + ("✓ TODO OK" if FAILED==0 else f"✗ {FAILED} fallos"))
sys.exit(FAILED)
