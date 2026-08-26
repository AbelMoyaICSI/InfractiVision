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

def test_spec_has_preset_and_secrets():
    for f in ["InfractiVision.spec","InfractiVision-CPU.spec","InfractiVision-CUDA.spec"]:
        text = (ROOT / f).read_text()
        assert "infractions_preset.db" in text, f"{f}: debe empaquetar presets/infractions_preset.db"
        assert "demo_videos.json" in text, f"{f}: debe empaquetar config/demo_videos.json"
        assert "infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json" in text, f"{f}: debe empaquetar la Service Account de migraciones"
        assert '".env"' in text or ".env" in text, f"{f}: debe empaquetar .env (token Plate Recognizer)"

def test_demo_manifest_valid():
    import json
    m = json.loads((ROOT / "config" / "demo_videos.json").read_text(encoding="utf-8"))
    videos = m["videos"]
    assert len(videos) == 5, "el manifest debe listar 5 videos demo"
    for v in videos:
        assert v["filename"] and v["gcs_path"] and v["sha256"] and v["size"], f"manifest incompleto: {v}"
        assert "firebasestorage.googleapis.com" in v["url"] or "storage.googleapis.com" in v["url"], f"URL inválida: {v}"
    # cada video demo debe tener config en el preset
    import sqlite3
    preset = ROOT / "presets" / "infractions_preset.db"
    assert preset.exists(), "presets/infractions_preset.db debe existir"
    conn = sqlite3.connect(preset)
    names = {r[0] for r in conn.execute("SELECT video_name FROM video_configs")}
    conn.close()
    for v in videos:
        assert v["filename"] in names, f"video demo sin config en preset: {v['filename']}"

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
check("spec preset + secrets", test_spec_has_preset_and_secrets)
check("demo manifest válido", test_demo_manifest_valid)
check("spec compile", test_spec_compile)
check("linux sh syntax", test_linux_sh_syntax)
check("win iss content", test_win_iss_syntax)
check("mac sh syntax", test_mac_sh_syntax)
check("build_online.py --help", test_build_helper)
check("release.yml exists", test_workflow_exists)
check("gpu detection logic", test_gpu_detection_unit)

print("\n" + ("✓ TODO OK" if FAILED==0 else f"✗ {FAILED} fallos"))
sys.exit(FAILED)
