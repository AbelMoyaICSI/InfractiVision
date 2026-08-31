# -*- mode: python ; coding: utf-8 -*-
"""
InfractiVision - Especificacion PyInstaller ONLINE (CUDA 12.4 / RTX 5050 sm_120)
Autor: InfractiVision Team - 2025/2026
Modo: ONEFILE, recursos minimos. Videos y datasets NO se empaquetan.
      Este spec BUNDLEA libs CUDA (nvidia-* cu124) y requiere
      requirements.txt con torch==2.6.0+cu124. Ver requirements-cpu.txt para
      el spec CPU que NO bundlea CUDA.
"""

import struct
import sys
import os
from pathlib import Path

if struct.calcsize("P") * 8 != 64:
    raise SystemExit(
        "[spec CUDA] ERROR: Se requiere Python 3.10 64-bit. "
        f"Detectado: {struct.calcsize('P')*8}-bit ({sys.version}). "
        "Reinstala Python x64."
    )

BASE_DIR = Path(os.getcwd())
SRC_DIR = BASE_DIR / 'src'
main_script = str(BASE_DIR / 'main.py')

# ---- Datas con verificacion de existencia (evita crash si falta archivo) ----
def _add_data(src: Path, dst: str, datas_list: list):
    if src.exists():
        datas_list.append((str(src), dst))
    else:
        print(f"[spec] skip missing: {src}")

datas: list[tuple[str, str]] = []
# ONLINE ligero: modelos NO se bundlean (descarga selectiva via model_downloader).
# Se incluye solo el manifest para que el downloader sepa que bajar.
# Los 21MB de .pt se descargan on-demand a APPDATA/InfractiVision/models.
for img in ["welcome_bg.png", "icon.ico", "InfractiVision-logo.png"]:
    _add_data(BASE_DIR / "img" / img, "img", datas)

# Configs por defecto (solo JSON, no secrets/data/videos)
for cfg in ["avenue_config.json", "camera_config.json", "polygon_config.json",
            "time_presets.json", "zones.json", "demo_videos.json", "models_manifest.json"]:
    _add_data(BASE_DIR / "config" / cfg, "config", datas)

# Preset de BD (seed versionable con video_configs de los videos demo)
_add_data(BASE_DIR / "presets" / "infractions_preset.db", "presets", datas)

# Secretos necesarios POST-instalacion (migraciones Firestore + Plate Recognizer).
# Se empaquetan SOLO si existen al compilar (el CI sin secrets no rompe).
# ADVERTENCIA: quedan extraibles del onefile; rotar keys requiere re-build.
_add_data(BASE_DIR / "infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json", ".", datas)
_add_data(BASE_DIR / ".env", ".", datas)

# Nota: data/, videos/ se excluyen a proposito para instalador ONLINE
#       (los videos demo se descargan al instalar desde config/demo_videos.json)

hiddenimports = [
    'tkinter', 'tkinter.messagebox', 'tkinter.filedialog', 'tkinter.ttk',
    'cv2', 'numpy', 'numpy._core', 'numpy._core.multiarray', 'numpy._core.numeric',
    'numpy._core.umath', 'numpy.lib', 'numpy.lib.format',
    'numpy.testing', 'numpy.f2py', 'numpy.distutils',
    'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageDraw', 'PIL.ImageFont',
    'torch', 'torch.nn', 'torch.nn.functional', 'torch._dynamo', 'torch.fx',
    'torch.fx.experimental', 'torch.fx.experimental.symbolic_shapes',
    'torchvision', 'torchvision.transforms',
    'ultralytics', 'ultralytics.models', 'ultralytics.models.yolo', 'ultralytics.nn', 'ultralytics.utils',
    'sklearn', 'sklearn.cluster', 'sklearn.cluster._kmeans', 'sklearn.preprocessing',
    'sklearn.preprocessing._data', 'sklearn.metrics', 'sklearn.neighbors', 'sklearn.utils',
    'sympy', 'sympy.core', 'sympy.utilities',
    'requests', 'urllib3', 'urllib3.util', 'urllib3.poolmanager',
    'firebase_admin', 'firebase_admin.firestore', 'firebase_admin.storage', 'firebase_admin.credentials',
    'google.cloud.firestore', 'google.cloud.storage', 'google.auth', 'google.auth.transport', 'google.oauth2',
    'flask', 'flask.helpers',
    'pandas', 'openpyxl',
    'tkcalendar',
    'psutil', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends', 'matplotlib.backends.backend_agg',
    'scipy', 'scipy.spatial', 'scipy.spatial.distance', 'scipy.cluster', 'scipy.cluster.hierarchy',
    'scipy.cluster._hierarchy', 'scipy.linalg', 'scipy.optimize', 'scipy.stats', 'scipy.ndimage',
    # InfractiVision
    'src.path_helper',
    'src.composition_root',
    'src.presentation.gui.main_window',
    'src.gui.app_manager', 'src.gui.welcome_window', 'src.gui.video_selector_window',
    'src.gui.preprocessing_dialog', 'src.gui.red_light_violation_window',
    'src.gui.infractions_management_window', 'src.gui.manual_window',
    'src.core.video.videoplayer_opencv',
    'src.core.detection.vehicle_detector', 'src.core.detection.plate_detector', 'src.core.detection.anpr',
    'src.core.ocr.recognizer', 'src.core.processing.plate_processing',
    'src.core.processing.plate_ocr_enhancer', 'src.core.processing.resolution_process',
    'src.core.processing.superresolution', 'src.core.traffic_signal.semaphore',
    'src.core.utils.paths', 'src.core.utils.timestamp', 'src.core.utils.icon', 'src.core.utils.audio',
    'src.automations.cloud_migrator',
    'src.infrastructure.ai.yolo_detector', 'src.infrastructure.ai.plate_detector',
    'src.infrastructure.ocr.lprnet_reader', 'src.infrastructure.database.sqlite_repository',
    'src.infrastructure.storage.demo_video_downloader',
    'src.infrastructure.storage.model_downloader',
    'src.infrastructure.ocr.cloud_plate_readers',
]

pathex = [str(BASE_DIR), str(SRC_DIR)]
binaries = []
# CUDA build: intentar bundlear libs nvidia si están instaladas (torch 2.6+cu124).
# En CI cuda estos paquetes existen (pip install -r requirements.txt); en dev sin cuda se ignora.
try:
    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
    _nvidia_datas = []
    _nvidia_bins = []
    for _pkg in ["nvidia", "nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn",
                 "nvidia.cusparse", "nvidia.cufft", "nvidia.curand", "nvidia.cusolver",
                 "nvidia.nccl", "nvidia.nvjitlink", "nvidia.nvtx", "triton"]:
        try:
            _nvidia_bins += collect_dynamic_libs(_pkg)
            _nvidia_datas += collect_data_files(_pkg)
        except Exception:
            pass
    datas += _nvidia_datas
    binaries += _nvidia_bins
    if _nvidia_bins:
        print(f"[spec CUDA] nvidia/triton libs bundled: {len(_nvidia_bins)} binaries, {len(_nvidia_datas)} datas")
except Exception as e:
    print(f"[spec CUDA] nvidia collect skipped (no cuda venv): {e}")

excludes = [
    'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'setuptools', 'pip', 'wheel',
    'docstring_parser', 'coverage', 'black', 'mypy', 'flake8', 'pylint', 'autopep8', 'isort', 'bandit', 'safety',
    # Mantener tensorflow/onnx excluidos, PERO triton NO se excluye en CUDA
    # (torch 2.6 Inductor lo usa si está; si no está, torch hace fallback sin error).
    'tensorboard', 'tensorflow', 'keras', 'onnx', 'numba', 'jax', 'cupy', 'dask', 'xarray',
    'bokeh', 'plotly', 'seaborn', 'statsmodels', 'networkx',
    'gensim', 'nltk', 'spacy', 'transformers', 'datasets', 'huggingface_hub',
]

a = Analysis(
    [main_script],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='InfractiVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="64bit",
    codesign_identity=None,
    entitlements_file=None,
    icon=str(BASE_DIR / 'img' / 'icon.ico'),
    version=None,
    uac_admin=False,
    uac_uiaccess=False,
)

print("InfractiVision-CUDA.spec ONLINE listo (CUDA 12.4 / RTX 5050 sm_120, nvidia libs bundled si disponible) [64bit]")
