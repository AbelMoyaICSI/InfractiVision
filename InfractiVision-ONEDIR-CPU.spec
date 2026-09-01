# -*- mode: python ; coding: utf-8 -*-
"""
InfractiVision - Especificacion PyInstaller ONEDIR (Setup-Online CPU)
ONEDIR ligero sin libs CUDA. Para Setup-Online por defecto.
"""
import struct
import sys
import os
from pathlib import Path

if struct.calcsize("P") * 8 != 64:
    raise SystemExit(
        "[spec ONEDIR-CPU] ERROR: Se requiere Python 3.10 64-bit. "
        f"Detectado: {struct.calcsize('P')*8}-bit ({sys.version}). "
        "Reinstala Python x64."
    )
try:
    import importlib.metadata as _md
    _md.version("opencv-python-headless")
    raise SystemExit(
        "[spec ONEDIR-CPU] ERROR: opencv-python-headless detectado. EasyOCR solo en tests, no en build prod. "
        "Ejecuta: pip uninstall opencv-python-headless opencv-python -y && pip install --no-cache --force-reinstall opencv-python==4.9.0.80"
    )
except Exception as _e:
    if isinstance(_e, SystemExit):
        raise
    pass

BASE_DIR = Path(os.getcwd())
SRC_DIR = BASE_DIR / 'src'
main_script = str(BASE_DIR / 'main.py')

def _add_data(src: Path, dst: str, datas_list: list):
    if src.exists():
        datas_list.append((str(src), dst))
    else:
        print(f"[spec ONEDIR-CPU] skip missing: {src}")

datas: list[tuple[str, str]] = []
for img in ["welcome_bg.png", "icon.ico", "InfractiVision-logo.png"]:
    _add_data(BASE_DIR / "img" / img, "img", datas)
for cfg in ["avenue_config.json", "camera_config.json", "polygon_config.json",
            "time_presets.json", "zones.json", "demo_videos.json", "models_manifest.json"]:
    _add_data(BASE_DIR / "config" / cfg, "config", datas)
_add_data(BASE_DIR / "presets" / "infractions_preset.db", "presets", datas)
_add_data(BASE_DIR / "infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json", ".", datas)
_add_data(BASE_DIR / ".env", ".", datas)

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
excludes = [
    'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'setuptools', 'pip', 'wheel',
    'docstring_parser', 'coverage', 'black', 'mypy', 'flake8', 'pylint', 'autopep8', 'isort', 'bandit', 'safety',
    'tensorboard', 'tensorflow', 'keras', 'onnx', 'triton', 'numba', 'jax', 'cupy', 'dask', 'xarray',
    'bokeh', 'plotly', 'seaborn', 'statsmodels', 'networkx',
    'gensim', 'nltk', 'spacy', 'transformers', 'datasets', 'huggingface_hub',
    # Recorte peso para mantener zip <2GB sin GCS
    'matplotlib.tests', 'mpl_toolkits.tests', 'numpy.tests', 'scipy.tests',
    'sklearn.tests', 'sklearn.datasets', 'sklearn.experimental',
    'pandas.tests', 'PIL.tests', 'tkinter.test', 'test', 'tests',
    'unittest', 'distutils.tests', 'email.tests',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
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
    [],
    exclude_binaries=True,
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='InfractiVision',
)

print("InfractiVision-ONEDIR-CPU.spec listo (ONEDIR Setup-Online ligero)")
