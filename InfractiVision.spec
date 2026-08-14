# -*- mode: python ; coding: utf-8 -*-

"""
InfractiVision - Especificación PyInstaller
Ejecutable robusto con TODOS los recursos incluidos
Autor: InfractiVision Team
Fecha: 2025
"""

import sys
import os
from pathlib import Path

# Configurar rutas base
BASE_DIR = Path(os.getcwd())
SRC_DIR = BASE_DIR / 'src'

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS
# ============================================================================

# Scripts principales
main_script = str(BASE_DIR / 'main.py')

# Datos adicionales (archivos que NO son .py)
datas = [
    # === MODELOS AI CRÍTICOS ===
    (str(BASE_DIR / 'models' / 'yolov8n.pt'), 'models'),
    (str(BASE_DIR / 'models' / 'license_plate_detector.pt'), 'models'),
    # === ARCHIVADOS EN LIMPIEZA ===
    # (str(BASE_DIR / 'models' / 'sort'), 'models/sort'),  # ARCHIVADO
    # (str(BASE_DIR / 'models' / 'automatic-number-plate-recognition-python-yolov8-main'), 'models/automatic-number-plate-recognition-python-yolov8-main'),  # ARCHIVADO
    
    # === IMÁGENES Y RECURSOS VISUALES ===
    (str(BASE_DIR / 'img' / 'welcome_bg.png'), 'img'),
    (str(BASE_DIR / 'img' / 'icon.ico'), 'img'),
    (str(BASE_DIR / 'img' / 'InfractiVision-logo.png'), 'img'),
    
    # === CONFIGURACIONES JSON ===
    (str(BASE_DIR / 'config'), 'config'),
    
    # === CARPETAS DE DATOS ===
    (str(BASE_DIR / 'data'), 'data'),
    
    # === CREDENCIALES ===
    (str(BASE_DIR / 'secrets'), 'secrets'),
    
    # === VIDEOS DE DEMO ===
    (str(BASE_DIR / 'videos'), 'videos'),
]

# Módulos ocultos requeridos
hiddenimports = [
    # === CORE PYTHON ===
    'tkinter',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.ttk',
    
    # === COMPUTER VISION ===
    'cv2',
    'numpy',
    'numpy._core',
    'numpy._core.multiarray',
    'numpy._core.numeric',
    'numpy._core.umath',
    'numpy.lib',
    'numpy.lib.format',
    'numpy.testing',
    'numpy.f2py',
    'numpy.distutils',  # ✅ REQUERIDO por scipy
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'imutils',
    'shapely',
    'shapely.geometry',
    'pyclipper',
    
    # === MACHINE LEARNING ===
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch._dynamo',
    'torch.fx',
    'torch.fx.experimental',
    'torch.fx.experimental.symbolic_shapes',
    'torchvision',
    'torchvision.transforms',
    'ultralytics',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.nn',
    'ultralytics.utils',
    'sklearn',
    'sklearn.cluster',
    'sklearn.cluster._kmeans',
    'sklearn.preprocessing',
    'sklearn.preprocessing._data',
    'sklearn.metrics',
    'sklearn.neighbors',
    'sklearn.utils',
    
    # === SYMBOLIC MATH (REQUERIDO POR PYTORCH) ===
    'sympy',
    'sympy.core',
    'sympy.utilities',
    
    'easyocr',
    'easyocr.easyocr',
    'easyocr.utils',
    'easyocr.craft_utils',
    'easyocr.imgproc',
    'pytesseract',
    
    # === NETWORK & API ===
    'requests',
    'urllib3',
    'urllib3.util',
    'urllib3.poolmanager',
    'firebase_admin',
    'firebase_admin.firestore',
    'firebase_admin.storage',
    'firebase_admin.credentials',
    'google.cloud.firestore',
    'google.cloud.storage',
    'google.auth',
    'google.auth.transport',
    'google.oauth2',
    'flask',
    'flask.helpers',
    
    # === DATA PROCESSING ===
    'pandas',
    'json',
    'csv',
    'openpyxl',
    
    # === UI COMPONENTS ===
    'tkcalendar',
    
    # === UTILITIES ===
    'psutil',
    'threading',
    'multiprocessing',
    'queue',
    'datetime',
    'pathlib',
    'shutil',
    'glob',
    're',
    'base64',
    'hashlib',
    'warnings',
    'collections',
    'itertools',
    'uuid',
    'getpass',
    'socket',
    'sys',
    'os',
    'time',
    'json',
    'csv',
    'logging',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
    
    # === SCIPY CRITICAL (NO EXCLUIR) ===
    'scipy',
    'scipy.spatial',
    'scipy.spatial.distance',
    'scipy.cluster',
    'scipy.cluster.hierarchy',
    'scipy.cluster._hierarchy',
    'scipy.linalg',
    'scipy.optimize',
    'scipy.stats',
    'scipy.ndimage',
    
    # === INFRACTIVISION MODULES ===
    'src.path_helper',
    'src.gui.app_manager',
    'src.gui.welcome_window',
    'src.gui.video_selector_window',
    'src.gui.preprocessing_dialog',
    'src.gui.red_light_violation_window',
    'src.gui.infractions_management_window',
    'src.gui.manual_window',
    'src.core.video.videoplayer_opencv',
    'src.core.detection.vehicle_detector',
    'src.core.detection.plate_detector',
    'src.core.detection.anpr',
    'src.core.ocr.recognizer',
    'src.core.processing.plate_processing',
    'src.core.processing.plate_ocr_enhancer',
    'src.core.processing.resolution_process',
    'src.core.processing.superresolution',
    'src.core.traffic_signal.semaphore',
    'src.core.utils.paths',
    'src.core.utils.timestamp',
    'src.automations.cloud_migrator',
]

# Rutas de módulos
pathex = [
    str(BASE_DIR),
    str(SRC_DIR),
]

# Binarios adicionales (archivos .dll, .so, etc.)
binaries = []

# Exclusiones (para reducir tamaño y acelerar arranque)
excludes = [
    # === HERRAMIENTAS DE DESARROLLO ===
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'pytest',
    'setuptools',
    'distutils',
    'pip',
    'wheel',
    'docstring_parser',
    'coverage',
    'black',
    'mypy',
    'flake8',
    'pylint',
    'autopep8',
    'isort',
    'bandit',
    'safety',
    
    # === ML FRAMEWORKS NO USADOS ===
    'tensorboard',
    'tensorflow',
    'keras',
    'onnx',
    'triton',
    'numba',
    'jax',
    'cupy',
    'dask',
    'xarray',
    
    # === VISUALIZACIÓN AVANZADA NO USADA ===
    'bokeh',
    'plotly',
    'seaborn',
    'statsmodels',
    'networkx',
    
    # === NLP NO USADO ===
    'gensim',
    'nltk',
    'spacy',
    'transformers',
    'datasets',
    'huggingface_hub',
    
    # 'sympy',  # ❌ NO EXCLUIR: PyTorch lo requiere
    # 'matplotlib',  # ❌ NO EXCLUIR: Requerido para análisis
    # 'numpy.f2py',     # ❌ NO EXCLUIR: scipy lo necesita
    # 'numpy.distutils', # ❌ NO EXCLUIR: scipy lo necesita
    # 'numpy.testing',  # ❌ NO EXCLUIR: scipy lo necesita
]

# ============================================================================
# ANÁLISIS PRINCIPAL
# ============================================================================

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
    # === OPTIMIZACIONES DESHABILITADAS POR NUMPY 2.x ===
    optimize=0,  # SIN optimización para evitar errores
)

# ============================================================================
# RECOLECCIÓN DE ARCHIVOS
# ============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ============================================================================
# CONFIGURACIÓN DEL EJECUTABLE
# ============================================================================

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
    strip=False,  # ❌ DESHABILITADO: Causa problemas con numpy 2.x
    upx=False,    # ❌ DESHABILITADO: Incompatible con numpy 2.x
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Ventana sin consola (GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(BASE_DIR / 'img' / 'icon.ico'),  # Icono del ejecutable
    version=None,
    uac_admin=False,  # No requiere permisos de administrador
    uac_uiaccess=False,
    # === OPTIMIZACIONES DESHABILITADAS ===
    # optimize=0,  # Ya configurado en Analysis
)

# ============================================================================
# DISTRIBUCIÓN FINAL (ONEFILE)
# ============================================================================

# Si quieres un solo archivo ejecutable, descomenta la siguiente sección:
"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InfractiVision_Portable'
)
"""

print("✅ InfractiVision.spec creado exitosamente!")
print("📁 Incluye TODOS los recursos necesarios:")
print("   🤖 Modelos AI (YOLO, License Plate)")
print("   🖼️ Imágenes (logos, iconos, fondos)")
print("   ⚙️ Configuraciones JSON")
print("   📂 Datos y estructuras")
print("   🔐 Credenciales Firebase")
print("   🎥 Videos de demostración")
print("🚀 Listo para construir el ejecutable!")
