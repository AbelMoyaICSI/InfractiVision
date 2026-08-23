# 🚦 InfractiVision

<div align="center">

![InfractiVision Logo](img/InfractiVision-logo.png)

**Sistema Inteligente de Detección de Infracciones por Cruce en Rojo**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)](https://ultralytics.com)
[![LPRNet](https://img.shields.io/badge/OCR-LPRNet%20Perú-orange.svg)](#-modelos-de-ia)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/AbelMoyaICSI/InfractiVision?style=social)](https://github.com/AbelMoyaICSI/InfractiVision)

*Detección automática de violaciones al semáforo en rojo con YOLOv8 + LPRNet y validación humana asistida*

[🚀 Instalación](#-instalación) • [📖 Manual de Usuario](#-manual-de-usuario) • [🎯 Características](#-características) • [🏗️ Arquitectura](#️-arquitectura) • [🔄 Flujo](#-flujo-de-procesamiento) • [☁️ Cloud](#️-integración-cloud)

</div>

---

## 📋 Descripción

**InfractiVision** detecta vehículos que cruzan en rojo a partir de video grabado. El flujo es **offline por video**: eliges un archivo, configuras la zona y los tiempos del semáforo, procesas, validas las placas y el sistema persiste métricas localmente y las migra a la nube.

### ¿Qué hace realmente?

- **🚦 Detección en rojo**: identifica cruces de un polígono configurable solo durante la fase roja (y un pequeño pre-rojo) del ciclo G → Y → R simulado.
- **🚗 Tracking**: asigna ID por vehículo (centroide / DeepSORT) para evitar duplicados.
- **🔍 Placas**: detecta placas con YOLO dedicado y guarda el mejor crop por infractor (con margen y scoring de calidad).
- **🔤 OCR principal LPRNet**: `LPRNet_Peru_MASTER_FINAL.pth` con contexto regional Trujillo y validación SIIV MTC Perú. Backends alternativos PaddleOCR/EasyOCR seleccionables por env.
- **☁️ Validación en la nube**: `PlateReviewWindow` valida cada crop secuencialmente contra **Plate Recognizer API** (`regions=pe`) con espera anti-límite.
- **📊 NID / NIE**: NID = placas validadas por el operador; NIE = pendientes sin placa o no validados. De ahí se derivan TI y TR.
- **💾 Persistencia local**: SQLite `data/infractions.sqlite` como fuente única.
- **☁️ Migración a Firebase**: Firestore `infractivision-e8c03`, colección `migraciones/{uuid}` con TI/TR/NID/NIE/settings/deteccion (ver [DOC.md §7.2](DOC.md#7-tecnologías-y-migración-a-firebase)).

> Ver el flujo completo con diagramas Mermaid en [DOC.md](DOC.md).

---

## 🌟 Características Principales

### 🧠 Inteligencia Artificial

- **YOLOv8 vehículos** (`yolov8n.pt`): clases 2/5/7 (car/bus/truck), solo en fase roja/pre-roja.
- **YOLO placas** (`license_plate_detector.pt`): sobre cuadrante inferior del vehículo.
- **LPRNet Perú** (`LPRNet_Peru_MASTER_FINAL.pth`): singleton thread-safe con `get_lprnet_predictor()`, precarga en background al arrancar la GUI.
- **SmartPlateCorrector** (`src/infrastructure/ocr/plate_corrector.py`): mapas de confusión (0↔O, 1↔I, 8↔B, etc.) con cache y validación SIIV.
- **Detección nocturna**: por nombre de video (`night`/`nocturno`) o análisis de brillo (< 60) + áreas oscuras. Ajusta confianza y realce de visibilidad.
- **Scoring de calidad** del crop: contraste + bordes + nitidez (Laplaciano) + tamaño.

### 🖥️ Interfaz Gráfica (Tkinter)

- **Welcome** (`src/gui/welcome_window.py`): 3 acciones — *Manual de Usuario*, *Foto Rojo*, *Gestión de Infracciones*. Fondo redimensionado con debounce.
- **Foto Rojo** (`src/gui/red_light_violation_window.py` + `src/core/video/videoplayer_opencv.py`): selector visual de videos (`src/gui/video_selector_window.py`), configuración de polígono/tiempos/avenida, reproductor OpenCV y diálogo de preprocesamiento.
- **PreprocessingDialog** (`src/gui/preprocessing_dialog.py`): orquesta el `OfficialVideoProcessor` en hilo worker y drena resultados a Tk vía cola (`result_queue`). Congela semáforo/timer al entrar a validación.
- **PlateReviewWindow** (`src/presentation/gui/plate_review_window.py`): revisión secuencial de best-crops con Plate Recognizer (cola `queue.Queue` + poller `after(50)` — nunca toca Tk desde el worker).
- **Gestión de Infracciones** (`src/gui/infractions_management_window.py`): tarjetas NID/NIE + paginación de 10 en 10 desde `AppRepository`.

### 🏗️ Arquitectura

- **Clean Architecture** por extracción: `domain/` (entidades, puertos, servicios), `application/` (casos de uso, DTOs), `infrastructure/` (YOLO, OCR, tracking, DB, video), `presentation/` (GUI/API).
- **Composition Root** (`main.py` + `src/composition_root.py`): único lugar que cablea dependencias. `Lazy` proxy para no cargar modelos al arrancar.
- **Multiplataforma**: `src/core/utils/icon.py` (`.ico`/`.png`), `src/core/utils/audio.py` (`winsound`/`paplay`/`afplay`), sin `state("zoomed")` forzado.

### ☁️ Cloud

- **Firebase Firestore** `infractivision-e8c03` vía `firebase-admin 7.5.0` + `firestore_migrator.py:153`. Un documento por ejecución (`migraciones/{uuid}`), no por nombre de video, para no sobrescribir re-procesamientos. Ver [DOC.md §7.2](DOC.md#7-tecnologías-y-migración-a-firebase).
- **Migración al completar validación** (no al terminar el video): un hilo `daemon` (`src/gui/preprocessing_dialog.py:1388`) llama `migrate_single_video_to_firestore()` y registra en `migrations` local. **Destino: Firebase**.

### 📦 Distribución

- **PyInstaller**: `InfractiVision.spec` / `InfractiVision-CPU.spec` / `InfractiVision-CUDA.spec`.
- **CI**: `.github/workflows/deps.yml` (matrix ubuntu+windows, Python 3.10) y `release.yml`.
- **Installer**: `installer/` (stub online multi-OS con detección de GPU).
- **Scripts**: `scripts/` (comparadores OCR, regeneración de indicadores, smoke tests).

---

## 💻 Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **SO** | Windows 10, Ubuntu 18.04, macOS 10.14 | Windows 11, Ubuntu 20.04+ |
| **CPU** | Intel i3 / Ryzen 3 | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **Disco** | 10 GB libres | 50 GB SSD |
| **GPU** | Opcional (CPU fallback) | NVIDIA GTX 1650 Ti+ con driver ≥ 450 (CUDA 11.7) |
| **Python** | 3.10 | 3.10 (fijado en `mise.toml`) |

### Stack pineado (no cambiar sin testear)

`torch 1.13.1` fue compilado contra NumPy 1.x → exige `numpy<2`, mientras `opencv-python ≥5` exige `numpy≥2`. Por eso:

| Paquete | Versión fijada | Motivo |
|---------|---------------|--------|
| `numpy` | `1.26.4` | Compatibilidad torch 1.13.1 |
| `opencv-python` | `4.9.0.80` | No 5.x; no headless |
| `torch` | `1.13.1+cu117` | Índice CUDA 11.7 |
| `torchvision` | `0.14.1+cu117` | A juego con torch |
| `scikit-image` | `0.23.2` | Procesamiento |
| `scikit-learn` | `1.4.2` | Métricas |

Ver `requirements.txt:1` y `requirements-cpu.txt` / `requirements-ocr.txt` para variantes. Tkinter viene con Python en Windows/macOS; en Linux: `sudo apt install python3-tk`.

---

## 🚀 Instalación

### Opción 1: Ejecutable

Descarga el ZIP de [Releases](https://github.com/AbelMoyaICSI/InfractiVision/releases), extrae y ejecuta `InfractiVision.exe` (Windows) o `./InfractiVision` (Linux/macOS).

### Opción 2: Desde código

```bash
git clone https://github.com/AbelMoyaICSI/InfractiVision.git
cd InfractiVision

# Python 3.10 (ver mise.toml:1)
python --version  # 3.10.x

# Crear venv (recomendado) e instalar
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # con CUDA 11.7 (usa --extra-index-url del archivo)
# Alternativas:
# pip install -r requirements-cpu.txt   # sin CUDA
# pip install -r requirements-ocr.txt   # solo OCR (PaddleOCR/EasyOCR)

# Variable para validación en la nube (requerida para Plate Recognizer)
# .env en la raíz (ver .env:1):
# PLATE_RECOGNIZER_API_TOKEN="..."

python main.py
```

### Cloud opcional (Firestore)

1. Proyecto GCP `infractivision-e8c03` con Firestore habilitado.
2. Service Account con `datastore.user` y su JSON en la raíz como `infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json` (ver `src/automations/firestore_migrator.py:30`).
3. Sin credenciales, la app funciona 100% offline; solo falla la migración.

---

## 📖 Manual de Usuario

### Pantalla de Inicio

`src/gui/welcome_window.py:95` — panel izquierdo con imagen `img/welcome_bg.png` redimensionada (máx. 1920×1080, debounce 200 ms) y panel derecho con:

- **Manual de Usuario** → `src/gui/manual_window.py:8`
- **Foto Rojo** → `src/gui/app_manager.py:43`
- **Gestión de Infracciones** → `src/gui/app_manager.py:54`

### Módulo Foto Rojo

1. **Selector visual** (`src/gui/video_selector_window.py:23`): miniaturas, metadatos (duración/resolución/tamaño), estado de configuración (polígono/semáforo/avenida). Acciones: *Seleccionar*, *Configurar*, *Limpiar*, *Eliminar*, *Importar*, *Actualizar*.
2. **Configurar zona**: polígono de la intersección + margen de peligro (`danger_zone_margin_pixels`, `src/infrastructure/configuration/video_config_repository.py:17`). Se guarda en `config/polygon_config.json`.
3. **Configurar semáforo y avenida**: tiempos G/Y/R + `pre_red_seconds`/`green_skip_rate` en `config/time_presets.json`; avenida en `config/avenue_config.json`. Todo vía `VideoConfigRepository:26`.
4. **Reproductor** (`src/core/video/videoplayer_opencv.py:1357`): `play_video()`, overlay del estado, timer HH:mm:ss y `G/Y/R` en segundos, beeps por `src/core/utils/audio.py:1`.
5. **Iniciar procesamiento** (`src/gui/preprocessing_dialog.py:1247`): `_process_video_official()` crea `OfficialVideoProcessor` con los detectores del player y un `callback` que encola `official_frame` / `official_infraction` / `official_complete`. La UI se drena en el hilo de Tk (`_process_results_queue`, `_poll_display_queue`). Al terminar, congela semáforo/timer y abre `PlateReviewWindow`.
6. **Validación** (`src/presentation/gui/plate_review_window.py:17`): un worker por crop llama `PlateRecognizerSnapshotReader.read()` (`src/infrastructure/ocr/cloud_plate_readers.py:23`, `regions=pe`, espera mínima 2 s, reintentos con `Retry-After`), publica en `queue.Queue` y el poller de Tk muestra `Confianza: 0.xx` y habilita el check *Validar*. Botones: *Reintentar actual*, *Exportar validados*, *Completado* (dispara persistencia + migración).

### Gestión de Infracciones

`src/gui/infractions_management_window.py:1` — lee de `AppRepository.list_infractions()` (`src/infrastructure/database/app_repository.py:226`), renderiza tarjetas a color por validación (grid 3/2/1 columnas según ancho), paginación de 10 en 10 con *Cargar más*. Acciones: filtrar por fecha/placa, exportar CSV/Excel/PDF, eliminar, y ver historial de migraciones (`migrations`).

### Configuraciones Avanzadas

| Parámetro | Rango | Default | Dónde |
|-----------|-------|---------|-------|
| Confianza vehículos (YOLO) | 0.1–0.9 | 0.40 | `OfficialVideoProcessor.process:140` |
| Confianza placas (YOLO) | 0.1–0.9 | 0.40 | `src/application/use_cases/process_violation_video.py:221` |
| Tamaño mínimo crop placa | w≥55, h≥30 | 55×30 | `src/application/use_cases/process_violation_video.py:41` |
| Margen de contexto del crop | 0.0–1.0 | 0.5 | `src/application/use_cases/process_violation_video.py:43` |
| Margen zona de peligro | px | 80 | `VideoConfig:17` |
| Pre-rojo | s | 0.5 | `VideoConfig:18` |
| Green skip rate | frames | 60 | `VideoConfig:19` |
| OCR backend | lprnet/paddleocr/easyocr | lprnet | `config/settings.py:32` |
| Confianza mínima OCR | 0.0–1.0 | 0.55 | `config/settings.py:33` |

---

## 🏗️ Arquitectura del Sistema

### Estructura del Proyecto

```
InfractiVision/
├── main.py                              # Composition Root (Tk + DI)
├── config/settings.py                   # Settings inmutables por DI
├── src/composition_root.py              # build_container() + Lazy proxies
│
├── src/presentation/gui/
│   ├── main_window.py                   # Envuelve AppManager legacy
│   ├── plate_review_window.py           # Validación secuencial cloud OCR
│   └── popups/preprocessing_popups.py   # Mixin de popups del diálogo
│
├── src/gui/                             # GUI legacy (Tkinter)
│   ├── app_manager.py                   # Navegación Welcome ↔ Foto Rojo ↔ Gestión
│   ├── welcome_window.py
│   ├── video_selector_window.py
│   ├── red_light_violation_window.py
│   ├── preprocessing_dialog.py          # Orquestación del pipeline oficial
│   ├── infractions_management_window.py
│   └── manual_window.py
│
├── src/application/
│   ├── use_cases/
│   │   ├── process_violation_video.py   # OfficialVideoProcessor (pipeline por video)
│   │   └── process_frame.py             # ProcessFrameUseCase (por frame, Clean)
│   ├── dto/violation_dto.py
│   └── services/
│       ├── traffic_processing_planner.py
│       └── metrics_calculator.py
│
├── src/domain/
│   ├── entities/                        # Violation, Vehicle, TrafficLight, PlateEvidence
│   ├── interfaces/                      # Puertos: Detector, OCR, Tracker, Repository
│   └── services/                        # ViolationService, TrackingService, EvidenceService
│
├── src/infrastructure/
│   ├── ai/                              # YoloVehicleDetector, YoloPlateDetector, VirtualTrafficLightDetector
│   ├── ocr/                             # LPRNetReader, PaddleOCRReader, EasyOCRReader, cloud_plate_readers, plate_corrector
│   ├── tracking/                        # DeepSortTracker (fallback centroide)
│   ├── video/                           # FrameExtractor, VideoReader
│   ├── database/                        # AppRepository (SQLite), SQLite/MySQL repositories
│   ├── configuration/                   # VideoConfig, VideoConfigRepository
│   └── reports/                         # ReportRepository
│
├── src/core/                            # Núcleo legacy (detección, OCR LPRNet, video, tracking)
│   ├── detection/  (vehicle_detector, plate_detector, model_guard)
│   ├── ocr/        (lprnet_engine, recognizer, super_resolution)
│   ├── video/      (videoplayer_opencv)
│   └── traffic/    (vehicle_tracker, semaphore)
│
├── src/automations/
│   ├── firestore_migrator.py            # Migración a migraciones/{uuid}
│   └── cloud_migrator.py                # Legacy
│
├── config/                              # polygon_config.json, time_presets.json, avenue_config.json, zones.json
├── data/                                # infractions.sqlite, output/official/, evidences/, reports/
├── models/                              # yolov8n.pt, license_plate_detector.pt, LPRNet_Peru_MASTER_FINAL.pth, FSRCNN_x3.pb
├── docs/                                # DOC.md (flujo con Mermaid)
├── scripts/                             # Comparadores OCR, regeneración de indicadores, smoke tests
└── .github/workflows/                   # deps.yml, release.yml
```

### Capas (Clean Architecture)

- **Domain**: entidades puras (`Violation`, `Vehicle`, `TrafficLight`, `PlateEvidence:38`) y puertos. Sin dependencias a frameworks.
- **Application**: casos de uso (`OfficialVideoProcessor`, `ProcessFrameUseCase:28`) y servicios (`TrafficProcessingPlanner:1`). Orquestan dominio + puertos.
- **Infrastructure**: adaptadores concretos (YOLO, LPRNet, DeepSORT, SQLite, OpenCV, Firestore).
- **Presentation**: `MainWindow` + `AppManager` + ventanas Tk. Recibe `process_frame_uc` y `traffic_light_state` por DI.

---

## 🧰 Stack Tecnológico del Flujo

Cada paso del flujo usa una tecnología concreta. Detalle completo con diagrama de migración en [DOC.md §7](DOC.md#7-tecnologías-y-migración-a-firebase).

| Capa del flujo | Tecnología | Versión | Uso |
|---------------|-----------|---------|-----|
| GUI | **Tkinter** + `tkcalendar` | stdlib / 1.6.1 | Welcome, selector de vídeos, reproductor, `PlateReviewWindow` |
| Vídeo | **OpenCV** | 4.9.0.80 | `VideoCapture`/`VideoWriter`, overlays, `pointPolygonTest` del polígono |
| Detección | **YOLOv8** (`ultralytics`) | 8.4.120 | Vehículos (car/bus/truck) y placas — dos modelos YOLO |
| Deep Learning | **PyTorch** + CUDA 11.7 | 1.13.1+cu117 | Inferencia YOLO y LPRNet |
| OCR primario | **LPRNet Perú** | `LPRNet_Peru_MASTER_FINAL.pth` | OCR principal con contexto Trujillo + SIIV |
| OCR alternos | PaddleOCR / EasyOCR | opcionales (`requirements-ocr.txt`) | Seleccionables por `INFRACTI_OCR_BACKEND` |
| OCR validación | **Plate Recognizer API** | `requests` | `cloud_plate_readers.py:23` — `regions=pe`, 2 s entre requests |
| Tracking | **DeepSORT** (`deep-sort-realtime`) | 1.3.2 | IDs por vehículo; pipeline oficial usa centroide con fallback |
| Datos | NumPy / SciPy / scikit-learn / scikit-image / pandas / openpyxl | 1.26.4 / 1.11.4 / 1.4.2 / 0.23.2 / 2.1.4 / 3.1.5 | Scoring del crop, exportación |
| Persistencia | **SQLite** (`AppRepository`) | stdlib | Fuente única — `infractions`, `video_configs`, `indicators`, `migrations` |
| **Migración** | **Firebase** (`firebase-admin` + `google-cloud-firestore`) | 7.5.0 / 2.28.1 | **Destino de todas las migraciones → Firestore `infractivision-e8c03`** |

> ☁️ **Destino de migraciones: Firebase** — proyecto `infractivision-e8c03`, colección `migraciones/{uuid}`. Se dispara al completar la validación (`PlateReviewWindow` → `migrate_single_video_to_firestore` en hilo `daemon`). Esquema: `ti`, `tr`, `NID`, `NIE`, `video-name`, `fecha`, `settings{red,green,yellow,polygon}`, `deteccion[{placa,timestamp,confianza,validate}]`. Ver [DOC.md §7.2](DOC.md#7-tecnologías-y-migración-a-firebase) y `src/automations/firestore_migrator.py:153`.

---

## 🔄 Flujo de Procesamiento

El flujo completo con decisiones está documentado en [DOC.md](DOC.md). Resumen del pipeline oficial (`src/application/use_cases/process_violation_video.py:140`):

```mermaid
flowchart TD
    A[main.py: Composition Root<br/>traffic_light_state dict + build_container] --> B[MainWindow / AppManager<br/>Welcome]
    B --> C[Foto Rojo<br/>VideoSelectorWindow]
    C --> D{Video configurado<br/>poligono + G/Y/R + avenida}
    D -- No --> E[Configurar poligono y semaforo<br/>VideoConfigRepository]
    E --> D
    D -- Si --> F[PreprocessingDialog<br/>_process_video_official]
    F --> G[OfficialVideoProcessor.process<br/>TrafficProcessingPlanner]
    G --> H{state_at frame<br/>G / Y / R}
    H -- G/Y sin pre-rojo --> I[should_detect=false<br/>display cada 60 frames]
    I --> H
    H -- R o pre-rojo --> J[YOLO vehiculos<br/>conf 0.40]
    J --> K[CentroidVehicleTracker<br/>update]
    K --> L{_near_polygon<br/>centro inferior vs poligono}
    L -- No --> J
    L -- Si --> M[YOLO placas<br/>sobre cuadrante inferior]
    M --> N{crop viable<br/>w>=55 h>=30 con margen 50%}
    N -- No --> O[Guardar pending<br/>crop vehiculo -> NIE]
    N -- Si --> P[Scoring calidad<br/>contraste+bordes+nitidez+tamano]
    P --> Q{pending en rojo<br/>y placa encontrada}
    Q -- Primera vez --> R[Confirmar infractor<br/>callback infraction_detected]
    R --> S[Guardar best evidence<br/>quality >= best previo]
    Q -- Ya confirmado --> S
    S --> T[Video anotado<br/>banner SEMAFORO + T + G/Y/R]
    T --> U{Fin de video}
    U -- No --> H
    U -- Si --> V[Payload<br/>evidence + pending_infractions<br/>+ report JSON]
    V --> W[PlateReviewWindow<br/>validacion secuencial]
    W --> X[Plate Recognizer API<br/>regions=pe, 2s entre requests]
    X --> Y{Texto reconocido}
    Y -- Si --> Z[NID validado]
    Y -- No --> AA[NIE]
    Z --> AB[AppRepository<br/>infractions + indicators]
    AA --> AB
    AB --> AC[Firestore<br/>migraciones/uuid<br/>TI/TR/NID/NIE/settings/deteccion]
```

Indicadores: **TI** = NID/(NID+NIE)*100, **TR** = duración total / NID validadas (min por infracción, `src/infrastructure/database/app_repository.py:379`).

---

## 🧠 Modelos de IA

| Modelo | Archivo | Rol | Notas |
|--------|---------|-----|-------|
| YOLOv8n vehículos | `models/yolov8n.pt` | Detecta car/bus/truck | `YoloVehicleDetector` vía `ultralytics` |
| YOLO placas | `models/license_plate_detector.pt` | Detecta bbox de placa | Fine-tuned YOLOv8 |
| LPRNet Perú | `models/LPRNet_Peru_MASTER_FINAL.pth` | OCR principal | Singleton `get_lprnet_predictor()` con precarga background |
| FSRCNN x3 | `models/FSRCNN_x3.pb` | Super-resolución opcional | `src/core/ocr/super_resolution.py` |
| Otros LPRNet | `LPRNet_CONSENSO_V2.pth`, `LPRNet_V3_ESPECIALISTA.pth`, `LPRNet_V4_CORREGIDO.pth` | Variantes de entrenamiento | No usados por defecto |

**OCR backends seleccionables** (`config/settings.py:32`, `INFRACTI_OCR_BACKEND`):

- `lprnet` (default) → `LPRNetReader` con `regional_context="Trujillo"` y validación SIIV (`src/core/ocr/recognizer.py:86`).
- `paddleocr` → `PaddleOCRReader` (`src/infrastructure/ocr/paddleocr_reader.py:16`).
- `easyocr` → `EasyOCRReader`.

**Validación cloud** (post-procesamiento, no durante el barrido de frames):

- `PlateRecognizerSnapshotReader` (`src/infrastructure/ocr/cloud_plate_readers.py:23`, `PLATE_RECOGNIZER_API_TOKEN` en `.env:1`).
- `GoogleVisionReader` (alternativa con `GOOGLE_API_KEY`).

---

## 💾 Base de Datos

**SQLite por defecto** (`data/infractions.sqlite`, `config/settings.py:25`), MySQL opcional vía `INFRACTI_MYSQL_URL`.

Tablas en `src/infrastructure/database/app_repository.py:45`:

- `infractions` — una fila por infracción NID/NIE
- `video_configs` — polígono/tiempos/avenida por video
- `indicators` — reporte TI/TR/NID/NIE serializado
- `migrations` — historial local de migraciones a Firestore
- `meta` — clave/valor interno

`AppRepository` es la fuente única para `VideoConfigRepository` y `firestore_migrator`.

---

## ☁️ Integración Cloud — Migración a Firebase

**Destino: Firebase Firestore** (proyecto `infractivision-e8c03`, `src/automations/firestore_migrator.py:30`). Sin Cloud Storage en el flujo oficial. Detalle y stack completo en [DOC.md §7](DOC.md#7-tecnologías-y-migración-a-firebase).

**Documento `migraciones/{uuid}`:**

```json
{
  "ti": 75.0,
  "tr": 0.42,
  "NID": 3,
  "NIE": 1,
  "video-name": "Av-Condorcanqui.mp4",
  "fecha": "2026-08-19T14:30:00",
  "settings": {
    "red": 10, "green": 12, "yellow": 2,
    "polygon": [{"x": 200, "y": 300}, {"x": 1080, "y": 300}, ...]
  },
  "deteccion": [
    {"placa": "T4A-123", "timestamp": "2026-08-19T14:30:05", "confianza": 0.87, "validate": true}
  ]
}
```

Se crea con `migrate_single_video_to_firestore()` al completar la validación; un hilo `daemon` (`src/gui/preprocessing_dialog.py:1388`) y `add_migration_to_history()` en `src/gui/infractions_management_window.py:21` registra el historial local. Legacy `cloud_migrator.py:61` (Cloud Storage `infractivision-474103`) no es parte del flujo oficial.

---

## 📊 Indicadores (NID / NIE / TI / TR)

Calculados en `AppRepository.compute_indicators_report()` (`src/infrastructure/database/app_repository.py:379`) y mostrados en el diálogo y en Gestión:

| Indicador | Fórmula |
|-----------|---------|
| **NID** | evidencias validadas (✓) con placa reconocida |
| **NIE** | pendientes sin placa + no validados |
| **TI** | `NID / (NID+NIE) * 100` (%) |
| **TR** | `duración total del video / NID` (min por infracción) |

Coherentes entre SQLite, el panel y Firestore (misma sesión).

---

## 🧪 Testing y Calidad

```bash
# Suite completa
python -m pytest tests/ -v

# Test del pipeline oficial
python -m pytest tests/test_official_video_processing.py -v

# Smoke test de instalación (usado en CI)
python scripts/ci_smoke_test.py

# Verificación de dependencias multiplataforma (CI matrix ubuntu+windows, Python 3.10)
# .github/workflows/deps.yml:25

# Regenerar indicadores desde SQLite
python scripts/regenerar_indicadores.py
```

---

## 🛠️ Desarrollo

```bash
pip install pre-commit
pre-commit install
pip install -r requirements.txt
```

- Estilo: Black + isort
- Linting: flake8 / pylint
- Tests: pytest + coverage
- Commits: convencionales, sin `Co-Authored-By`

### Añadir un nuevo backend OCR

Implementa `OCRReaderPort` (`src/domain/interfaces/ocr_interface.py:1`) y regístralo en `src/composition_root.py:84` (`_build_ocr`).

---

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).

- ✅ Uso comercial, modificación, distribución, uso privado
- ❗ Sin garantía

---

## 🙏 Agradecimientos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Plate Recognizer](https://platerecognizer.com/) — validación de placas
- [Firebase / Google Cloud](https://cloud.google.com/) — Firestore

---

<div align="center">

### 🌟 ¡Si InfractiVision te resulta útil, deja una estrella! ⭐

**Desarrollado por [Abel Moya](https://github.com/AbelMoyaICSI)**

[🔝 Volver al inicio](#-infractivision)

</div>
