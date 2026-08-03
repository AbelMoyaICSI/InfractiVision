# Comparativa GUI + dependencias: `main` vs `feature/cli_and_test`

Rango: `git diff main...HEAD` (5 commits). 148 archivos tocados totales; aquí solo los archivos de la GUI y sus dependencias directas.

## Resumen ejecutivo

La rama migra la GUI legacy a **Clean Architecture por extracción**: el `preprocessing_dialog.py` pasó de ~4.300 a ~4.700 líneas pero perdió ~4.099 (el diálogo ahora delega en ~10 módulos nuevos). El hilo conductor es:

1. **Extracción de clases gigantes** a capas (`domain/`, `application/`, `infrastructure/`).
2. **Inyección opcional y retro-compatible** del `ProcessFrameUseCase` y un `traffic_light_state` dict en la GUI (seams Clean Architecture).
3. **Portabilidad multiplataforma**: se eliminan `winsound`, `iconbitmap` directo y `state("zoomed")` (evita segfault en Linux).
4. **Seguridad de hilos en Tk**: wrappers `_safe_after` / `_safe_*` con `winfo_exists()`.

---

## 1. Archivos de la GUI (`src/gui/`)

| Archivo | Cambio | Insert/Delete |
|---|---|---|
| `app_manager.py` | Se quita auto-maximizado `state("zoomed")` y los imports cv2/numpy/PIL; acepta `process_frame_uc` y `traffic_light_state` y los propaga a la ventana de Foto Rojo | +18 / −28 |
| `infractions_management_window.py` | Se desactiva `window.state("zoomed")` (comentado) por portabilidad | +6 / −3 |
| `manual_window.py` | Icono centralizado en `set_window_icon()` (multiplataforma) | +2 / −4 |
| `preprocessing_dialog.py` | **Refactor principal** (detalle abajo) | +205 / −4.099 |
| `red_light_violation_window.py` | `create_violation_window()` acepta kwargs opcionales `process_frame_uc` / `traffic_light_state` y los pasa al `VideoPlayerOpenCV` | +11 / −2 |
| `video_selector_window.py` | Icono centralizado; wrappers `_safe_display_videos`, `_safe_update_thumbnail`, `_safe_display_video_info`, `_safe_display_status` (evitan `TclError` al actualizar Tk desde hilos) | +79 / −44 |
| `welcome_window.py` | Carga de imagen rediseñada: límite preventivo 1920×1080, resize con debounce 200 ms, `destroy()` cancela jobs pendientes, se quitan `cursor="hand2"` y emojis | +104 / −34 |

### Detalle: `preprocessing_dialog.py` (el cambio grande)

- `PreprocessingDialog` ahora hereda de **`PreprocessingPopupsMixin`** (popups extraídos).
- Se **eliminan del archivo** las clases que ahora viven en capas:
  - `SmartPlateCorrector` → `infrastructure/ocr/plate_corrector.py`
  - `ThesisMetricsCalculator` → `application/services/metrics_calculator.py`
  - `IntelligentTrafficOptimizer` → `application/services/processing_planner.py`
  - `IntelligentVehicleTracker` → `domain/services/intelligent_tracker.py`
  - `PlateClassificationSystem` → `domain/services/plate_classification.py`
- **Nuevo flujo oficial**: `_process_video_official()` delega en `OfficialVideoProcessor` (use case) y alimenta Tk vía cola; `_open_official_review()` abre `PlateReviewWindow` con `PlateEvidence` construidos del payload. El flujo legacy queda en `_process_video_legacy()` como respaldo.
- **Gestión de callbacks Tk**: `_safe_after`, `_safe_after_idle`, `_cancel_all_after` y `_after_ids` para cancelar timers al cerrar.

---

## 2. Dependencias modificadas

| Archivo | Cambio | Insert/Delete |
|---|---|---|
| `src/core/video/videoplayer_opencv.py` | **2º refactor grande**: `winsound`→`play_beep`; `iconbitmap`→`set_window_icon`; nuevo `_start_semaforo_state_bridge()` (sincroniza el semáforo Tk con el dict cada 250 ms); kwargs `process_frame_uc`/`traffic_light_state` opcionales; `_sync_hardware_state()`; se **eliminan ~17 métodos legacy** (polígonos, cámaras, hardware, super-resolución, métricas: `save_polygon`, `gestionar_camaras`, `detect_hardware`, `_apply_super_resolution`, etc.) | +121 / −673 |
| `src/core/traffic_signal/semaphore.py` | Se elimina método `show_state()` (sin callers tras el refactor) | 0 / −8 |
| `config/settings.py` | **Nuevo**: clases `ModelPaths`, `DatabaseSettings`, `OCRSettings`, `DetectionSettings`, `StoragePaths`, `ConfigPaths`, `Settings` + `load_settings()` | +76 |
| `config/avenue_config.json` | Entrada para "Night Time Traffic Camera video (DEMO 4).mp4" | +1 |
| `config/time_presets.json` | Preset de tiempos para el mismo video demo | +6 |
| `requirements.txt` | `torch==1.13.1+cu117` / `torchvision==0.14.1+cu117` (índice CUDA); añade `deep-sort-realtime`, `easyocr`, `python-dotenv`; comentarios de instalación multiplataforma | +44 / −4 |

---

## 3. Dependencias nuevas (creadas en esta rama)

| Archivo | Líneas | Rol |
|---|---|---|
| `src/core/utils/icon.py` | 51 | `set_window_icon()` — helper multiplataforma (`.ico` en Windows, `.png` en Linux/macOS) |
| `src/core/utils/audio.py` | 77 | `play_beep()` / `play_sequence()` — reemplaza `winsound` (usa `paplay`/`aplay`/`afplay` según SO) |
| `src/infrastructure/ocr/plate_corrector.py` | 455 | `SmartPlateCorrector` (extraído de la GUI) |
| `src/application/services/metrics_calculator.py` | 73 | `ThesisMetricsCalculator` |
| `src/application/services/processing_planner.py` | 227 | `IntelligentTrafficOptimizer` (plan de procesamiento por segmentos) |
| `src/domain/services/intelligent_tracker.py` | 309 | `IntelligentVehicleTracker` (tracking + infracciones por polígono) |
| `src/domain/services/plate_classification.py` | 365 | `PlateClassificationSystem` (clasificación y validación de placas) |
| `src/presentation/gui/popups/preprocessing_popups.py` | 1343 | `PreprocessingPopupsMixin` — todos los popups/sonidos/errores del diálogo |
| `src/presentation/gui/plate_review_window.py` | 186 | `PlateReviewWindow` — revisión manual de evidencia + exportación |
| `src/presentation/gui/main_window.py` | 59 | `MainWindow` — composition root de presentación; embebe `AppManager` legacy y expone `process_frame_use_case` |
| `src/application/use_cases/process_violation_video.py` | 253 | `OfficialVideoProcessor` — use case del pipeline oficial (antes `annotate_video`) |
| `src/domain/entities/plate_evidence.py` | 38 | `PlateEvidence` — DTO entre use case y ventana de revisión |
| `src/composition_root.py` | 158 | `build_container()` — DI: repositorio, OCR, detectores, tracker, semáforo virtual |

---

## 4. Entry point y orquestación

`main.py` reescrito como **Composition Root**: crea un dict `traffic_light_state = {"value": "green"}`, arma el `Container` vía `build_container(traffic_light_state_provider=...)`, instancia `MainWindow` (que monta el `AppManager` legacy) y expone `root.tk_infractivision = {"container", "traffic_light_state"}`. La precarga LPRNet en background se mantiene pero con logger.

---

## 5. Dependencias de la GUI **sin cambios**

`src/automations/cloud_migrator.py`, `src/path_helper.py`, `src/core/utils/timestamp.py` existen en `main` y no fueron modificadas en la rama.

---

## Observaciones

- **Puente GUI ↔ Clean Architecture**: el semáforo Tk no se acopla al use case; el bridge `_start_semaforo_state_bridge` copia `current_state` al dict cada 250 ms, y `VirtualTrafficLightDetector` lo lee. La GUI **funciona idéntica** si los kwargs van `None` (retro-compat 100%).
- **Deuda**: hay restos del refactor en la GUI — imports redundantes (dos docstrings duplicados en `PreprocessingDialog`, `Path` importado dos veces en `preprocessing_dialog.py`) y el flujo legacy convive con el oficial.
- **Riesgo**: `videoplayer_opencv.py` perdió `gestionar_poligonos`, `gestionar_camaras` y `_apply_super_resolution`; conviene confirmar que ningún caller legacy las usaba (los scripts CLI no las referencian).
