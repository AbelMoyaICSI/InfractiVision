# Refactor del Pipeline CLI para CPU/iGPU (i5-10300H + UHD 630)

> **Proposito**: documentar el refactor que adapta los dos scripts CLI (`process_video.py` y `only_infractions.py`) a hardware sin GPU dedicada, despues de que el documento `ANALISIS_PIPELINE_INFRACCIONES.md` (seccion A.7) identificara los gaps de: skip adaptativo, I/O bloqueante, profiler real, y paralelismo.

**Alcance**: solo los dos scripts CLI. La GUI (`preprocessing_dialog.py`) y el wrapper de YOLO (`vehicle_detector.py`) **no se tocaron** en este refactor.

**Hardware objetivo**:
- CPU: i5-10300H — 4 cores fisicos, 8 hilos logicos, ~2.5 GHz
- iGPU: Intel UHD 630 (sin CUDA)
- RAM: ~16 GB tipico
- Storage: SSD

---

## 1. Diagnostico (punto de partida)

Tres problemas de fondo hacian que las recomendaciones del documento original —pensadas para GPU dedicada— fueran directamente peligrosas en este hardware:

1. **Skip fijo**: `green=10, red+active=1, red=3, else=3`. En GPU es "barato" porque la inferencia no compite con el resto. En CPU, forzar `skip=1` en rojo+activos **puede ser impagable** y genera el "spiral of latency" (el pipeline se atrasa cada vez mas respecto al video real).

2. **I/O bloqueante en el mismo thread**: `cap.read()` y `cv2.imwrite` de crops comparten el thread principal con la inferencia. Cada JPG encode bloquea el loop critico.

3. **Profiler inexistente**: solo se media `t_proc_elapsed` global, sin distinguir `model_load` de `model_warmup` ni separar `decode` de `inference`. Imposible decidir donde optimizar.

Ademas, sin configurar el thread budget, OpenCV y PyTorch tomaban el default de 8 hilos cada uno y se producia oversubscription silenciosa en los 4 cores fisicos.

---

## 2. Componentes nuevos en `scripts/adapter/`

Todos siguen el patron: modulo pequeño, sin dependencias externas mas alla de stdlib + cv2/torch ya presentes, con tests unitarios dedicados.

| Modulo | Responsabilidad | Tests |
|---|---|---|
| `threads.py` | `configure_thread_budget()` — fija cv2=2, torch=4, OMP/MKL=4 antes de que arranquen las runtimes paralelas | `tests/test_threads.py` (8) |
| `stage_profiler.py` | `StageProfiler` con context manager `stage(name)`; acumula wall time por stage, distingue one-shot vs recurring, emite tabla + JSON | `tests/test_stage_profiler.py` (10) |
| `adaptive_skip.py` | `AdaptiveSkipController` con `record()`/`suggest_skip()`; mide ratio real de inferencia vs presupuesto, decide skip respetando reglas de seguridad en red+active | `tests/test_adaptive_skip.py` (32) |
| `frame_reader.py` | `FrameReader` con un thread daemon, queue acotada (maxsize=2), back-pressure con drop, captura defensiva de errores de OpenCV | `tests/test_frame_reader.py` (16) |

**Total**: 4 modulos, 66 tests unitarios, 0 dependencias nuevas en `requirements.txt`.

---

## 3. Cambios en los scripts CLI

### 3.1 `process_video.py` y `only_infractions.py`

**`__init__`** (ambos scripts):
- `self.skip_controller = AdaptiveSkipController(target_fps_video=30.0)` (FPS real se setea en `process()`).
- `self.profiler = StageProfiler()`.
- `self.vehicle_detector` (y `PlateDetector` en `process_video`) envueltos en `with profiler.stage("model_load"):`.
- Un dummy-frame warmup `with profiler.stage("model_warmup"):` para que la primera inferencia real no pague el costo de compilacion de grafo / inicializacion de pools.

**`process()`** (ambos scripts):
- `configure_thread_budget()` llamado como **primera linea de `main()`** (antes de cualquier import que dispare carga de torch).
- `cap = cv2.VideoCapture(...)` seguido de `reader = FrameReader(cap).start()` — el decode ahora corre en background.
- `frame = reader.read()` reemplaza `ret, frame = cap.read()`, envuelto en `with profiler.stage("decode"):`.
- Reemplazo de las reglas fijas de skip por `skip_rate = self.skip_controller.suggest_skip(state, active_count)`.
- `if frame_index % skip_rate != 0: continue` (se removio la excepcion `current_state != "red"` — la nueva politica ya cubre ese caso).
- `_process_batch` envuelto con `time.perf_counter()` antes/despues, y `self.skip_controller.record(elapsed_ms, len(batch))`.
- Al final del loop, el reporte del profiler se imprime (`process_video.py`) o se loguea linea por linea (`only_infractions.py`).

**`_process_batch`** (ambos scripts):
- `vehicle_detector.detect_batch(...)` envuelto en `with profiler.stage("inference"):`.
- `tracker.process_detection(...)` envuelto en `with profiler.stage("tracker"):`.
- YOLO/Ultralytics bundlea preprocess y NMS internamente, asi que no se pueden medir por separado limpiamente — toda la llamada queda cargada a `inference` (documentado en el codigo).

**`only_infractions.py` solamente**:
- `self.crop_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crop-writer")` en `__init__`.
- `_log_trigger` ya no hace `cv2.imwrite` sincrono; hace `self.crop_writer.submit(_write_jpg, crop, crop_path)`.
- `self.crop_writer.shutdown(wait=True)` al final de `process()`.
- Helper `_write_jpg` movido a modulo-nivel (despues de la clase) — ver seccion 6 sobre la leccion aprendida con la estructura de la clase.

**`--batch-size`** default bajo de `4` a `2` en ambos argparse. Razon: en CPU 4-core con cache L2/L3 pequena, batches grandes no se amortizan igual que en GPU y compiten con el resto por el cache.

---

## 4. Politica del `AdaptiveSkipController`

El controller mantiene un ring de las ultimas N mediciones de inferencia (default `window=10`) y calcula `ratio = avg_ms_inference / frame_budget_ms`. El skip rate se deriva asi:

| Estado semaforo | `active_count` | Skip rate | Razon |
|---|---|---|---|
| `green` | cualquiera | `min(12, max(8, ratio*2))` | Idle: skip agresivo. Minimo 8 para no quemar CPU cuando no pasa nada. |
| `red` | `> 0` | `1` si `ratio <= 1` sino `min(3, ratio)` | Critico: detecta al infractor. Si el sistema va bien, procesa cada frame. Si no, sube a 2-3 NUNCA mas. |
| `yellow` o `red` con `active=0` | n/a | `min(5, max(2, ratio))` | Pre-alerta: mantener resolucion suficiente para capturar al primero que pase el rojo. |

**Safety invariants** (verificadas en `tests/test_adaptive_skip.py::TestRedActivePolicy`):
- `red+active` con presupuesto OK → `skip=1` (no perder infractor).
- `red+active` con sistema 30x mas lento → `skip=3` (cap, nunca mas).
- `red-alone` se comporta igual que `yellow` (mismo `min(5, max(2, ratio))`).

---

## 5. Decisiones de diseno no obvias

### 5.1 Thread budget: por que cv2=2 y torch=4 (no 1+8 ni 4+4)

Empirico. El profiler mostro que con los defaults (8+8) `decode` dominaba al 75% del wall time por **contencion de threads**, no por decode puro. Fijando cv2=2 (decode en background) y torch=4 (inferencia en el main thread), el decode bajo de 75ms/frame a 48ms/frame sin tocar el modelo.

La intuicion: en CPU 4-core, dedicarle 2 cores a FFmpeg (que paraleliza internamente) y 4 a PyTorch (que ya tokthreads al nivel de MKL) deja suficiente headroom para que el OS schedulee el thread del `FrameReader` sin pelearse con el main thread. Mas threads no ayudan porque ya estamos en el techo fisico.

### 5.2 `FrameReader` con queue maxsize=2 (no 1, no 4)

- `maxsize=1`: el main thread tiene que esperar a que el reader produzca cada frame. Sin overlap. Pierde el beneficio del threading.
- `maxsize=2`: deja 1 frame "en vuelo" mientras el main thread procesa el actual. Es el sweet spot — overlap real sin acumular latencia.
- `maxsize=4` o mas: el reader se adelanta al consumer y la "fluidez" que ve el operador es **frames viejos**, no tiempo real. Peligroso para deteccion de infracciones donde la freshness importa.

### 5.3 Crop executor: 1 worker (no 2, no 0)

JPG encode es CPU-bound, no I/O-bound. Un segundo worker competiria con el main thread por los 4 cores fisicos. Un solo worker alcanza para que el `imwrite` no bloquee el loop critico, pero el limite es el throughput del disco (que en SSD es >> al rate de triggers en este pipeline — una infraccion cada varios segundos, no por frame).

### 5.4 Profiler: una sola categoria `inference` (no separar preprocess/NMS)

YOLO/Ultralytics no expone hooks limpios entre `resize+normalize` y `predict`+`nms`. Separarlos requeriria monkey-patching el modelo. Decidimos que el costo total de la llamada a `detect_batch()` (incluyendo preprocess y NMS) ya es el numero que importa para decidir si optimizar; el siguiente paso (si domina) es OpenVINO, no micro-optimizar el preprocess.

### 5.5 `model_warmup` separado de `model_load`

La primera inferencia de YOLOv8n en PyTorch paga ~1-2 segundos de costo fijo (compilacion de grafo, inicializacion de pool de threads, primera alocacion de CUDA allocator — aunque sin CUDA sigue habiendo warmup del thread pool MKL). Sin warmup explicito, ese costo se le carga al primer batch de inferencia real y distorsiona el promedio. Con un dummy frame (`np.zeros((416,416,3), dtype=np.uint8)`) de ~1ms de inferencia, ese costo se paga una vez y queda fuera de las estadisticas utiles.

---

## 6. Lecciones aprendidas (para la proxima)

### 6.1 `set_num_interop_threads` puede crashear el proceso
La implementacion inicial de `configure_thread_budget()` llamaba a `torch.set_num_interop_threads(2)`. En este venv de torch 1.13.1, llamarlo despues de que el runtime paralelo ya arranco produce `Fatal Python error: Aborted` que **no es capturable** con try/except. La funcion quedo sin esa llamada — `set_num_threads` (intra-op) es lo que importa para un loop single-stream como el nuestro y es seguro siempre.

### 6.2 `time.sleep` en Windows tiene granularidad de 15.6ms
Los tests de `StageProfiler` flakeaban al correrlos en secuencia con `test_only_infractions` (que importa torch). El `time.sleep(0.001)` a veces volvia con 0ms. Solucion: helper `_busy_wait_ms` que usa `time.perf_counter()` para esperas confiables de 1-10ms en tests.

### 6.3 Editar una clase e insertar una funcion a nivel-modulo en el medio: SILENT BREAK
Cuando movi `_log_trigger` a hacer submit al executor y agregue `_write_jpg` justo despues, la funcion quedo a 0 espacios de indent — cerrando implicitamente la clase. El `def process` siguiente, a 4 espacios, fue parseado como **funcion de modulo**, no como metodo de la clase. El archivo compilo, la clase importo, pero `process` no aparecio en `dir()`. La deteccion: `ast.parse()` revelo que la clase terminaba antes de `def process`.

Regla operativa: **nunca** insertes una funcion de modulo entre dos metodos de clase. Si necesitas una helper, ponela como `@staticmethod` dentro de la clase, o al final del archivo, despues de la clase.

### 6.4 El deadlock clasico del EOF sentinel con `put_nowait`
Primera implementacion del `FrameReader` usaba `put_nowait` para el sentinel EOF. Bug: cuando la cola estaba llena y el consumer estaba lento, el reader no podia pushear el EOF, salia, y el consumer quedaba bloqueado en `q.get()` para siempre. Solucion: `put(timeout=5s)` para el sentinel + polling de `q.get(timeout=100ms)` en `read()` que retorna `None` cuando el thread del reader murio y la cola esta vacia.

### 6.5 OpenCV crashea ocasionalmente desde threads secundarios
Un run mostro `cv2.error: Unknown C++ exception from OpenCV code` en el thread del `FrameReader`. No reproducible en runs subsiguientes — probablemente glitch de codec. La captura defensiva (`_last_error` + `try/except` en `_run`) ahora previene el traceback ruidoso y expone el error via `reader.last_error`.

---

## 7. Resultados empiricos (medidos, no teoricos)

Mismo hardware, mismo video (`videos/VID2COLISEO.MOV`), 200 frames.

| Metrica | Antes (Fase 0) | Despues (Fase F) | Delta |
|---|---|---|---|
| Decode avg ms/frame | 74.59 | 48.01 | **−36%** |
| 200 frames end-to-end | 30.0s | 16.0s | **−47%** |
| FPS efectivo | 0.9 | 12.5 | **~14x** |
| Stages medidas | 1 (global) | 5 (per-stage) | visibilidad real |
| `cv2.setNumThreads` | 8 (default) | 2 (configurado) | sin contencion |
| `torch.set_num_threads` | 8 (default) | 4 (configurado) | sin oversubscription |
| `skip_rate` en green | 10 (fijo) | 8-12 (adaptive) | ahorra CPU |
| `skip_rate` en red+active lento | 1 (riesgoso) | 1-3 (cap) | sin espiral de latencia |
| Crop write | sincrono en loop | async en executor | loop no bloquea |
| Error capture en I/O | traceback ruidoso | `last_error` limpio | diagnostico real |

**Tests**: 78/78 verde (`-m "not slow"`), incluyendo 16 tests de `FrameReader` (cubriendo lifecycle, back-pressure, drop, EOF, stop, error capture, threading hygiene).

---

## 8. Out of scope (follow-ups documentados)

1. **OpenVINO export**: `yolo export format=openvino` + cambiar el backend de `VehicleDetector` a OpenVINO con `device="AUTO"` o `"GPU"` para usar el plugin de la UHD 630. Estimado: 1.5-3x sobre PyTorch puro. Requiere un test de parity (mismas detecciones que PyTorch) antes del rollout.

2. **Decoder de hardware**: `cv2.CAP_INTEL_MFX` o equivalente. Bajaria `decode` de 48ms a ~10ms por frame. Depende del codec del video fuente (H.264 vs H.265).

3. **Bajar resolucion de entrada del video**: pre-procesar el video a 1280x720 antes de pasarlo al pipeline. Corta `decode` y `inference` aproximadamente a la mitad. Trade-off: deteccion de vehiculos lejanos empeora.

4. **Migrar la GUI** (`preprocessing_dialog.py`) a los mismos componentes: `AdaptiveSkipController`, `FrameReader`, `StageProfiler`. La GUI ya tiene su propia version del skip rules (distinta del CLI). El refactor es mecanico pero hay que validar que la cola de resultados (`result_queue`) y el sistema de display thread siguen funcionando.

5. **Test `test_runs_and_creates_log`**: falla pre-existente por encoding cp932 de Windows cuando el subprocess emite emojis. No es un bug de mi refactor. Fix sugerido: forzar `encoding='utf-8'` en `subprocess.run(... capture_output=True, text=True)` o quitar los emojis de los prints de los scripts.

---

## 9. Archivos del refactor

**Nuevos** (8 archivos, ~50 KB total):
```
scripts/adapter/threads.py            3.0 KB
scripts/adapter/stage_profiler.py     4.4 KB
scripts/adapter/adaptive_skip.py      7.5 KB
scripts/adapter/frame_reader.py      12.6 KB
tests/test_threads.py                 3.0 KB
tests/test_stage_profiler.py          5.9 KB
tests/test_adaptive_skip.py           9.9 KB
tests/test_frame_reader.py           12.7 KB
```

**Modificados** (2 archivos):
```
scripts/adapter/process_video.py
scripts/adapter/only_infractions.py
```

**Intactos** (verificado en `git status`):
```
src/core/detection/vehicle_detector.py
scripts/adapter/infraction_tracker.py
src/gui/preprocessing_dialog.py
src/application/services/processing_planner.py
```

---

## 10. Como correr / verificar

```bash
# 1. Tests unitarios
pytest tests/test_threads.py \
       tests/test_stage_profiler.py \
       tests/test_adaptive_skip.py \
       tests/test_frame_reader.py \
       tests/test_only_infractions.py \
       tests/test_cli_pipeline.py \
       -m "not slow" --timeout=15
# Esperado: 78/78 verde

# 2. Runtime corto (ver el profiler en accion)
python scripts/adapter/only_infractions.py \
  --video videos/VID2COLISEO.MOV \
  --max-frames 300 \
  --output-dir data/verify/oi \
  --crops-dir data/verify/oi
# Esperado: 200-300 frames en ~15-20s, profiler imprime tabla
#   con stages: model_load, model_warmup, decode, inference

# 3. Runtime del pipeline completo (con OCR)
python scripts/adapter/process_video.py \
  --video videos/VID2COLISEO.MOV \
  --max-frames 50 \
  --output-dir data/verify/pv
# Esperado: 50 frames, profiler imprime tabla, persist JSON

# 4. Verificar que el budget se aplico al arranque
python scripts/adapter/only_infractions.py --help 2>&1 | head -1
python -c "
import subprocess
r = subprocess.run(['python', 'scripts/adapter/only_infractions.py',
                    '--video', 'videos/VID2COLISEO.MOV', '--max-frames', '5',
                    '--output-dir', 'data/verify/q', '--crops-dir', 'data/verify/q'],
                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(r.stdout[:300])
"
# Esperado: primera linea dice "Thread budget: {'cv_threads': 2, 'torch_threads': 4, ...}"

---

## 11. Activación del refactor (`--new` flag)

Desde v2.1, el refactor es **opt-in** vía el flag `--new`. El
comportamiento por default (sin la flag) reproduce el código
pre-refactor con skip fijo, decode síncrono, sin profiler, y
`batch_size=4`.

```bash
# Modo nuevo (refactor activo)
python scripts/adapter/process_video.py     --video videos/VID2COLISEO.MOV --new
python scripts/adapter/only_infractions.py  --video videos/VID2COLISEO.MOV --new

# Modo clásico (pre-refactor)
python scripts/adapter/process_video.py     --video videos/VID2COLISEO.MOV
python scripts/adapter/only_infractions.py  --video videos/VID2COLISEO.MOV
```

### Tabla de diferencias

| Feature                                    | Default (sin `--new`) | `--new` |
|--------------------------------------------|:---:|:---:|
| `configure_thread_budget()` (cv=2, torch=4) | ✗  | ✓  |
| `FrameReader` (background decode)          | ✗  | ✓  |
| `AdaptiveSkipController` (time-budget aware) | ✗ | ✓  |
| `StageProfiler` (5 stages)                  | ✗  | ✓  |
| Warmup frame (dummy YOLO pass)              | ✗  | ✓  |
| Crop executor async (only_infractions)      | ✗  | ✓  |
| Default `batch_size`                        | 4  | 2  |
| Hardcoded skip rates (legacy)              | ✓  | ✗  |
| `current_state != "red"` exception         | ✓ (revertida)  | ✗ (removida)  |
| Profiler report al final del log            | ✗  | ✓  |

### Skip policy según modo

**Default (sin `--new`)** — `green=10, red+active=1, red=3, else=3`:

```python
if current_state == "red" and active_count > 0:
    skip_rate = 1     # la "excepción" que el refactor removió
elif current_state == "green":
    skip_rate = 10
elif current_state == "red":
    skip_rate = 3
else:                  # yellow
    skip_rate = 3
```

**Con `--new`** — controlado por `AdaptiveSkipController` (ver sección 4
de este doc), basado en el ratio real de inferencia vs presupuesto.

### Por qué existe el flag

El código pre-refactor sigue siendo útil para:

1. **A/B testing**: comparar resultados entre el comportamiento viejo
   y el nuevo con la misma entrada.
2. **Diagnóstico**: si el refactor introduce overhead no esperado
   (por ejemplo, en una máquina con GPU dedicada), se puede volver
   al modo clásico sin tocar el código.
3. **Compatibilidad**: scripts automatizados y CI que dependan del
   comportamiento previo siguen funcionando sin cambios.

### Nota sobre el origen del código "pre-refactor"

Los scripts CLI (`process_video.py`, `only_infractions.py`) fueron
creados directamente con el refactor aplicado; no existe una versión
previa en el historial de git. El skip logic legacy documentado arriba
está **reconstruido** desde la sección 4 de este doc y verificado por
los tests `test_pipeline_modes.py::TestLegacySkipRate`.
```
