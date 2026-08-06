# Analisis del Pipeline de Deteccion de Infracciones (`only_infractions.py`)

> **Proposito del documento**: servir como contexto autonomo para que otra IA pueda proponer optimizaciones sobre el pipeline de deteccion de infracciones sin placa, sin necesidad de leer todo el codigo. Incluye ademas un catalogo priorizado de opciones de mejora.

**Archivos principales analizados:**

- `scripts/adapter/only_infractions.py` (617 lineas) — pipeline segmentado (Fase 1)
- `scripts/adapter/infraction_tracker.py` (242 lineas) — logica de tracking + triggers
- `scripts/adapter/process_video.py` (711 lineas) — pipeline completo (Fase 1 + Fase 2 OCR)
- `src/core/detection/vehicle_detector.py` — YOLOv8 wrapper
- `src/core/traffic_signal/semaphore.py` — estado del semaforo

---

# A. Resumen detallado de `scripts/adapter/only_infractions.py`

## A.1 Proposito y motivacion

`only_infractions.py` es una **version segmentada del pipeline completo** (`process_video.py`) que ejecuta **solo la Fase 1** del sistema InfractiVision: deteccion de vehiculos, tracking y disparo de triggers de infraccion. **No ejecuta OCR ni reconocimiento de placas**.

Fue creado para:

- **Testeo independiente** de la logica de triggers sin pagar el costo (ni la complejidad) del OCR.
- **Benchmarking** de la logica de deteccion pura.
- **Debugging visual** de cuando y por que se dispara un trigger.

A diferencia de `process_video.py`, este script **no persiste infracciones en JSON**, **no clasifica NID/NIE**, y **no llama a la nube**.

## A.2 Arquitectura general

```
Video input
   |
   v
[VideoSemaphore]  --> estado del semaforo por segundo (green/yellow/red)
   |
   v
[VehicleDetector YOLOv8]  --> detecciones por frame (batch)
   |
   v
[PPI V50]  --> indice de cercania (vertical-primary)
   |
   v
[3-vertex polygon test]  --> contencion en zona de infraccion
   |
   v
[InfractionTracker]  --> asociacion + MMRP + 4 tipos de trigger
   |
   v
[File logger + crops]  --> data/logs/infractions_<ts>.log + data/output/only_infractions/run_<ts>/*.jpg
```

## A.3 Componentes principales

### A.3.1 `OnlyInfractionsPipeline` (clase principal)

**Ubicacion**: `scripts/adapter/only_infractions.py:158-493`

**Atributos de configuracion:**

- `config` (dict): configuracion cargada (semaforo, poligono, batch, conf)
- `log` (Logger): logger de archivo con timestamp
- `crops_dir` (str): directorio para crops de vehiculos infractores
- `semaphore` (VideoSemaphore): estado del semaforo por segundo de video
- `polygon` (np.ndarray | None): zona de infraccion (4 puntos tipicamente)
- `conf_vehicle` (float, default 0.50): umbral YOLO
- `batch_size` (int, default 4): tamano de batch para inferencia
- `vehicle_detector` (VehicleDetector): instancia de YOLOv8
- `plate_detector = None`: **explicitamente deshabilitado** (clave del diseno)

### A.3.2 Metodo `process()` — el loop principal

**Ubicacion**: `only_infractions.py:325-493`

**Flujo paso a paso:**

1. **Apertura del video** con OpenCV (`cv2.VideoCapture`)
   - Lee `total_frames`, `fps_video`, calcula `video_duration_str` (formato `MM:SS`)
   - Aplica `skip_start` segundos al inicio (mueve `CAP_PROP_POS_FRAMES`)

2. **Loop de frames** (lineas 386-449):
   - **Skip adaptativo por estado de semaforo** (lineas 411-421):

     ```python
     if current_state == "green":                              skip_rate = 10
     elif current_state == "red" and tracker.active_count > 0: skip_rate = 1
     elif current_state == "red":                              skip_rate = 3
     else:                                                     skip_rate = 3
     ```

     Idea: en verde se salta 9/10 frames (ahorro masivo), en rojo con infractores activos se procesa cada frame (precision maxima).

   - **Recolecta batches** de hasta `batch_size` frames, llama a `_process_batch`

3. **Calculo de metricas finales** (lineas 454-477):
   - `t_proc_elapsed` (sin carga de modelos)
   - `t_total_elapsed = t_proc_elapsed + t_model_elapsed`
   - Cuenta `PEAK` y `PERSIST` (los unicos tipos que se disparan sin placa)
   - Calcula `fps_avg = processed / t_proc_elapsed`

4. **Retorna dict** con: video, duration, frames, infractions raw, peak_count, heavy_count, total_triggers, t_model, t_processing, t_total, fps_avg.

### A.3.3 Metodo `_process_batch()` — el worker por batch

**Ubicacion**: `only_infractions.py:201-290`

**Pasos por batch:**

1. **YOLO batch inference** (`self.vehicle_detector.detect_batch(frames, conf=0.50)`) — una sola llamada GPU con N frames.

2. **Por cada frame del batch** (zip con detecciones):
   - Calcula `current_state` del semaforo para ese frame
   - **Filtro de distancia**: `if ppi < 0.20: continue` (rechaza vehiculos lejanos)
   - **Si NO es rojo O no hay poligono**: `continue` (no hay infraccion posible)
   - **`has_plate = False`** (siempre — este script nunca confirma placa)
   - Llama a `tracker.process_detection(...)` con todos los datos

3. **Si `display=True`**: dibuja detecciones + poligono + banner semaforo con `cv2.imshow`

### A.3.4 `_log_trigger()` — el reporter

**Ubicacion**: `only_infractions.py:292-321`

Por cada trigger disparado:

- **Recorta el vehiculo** del snapshot (con el bbox del `mmrp_frame`)
- Guarda `vehicle_inf<track_id>_t<track_id>_f<frame>.jpg` en `crops_dir`
- Loguea el evento con: tipo, track_id, frame, segundo, PPI, num_frames

## A.4 Mecanica del trigger (`InfractionTracker`)

> **Nota**: `only_infractions.py` **delega toda la decision al `InfractionTracker`** (en `infraction_tracker.py`). Entender esa logica es clave para optimizar.

### A.4.1 PPI V50 (Proximity Proximity Index)

```python
y_factor = max(0, (bumper_y - frame_h*0.35) / (frame_h*0.60))
x_center_factor = 1.0 - abs(bumper_x - frame_w*0.5) / (frame_w*0.5) * 0.3
PPI = clamp(y_factor * x_center_factor, 0.01, 1.0)
```

- **Y-primary**: la cercania real la da la posicion vertical (auto abajo = mas cerca)
- **X-secondary**: penalizacion leve por salirse del centro horizontal
- Rango: 0.01 (lejos) → 1.0 (muy cerca)

### A.4.2 Asociacion de tracks

- Distancia euclidiana centro-a-centro entre detecciones del frame actual y el ultimo centro visto de cada `_active[inf_N]`.
- Si `dist < track_dist_threshold` (default 140 px) → se asocia al mismo infractor.
- Si no → es nuevo infractor (y debe cumplir `in_polygon and PPI > 0.40 and area > 25000`).

### A.4.3 Los 4 tipos de trigger

| Tipo          | Condicion                                                       | Requiere placa |
|---------------|-----------------------------------------------------------------|----------------|
| `PANIC`       | `ppi >= 0.88 and has_plate`                                     | Si             |
| `SECURE`      | `num_frames >= 3 and ppi >= 0.85 and has_plate`                | Si             |
| `PEAK_GOLD`   | `num_frames >= 5 and mmrp_reached and ppi >= 0.78`             | No             |
| `HEAVY (PERSIST)` | `num_frames >= 22 and ppi >= 0.75`                          | No             |

**En `only_infractions.py` solo se disparan `PEAK_GOLD` y `HEAVY`** (porque `has_plate=False` siempre). Por eso el reporte final distingue `peak_count` y `heavy_count`.

### A.4.4 MMRP (Maximal Mid-Range Peak)

Deteccion de "pico" en el area del vehiculo:

- Requiere al menos 6 frames trackeados
- Mira los ultimos 5: `sum(ultimos 3) / 3 < sum(primeros 3) / 3 * 0.98` → "el auto dejo de crecer" → alcanzo su pico
- Es lo que distingue un vehiculo **estacionado/avanzando lento** (PEAK) de uno **simplemente cerca** (todavia no triggerea)

## A.5 Flujo de ejecucion (CLI)

```bash
python scripts/adapter/only_infractions.py \
  --video videos/VID2COLISEO.MOV \
  --config config/polygon_config.json \     # opcional
  --output-dir data/logs \                  # default
  --crops-dir data/output/only_infractions \ # default
  --max-frames 500 \                        # default: sin limite
  --skip-start 0 \                          # default
  --skip-end 0 \                            # default
  --conf-vehicle 0.50 \                     # default
  --batch-size 4 \                          # default
  --display \                               # ventana cv2 en vivo
  --quiet                                   # solo al archivo, sin stdout
```

## A.6 Outputs y observabilidad

| Salida                          | Ubicacion                                            | Formato                                  |
|---------------------------------|------------------------------------------------------|------------------------------------------|
| Log de ejecucion                | `data/logs/infractions_<ISO_ts>.log`                 | texto con timestamps                     |
| Crops de vehiculos              | `data/output/only_infractions/run_<ts>/*.jpg`        | JPG con bbox del MMRP                    |
| Stdout (sin `--quiet`)          | consola                                              | resumen con conteos, tiempos, FPS        |

## A.7 Deuda tecnica / gaps detectados

1. **Asignacion manual de `track_id`**: el `_counter` se incrementa localmente por ciclo, no sobrevive entre runs.
2. **`track_dist_threshold = 140` esta hard-coded** en `infraction_tracker.py:44`. No se puede tunear por video.
3. **El "skip adaptativo" no considera resolucion ni hardware**: en GPU barata con batches grandes, `skip_rate=1` puede saturarse.
4. **No hay un profiler real**: el `t_total_elapsed` solo mide el loop, no la carga de modelos ni el display.
5. **`has_plate = False` siempre** desactiva PANIC/SECURE. Si en algun momento se quisiera testear la cadena completa con placa, hay que ir al `process_video.py` (no es trivial parametrizar).
6. **Sin paralelismo real**: el batch se procesa secuencial en GPU pero la I/O del video y los crops son bloqueantes.
7. **Logging en archivo siempre, nunca a stdout en `--quiet`**: en Windows cp932 los emojis del logger pueden romper la salida si se redirige.

