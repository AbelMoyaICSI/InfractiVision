#!/usr/bin/env python
"""
process_video.py — CLI pipeline for infraction detection.

Runs the full detection pipeline on a video file WITHOUT opening Tk windows.
Accepts a JSON config file and/or reads the existing config/*.json files.

Usage:
    python scripts/adapter/process_video.py \\
        --video videos/VID2COLISEO.MOV \\
        --config config/sample_config.json

    python scripts/adapter/process_video.py \\
        --video videos/VID2COLISEO.MOV \\
        --output-dir data/cli_output \\
        --conf-vehicle 0.50 --conf-plate 0.40 \\
        --max-frames 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detection.vehicle_detector import VehicleDetector
from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.recognizer import recognize_plate, format_siiv_plate
from src.path_helper import resource_path
from scripts.adapter.video_semaphore import VideoSemaphore
from scripts.adapter.infraction_tracker import InfractionTracker, calculate_ppi
from scripts.adapter.adaptive_skip import AdaptiveSkipController
from scripts.adapter.frame_reader import FrameReader
from scripts.adapter.stage_profiler import StageProfiler
from scripts.adapter.threads import configure_thread_budget
from scripts.adapter.persistence import (
    save_infractions_json,
    save_nie_infractions_json,
    save_indicators_json,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    """Load a JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_configs(
    video_basename: str,
) -> tuple[dict | None, dict | None, str | None]:
    """Load polygon, semaphore times, and avenue name for a video from config/*.json."""
    polygon = None
    semaphore = None
    avenue = None

    poly_path = PROJECT_ROOT / "config" / "polygon_config.json"
    if poly_path.exists():
        poly_data = json.loads(poly_path.read_text(encoding="utf-8"))
        if video_basename in poly_data:
            polygon = poly_data[video_basename]

    time_path = PROJECT_ROOT / "config" / "time_presets.json"
    if time_path.exists():
        time_data = json.loads(time_path.read_text(encoding="utf-8"))
        if video_basename in time_data:
            semaphore = time_data[video_basename]

    aven_path = PROJECT_ROOT / "config" / "avenue_config.json"
    if aven_path.exists():
        aven_data = json.loads(aven_path.read_text(encoding="utf-8"))
        if video_basename in aven_data:
            avenue = aven_data[video_basename]

    return polygon, semaphore, avenue


def _crop_vehicle_context(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_pct: float = 0.10,
) -> np.ndarray:
    """Crop vehicle with proportional margin (Phase 2 standard)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    mw = int((x2 - x1) * margin_pct)
    mh = int((y2 - y1) * margin_pct)
    vx1 = max(0, x1 - mw)
    vy1 = max(0, y1 - mh)
    vx2 = min(w, x2 + mw)
    vy2 = min(h, y2 + mh)
    return frame[vy1:vy2, vx1:vx2].copy()


def _quick_plate_check(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    plate_detector: PlateDetector,
    conf: float = 0.40,
) -> bool:
    """Fast yes/no plate presence check (same as V50 inline)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    tm = max(30, int(min(x2 - x1, y2 - y1) * 0.15))
    ry1 = max(0, y1 - tm)
    ry2 = min(h, y2 + tm)
    rx1 = max(0, x1 - tm)
    rx2 = min(w, x2 + tm)
    v_roi = frame[ry1:ry2, rx1:rx2]
    if v_roi.size == 0:
        return False
    p_det = plate_detector.detect_plates(v_roi, confidence=conf)
    return bool(p_det)


def _do_ocr(
    plate_img: np.ndarray,
    regional_context: str = "Trujillo",
) -> dict:
    """Run full OCR on a plate image. Returns {plate, confidence, surgical_crop}."""
    try:
        txt, conf, crop = recognize_plate(
            plate_img,
            return_processed=True,
            autocrop=True,
            regional_context=regional_context,
        )
        return {"plate": txt or "", "confidence": conf, "crop": crop}
    except Exception:
        return {"plate": "", "confidence": 0.0, "crop": None}


def _legacy_skip_rate(current_state: str, active_count: int) -> int:
    """Pre-refactor skip policy (reconstructed from docs/REFACTOR_CPU_PIPELINE.md).

    The CLI scripts in this repo were created with the refactor already
    applied; there is no pre-refactor version in git history. The values
    below come from the section 4 table of REFACTOR_CPU_PIPELINE.md:

        green         → 10   (idle, save CPU)
        red+active    →  1   (the "exception" that the refactor removed:
                             force every frame while an infractor is tracked)
        red, no active → 3   (pre-alert)
        yellow        →  3   (pre-alert)

    Unlike the refactor's AdaptiveSkipController, these rates are static
    and do not adapt to the actual inference latency. On a 4-core CPU
    without a discrete GPU, they can be unsafe in red+active (the
    "spiral of latency" pattern documented in section 1 of the refactor
    doc). Pass --new to enable the adaptive policy.
    """
    if current_state == "red" and active_count > 0:
        return 1
    if current_state == "green":
        return 10
    if current_state == "red":
        return 3
    return 3  # yellow / unknown


# ── Pipeline ──────────────────────────────────────────────────────────


class CLIInfractionPipeline:
    """Headless infraction detection pipeline."""

    def __init__(self, config: dict, use_new: bool = False) -> None:
        self.config = config
        self.use_new = use_new

        # Semaphore
        sem_cfg = config.get("semaphore", {"green": 30, "yellow": 5, "red": 40})
        self.semaphore = VideoSemaphore(sem_cfg)

        # Polygon
        poly_raw = config.get("polygon")
        self.polygon: np.ndarray | None = (
            np.array(poly_raw, np.int32) if poly_raw else None
        )

        # Avenue & time slot
        self.avenue = config.get("avenue", "Desconocida")
        self.time_slot = config.get("time_slot", "No especificada")

        # Confidence thresholds
        self.conf_vehicle = config.get("conf_vehicle", 0.50)
        self.conf_plate = config.get("conf_plate", 0.40)

        # Rectification toggle
        self.use_rectification = config.get("rectification", True)

        # Batch size for vehicle detection. Defaults differ by mode:
        # --new (refactor) → 2 (tuned for 4-core CPU; refactor's skip
        #                     controller absorbs latency spikes).
        # default (legacy) → 4 (pre-refactor value).
        default_batch = 2 if use_new else 4
        batch_cfg = config.get("batch_size")
        self.batch_size = batch_cfg if batch_cfg is not None else default_batch

        if use_new:
            # Skip-rate controller. Target FPS is set in process()
            # once we know the real video FPS. The controller is
            # created here so that the rest of __init__ can reference
            # it.
            self.skip_controller = AdaptiveSkipController(target_fps_video=30.0)
            # Per-stage profiler. Wraps model load, model warmup,
            # decode, inference and tracker so we can see where the
            # CPU budget actually goes. See stage_profiler.py.
            self.profiler = StageProfiler()
        else:
            # Legacy mode: no skip controller, no profiler. The
            # legacy skip logic uses hardcoded rates and the decode
            # happens synchronously on the main thread.
            self.skip_controller = None
            self.profiler = None

        # Models (loaded lazily or at init). With --new we separate
        # `model_load` (file -> memory) from `model_warmup` (one
        # inference pass on a dummy frame) so the warmup cost does
        # not skew the average of subsequent inference calls. In
        # legacy mode both stages are charged directly to model load.
        print("🔧 Cargando modelos...")
        if use_new:
            with self.profiler.stage("model_load"):
                self.vehicle_detector = VehicleDetector(
                    str(PROJECT_ROOT / "models" / "yolov8n.pt")
                )
                self.plate_detector = PlateDetector()
            print("✅ Modelos listos.")
            # Warmup: one inference on a synthetic frame to amortize
            # the first-call setup of PyTorch (graph compilation,
            # allocator, thread pool warm-up). This cost would
            # otherwise be charged to the first real frame and
            # distort the inference average.
            with self.profiler.stage("model_warmup"):
                _warmup_frame = np.zeros((416, 416, 3), dtype=np.uint8)
                self.vehicle_detector.detect_batch(
                    [_warmup_frame], conf=0.25
                )
        else:
            self.vehicle_detector = VehicleDetector(
                str(PROJECT_ROOT / "models" / "yolov8n.pt")
            )
            self.plate_detector = PlateDetector()
            print("✅ Modelos listos.")

    def process(
        self,
        video_path: str,
        *,
        max_frames: int | None = None,
        skip_start: int = 0,
        skip_end: int = 0,
        display: bool = False,
    ) -> dict:
        """
        Run the full pipeline and return aggregated metrics.

        Returns dict with:
            infractions, nid_count, nie_count, tir, ti_pct, tr_minutes,
            tr_seconds_per_nid, total_frames, processed_frames,
            elapsed_seconds, avg_fps, video_duration
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.use_new:
            # Tell the skip controller the real frame budget now
            # that we know the video FPS. This is the ONLY place we
            # call set_target_fps, so all subsequent suggest_skip()
            # decisions are framed against this video's own budget.
            self.skip_controller.set_target_fps(fps_video)
        video_duration_sec = total_frames / fps_video if fps_video else 0
        video_duration_str = f"{int(video_duration_sec // 60):02d}:{int(video_duration_sec % 60):02d}"

        # Hand the cap to a single background reader. Decode now
        # overlaps with inference in the main thread. The reader's
        # bounded queue (maxsize=2) prevents the producer from
        # racing ahead and accumulating stale frames. Only in --new
        # mode; legacy mode decodes synchronously.
        reader = FrameReader(cap).start() if self.use_new else None

        video_basename = os.path.basename(video_path)

        # Apply skip_start
        if skip_start > 0:
            start_frame = int(skip_start * fps_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        tracker = InfractionTracker()

        frame_index = 0
        processed = 0
        detection_cache: list[list[tuple]] = []

        t_start = time.time()
        phase1_start = t_start

        print(f"🎬 Procesando: {video_basename}")
        print(f"   📐 Frames totales: {total_frames}")
        print(f"   ⏱️  FPS video: {fps_video:.1f}")

        infractions_raw: list[dict] = []  # triggered events

        # ── FRAME LOOP ──────────────────────────────────────────────
        frames_batch: list[np.ndarray] = []
        frame_indices_batch: list[int] = []

        while True:
            # Decode: in --new mode it runs in the background
            # reader thread; what we measure is "how long did the
            # main thread wait to get a frame from the reader's
            # queue". In legacy mode decode is a synchronous
            # cap.read().
            if self.use_new:
                with self.profiler.stage("decode"):
                    frame = reader.read()
                if frame is None:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_index += 1
            if max_frames and frame_index > max_frames:
                break
            if skip_end and frame_index > (total_frames - int(skip_end * fps_video)):
                break

            video_second = frame_index / fps_video
            current_state = self.semaphore.get_state(video_second)

            # ---- Skip decision ----
            # --new: AdaptiveSkipController (time-budget aware)
            # default: hardcoded rates (reconstructed from
            #          docs/REFACTOR_CPU_PIPELINE.md section 4,
            #          since there is no pre-refactor version in
            #          git history):
            #            red+active=1, green=10, red=3, else=3
            if self.use_new:
                skip_rate = self.skip_controller.suggest_skip(
                    current_state, tracker.active_count
                )
            else:
                skip_rate = _legacy_skip_rate(
                    current_state, tracker.active_count
                )

            if frame_index % skip_rate != 0:
                continue

            h_frame, w_frame = frame.shape[:2]

            # BATCH collection
            frames_batch.append(frame)
            frame_indices_batch.append(frame_index)

            if len(frames_batch) >= self.batch_size:
                t0 = time.perf_counter()
                self._process_batch(
                    frames_batch,
                    frame_indices_batch,
                    infractions_raw,
                    tracker,
                    fps_video,
                    display,
                )
                if self.use_new:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    self.skip_controller.record(
                        elapsed_ms, len(frames_batch)
                    )
                processed += len(frames_batch)
                frames_batch = []
                frame_indices_batch = []

        # Drain remaining batch
        if frames_batch:
            t0 = time.perf_counter()
            self._process_batch(
                frames_batch,
                frame_indices_batch,
                infractions_raw,
                tracker,
                fps_video,
                display,
            )
            if self.use_new:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.skip_controller.record(elapsed_ms, len(frames_batch))
            processed += len(frames_batch)

        cap.release()
        if reader is not None:
            reader.stop()
        cv2.destroyAllWindows()

        # Print the per-stage profile so the operator can see where
        # the time really went. Only in --new mode.
        if self.use_new:
            print("\n" + self.profiler.report())

        phase1_elapsed = time.time() - phase1_start
        print(f"\n📊 Fase 1 completada: {processed} frames en {phase1_elapsed:.1f}s")
        print(f"   🚨 Triggers disparados: {len(infractions_raw)}")

        # ── PHASE 2: OCR + classification ───────────────────────────
        print("\n🔬 Fase 2: OCR y clasificación...")
        infractions_final: list[dict] = []
        tr_times_minutes: list[float] = []

        for i, raw in enumerate(infractions_raw):
            snap = raw["snapshot"]
            track_id = raw["track_id"]
            t_ocr_start = time.time()

            # Fast path: use pre-rectified plate if available
            plate_stripped = snap.get("plate_stripped")
            vehicle_ctx = snap.get("vehicle_context")

            if plate_stripped is not None and plate_stripped.size > 0:
                ocr_result = _do_ocr(plate_stripped)
            elif vehicle_ctx is not None and vehicle_ctx.size > 0:
                # Fallback: run OCR on the vehicle context
                ocr_result = _do_ocr(vehicle_ctx)
            else:
                # Last resort: crop the snapshot image with the bbox
                bbox = snap.get("bbox", (0, 0, 100, 100))
                vehicle_crop = _crop_vehicle_context(snap["img"], bbox, margin_pct=0.10)
                ocr_result = _do_ocr(vehicle_crop)

            plate_text = ocr_result["plate"]
            confidence = ocr_result["confidence"]

            t_ocr_end = time.time()
            tr_minutes = (t_ocr_end - t_ocr_start) / 60.0
            tr_times_minutes.append(tr_minutes)

            # Classify NID/NIE
            if plate_text and confidence >= 0.70:
                classification = "NID"
            else:
                classification = "NIE"

            # Generate semaphore config ID
            sem_cfg = self.semaphore.get_cycle_durations()
            config_id = f"{sem_cfg['green']}-{sem_cfg['yellow']}-{sem_cfg['red']}"

            infractions_final.append({
                "plate": plate_text or f"NIE_{track_id}",
                "confidence": confidence,
                "clasificacion": classification,
                "track_id": track_id,
                "frame": snap.get("f", 0),
                "time": snap.get("f", 0) / fps_video if fps_video else 0,
                "tiempo_procesamiento": tr_minutes * 60,
                "video_duration": video_duration_str,
                "sistema_version": "InfractiVision_v2.0",
                "metadata_clasificacion": {
                    "placa_final": plate_text,
                    "confianza": round(confidence, 3),
                    "calidad_deteccion": (
                        "alta" if confidence >= 0.85 else "media"
                    ),
                    "justificacion": (
                        "✅ Placa leída correctamente"
                        if classification == "NID"
                        else "No cumple criterios técnicos"
                    ),
                },
            })

            print(
                f"   [{i+1}/{len(infractions_raw)}] "
                f"Track #{track_id}: {plate_text or '(vacío)'} "
                f"({classification} | {confidence:.2f})"
            )

        # ── METRICS ─────────────────────────────────────────────────
        nid_count = sum(1 for i in infractions_final if i["clasificacion"] == "NID")
        nie_count = sum(1 for i in infractions_final if i["clasificacion"] == "NIE")
        total_detectadas = nid_count + nie_count
        ti_pct = (
            (nid_count / total_detectadas * 100)
            if total_detectadas > 0
            else 0.0
        )

        total_elapsed = time.time() - t_start
        avg_fps = processed / phase1_elapsed if phase1_elapsed > 0 else 0

        tr_seconds_per_nid = (
            total_elapsed / nid_count if nid_count > 0 else 0.0
        )

        return {
            "infractions": infractions_final,
            "nid_count": nid_count,
            "nie_count": nie_count,
            "tir": total_detectadas,
            "ti_pct": ti_pct,
            "tr_minutes": tr_times_minutes,
            "tr_seconds_per_nid": tr_seconds_per_nid,
            "total_frames": total_frames,
            "processed_frames": processed,
            "elapsed_seconds": total_elapsed,
            "avg_fps": avg_fps,
            "video_duration": video_duration_str,
            "profiler": self.profiler.as_dict() if self.use_new else None,
        }

    # ── internal: batch worker ─────────────────────────────────────

    def _process_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        infractions_out: list[dict],
        tracker: InfractionTracker,
        fps_video: float,
        display: bool,
    ) -> None:
        """Run vehicle detection on a batch of frames and feed the tracker."""

        # Batch YOLO inference. YOLO bundles preprocessing (resize /
        # normalize) and NMS internally, so they cannot be separated
        # out cleanly. We charge the whole call to "inference"; if
        # the profile shows inference is dominant, the right next
        # step is to lower the input resolution or move to OpenVINO.
        # In legacy mode we skip the profiler wrapping.
        if self.use_new:
            with self.profiler.stage("inference"):
                all_dets = self.vehicle_detector.detect_batch(
                    frames, conf=self.conf_vehicle
                )
        else:
            all_dets = self.vehicle_detector.detect_batch(
                frames, conf=self.conf_vehicle
            )

        for frame, fidx, detections in zip(frames, frame_indices, all_dets):
            h, w = frame.shape[:2]
            video_second = fidx / fps_video
            current_state = self.semaphore.get_state(video_second)

            if display:
                display_frame = frame.copy()

            # Process each vehicle detection
            for det in detections:
                x1, y1, x2, y2, cls = det
                bumper_x = (x1 + x2) // 2
                bumper_y = y2
                vehicle_area = (x2 - x1) * (y2 - y1)
                v_left = (x1, y2)
                v_right = (x2, y2)

                ppi = calculate_ppi(bumper_x, bumper_y, h, w)

                # Distance filter
                if ppi < 0.20:
                    continue

                # Quick plate check
                has_plate = False
                if self.plate_detector is not None:
                    has_plate = _quick_plate_check(
                        frame, (x1, y1, x2, y2), self.plate_detector, self.conf_plate
                    )

                # Only track if RED or if already tracked
                if current_state == "red" and self.polygon is not None:
                    if self.use_new:
                        with self.profiler.stage("tracker"):
                            trigger = tracker.process_detection(
                                vehicle_center=(bumper_x, bumper_y),
                                vehicle_area=vehicle_area,
                                h_frame=h,
                                w_frame=w,
                                polygon=self.polygon,
                                proximity_factor=ppi,
                                has_plate_score=has_plate,
                                frame_img=frame,
                                bbox=(x1, y1, x2, y2),
                                v_left=v_left,
                                v_right=v_right,
                                frame_index=fidx,
                                plate_detector=self.plate_detector,
                            )
                    else:
                        trigger = tracker.process_detection(
                            vehicle_center=(bumper_x, bumper_y),
                            vehicle_area=vehicle_area,
                            h_frame=h,
                            w_frame=w,
                            polygon=self.polygon,
                            proximity_factor=ppi,
                            has_plate_score=has_plate,
                            frame_img=frame,
                            bbox=(x1, y1, x2, y2),
                            v_left=v_left,
                            v_right=v_right,
                            frame_index=fidx,
                            plate_detector=self.plate_detector,
                        )
                    if trigger:
                        infractions_out.append(trigger)

                if display:
                    color = (0, 0, 255) if current_state == "red" else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display_frame,
                        f"PPI:{ppi:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            if display:
                # Draw polygon
                if self.polygon is not None:
                    pts = self.polygon.reshape(-1, 1, 2)
                    cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)

                # Semaphore banner
                state_colors = {
                    "red": ((0, 0, 255), (255, 255, 255)),
                    "yellow": ((0, 255, 255), (0, 0, 0)),
                    "green": ((0, 255, 0), (0, 0, 0)),
                }
                tc, bc = state_colors.get(current_state, ((255, 255, 255), (0, 0, 0)))
                txt = f" SEMAFORO: {current_state.upper()} "
                cv2.rectangle(display_frame, (10, 10), (350, 50), bc, -1)
                cv2.putText(
                    display_frame, txt, (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, tc, 2,
                )

                cv2.imshow("InfractiVision CLI", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    display = False  # stop display on 'q'


# ── Main ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for process_video.py.

    Extracted into a function so tests can introspect the parser
    without invoking main().
    """
    parser = argparse.ArgumentParser(
        description="InfractiVision CLI — Procesamiento de infracciones sin GUI"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al archivo de video a procesar",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Activa el refactor CPU/iGPU: thread budget (cv=2, torch=4), "
             "FrameReader (background decode), AdaptiveSkipController, "
             "StageProfiler, warmup frame, batch_size=2. "
             "Default (sin --new): comportamiento pre-refactor con skip "
             "fijo, decode síncrono, sin profiler, batch_size=4.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta al archivo JSON de configuración (polígono, semáforo, avenida). "
             "Si no se provee, se buscan en config/*.json por nombre de video.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directorio de salida para JSON e imágenes (default: data)",
    )
    parser.add_argument(
        "--conf-vehicle",
        type=float,
        default=0.50,
        help="Umbral de confianza para detección de vehículos (default: 0.50)",
    )
    parser.add_argument(
        "--conf-plate",
        type=float,
        default=0.40,
        help="Umbral de confianza para detección de placas (default: 0.40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño del batch para inferencia YOLO. Default: 2 con --new, "
             "4 sin --new (legacy). Subir si hay GPU.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Número máximo de frames a procesar (default: sin límite)",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        help="Segundos a saltar al inicio del video (default: 0)",
    )
    parser.add_argument(
        "--skip-end",
        type=int,
        default=0,
        help="Segundos a omitir al final del video (default: 0)",
    )
    parser.add_argument(
        "--no-rectification",
        action="store_true",
        help="Desactivar la homografía v6.3 (usa solo autocrop)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Mostrar visualización en vivo (cv2.imshow)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Pin the thread budget before argparse or any heavy import so
    # OpenCV / PyTorch / BLAS pick up the right defaults before
    # their internal thread pools start. See scripts/adapter/threads.py
    # for the rationale (4-core CPU with no discrete GPU: cv=2, torch=4).
    # Only with --new; in legacy mode the defaults (8+8) are kept.
    if args.new:
        budget = configure_thread_budget()
        print("⚙️  Thread budget:", budget)
        print("✨ Modo --new: refactor CPU/iGPU activo")
    else:
        print("ℹ️  Modo clásico (sin --new): skip fijo, decode síncrono, "
              "sin profiler, batch_size=4")

    # ── Build config ────────────────────────────────────────────────
    video_basename = os.path.basename(args.video)
    polygon, semaphore_from_file, avenue_from_file = _load_existing_configs(
        video_basename
    )

    config: dict[str, Any] = {
        "conf_vehicle": args.conf_vehicle,
        "conf_plate": args.conf_plate,
        "batch_size": args.batch_size,
        "rectification": not args.no_rectification,
    }

    if args.config:
        user_cfg = _load_config(args.config)
        config["polygon"] = user_cfg.get("polygon")
        config["semaphore"] = user_cfg.get("semaphore", {})
        config["avenue"] = user_cfg.get("avenue", "Desconocida")
        config["time_slot"] = user_cfg.get("time_slot", "No especificada")
    else:
        config["polygon"] = polygon
        config["semaphore"] = semaphore_from_file or {
            "green": 30, "yellow": 5, "red": 40
        }
        if semaphore_from_file:
            config["time_slot"] = semaphore_from_file.get(
                "time_slot", "No especificada"
            )
        else:
            config["time_slot"] = "No especificada"
        config["avenue"] = avenue_from_file or "Desconocida"

    if config.get("polygon") is None:
        print("⚠️  ADVERTENCIA: No se encontró polígono para este video.")
        print("   La detección de infracciones NO funcionará sin polígono.")
        print(
            "   Provee --config con un JSON que incluya \"polygon\" "
            "o configura config/polygon_config.json."
        )

    if not config.get("semaphore"):
        config["semaphore"] = {"green": 30, "yellow": 5, "red": 40}

    # ── Run pipeline ────────────────────────────────────────────────
    pipeline = CLIInfractionPipeline(config, use_new=bool(args.new))

    result = pipeline.process(
        video_path=args.video,
        max_frames=args.max_frames,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        display=args.display,
    )

    # ── Save results ────────────────────────────────────────────────
    sem_cfg = config["semaphore"]
    config_id = f"{sem_cfg.get('green',30)}-{sem_cfg.get('yellow',5)}-{sem_cfg.get('red',40)}"

    # Separate NID and NIE
    nid_list = [i for i in result["infractions"] if i["clasificacion"] == "NID"]
    nie_list = [i for i in result["infractions"] if i["clasificacion"] == "NIE"]

    # Save images
    plates_dir = os.path.join(args.output_dir, "output", "placas")
    vehicles_dir = os.path.join(args.output_dir, "output", "autos")
    os.makedirs(plates_dir, exist_ok=True)
    os.makedirs(vehicles_dir, exist_ok=True)

    # Persist JSON
    nid_path = save_infractions_json(
        nid_list + nie_list,  # all in one file, classified
        output_dir=args.output_dir,
        filename="infracciones.json",
        avenue_name=config.get("avenue", "Desconocida"),
        time_slot=config.get("time_slot", "No especificada"),
        video_name=video_basename,
        semaphore_config_id=config_id,
    )

    nie_path = save_nie_infractions_json(
        nie_list,
        output_dir=args.output_dir,
        filename="nie_infracciones.json",
        avenue_name=config.get("avenue", "Desconocida"),
        time_slot=config.get("time_slot", "No especificada"),
        video_name=video_basename,
        semaphore_config_id=config_id,
    )

    indicators_path = save_indicators_json(
        nid_count=result["nid_count"],
        nie_count=result["nie_count"],
        ti_percentage=result["ti_pct"],
        tr_individual_minutes=result["tr_minutes"],
        tr_overall_minutes=result["tr_seconds_per_nid"] / 60.0,
        output_dir=args.output_dir,
    )

    # ── Print report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INFRACTIVISION CLI — REPORTE DE PROCESAMIENTO")
    print("=" * 60)
    print(f"  Video:              {video_basename}")
    print(f"  Duración:           {result['video_duration']}")
    print(f"  Frames totales:     {result['total_frames']}")
    print(f"  Frames procesados:  {result['processed_frames']}")
    print(f"  Tiempo total:       {result['elapsed_seconds']:.1f}s")
    print(f"  FPS promedio:       {result['avg_fps']:.1f}")
    print()
    print(f"  --- Infracciones ---")
    print(f"  Total detectadas:   {result['tir']}")
    print(f"    NID:              {result['nid_count']} ({result['ti_pct']:.1f}%)")
    print(f"    NIE:              {result['nie_count']}")
    print(f"    TIR:              {result['tir']}")
    print()
    print(f"  --- Indicadores ---")
    print(f"  TI (acierto):       {result['ti_pct']:.1f}%")
    print(f"  TR:                 {result['tr_seconds_per_nid']:.2f} s/NID")
    print(f"    Tiempo ejecución: {result['elapsed_seconds']:.1f}s")
    print(f"    Duración video:   {result['video_duration']}")
    print()
    print(f"  --- Archivos ---")
    print(f"  {nid_path}")
    print(f"  {nie_path}")
    print(f"  {indicators_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
