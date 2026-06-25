#!/usr/bin/env python
"""
only_infractions.py — Vehicle-level infraction detection (no OCR / no plate).

This is a SEGMENTED version of the full pipeline. It runs Phase 1 only:
  - Vehicle detection (YOLO, batched)
  - Centroid tracking across frames
  - Polygon containment
  - PPI V50 proximity index
  - MMRP peak detection
  - V50 triggers: PEAK_GOLD and HEAVY (the two that do NOT require
    plate confirmation)

It does NOT run:
  - Plate YOLO detector
  - LPRNet OCR
  - Perspective rectification
  - NID/NIE classification
  - JSON infraction persistence

The purpose is independent testing of the infraction detection step
without paying the cost (or complexity) of plate recognition.

All events are written to a timestamped log file:
  data/logs/infractions_<ISO_timestamp>.log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detection.vehicle_detector import VehicleDetector
from scripts.adapter.video_semaphore import VideoSemaphore
from scripts.adapter.infraction_tracker import InfractionTracker, calculate_ppi
from scripts.adapter.adaptive_skip import AdaptiveSkipController
from scripts.adapter.frame_reader import FrameReader
from scripts.adapter.stage_profiler import StageProfiler
from scripts.adapter.threads import configure_thread_budget


# ── Logging helpers ──────────────────────────────────────────────────


def make_run_id() -> str:
    """Return ISO timestamp safe for filenames (no colons)."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def setup_file_logger(log_path: str, quiet: bool = False) -> logging.Logger:
    """Create a logger that writes to a timestamped file + optional stdout."""
    logger = logging.getLogger("only_infractions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ── Config loading ──────────────────────────────────────────────────


def _load_config_json(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_configs(video_basename: str) -> tuple[Any, Any, Any]:
    polygon, semaphore, avenue = None, None, None

    poly_path = PROJECT_ROOT / "config" / "polygon_config.json"
    if poly_path.exists():
        poly = json.loads(poly_path.read_text(encoding="utf-8"))
        if video_basename in poly:
            polygon = poly[video_basename]

    time_path = PROJECT_ROOT / "config" / "time_presets.json"
    if time_path.exists():
        times = json.loads(time_path.read_text(encoding="utf-8"))
        if video_basename in times:
            semaphore = times[video_basename]

    aven_path = PROJECT_ROOT / "config" / "avenue_config.json"
    if aven_path.exists():
        aven = json.loads(aven_path.read_text(encoding="utf-8"))
        if video_basename in aven:
            avenue = aven[video_basename]

    return polygon, semaphore, avenue


def _build_config(args: argparse.Namespace) -> dict:
    """Merge CLI args, --config file, and config/*.json auto-discovery."""
    video_basename = os.path.basename(args.video)
    polygon, semaphore, avenue = _load_existing_configs(video_basename)

    cfg: dict[str, Any] = {
        "conf_vehicle": args.conf_vehicle,
        "batch_size": args.batch_size,
    }

    if args.config:
        user_cfg = _load_config_json(args.config)
        cfg["polygon"] = user_cfg.get("polygon")
        cfg["semaphore"] = user_cfg.get(
            "semaphore", {"green": 30, "yellow": 5, "red": 40}
        )
        cfg["avenue"] = user_cfg.get("avenue", avenue or "Desconocida")
        cfg["time_slot"] = user_cfg.get("time_slot", "No especificada")
    else:
        cfg["polygon"] = polygon
        cfg["semaphore"] = semaphore or {"green": 30, "yellow": 5, "red": 40}
        cfg["time_slot"] = (
            semaphore.get("time_slot", "No especificada")
            if semaphore
            else "No especificada"
        )
        cfg["avenue"] = avenue or "Desconocida"

    if cfg.get("polygon") is None:
        print(
            "⚠️  ADVERTENCIA: sin polígono → la detección de infracciones no funcionará."
        )
    if not cfg.get("semaphore"):
        cfg["semaphore"] = {"green": 30, "yellow": 5, "red": 40}

    return cfg


# ── Pipeline ─────────────────────────────────────────────────────────


class OnlyInfractionsPipeline:
    """Phase 1 only: detect vehicles, track, and fire infractions.

    No plate detection, no OCR, no JSON persistence.
    Outputs a structured log + vehicle crops for visual inspection.
    """

    def __init__(
        self,
        config: dict,
        log: logging.Logger,
        crops_dir: str,
        use_new: bool = False,
    ) -> None:
        self.config = config
        self.log = log
        self.crops_dir = crops_dir
        self.use_new = use_new
        os.makedirs(self.crops_dir, exist_ok=True)

        # Single-thread executor for the JPG crop writes. JPEG
        # encoding is CPU-bound, so a pool would only steal cycles
        # from the inference loop on a 4-core box. One worker is
        # enough to remove the imwrite stall from the critical
        # path. Shut down at the end of process(). Only created in
        # --new mode; in legacy mode crops are written synchronously.
        self.crop_writer: ThreadPoolExecutor | None
        if use_new:
            self.crop_writer = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="crop-writer"
            )
        else:
            self.crop_writer = None

        self.semaphore = VideoSemaphore(config["semaphore"])
        self.polygon: np.ndarray | None = (
            np.array(config["polygon"], np.int32)
            if config.get("polygon")
            else None
        )

        self.conf_vehicle = config.get("conf_vehicle", 0.50)
        # Batch size: --new (refactor) → 2, default (legacy) → 4.
        default_batch = 2 if use_new else 4
        batch_cfg = config.get("batch_size")
        self.batch_size = batch_cfg if batch_cfg is not None else default_batch

        if use_new:
            # Skip-rate controller. Target FPS is set in process()
            # once we know the real video FPS.
            self.skip_controller = AdaptiveSkipController(
                target_fps_video=30.0
            )
            # Per-stage profiler. See stage_profiler.py.
            self.profiler = StageProfiler()
        else:
            # Legacy mode: no skip controller, no profiler. The
            # legacy skip logic uses hardcoded rates and decode
            # happens synchronously on the main thread.
            self.skip_controller = None
            self.profiler = None

        # Model load + warmup are separated so the warmup cost does
        # not skew the average of subsequent inference calls. Only
        # the --new path records these in the profiler.
        if use_new:
            with self.profiler.stage("model_load"):
                self.vehicle_detector = VehicleDetector(
                    str(PROJECT_ROOT / "models" / "yolov8n.pt")
                )
            with self.profiler.stage("model_warmup"):
                _warmup_frame = np.zeros((416, 416, 3), dtype=np.uint8)
                self.vehicle_detector.detect_batch(
                    [_warmup_frame], conf=0.25
                )
        else:
            self.vehicle_detector = VehicleDetector(
                str(PROJECT_ROOT / "models" / "yolov8n.pt")
            )
        self.plate_detector = None  # NEVER load plate detector

    def _quick_plate_check_stub(self) -> bool:
        """By design, this script never confirms plates.

        InfractionTracker is still called with has_plate_score=False,
        so the PANIC and SECURE triggers are naturally disabled.
        """
        return False

    # ── per-batch processing ──────────────────────────────────────

    def _process_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        fps_video: float,
        tracker: InfractionTracker,
        infractions_out: list[dict],
        display: bool,
    ) -> None:
        # YOLO inference bundles preprocess + NMS internally on
        # Ultralytics; we charge the whole call to "inference". If
        # the profiler shows this dominates, the next step is to
        # lower the input resolution or switch to OpenVINO. In
        # legacy mode we skip the profiler wrapping.
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

            for det in detections:
                x1, y1, x2, y2, cls = det
                bumper_x = (x1 + x2) // 2
                bumper_y = y2
                vehicle_area = (x2 - x1) * (y2 - y1)
                v_left = (x1, y2)
                v_right = (x2, y2)

                ppi = calculate_ppi(bumper_x, bumper_y, h, w)
                if ppi < 0.20:
                    continue

                # Only process if RED and polygon is configured
                if current_state != "red" or self.polygon is None:
                    continue

                # Plate score is ALWAYS False in this mode
                has_plate = self._quick_plate_check_stub()

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
                            plate_detector=None,  # never use plate in this mode
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
                        plate_detector=None,  # never use plate in this mode
                    )

                if trigger:
                    infractions_out.append(trigger)
                    self._log_trigger(trigger, fidx, video_second)

            if display:
                if self.polygon is not None:
                    pts = self.polygon.reshape(-1, 1, 2)
                    cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
                state_colors = {
                    "red": ((0, 0, 255), (255, 255, 255)),
                    "yellow": ((0, 255, 255), (0, 0, 0)),
                    "green": ((0, 255, 0), (0, 0, 0)),
                }
                tc, bc = state_colors.get(
                    current_state, ((255, 255, 255), (0, 0, 0))
                )
                txt = f" SEMAFORO: {current_state.upper()} "
                cv2.rectangle(display_frame, (10, 10), (350, 50), bc, -1)
                cv2.putText(
                    display_frame,
                    txt,
                    (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    tc,
                    2,
                )
                # Draw all detections in red when state=red
                for det in detections:
                    x1, y1, x2, y2, _ = det
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                    )
                cv2.imshow("only_infractions", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return  # stop display, but keep processing

    def _log_trigger(self, trigger: dict, frame_index: int, video_second: float) -> None:
        """Log a trigger event and save the vehicle crop.

        In --new mode the actual `cv2.imwrite` is submitted to the
        single-thread `crop_writer` executor so the encode cost does
        not block the frame loop. In legacy mode the imwrite runs
        synchronously on the main thread. We capture the relevant
        numpy arrays in the closure — do NOT pass the whole `trigger`
        dict (it would keep the (potentially large) `snap["img"]`
        alive in the worker for longer than needed).
        """
        snap = trigger["snapshot"]
        track_id = trigger["track_id"]
        ppi = trigger["proximity_factor"]
        num_frames = trigger["num_frames"]
        trigger_type = trigger["trigger_type"]

        # Build the crop file path up front; the actual encode is
        # done by the executor (--new) or synchronously (legacy).
        crop_filename = f"vehicle_inf{track_id}_t{track_id}_f{frame_index}.jpg"
        crop_path = os.path.join(self.crops_dir, crop_filename)

        img = snap.get("img")
        bbox = snap.get("bbox")
        if img is not None and bbox is not None:
            bx1, by1, bx2, by2 = bbox
            # Slice eagerly so the worker does not need access to
            # the full frame.
            try:
                crop = img[by1:by2, bx1:bx2].copy()
            except Exception:
                crop = None
            if crop is not None and crop.size > 0:
                if self.use_new and self.crop_writer is not None:
                    # --new: async crop write
                    self.crop_writer.submit(_write_jpg, crop, crop_path)
                else:
                    # Legacy: sync imwrite on the main thread
                    try:
                        cv2.imwrite(crop_path, crop)
                    except Exception:
                        pass

        self.log.info(
            f"🚨 {trigger_type} TRIGGER #{track_id} "
            f"at frame {frame_index} (t={video_second:.1f}s) "
            f"PPI={ppi:.2f} num_frames={num_frames}"
        )
        self.log.info(f"   Crop queued: {crop_path}")

    # ── main run ──────────────────────────────────────────────────

    def process(
        self,
        video_path: str,
        max_frames: int | None = None,
        skip_start: int = 0,
        skip_end: int = 0,
        display: bool = False,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.use_new:
            # Tell the skip controller the real frame budget now.
            self.skip_controller.set_target_fps(fps_video)
        video_basename = os.path.basename(video_path)
        video_duration_sec = total_frames / fps_video if fps_video else 0
        video_duration_str = (
            f"{int(video_duration_sec // 60):02d}:{int(video_duration_sec % 60):02d}"
        )

        # Background reader so decode overlaps with the inference
        # batches. Bounded queue (maxsize=2) prevents stale frames.
        # Only in --new mode; legacy mode decodes synchronously.
        reader = FrameReader(cap).start() if self.use_new else None

        if skip_start > 0:
            start_frame = int(skip_start * fps_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.log.info(f"⏩ Skipping first {skip_start}s → frame {start_frame}")

        # ── HEADER ────────────────────────────────────────────────
        self.log.info("=" * 60)
        self.log.info("=== START OnlyInfractions ===")
        self.log.info(f"Video: {video_path}")
        self.log.info(
            f"Polígono: {len(self.polygon) if self.polygon is not None else 'NONE'}"
        )
        sem = self.semaphore.get_cycle_durations()
        self.log.info(
            f"Semáforo: green={sem['green']} yellow={sem['yellow']} red={sem['red']} "
            f"offset={self.semaphore.offset}"
        )
        self.log.info(
            f"Frames totales: {total_frames} | FPS video: {fps_video:.1f}"
        )
        self.log.info(f"Conf vehicle: {self.conf_vehicle} | Batch: {self.batch_size}")
        self.log.info("=" * 60)

        # ── Load models ───────────────────────────────────────────
        t_model_start = time.time()
        # Vehicle detector was already loaded in __init__
        # We re-measure for reporting consistency
        t_model_elapsed = time.time() - t_model_start
        self.log.info(f"Modelos cargados ({t_model_elapsed:.1f}s)")

        # ── Frame loop ────────────────────────────────────────────
        tracker = InfractionTracker()
        frame_index = 0
        processed = 0
        infractions_raw: list[dict] = []
        frames_batch: list[np.ndarray] = []
        idx_batch: list[int] = []

        t_proc_start = time.time()
        last_log_second = -1

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
            if skip_end:
                cutoff = total_frames - int(skip_end * fps_video)
                if frame_index > cutoff:
                    break

            video_second = frame_index / fps_video
            current_state = self.semaphore.get_state(video_second)

            # Log state transitions (at integer second boundaries)
            cur_sec = int(video_second)
            if cur_sec != last_log_second:
                last_log_second = cur_sec
                self.log.debug(
                    f"t={video_second:.1f}s frame={frame_index} state={current_state}"
                )

            # Skip decision
            if self.use_new:
                # AdaptiveSkipController (time-budget aware). See
                # adaptive_skip.py for the policy. The old
                # "current_state != red" exception is dropped
                # because the controller's red+active branch
                # already returns 1 when the system keeps up with
                # budget.
                skip_rate = self.skip_controller.suggest_skip(
                    current_state, tracker.active_count
                )
            else:
                # Legacy hardcoded skip rates (reconstructed from
                # docs/REFACTOR_CPU_PIPELINE.md section 4 — no
                # pre-refactor version exists in git history).
                skip_rate = _legacy_skip_rate(
                    current_state, tracker.active_count
                )

            if frame_index % skip_rate != 0:
                continue

            frames_batch.append(frame)
            idx_batch.append(frame_index)

            if len(frames_batch) >= self.batch_size:
                t0 = time.perf_counter()
                self._process_batch(
                    frames_batch,
                    idx_batch,
                    fps_video,
                    tracker,
                    infractions_raw,
                    display,
                )
                if self.use_new:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    self.skip_controller.record(
                        elapsed_ms, len(frames_batch)
                    )
                processed += len(frames_batch)
                frames_batch = []
                idx_batch = []

        # Drain remaining batch
        if frames_batch:
            t0 = time.perf_counter()
            self._process_batch(
                frames_batch,
                idx_batch,
                fps_video,
                tracker,
                infractions_raw,
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

        # Wait for all pending crop writes to finish before we
        # return. `wait=True` blocks until they all complete; the
        # `cancel_futures=False` (default) lets the queued work run.
        # Only in --new mode; legacy mode has no executor.
        if self.crop_writer is not None:
            self.crop_writer.shutdown(wait=True)

        t_proc_elapsed = time.time() - t_proc_start
        t_total_elapsed = time.time() - t_proc_start + t_model_elapsed

        # ── Trigger summary ───────────────────────────────────────
        triggers_peak = sum(1 for t in infractions_raw if t["trigger_type"] == "PEAK")
        triggers_heavy = sum(1 for t in infractions_raw if t["trigger_type"] == "PERSIST")
        # Note: in plate-agnostic mode, the only triggers are PEAK and PERSIST
        # (PANIC/SECURE need plate confirmation, which is always False here)

        fps_avg = processed / t_proc_elapsed if t_proc_elapsed > 0 else 0

        # Emit the per-stage profile so the operator can see where
        # the CPU budget really went. One-shot stages (model_load,
        # model_warmup) are reported separately from recurring ones
        # (decode, inference, tracker) so percentages are honest.
        # Only in --new mode; legacy mode has no profiler.
        if self.use_new:
            self.log.info("")
            self.log.info("--- Profile (per-stage wall time) ---")
            for line in self.profiler.report().splitlines():
                self.log.info(line)
            self.log.info("")

        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("=== END OnlyInfractions ===")
        self.log.info(f"Frames procesados: {processed} de {total_frames}")
        self.log.info(f"Infractores: {len(infractions_raw)}")
        self.log.info(f"  PEAK_GOLD:   {triggers_peak}")
        self.log.info(f"  HEAVY:       {triggers_heavy}")
        self.log.info(
            f"Tiempos: model_load={t_model_elapsed:.1f}s "
            f"processing={t_proc_elapsed:.1f}s total={t_total_elapsed:.1f}s"
        )
        self.log.info(f"FPS promedio: {fps_avg:.1f}")
        self.log.info("=" * 60)

        return {
            "video": video_basename,
            "video_path": video_path,
            "video_duration": video_duration_str,
            "total_frames": total_frames,
            "processed_frames": processed,
            "infractions": infractions_raw,
            "peak_count": triggers_peak,
            "heavy_count": triggers_heavy,
            "total_triggers": len(infractions_raw),
            "t_model": t_model_elapsed,
            "t_processing": t_proc_elapsed,
            "t_total": t_total_elapsed,
            "fps_avg": fps_avg,
            "profiler": self.profiler.as_dict() if self.use_new else None,
        }


# ── Main ─────────────────────────────────────────────────────────────


def _write_jpg(img: np.ndarray, path: str) -> None:
    """Module-level helper so it pickles / captures cleanly in the executor."""
    try:
        cv2.imwrite(path, img)
    except Exception:
        # Swallow exceptions in the worker — the main thread has
        # already logged the trigger. A failed crop write must not
        # kill the pipeline.
        pass


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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for only_infractions.py.

    Extracted into a function so tests can introspect the parser
    without invoking main().
    """
    parser = argparse.ArgumentParser(
        description="OnlyInfractions — Detección de infracciones SIN OCR / placa"
    )
    parser.add_argument(
        "--video", required=True, help="Ruta al archivo de video"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Activa el refactor CPU/iGPU: thread budget (cv=2, torch=4), "
             "FrameReader (background decode), AdaptiveSkipController, "
             "StageProfiler, warmup frame, crop_writer async executor, "
             "batch_size=2. Default (sin --new): comportamiento pre-refactor "
             "con skip fijo, decode síncrono, sin profiler, batch_size=4.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON de configuración (polígono + semáforo + avenida). "
             "Si no se provee, busca en config/*.json por nombre de video.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/logs",
        help="Directorio de logs (default: data/logs)",
    )
    parser.add_argument(
        "--crops-dir",
        default="data/output/only_infractions",
        help="Directorio de crops de vehículo (default: data/output/only_infractions)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Frames máximos a procesar",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        help="Segundos a saltar al inicio",
    )
    parser.add_argument(
        "--skip-end",
        type=int,
        default=0,
        help="Segundos a omitir al final",
    )
    parser.add_argument(
        "--conf-vehicle",
        type=float,
        default=0.50,
        help="Umbral confianza YOLO vehículos (default: 0.50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño del batch YOLO. Default: 2 con --new, 4 sin --new (legacy).",
    )
    parser.add_argument(
        "--display", action="store_true", help="Mostrar ventana cv2 en vivo"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Solo loguear al archivo, sin stdout"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Pin the thread budget BEFORE argparse or any heavy import so
    # OpenCV / PyTorch / BLAS pick up the right defaults before their
    # internal thread pools start. cv2.setNumThreads is cheap and
    # idempotent; torch.set_num_threads is too. The returned dict
    # tells us exactly what was applied, which is useful for
    # diagnosing "why is this so slow on my machine?" later.
    # Only with --new; in legacy mode the defaults (8+8) are kept.
    if args.new:
        budget = configure_thread_budget()
        print("⚙️  Thread budget:", budget)
        print("✨ Modo --new: refactor CPU/iGPU activo")
    else:
        print("ℹ️  Modo clásico (sin --new): skip fijo, decode síncrono, "
              "sin profiler, batch_size=4")

    # ── Run id + log path ─────────────────────────────────────────
    run_id = make_run_id()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"infractions_{run_id}.log")

    # Each run gets its own crops subdir
    crops_dir = os.path.join(args.crops_dir, f"run_{run_id}")
    os.makedirs(crops_dir, exist_ok=True)

    log = setup_file_logger(log_path, quiet=args.quiet)
    log.info(f"Log file: {log_path}")
    log.info(f"Crops dir: {crops_dir}")

    # ── Build config ─────────────────────────────────────────────
    config = _build_config(args)
    log.info(f"Config: avenue={config['avenue']} time_slot={config['time_slot']}")
    if config.get("polygon") is None:
        log.warning("Sin polígono → no se detectarán infracciones")

    # ── Run pipeline ─────────────────────────────────────────────
    pipeline = OnlyInfractionsPipeline(
        config, log, crops_dir, use_new=bool(args.new)
    )

    t_wall_start = time.time()
    result = pipeline.process(
        video_path=args.video,
        max_frames=args.max_frames,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        display=args.display,
    )
    t_wall_total = time.time() - t_wall_start

    # ── Stdout summary (always printed, even with --quiet) ───────
    print()
    print("=" * 60)
    print("  INFRACTIVISION — ONLY INFRACTIONS (no OCR)")
    print("=" * 60)
    print(f"  Video:           {result['video']}")
    print(f"  Duración:        {result['video_duration']}")
    print(f"  Frames:          {result['total_frames']} totales | "
          f"{result['processed_frames']} procesados")
    print(f"  Tiempos:")
    print(f"    Carga modelos:  {result['t_model']:.1f}s")
    print(f"    Procesamiento:  {result['t_processing']:.1f}s")
    print(f"    Wall clock:     {t_wall_total:.1f}s")
    print(f"  FPS promedio:    {result['fps_avg']:.1f}")
    print(f"  Infractores:     {result['total_triggers']}")
    print(f"    PEAK_GOLD:     {result['peak_count']}")
    print(f"    HEAVY:         {result['heavy_count']}")
    print()
    print(f"  Log:     {log_path}")
    print(f"  Crops:   {crops_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
