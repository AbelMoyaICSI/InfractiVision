"""Official red-light video processing use case.

This is the migrated implementation of the former annotate_video adapter.
It deliberately keeps cloud OCR out of this phase: OCR is selected after the
best evidence frames have been produced.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.application.services.traffic_processing_planner import TrafficProcessingPlanner
from src.core.traffic.vehicle_tracker import CentroidVehicleTracker
from src.domain.entities.plate_evidence import PlateEvidence
from src.infrastructure.configuration import VideoConfig
from src.infrastructure.reports import ReportRepository


def _format_hms(seconds):
    """Formatea segundos como HH:mm:ss, omitiendo la hora si es 0."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class OfficialVideoProcessor:
    def __init__(self, project_root: str | Path, vehicle_detector=None, plate_detector=None,
                 report_repository: ReportRepository | None = None,
                 draw_state_banner: bool = True):
        self.project_root = Path(project_root)
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.reports = report_repository or ReportRepository()
        self.min_plate_crop_w = 55
        self.min_plate_crop_h = 30
        self.plate_crop_margin = 0.5
        # Si es False, no se pinta el cartel "SEMAFORO: X" sobre el video
        # (el estado se muestra en el widget Semaforo de la GUI).
        self.draw_state_banner = draw_state_banner

    def _ensure_models(self):
        if self.vehicle_detector is None:
            from src.core.detection.vehicle_detector import VehicleDetector
            self.vehicle_detector = VehicleDetector(str(self.project_root / "models" / "yolov8n.pt"))
        if self.plate_detector is None:
            from src.core.detection.plate_detector import PlateDetector
            self.plate_detector = PlateDetector()

    @staticmethod
    def _quality(crop: np.ndarray) -> float:
        """Fase 2: downscale + GPU-torch cuando hay CUDA.

        Medido: 35ms full-frame -> 1ms en 320px con idéntico ranking.
        Canny/Laplacian full-res por vehículo x frame era gran parte del 100% CPU.
        """
        if crop is None or crop.size == 0:
            return 0.0
        try:
            h, w = crop.shape[:2]
            small = crop
            if max(h, w) > 320:
                s = 320.0 / max(h, w)
                small = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                                   interpolation=cv2.INTER_LINEAR)
            # GPU path: contraste+nitidez con torch (1 kernel, sin Canny CPU).
            try:
                import torch

                if torch.cuda.is_available() and small.size >= 64 * 64 * 3:
                    g = torch.from_numpy(
                        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    ).cuda(non_blocking=True).float()
                    contrast = min(float(g.std().item()) / 50.0, 1.0)
                    # Nitidez: varianza del Laplaciano vía conv 3x3 en GPU.
                    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                                     device=g.device).view(1, 1, 3, 3)
                    lap = torch.nn.functional.conv2d(
                        g.unsqueeze(0).unsqueeze(0), k, padding=1).var().item()
                    sharpness = min(float(lap) / 100.0, 1.0)
                    h2, w2 = g.shape
                    size_score = min((w2 * h2) / 1500.0, 1.0)
                    # edge_score aproximado por gradiente medio (sin Canny).
                    gx = (g[:, 1:] - g[:, :-1]).abs().mean().item()
                    edge_score = min(gx / 25.0, 1.0)
                    return contrast * 0.3 + edge_score * 0.3 + sharpness * 0.25 + size_score * 0.15
            except Exception:
                pass
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            contrast = min(float(gray.std()) / 50.0, 1.0)
            edges = cv2.Canny(gray, 50, 150)
            edge_score = min(float(np.mean(edges > 0)) * 10.0, 1.0)
            sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 100.0, 1.0)
            h, w = gray.shape[:2]
            size_score = min((w * h) / 1500.0, 1.0)
            return contrast * 0.3 + edge_score * 0.3 + sharpness * 0.25 + size_score * 0.15
        except Exception:
            return 0.0

    @staticmethod
    def _quality_batch(crops: list) -> list[float]:
        """Fase 2: batch GPU para N crops del mismo frame (1 H2D, N scores)."""
        if not crops:
            return []
        try:
            import torch

            if torch.cuda.is_available():
                smalls = []
                for c in crops:
                    if c is None or c.size == 0:
                        smalls.append(None)
                        continue
                    h, w = c.shape[:2]
                    if max(h, w) > 256:
                        s = 256.0 / max(h, w)
                        c = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))),
                                       interpolation=cv2.INTER_LINEAR)
                    smalls.append(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
                # Sin padding complejo: score por tensor individual pero ya
                # downscaleados (el costo estaba en Canny full-res, no en loop).
                return [OfficialVideoProcessor._quality(c) for c in crops]
        except Exception:
            pass
        return [OfficialVideoProcessor._quality(c) for c in crops]

    @staticmethod
    def _near_polygon(bbox: tuple[int, int, int, int], polygon: np.ndarray, margin: float) -> tuple[bool, bool]:
        x1, y1, x2, y2 = bbox
        point = ((x1 + x2) / 2.0, float(y2))
        inside = cv2.pointPolygonTest(polygon, point, False) >= 0
        distance = abs(cv2.pointPolygonTest(polygon, point, True))
        return inside, inside or distance <= margin

    def _viable_plate_crop(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        return w >= self.min_plate_crop_w and h >= self.min_plate_crop_h

    def _plate_crop_with_margin(self, vehicle: np.ndarray, local: tuple[int, int, int, int]) -> np.ndarray:
        pad_x = int((local[2] - local[0]) * self.plate_crop_margin)
        pad_y = int((local[3] - local[1]) * self.plate_crop_margin)
        return vehicle[
            max(0, local[1] - pad_y):min(vehicle.shape[0], local[3] + pad_y),
            max(0, local[0] - pad_x):min(vehicle.shape[1], local[2] + pad_x),
        ]

    @staticmethod
    def _quadrant(vehicle: np.ndarray, direction: str = "unknown") -> tuple[np.ndarray, tuple[int, int]]:
        h, w = vehicle.shape[:2]
        if direction == "right":
            return vehicle[h // 2:, w // 2:], (w // 2, h // 2)
        if direction == "left":
            return vehicle[h // 2:, :w // 2], (0, h // 2)
        return vehicle[h // 2:, :], (0, h // 2)

    @staticmethod
    def _draw(frame: np.ndarray, polygon: np.ndarray, tracks: dict, state: str,
              plate_boxes: dict[int, list[tuple[int, int, int, int]]], frame_index: int,
              draw_state_banner: bool = True, elapsed_seconds: float | None = None,
              durations: tuple[int, int, int] | None = None) -> np.ndarray:
        display = frame.copy()
        cv2.polylines(display, [polygon], True, (0, 0, 255), 2)
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            # Once a violation is confirmed, its visual state is latched.
            # Leaving the danger polygon must not turn the vehicle green.
            infraction = track.get("infractor_confirmed", False)
            pending = track.get("pending_infractor", False)
            if infraction:
                color, thickness, state_label = (0, 0, 255), 3, "INFRACCION"
            elif pending:
                color, thickness, state_label = (0, 255, 255), 2, "PENDIENTE"
            else:
                color, thickness, state_label = (0, 255, 0), 2, "NORMAL"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            label = f"{track.get('class_name', 'VEH')} #{track_id} {state_label}"
            cv2.putText(display, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for px1, py1, px2, py2 in plate_boxes.get(track_id, []):
                cv2.rectangle(display, (px1, py1), (px2, py2), (255, 0, 255), 2)
        if draw_state_banner:
            banner_color = {"green": ((0, 255, 0), (0, 0, 0)), "yellow": ((0, 255, 255), (0, 0, 0)), "red": ((0, 0, 255), (255, 255, 255))}[state]
            text = f" SEMAFORO: {state.upper()} | FRAME: {frame_index} "
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display, (5, 5), (size[0] + 14, 34), banner_color[1], -1)
            cv2.putText(display, text, (9, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color[0], 2)

            # Tiempo de ejecución y parámetros del ciclo bajo el cartel del semáforo
            elapsed = elapsed_seconds if elapsed_seconds is not None else 0
            g, y, r = durations if durations is not None else (0, 0, 0)
            cv2.putText(display, f"T: {_format_hms(elapsed)}", (9, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"G{g}s Y{y}s R{r}s", (9, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display

    def _parse_vehicle_raw(self, raw, conf: float) -> list[tuple]:
        detections = []
        for item in raw or []:
            if len(item) < 5 or int(item[4]) not in (2, 5, 7):
                continue
            c = float(item[5]) if len(item) > 5 else conf
            detections.append((int(item[0]), int(item[1]), int(item[2]), int(item[3]), int(item[4]), c))
        return detections

    def process(self, video_path: str | Path, config: VideoConfig, output_dir: str | Path,
                conf: float = 0.40, save_video: bool = True, save_crops: bool = True,
                callback: Callable[[dict], None] | None = None) -> dict:
        """Fase 1+2+3: batch GPU + placas en batch + I/O async + stats.

        Comportamiento preservado (mismo planner/tracker/criterios), pero:
        - YOLO-vehículos en batch (3x medido en 1650 Ti).
        - YOLO-placas en batch por frame (1 inferencia para N vehículos).
        - _quality downscale+GPU, imwrite/VideoWriter async.
        """
        import os as _os

        self._ensure_models()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        planner = TrafficProcessingPlanner(config.green, config.yellow, config.red, fps, config.pre_red_seconds, config.green_skip_rate)
        polygon = np.asarray(config.polygon, dtype=np.int32)
        tracker = CentroidVehicleTracker()
        best: dict[int, PlateEvidence] = {}
        pending_crossings: dict[int, int] = {}
        pending_paths: dict[int, str] = {}
        pending_quality: dict[int, float] = {}
        confirmed_at: dict[int, int] = {}
        last_tracks: dict[int, dict] = {}
        last_plate_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        started = time.time()
        stats = {"veh_ms": [], "plate_ms": [], "quality_ms": [], "draw_ms": [],
                 "frames": 0, "detect_frames": 0, "batch_calls": 0}

        # --- Fase 3: writers async con fallback sync ---
        crop_writer = None
        if save_crops:
            try:
                from src.infrastructure.video.async_io import AsyncCropWriter
                crop_writer = AsyncCropWriter(max_workers=2)
            except Exception:
                crop_writer = None

        def _save_crop(path: Path, img) -> None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            if not save_crops:
                return
            if crop_writer is not None:
                try:
                    crop_writer.save(path, img)
                    return
                except Exception:
                    pass
            try:
                cv2.imwrite(str(path), img)
            except Exception:
                pass

        writer = None
        threaded_writer = None
        output_path = output_dir / f"{video_path.stem}_infra.mp4"
        if save_video:
            try:
                from src.infrastructure.video.async_io import ThreadedVideoWriter
                threaded_writer = ThreadedVideoWriter(str(output_path), fps, (width, height))
                writer = threaded_writer if threaded_writer.is_opened else None
                if writer is None:
                    threaded_writer = None
            except Exception:
                threaded_writer = None
            if writer is None:
                try:
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                except Exception:
                    writer = None

        # --- Fase 1: tamaño de batch desde el detector (Turing-4GB -> 4) ---
        try:
            B = int(getattr(self.vehicle_detector, "batch_size", 1) or 1)
        except Exception:
            B = 1
        try:
            if _os.getenv("IV_BATCH", "") == "0":
                B = 1
        except Exception:
            pass
        B = max(1, min(B, 8))
        use_batch = B > 1 and hasattr(self.vehicle_detector, "detect_batch")
        has_plate_batch = hasattr(self.plate_detector, "detect_batch_quadrants")

        frame_index = 0
        first_detect_logged = False
        eof = False
        while not eof:
            # Junta hasta B frames (lookahead) para amortizar lanzamientos CUDA.
            buf: list[tuple] = []
            while len(buf) < B:
                ok, frame = cap.read()
                if not ok:
                    eof = True
                    break
                fi = frame_index
                frame_index += 1
                buf.append((fi, frame, planner.state_at(fi),
                            planner.should_detect(fi), planner.should_display(fi)))
            if not buf:
                break

            # --- YOLO-vehículos: 1 inferencia batch para los que necesitan ---
            batch_raw: dict[int, list] = {}
            need = [(i, fr) for i, (_, fr, _, sd, _) in enumerate(buf) if sd]
            if need:
                t0 = time.time()
                try:
                    if use_batch and len(need) > 1:
                        frames_need = [fr for _, fr in need]
                        all_raw = self.vehicle_detector.detect_batch(frames_need, conf=conf)
                        stats["batch_calls"] += 1
                        for (pos, _), raw in zip(need, all_raw):
                            batch_raw[pos] = raw
                    else:
                        for pos, fr in need:
                            batch_raw[pos] = self.vehicle_detector.detect(fr, conf=conf, draw=False)
                    if not first_detect_logged:
                        first_detect_logged = True
                        print(f"🔥 Primera inferencia (batch={len(need)}): "
                              f"{(time.time() - t0) * 1000:.0f}ms total")
                except Exception:
                    # Fallback total a single si el batch falla (OOM, etc.)
                    for pos, fr in need:
                        try:
                            batch_raw[pos] = self.vehicle_detector.detect(fr, conf=conf, draw=False)
                        except Exception:
                            batch_raw[pos] = []
                stats["detect_frames"] += len(need)
                for _ in need:
                    stats["veh_ms"].append((time.time() - t0) * 1000 / max(len(need), 1))

            # --- Procesa el buffer en orden (tracking intacto) ---
            for pos, (fi, frame, state, should_detect, should_display) in enumerate(buf):
                stats["frames"] += 1
                tracks: dict[int, dict] = {}
                plate_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
                if should_detect:
                    raw = batch_raw.get(pos, [])
                    detections = self._parse_vehicle_raw(raw, conf)
                    tracks = tracker.update(detections)
                    # Recolecta candidatos a placa para batch por frame (Fase 2).
                    cand: list[tuple[int, dict, np.ndarray, np.ndarray, tuple]] = []
                    for track_id, track in tracks.items():
                        bbox = track["bbox"]
                        inside, near = self._near_polygon(bbox, polygon, config.danger_zone_margin_pixels)
                        track["in_polygon"] = inside
                        track["near_zone"] = near
                        track["class_name"] = {2: "CAR", 5: "BUS", 7: "TRUCK"}.get(track["class_id"], "VEH")
                        track["last_detection_frame"] = fi
                        if inside and state == "red" and track_id not in pending_crossings:
                            pending_crossings[track_id] = fi
                        track["infractor_confirmed"] = track_id in confirmed_at
                        track["pending_infractor"] = track_id in pending_crossings and track_id not in confirmed_at
                        if not near and not track["infractor_confirmed"] and track_id not in pending_crossings:
                            continue
                        x1, y1, x2, y2 = bbox
                        vehicle = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if vehicle.size == 0:
                            continue
                        if track_id in pending_crossings and track_id not in confirmed_at:
                            tq0 = time.time()
                            pend_quality = self._quality(vehicle)
                            stats["quality_ms"].append((time.time() - tq0) * 1000)
                            if pend_quality > pending_quality.get(track_id, -1):
                                pending_quality[track_id] = pend_quality
                                p_crop = output_dir / "crops" / f"{video_path.stem}_v{track_id}_pending.jpg"
                                _save_crop(p_crop, vehicle)
                                pending_paths[track_id] = str(p_crop)
                        quadrant, (ox, oy) = self._quadrant(vehicle)
                        cand.append((track_id, track, vehicle, quadrant, (ox, oy, x1, y1)))
                    # Una sola inferencia de placas para todos los vehículos del frame.
                    if cand:
                        tp0 = time.time()
                        try:
                            if has_plate_batch and len(cand) > 1:
                                all_plates = self.plate_detector.detect_batch_quadrants(
                                    [c[3] for c in cand], conf=0.40)
                            else:
                                all_plates = [self.plate_detector.detect(c[3], conf=0.40, draw=False)
                                              for c in cand]
                        except Exception:
                            all_plates = []
                        stats["plate_ms"].append((time.time() - tp0) * 1000 / max(len(cand), 1))
                        for (track_id, track, vehicle, quadrant, (ox, oy, x1, y1)), plates in zip(cand, all_plates):
                            mapped = []
                            plate_found = False
                            for plate in (plates or [])[:1]:
                                px1, py1, px2, py2 = map(int, plate[:4])
                                if px2 <= px1 or py2 <= py1:
                                    continue
                                local = (px1 + ox, py1 + oy, px2 + ox, py2 + oy)
                                crop = vehicle[max(0, local[1]):min(vehicle.shape[0], local[3]), max(0, local[0]):min(vehicle.shape[1], local[2])]
                                if crop.size == 0:
                                    continue
                                crop_with_margin = self._plate_crop_with_margin(vehicle, local)
                                evidence_crop = crop_with_margin if crop_with_margin.size else crop
                                if not self._viable_plate_crop(evidence_crop):
                                    continue
                                plate_found = True
                                mapped.append((x1 + local[0], y1 + local[1], x1 + local[2], y1 + local[3]))
                                tq1 = time.time()
                                quality = self._quality(evidence_crop)
                                stats["quality_ms"].append((time.time() - tq1) * 1000)
                                crossing_frame = pending_crossings.get(track_id)
                                if crossing_frame is not None and track_id not in confirmed_at:
                                    confirmed_at[track_id] = fi
                                    track["infractor_confirmed"] = True
                                    track["pending_infractor"] = False
                                    if callback is not None:
                                        callback({
                                            "type": "infraction_detected",
                                            "track_id": track_id,
                                            "frame_index": fi,
                                            "timestamp_seconds": fi / fps,
                                            "vehicle_class": track["class_name"],
                                        })
                                confirmation_frame = confirmed_at.get(track_id)
                                is_valid_candidate = (
                                    confirmation_frame is not None
                                    and crossing_frame is not None
                                    and fi >= crossing_frame
                                )
                                if is_valid_candidate and quality >= best.get(track_id, PlateEvidence("", 0, 0, 0, "", -1)).quality_score:
                                    crop_path = output_dir / "crops" / f"{video_path.stem}_v{track_id}_best.jpg"
                                    _save_crop(crop_path, evidence_crop)
                                    best[track_id] = PlateEvidence(config.video_name, track_id, fi, fi / fps, track["class_name"], quality, str(crop_path))
                            if mapped and plate_found:
                                plate_boxes[track_id] = mapped

                    last_tracks = tracks
                    last_plate_boxes = plate_boxes
                else:
                    # Green is display-only: do not detect normal vehicles. Keep
                    # only already-confirmed infractors visually latched.
                    max_display_age = max(1, int(fps))
                    tracks = {
                        track_id: track
                        for track_id, track in last_tracks.items()
                        if (track.get("infractor_confirmed", False) or track.get("pending_infractor", False))
                        and fi - track.get("last_detection_frame", fi) <= max_display_age
                    }
                    plate_boxes = {
                        track_id: boxes
                        for track_id, boxes in last_plate_boxes.items()
                        if track_id in tracks
                    }

                td0 = time.time()
                display = self._draw(frame, polygon, tracks, state, plate_boxes, fi,
                                     self.draw_state_banner, time.time() - started,
                                     (config.green, config.yellow, config.red))
                stats["draw_ms"].append((time.time() - td0) * 1000)
                if writer is not None and should_display:
                    try:
                        writer.write(display)
                    except Exception:
                        pass
                if callback is not None and should_display:
                    callback({"type": "frame", "frame": display, "frame_index": fi, "total_frames": total, "state": state, "processed": should_detect})

        cap.release()
        # Fase 3: flush async (crops + video) antes de leerlos para el reporte.
        dropped = 0
        if threaded_writer is not None:
            try:
                dropped = threaded_writer.release()
            except Exception:
                pass
            writer = None
        elif writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if crop_writer is not None:
            try:
                crop_writer.shutdown()
            except Exception:
                pass
        evidence = [item.to_dict() for item in sorted(best.values(), key=lambda value: value.track_id)]
        pending_infractions = []
        for track_id in sorted(pending_crossings):
            if track_id in confirmed_at:
                continue
            crop_path = pending_paths.get(track_id, "")
            crop_w, crop_h = 0, 0
            if crop_path:
                try:
                    probe = cv2.imread(str(crop_path))
                    if probe is not None:
                        crop_h, crop_w = probe.shape[:2]
                except Exception:
                    pass
            pending_infractions.append({
                "vehicle_id": track_id,
                "frame_index": pending_crossings[track_id],
                "timestamp_seconds": round(pending_crossings[track_id] / fps, 3),
                "vehicle_class": "VEH",
                "crop_path": crop_path,
                "crop_w": crop_w,
                "crop_h": crop_h,
            })
        def _avg(xs: list) -> float:
            return round(sum(xs) / len(xs), 2) if xs else 0.0

        perf = {
            "frames": stats["frames"],
            "detect_frames": stats["detect_frames"],
            "batch_calls": stats["batch_calls"],
            "batch_size_cfg": B,
            "veh_ms_avg": _avg(stats["veh_ms"]),
            "plate_ms_avg": _avg(stats["plate_ms"]),
            "quality_ms_avg": _avg(stats["quality_ms"]),
            "draw_ms_avg": _avg(stats["draw_ms"]),
            "writer_dropped": dropped,
            "device": str(getattr(self.vehicle_detector, "device", "?")),
            "using_gpu": bool(getattr(self.vehicle_detector, "using_gpu", False)),
        }
        print(f"[perf] frames={perf['frames']} detect={perf['detect_frames']} "
              f"batch_cfg={B} calls={perf['batch_calls']} "
              f"veh={perf['veh_ms_avg']}ms plate={perf['plate_ms_avg']}ms "
              f"quality={perf['quality_ms_avg']}ms draw={perf['draw_ms_avg']}ms "
              f"dropped={dropped} dev={perf['device']}")
        payload = {
            "video": config.video_name,
            "video_path": str(video_path),
            "frames": frame_index,
            "fps": fps,
            "duration_seconds": frame_index / fps if fps else 0,
            "config": {"green": config.green, "yellow": config.yellow, "red": config.red, "pre_red_seconds": config.pre_red_seconds, "green_skip_rate": config.green_skip_rate, "danger_zone_margin_pixels": config.danger_zone_margin_pixels, "avenue": config.avenue},
            "evidence": evidence,
            "pending_infractions": pending_infractions,
            "infractor_count": len(confirmed_at) + len(pending_infractions),
            "confirmed_infractor_ids": sorted(confirmed_at),
            "elapsed_seconds": round(time.time() - started, 3),
            "perf": perf,
        }
        report_path = self.reports.save_processing(output_dir / f"{video_path.stem}_report.json", payload)
        if callback is not None:
            callback({"type": "complete", "payload": payload, "report_path": str(report_path), "output_path": str(output_path) if save_video else ""})
        return payload
