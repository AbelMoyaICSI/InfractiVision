"""Official red-light video processing use case.

Nuevo flujo:
  LIVE: solo YOLOv8-vehiculos. Todo vehiculo que cruza el poligono en rojo
  es infractor directo (rojo latched). No hay estados pending ni detector
  de placas en el loop: solo se guardan los crops de carro completo con
  mejor calidad por track.
  ACCION (post-proceso): sobre la mejor imagen de cada auto se aplica el
  cuadrante existente + localizacion de placa 1x y ese recorte se envia a
  la API de Plate Recognizer para obtener el texto (ver
  `src/application/services/plate_review_preparer.py`).
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
                 draw_state_banner: bool = True,
                 max_candidates: int = 5,
                 direction_min_px: float = 15.0):
        self.project_root = Path(project_root)
        self.vehicle_detector = vehicle_detector
        # Compat: se acepta pero YA NO se usa en live. La localizacion de
        # placa ocurre 1x por candidato en post-proceso
        # (`plate_review_preparer`), que carga su propio detector lazy.
        self.plate_detector = plate_detector
        self.reports = report_repository or ReportRepository()
        self.min_plate_crop_w = 55
        self.min_plate_crop_h = 30
        self.plate_crop_margin = 0.5
        # Legacy (sin efecto): antes limitaba a top-K crops por infractor.
        # Ahora se guardan TODAS las imagenes del infractor en disco.
        self.max_candidates = max(1, int(max_candidates))
        # Desplazamiento horizontal minimo en el history para decidir si el
        # carro va a la derecha o izquierda (cuadrante inferior der/izq).
        self.direction_min_px = float(direction_min_px)
        # Si es False, no se pinta el cartel "SEMAFORO: X" sobre el video
        # (el estado se muestra en el widget Semaforo de la GUI).
        self.draw_state_banner = draw_state_banner

    def _ensure_models(self):
        # Live = SOLO YOLOv8-vehiculos. El detector de placas NO se carga
        # aqui: el post-proceso lo pide bajo demanda (1x por infractor).
        if self.vehicle_detector is None:
            from src.core.detection.vehicle_detector import VehicleDetector
            self.vehicle_detector = VehicleDetector(str(self.project_root / "models" / "yolov8n.pt"))

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
            # i3+RTX5050: intentar siempre en GPU (sin umbral de tamaño); el
            # fallback CPU solo queda si CUDA falla o el crop es vacío.
            try:
                import torch

                if torch.cuda.is_available() and small.size > 0:
                    g = torch.from_numpy(
                        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    ).cuda(non_blocking=True).float()
                    contrast = min(float(g.std().item()) / 50.0, 1.0)
                    # Nitidez: varianza del Laplaciano con ops puntuales (sin
                    # conv2d: con cudnn.benchmark activo, cada forma nueva de
                    # crop disparaba autotune de cuDNN -> picos de 100-500ms
                    # por frame del infractor).
                    if min(g.shape) >= 3:
                        lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                               - g[1:-1, :-2] - g[1:-1, 2:])
                        sharpness = min(float(lap.var().item()) / 100.0, 1.0)
                    else:
                        sharpness = 0.0
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

    # ── Helpers de post-proceso (1x por infractor, fuera del live) ──
    # `_quadrant` + `_plate_crop_with_margin` + `_viable_plate_crop` los usa
    # `plate_review_preparer` sobre la mejor imagen guardada.
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
    def _direction_from_history(history, min_px: float = 15.0) -> str:
        """Direccion del carro desde el history del tracker (sin costo).

        `history` = ultimos centros (x, y). dx positivo = se mueve a la
        derecha => la placa trasera se busca en el cuadrante inferior
        derecho; dx negativo => izquierdo; bajo el umbral => "unknown"
        (cuadrante inferior completo).
        """
        try:
            if not history or len(history) < 2:
                return "unknown"
            dx = float(history[-1][0]) - float(history[0][0])
            if dx >= min_px:
                return "right"
            if dx <= -min_px:
                return "left"
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _draw(frame: np.ndarray, polygon: np.ndarray, tracks: dict, state: str,
              plate_boxes: dict[int, list[tuple[int, int, int, int]]], frame_index: int,
              draw_state_banner: bool = True, elapsed_seconds: float | None = None,
              durations: tuple[int, int, int] | None = None) -> np.ndarray:
        display = frame.copy()
        cv2.polylines(display, [polygon], True, (0, 0, 255), 2)
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            # Sin pending: el infractor queda en rojo latched al cruzar en
            # rojo. Salir del poligono no lo vuelve verde.
            infraction = track.get("infractor_confirmed", False)
            if infraction:
                color, thickness, state_label = (0, 0, 255), 3, "INFRACCION"
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
        """Fase 1+2+3: batch GPU + I/O async + stats.

        Nuevo flujo (sin YOLO-placas en live, sin pending):
        - YOLO-vehículos en batch + CentroidTracker + TrafficPlanner.
        - Cruce de polígono en rojo => infractor confirmado directo (rojo).
        - Solo se guardan crops de carro completo del mejor frame por track.
        - La placa se localiza 1x por infractor en post-proceso y el texto
          lo resuelve la API de Plate Recognizer en la revisión.
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
        # track_id -> frame del cruce (infractor directo, sin pending)
        infractors: dict[int, int] = {}
        # track_id -> mejor quality vista (gate de guardado por mejora:
        # solo el mejor crop va a disco, no todos los frames).
        best_quality: dict[int, float] = {}
        # track_id -> top-K candidatos {quality, frame, timestamp, direction, image}
        candidates: dict[int, list[dict]] = {}
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
                    # LIVE sin YOLO-placas: solo vehiculos. Cruce en rojo =>
                    # infractor directo (rojo). Se guarda el mejor crop de
                    # carro completo por track para el post-proceso.
                    for track_id, track in tracks.items():
                        bbox = track["bbox"]
                        inside, near = self._near_polygon(bbox, polygon, config.danger_zone_margin_pixels)
                        track["in_polygon"] = inside
                        track["near_zone"] = near
                        track["class_name"] = {2: "CAR", 5: "BUS", 7: "TRUCK"}.get(track["class_id"], "VEH")
                        track["last_detection_frame"] = fi
                        if inside and state == "red" and track_id not in infractors:
                            infractors[track_id] = fi
                            if callback is not None:
                                callback({
                                    "type": "infraction_detected",
                                    "track_id": track_id,
                                    "frame_index": fi,
                                    "timestamp_seconds": fi / fps,
                                    "vehicle_class": track["class_name"],
                                })
                        track["infractor_confirmed"] = track_id in infractors
                        if not near and not track["infractor_confirmed"]:
                            continue
                        if not track["infractor_confirmed"]:
                            continue
                        # Gate de zona: el mejor frame siempre está cerca del
                        # polígono. Confirmado-pero-lejos: display latched sin
                        # costo por frame (sin _quality ni imwrite).
                        if not near:
                            continue
                        x1, y1, x2, y2 = bbox
                        vehicle = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if vehicle.size == 0:
                            continue
                        tq0 = time.time()
                        quality = self._quality(vehicle)
                        stats["quality_ms"].append((time.time() - tq0) * 1000)
                        # Gate de mejora: solo el mejor crop va a disco (el
                        # primer frame near siempre supera el -1 inicial).
                        # Sin esto, cada frame del infractor pagaba JPEG +
                        # lista sin cota + N YOLOs extra en post-proceso.
                        if quality <= best_quality.get(track_id, -1.0):
                            continue
                        best_quality[track_id] = quality
                        # Solo mejoras van a disco de inmediato (sin
                        # YOLO-placas en live): el writer async absorbe el
                        # I/O y en RAM solo queda metadata liviana.
                        # El post-proceso elige la mejor CON placa.
                        direction = self._direction_from_history(
                            track.get("history"), self.direction_min_px)
                        cand_path = output_dir / "crops" / f"{video_path.stem}_v{track_id}_f{fi}.jpg"
                        _save_crop(cand_path, vehicle)
                        candidates.setdefault(track_id, []).append({
                            "path": str(cand_path),
                            "quality": quality,
                            "frame": fi,
                            "timestamp": fi / fps,
                            "direction": direction,
                            "class_name": track["class_name"],
                        })

                    last_tracks = tracks
                    last_plate_boxes = plate_boxes
                else:
                    # Green is display-only: do not detect normal vehicles. Keep
                    # only already-confirmed infractors visually latched.
                    max_display_age = max(1, int(fps))
                    tracks = {
                        track_id: track
                        for track_id, track in last_tracks.items()
                        if track.get("infractor_confirmed", False)
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
        # Arma la evidencia con TODOS los candidatos guardados: el
        # post-proceso revisa placa en CADA uno (cuadrante direccional)
        # y elige la mejor CON placa; fallback = mayor calidad.
        for track_id in sorted(candidates):
            ranking = sorted(candidates[track_id], key=lambda c: c["quality"], reverse=True)
            if not ranking:
                continue
            crops_meta: list[dict] = [
                {
                    "path": str(cand["path"]),
                    "quality": round(float(cand["quality"]), 4),
                    "frame": int(cand["frame"]),
                    "timestamp_seconds": round(float(cand["timestamp"]), 3),
                    "direction": cand.get("direction", "unknown"),
                }
                for cand in ranking
            ]
            top = ranking[0]
            best_path = output_dir / "crops" / f"{video_path.stem}_v{track_id}_best.jpg"
            best[track_id] = PlateEvidence(
                config.video_name, track_id, int(top["frame"]), float(top["timestamp"]),
                top.get("class_name", "VEH"), float(top["quality"]), str(best_path),
                review_notes="Carro completo (placa se localiza en post-proceso)",
                metadata={
                    "full_car": True,
                    "crossing_frame": infractors.get(track_id, int(top["frame"])),
                    "direction": top.get("direction", "unknown"),
                    "n_candidates": len(crops_meta),
                    "candidate_crops": crops_meta,
                },
            )
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
        # best.jpg = copia del mejor candidato (ya flusheado a disco).
        for track_id, item in best.items():
            try:
                cands = (item.metadata or {}).get("candidate_crops") or []
                if cands and save_crops:
                    import shutil as _shutil
                    _shutil.copyfile(cands[0]["path"], str(item.crop_path))
            except Exception:
                pass
        evidence = [item.to_dict() for item in sorted(best.values(), key=lambda value: value.track_id)]
        # Sin pending: todo infractor ya tiene su mejor crop de carro en
        # `evidence`. Se mantiene la clave vacia por compat con reportes
        # antiguos y callers externos.
        pending_infractions: list = []
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
              f"veh={perf['veh_ms_avg']}ms "
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
            "infractor_count": len(infractors),
            "confirmed_infractor_ids": sorted(infractors),
            "elapsed_seconds": round(time.time() - started, 3),
            "perf": perf,
        }
        report_path = self.reports.save_processing(output_dir / f"{video_path.stem}_report.json", payload)
        if callback is not None:
            callback({"type": "complete", "payload": payload, "report_path": str(report_path), "output_path": str(output_path) if save_video else ""})
        return payload
