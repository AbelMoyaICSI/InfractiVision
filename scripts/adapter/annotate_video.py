#!/usr/bin/env python
"""
annotate_video.py — Generate annotated videos with vehicle detection overlays.

Reads ground truth from tests/verdad.test.json (polygon, semaphore config)
and produces videos with bounding boxes matching the UI reproduction mode:
  - Green box  → normal vehicle
  - Red box    → infraction (bumper inside polygon + red light)
  - X marks    → plate detection points (before homography)
  - Polygon ROI overlay
  - Semaphore state banner

Usage:
    python scripts/adapter/annotate_video.py --video "VID4EDIT ‐ Hecho con Clipchamp.mp4"
    python scripts/adapter/annotate_video.py --all
    python scripts/adapter/annotate_video.py --video "VID2COLISEO.MOV" --speed 60
    python scripts/adapter/annotate_video.py --video "VID4EDIT ‐ Hecho con Clipchamp.mp4" --save-crops
    python scripts/adapter/annotate_video.py --video "VID2COLISEO.MOV" --save-crops --stack
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ocr.super_resolution import upscale_plate

VIDEOS_DIR = PROJECT_ROOT / "videos"
VERDAD_JSON = PROJECT_ROOT / "tests" / "verdad.test.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "video_track" / "output"
DEFAULT_CROPS_DIR = PROJECT_ROOT / "data" / "output" / "crops"

CODEC = "mp4v"

# ── Colors (BGR) ─────────────────────────────────────────────────────

BOX_NORMAL = (0, 255, 0)        # green
BOX_INFRACCION = (0, 0, 255)    # red
PLATE_MARKER = (255, 0, 255)    # magenta (X on plate points)
TEXT_WHITE = (255, 255, 255)
TEXT_BLACK = (0, 0, 0)
POLYGON_COLOR = (0, 0, 255)     # red

SEMAPHORE_COLORS = {
    "red":    ((0, 0, 255), (255, 255, 255)),   # text=red,    bg=white
    "yellow": ((0, 255, 255), (0, 0, 0)),       # text=yellow, bg=black
    "green":  ((0, 255, 0), (0, 0, 0)),         # text=green,  bg=black
}

VEHICLE_CLASSES = {2: "CAR", 5: "BUS", 7: "TRUCK"}

PLATE_YELLOW = (0, 255, 255)  # yellow for plate bbox on crops
PLATE_ZONE_COLOR = (200, 200, 0)  # cyan for plate detection zone


# ── Plate quality scoring ─────────────────────────────────────────────


def score_plate_quality(plate_crop: np.ndarray) -> float:
    """Evaluate plate image quality (0.0 - 1.0) based on 4 factors."""
    if plate_crop is None or plate_crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Contrast
        contrast = min(gray.std() / 50.0, 1.0)

        # Edge density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        edge_score = min(edge_density * 10, 1.0)

        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 100.0, 1.0)

        # Size (minimum 1500 px^2 for OCR)
        h, w = plate_crop.shape[:2]
        size = min((w * h) / 1500.0, 1.0)

        return contrast * 0.3 + edge_score * 0.3 + sharpness * 0.25 + size * 0.15
    except Exception:
        return 0.0


# ── Frame stacking composite ─────────────────────────────────────────


def create_plate_composite(plate_crops: list[np.ndarray]) -> np.ndarray:
    """Create a 2x2 quadrant grid from 2-4 plate crops.

    Layout adaptivo:
        4 crops → grid completo 2x2
        3 crops → top-left + top-right + bottom-center
        2 crops → fila horizontal (izq + der)
        1 crop  → solo la imagen (sin composite)

    El resultado final se redimensiona a (94, 24) para LPRNet.
    """
    if not plate_crops:
        return np.zeros((24, 94, 3), dtype=np.uint8)

    # Target half-size for each quadrant
    qw, qh = 47, 12

    def _resize(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((qh, qw, 3), dtype=np.uint8)
        return cv2.resize(img, (qw, qh), interpolation=cv2.INTER_CUBIC)

    # Resize all crops to quadrant size
    crops = [_resize(c) for c in plate_crops]

    if len(crops) >= 4:
        # 2x2 grid: TL=0, TR=1, BL=2, BR=3
        top_row = np.hstack([crops[0], crops[1]])
        bot_row = np.hstack([crops[2], crops[3]])
        composite = np.vstack([top_row, bot_row])
    elif len(crops) == 3:
        # L-shape: TL=0, TR=1, bottom-center=2
        top_row = np.hstack([crops[0], crops[1]])
        # Center bottom crop
        bot_canvas = np.zeros((qh, qw * 2, 3), dtype=np.uint8)
        x_off = (qw * 2 - qw) // 2
        bot_canvas[:, x_off:x_off + qw] = crops[2]
        composite = np.vstack([top_row, bot_canvas])
    elif len(crops) == 2:
        # Horizontal row
        composite = np.hstack([crops[0], crops[1]])
    else:
        composite = crops[0]

    # Final resize to 94x24 for LPRNet
    composite = cv2.resize(composite, (94, 24), interpolation=cv2.INTER_CUBIC)
    return composite


# ── Skew detection + rotation ────────────────────────────────────────


def calculate_skew_angle(plate_region: np.ndarray, max_angle: float = 15.0) -> float:
    """Detect plate inclination angle. Returns degrees (-45..45) or 0 if undetectable."""
    if plate_region is None or plate_region.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 5:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > max_angle:
            return 0.0
        return angle
    except Exception:
        return 0.0


def rotate_around_center(image: np.ndarray, angle: float,
                         center: tuple[int, int]) -> np.ndarray:
    """Rotate full image around a point. BORDER_REPLICATE avoids black edges."""
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def autocrop_plate_standalone(vehicle_crop: np.ndarray,
                              plate_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Detect skew -> rotate vehicle -> crop plate region directly.
    plate_bbox = (x1, y1, x2, y2) in vehicle_crop coords.
    Returns the plate region only, no artificial padding or borders.
    """
    px1, py1, px2, py2 = plate_bbox
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return vehicle_crop

    # 1. Skew angle from plate region
    plate_region = vehicle_crop[py1:py2, px1:px2]
    angle = calculate_skew_angle(plate_region)

    # 2. Rotate full vehicle around plate center
    plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
    rotated = rotate_around_center(vehicle_crop, angle, plate_center)

    # 3. Crop plate directly from rotated image (no borders)
    return rotated[py1:py2, px1:px2].copy()


def get_plate_quadrant(vehicle_crop: np.ndarray, direction: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Recort zona probable de placa según dirección del vehículo.
    Retorna (quadrant_image, (offset_x, offset_y))."""
    h, w = vehicle_crop.shape[:2]

    if direction == "right":
        return vehicle_crop[h // 2:, w // 2:], (w // 2, h // 2)
    elif direction == "left":
        return vehicle_crop[h // 2:, :w // 2], (0, h // 2)
    else:
        return vehicle_crop[h // 2:, :], (0, h // 2)


# ── Ground truth ──────────────────────────────────────────────────────


def load_ground_truth() -> list[dict]:
    with open(VERDAD_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos_verdad", [])


def find_entry(entries: list[dict], video_name: str) -> dict | None:
    for e in entries:
        if e["path_name"] == video_name:
            return e
    return None


# ── Tracking ──────────────────────────────────────────────────────────


class VehicleTracker:
    """Center-distance tracker with direction history for plate quadrant detection."""

    def __init__(self, tolerance: float = 80.0, max_lost: int = 6) -> None:
        self.tolerance = tolerance
        self.max_lost = max_lost
        self.vehicles: dict[int, dict] = {}
        self._counter = 0
        self._frame = 0
        self.max_history = 5

    def update(self, detections: list[tuple], polygon: np.ndarray | None) -> dict:
        """Match detections to tracked vehicles, return current tracks."""
        self._frame += 1
        current: dict[int, dict] = {}

        for det in detections:
            x1, y1, x2, y2, cls_id = det
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Find closest existing vehicle
            vehicle_id = None
            min_dist = float("inf")
            for vid, vdata in self.vehicles.items():
                ex = vdata["center"]
                dist = ((center_x - ex[0]) ** 2 + (center_y - ex[1]) ** 2) ** 0.5
                if dist < self.tolerance and dist < min_dist:
                    vehicle_id = vid
                    min_dist = dist

            if vehicle_id is None:
                vehicle_id = self._counter
                self._counter += 1

            # Center history for direction detection
            prev_data = self.vehicles.get(vehicle_id, {})
            history = prev_data.get("center_history", [])
            history.append((center_x, center_y))
            if len(history) > self.max_history:
                history.pop(0)

            # Check if bumper is inside polygon
            front_bumper_x = center_x
            front_bumper_y = y2 - 10
            in_polygon = False
            if polygon is not None and len(polygon) >= 3:
                in_polygon = (
                    cv2.pointPolygonTest(polygon, (float(front_bumper_x), float(front_bumper_y)), False) >= 0
                )

            current[vehicle_id] = {
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "center_history": history,
                "cls_id": cls_id,
                "in_polygon": in_polygon,
                "last_seen": self._frame,
            }

        # Keep recently seen vehicles
        for vid, vdata in list(self.vehicles.items()):
            if (self._frame - vdata["last_seen"]) <= self.max_lost:
                if vid not in current:
                    current[vid] = vdata

        self.vehicles = current
        return current

    def get_direction(self, vid: int) -> str:
        """Return movement direction: 'right', 'left', 'down', or 'unknown'."""
        history = self.vehicles.get(vid, {}).get("center_history", [])
        if len(history) < 2:
            return "unknown"

        first = history[0]
        last = history[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]

        threshold = 15
        if abs(dx) > abs(dy) and abs(dx) > threshold:
            return "right" if dx > 0 else "left"
        elif abs(dy) > threshold:
            return "down" if dy > 0 else "up"
        return "unknown"


# ── Drawing ───────────────────────────────────────────────────────────


def draw_annotations(
    frame: np.ndarray,
    tracked: dict,
    polygon: np.ndarray | None,
    sem_state: str,
    frame_idx: int,
    total_frames: int,
    plate_points: dict[int, list[tuple]] | None = None,
) -> np.ndarray:
    """Draw all overlays on the frame (matching UI reproduction mode)."""
    display = frame.copy()

    # Polygon ROI
    if polygon is not None and len(polygon) >= 3:
        cv2.polylines(display, [polygon], True, POLYGON_COLOR, 2)

    # Vehicle boxes
    for _vid, vdata in tracked.items():
        x1, y1, x2, y2 = vdata["bbox"]
        cls_id = vdata["cls_id"]
        in_polygon = vdata["in_polygon"]

        if in_polygon and sem_state == "red":
            box_color = BOX_INFRACCION
            label_text = "INFRACCION"
            text_color = TEXT_WHITE
            # Bumper indicator
            front_x, front_y = (x1 + x2) // 2, y2 - 10
            cv2.circle(display, (front_x, front_y), 8, BOX_INFRACCION, -1)
            thickness = 3
        else:
            box_color = BOX_NORMAL
            label_text = "NORMAL"
            text_color = TEXT_BLACK
            thickness = 2

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)

        # Vehicle type label
        vehicle_label = VEHICLE_CLASSES.get(cls_id, "VEH")
        cv2.putText(display, vehicle_label, (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Detection state label
        cv2.putText(display, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Draw X marks on plate detection points
        if plate_points and _vid in plate_points:
            for px1, py1, px2, py2 in plate_points[_vid]:
                corners = [
                    (px1, py1),  # top-left
                    (px2, py1),  # top-right
                    (px2, py2),  # bottom-right
                    (px1, py2),  # bottom-left
                ]
                for cx, cy in corners:
                    cv2.line(display, (cx - 6, cy - 6), (cx + 6, cy + 6), PLATE_MARKER, 2)
                    cv2.line(display, (cx + 6, cy - 6), (cx - 6, cy + 6), PLATE_MARKER, 2)

    # Semaphore banner
    text_color, bg_color = SEMAPHORE_COLORS.get(sem_state, ((255, 255, 255), (0, 0, 0)))
    sem_text = f" SEMAFORO: {sem_state.upper()} "
    text_size = cv2.getTextSize(sem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
    cv2.rectangle(display, (5, 5), (text_size[0] + 20, 40), bg_color, -1)
    cv2.putText(display, sem_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 3)

    # Frame info
    info_text = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(display, info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE, 2)

    return display


# ── Processing ────────────────────────────────────────────────────────


def process_video(
    entry: dict,
    output_dir: Path,
    speed: int | None = None,
    conf: float = 0.40,
    save_crops: bool = False,
    crops_dir: Path | None = None,
    best_only: bool = False,
    crops_only: bool = False,
    use_stack: bool = False,
    border_crop: bool = False,
) -> None:
    """Process a single video entry and write annotated output."""
    from src.core.detection.vehicle_detector import VehicleDetector
    from src.core.detection.plate_detector import PlateDetector
    from scripts.adapter.video_semaphore import VideoSemaphore

    video_name = entry["path_name"]
    video_path = VIDEOS_DIR / video_name
    polygon_points = entry.get("polygon", [])
    video_stem = Path(video_name).stem

    # Best-only tracking: {vid: (score, plate_img, direction)}
    best_by_vehicle: dict[int, tuple[float, np.ndarray, str]] = {} if best_only else None

    # Stack tracking: {vid: [{'score': S, 'img': I, 'frame': F}, ...]}
    stack_candidates: dict[int, list[dict]] = {} if use_stack else None

    if not video_path.exists():
        print(f"  SKIP: archivo no encontrado - {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  SKIP: no se pudo abrir - {video_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path (skip if crops_only)
    stem = Path(video_name).stem
    out = None
    if not crops_only:
        output_path = output_dir / f"{stem}_infra.mp4"
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Semaphore
    sem_config = {
        "green": entry.get("green", 30),
        "yellow": entry.get("yellow", 5),
        "red": entry.get("red", 40),
    }
    semaphore = VideoSemaphore(sem_config)

    # Polygon
    polygon = np.array(polygon_points, np.int32) if polygon_points and len(polygon_points) >= 3 else None

    # Detectors
    detector = VehicleDetector(str(PROJECT_ROOT / "models" / "yolov8n.pt"))
    plate_detector = PlateDetector() if save_crops else None

    # Crops dir
    if save_crops:
        if crops_dir is None:
            crops_dir = DEFAULT_CROPS_DIR
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Tracker
    tracker = VehicleTracker()

    # Process
    frame_idx = 0
    written = 0
    crops_saved = 0
    t_start = time.time()

    print(f"\n  Procesando: {video_name}")
    print(f"    FPS: {fps:.1f} | Frames: {total_frames} | Res: {width}x{height}")
    print(f"    Semaforo: green={sem_config['green']}s yellow={sem_config['yellow']}s red={sem_config['red']}s")
    if save_crops:
        print(f"    Crops: {crops_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Semaphore state (calculated BEFORE skip for adaptive speed)
        video_second = frame_idx / fps
        sem_state = semaphore.get_state(video_second)

        # Adaptive speed: skip only in green/yellow, process ALL frames in red
        if speed and sem_state != "red" and frame_idx % speed != 0:
            continue

        # Detect
        detections = detector.detect(frame, conf=conf, draw=False)

        # Track
        tracked = tracker.update(detections, polygon)

        # Plate detection + crop export for infractors
        plate_points: dict[int, list[tuple]] = {}
        for vid, vdata in tracked.items():
            x1, y1, x2, y2 = vdata["bbox"]
            in_polygon = vdata["in_polygon"]

            if not (in_polygon and sem_state == "red"):
                continue

            # Crop vehicle from frame
            vy1 = max(0, y1)
            vy2 = min(height, y2)
            vx1 = max(0, x1)
            vx2 = min(width, x2)
            vehicle_crop = frame[vy1:vy2, vx1:vx2]

            if vehicle_crop.size == 0:
                continue

            # Detect plates on dynamic quadrant (based on vehicle direction)
            direction = tracker.get_direction(vid)
            plate_dets = []
            if plate_detector is not None:
                try:
                    plate_quadrant, (q_off_x, q_off_y) = get_plate_quadrant(vehicle_crop, direction)
                    plate_dets_raw = plate_detector.detect_plates(plate_quadrant, confidence=0.40)
                    # Map coords back to vehicle_crop space
                    plate_dets = [
                        (px1 + q_off_x, py1 + q_off_y, px2 + q_off_x, py2 + q_off_y)
                        for px1, py1, px2, py2 in plate_dets_raw
                    ]
                except Exception:
                    plate_dets = []

                # Map plate coords back to frame space
                frame_plates = []
                for px1, py1, px2, py2 in plate_dets:
                    fx1 = vx1 + px1
                    fy1 = vy1 + py1
                    fx2 = vx1 + px2
                    fy2 = vy1 + py2
                    frame_plates.append((fx1, fy1, fx2, fy2))

                if frame_plates:
                    plate_points[vid] = frame_plates

            # Export plate crop (rotated + centered)
            if save_crops and crops_dir is not None:
                plate_score = 0.0
                plate_bbox_local = None
                plate_processed = None
                if plate_dets:
                    px1, py1, px2, py2 = [int(v) for v in plate_dets[0]]
                    plate_bbox_local = (px1, py1, px2, py2)
                    plate_processed = autocrop_plate_standalone(
                        vehicle_crop, plate_bbox_local
                    )
                    plate_score = score_plate_quality(plate_processed)
                    if plate_processed is not None:
                        plate_processed = upscale_plate(plate_processed, min_width=0)
                        if border_crop:
                            h, w = plate_processed.shape[:2]
                            bx = max(1, int(w * 0.05))
                            by = max(1, int(h * 0.05))
                            if w > bx * 2 + 2 and h > by * 2 + 2:
                                plate_processed = plate_processed[by:h - by, bx:w - bx]
                        plate_processed = cv2.resize(plate_processed, (94, 24), interpolation=cv2.INTER_LANCZOS4)

                if use_stack and stack_candidates is not None:
                    # Stack mode: accumulate top candidates per vehicle
                    if plate_processed is not None and plate_score >= 0.15:
                        if vid not in stack_candidates:
                            stack_candidates[vid] = []
                        stack_candidates[vid].append({
                            'score': plate_score,
                            'img': plate_processed,
                            'frame': frame_idx,
                        })
                        # Keep only top 4 by score
                        stack_candidates[vid] = sorted(
                            stack_candidates[vid],
                            key=lambda x: x['score'],
                            reverse=True,
                        )[:4]
                elif best_only:
                    if vid not in best_by_vehicle or plate_score > best_by_vehicle[vid][0]:
                        best_by_vehicle[vid] = (plate_score, plate_processed, direction)
                else:
                    if plate_processed is not None:
                        crop_name = f"{video_stem}_f{frame_idx}_v{vid}.jpg"
                        crop_path = crops_dir / crop_name
                        try:
                            cv2.imwrite(str(crop_path), plate_processed)
                            crops_saved += 1
                        except Exception:
                            pass

        # Annotate (with plate X marks) — skip if crops_only
        if not crops_only:
            annotated = draw_annotations(
                frame, tracked, polygon, sem_state, frame_idx, total_frames,
                plate_points=plate_points,
            )
            out.write(annotated)
            written += 1

        # Progress
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"    [{pct:5.1f}%] frame {frame_idx}/{total_frames} - {elapsed:.1f}s")

    cap.release()
    if out is not None:
        out.release()

    elapsed = time.time() - t_start
    if crops_only:
        print(f"    Listo: {frame_idx} frames procesados ({elapsed:.1f}s)")
    else:
        print(f"    Listo: {written} frames -> {output_path.name} ({elapsed:.1f}s)")

    # Save best crops per vehicle
    if best_only and best_by_vehicle is not None:
        saved_best = 0
        for vid, (score, plate_img, direction) in best_by_vehicle.items():
            if score < 0.3 or plate_img is None:
                continue
            crop_name = f"{video_stem}_v{vid}_best.jpg"
            crop_path = crops_dir / crop_name
            try:
                cv2.imwrite(str(crop_path), plate_img)
                saved_best += 1
            except Exception:
                pass
        print(f"    Best crops: {saved_best} imagenes guardadas en {crops_dir} (score >= 0.3)")
    elif use_stack and stack_candidates is not None:
        # Stack mode: save composites + individual top crops
        saved_stacks = 0
        for vid, candidates in stack_candidates.items():
            if not candidates:
                continue

            # Save individual top crops (ranked by score)
            for rank, cand in enumerate(candidates, 1):
                crop_name = f"{video_stem}_v{vid}_top{rank}.jpg"
                crop_path = crops_dir / crop_name
                try:
                    cv2.imwrite(str(crop_path), cand['img'])
                except Exception:
                    pass

            # Create and save composite 2x2
            composite = create_plate_composite([c['img'] for c in candidates])
            composite_name = f"{video_stem}_v{vid}_stack.jpg"
            composite_path = crops_dir / composite_name
            try:
                cv2.imwrite(str(composite_path), composite)
                saved_stacks += 1
            except Exception:
                pass

        print(f"    Stack crops: {saved_stacks} composites guardados en {crops_dir}")
        print(f"    Top-4 crops individuales también guardados")
    elif save_crops:
        print(f"    Crops: {crops_saved} imagenes guardadas en {crops_dir}")


# ── CLI ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InfractiVision — Generar videos anotados con detección de infracciones"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Nombre del video (path_name del JSON ground truth)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Procesar TODOS los videos del JSON ground truth",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directorio de salida (default: video_track/output/)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help="Procesar 1 de cada N frames (skip fijo)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Umbral de confianza YOLO (default: 0.40)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Exportar crops de infracciones a data/output/crops/",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Guardar solo la mejor crop por vehículo (score >= 0.3 de calidad de placa)",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Directorio de salida para crops (default: data/output/crops/)",
    )
    parser.add_argument(
        "--crops-only",
        action="store_true",
        help="Solo generar crops, no escribir video anotado (mucho más rápido)",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Frame stacking: guardar top-4 crops por vehículo y crear composite 2x2",
    )
    parser.add_argument(
        "--border",
        action="store_true",
        help="Recortar 5%% de los bordes del crop de placa para eliminar márgenes",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Se requiere --video o --all")

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_ground_truth()
    if not entries:
        print("No se encontraron videos en verdad.test.json")
        sys.exit(1)

    if args.all:
        targets = entries
    else:
        entry = find_entry(entries, args.video)
        if entry is None:
            print(f"Video no encontrado en verdad.test.json: {args.video}")
            print("Videos disponibles:")
            for e in entries:
                print(f"  - {e['path_name']}")
            sys.exit(1)
        targets = [entry]

    print(f"Annotate Video - {len(targets)} video(s) -> {output_dir}")

    crops_dir = Path(args.crops_dir) if args.crops_dir else DEFAULT_CROPS_DIR

    # --best-only and --crops-only imply --save-crops
    save_crops = args.save_crops or args.best_only or args.crops_only or args.stack

    for i, entry in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {entry['path_name']}")
        process_video(
            entry, output_dir,
            speed=args.speed, conf=args.conf,
            save_crops=save_crops, crops_dir=crops_dir,
            best_only=args.best_only,
            crops_only=args.crops_only,
            use_stack=args.stack,
            border_crop=args.border,
        )

    print(f"\nListo. Videos anotados en: {output_dir}")


if __name__ == "__main__":
    main()
