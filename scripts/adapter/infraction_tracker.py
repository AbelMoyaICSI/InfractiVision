"""
InfractionTracker — pure-logic tracking, PPI, MMRP and trigger engine.

Extracted from preprocessing_dialog.py:1245-1434. No Tk, no display.
Each public method receives raw per-frame data and returns decisions.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


# ── PPI V50 (Proximity Proximity Index) ──────────────────────────────


def calculate_ppi(
    bumper_x: float, bumper_y: float, frame_h: int, frame_w: int
) -> float:
    """
    Vertical-primary proximity index (V50 logic).

    Y-factor: real closeness (lower in frame = closer to camera).
    X-factor: slight horizontal penalty, NOT a closeness indicator.
    """
    y_factor = max(0.0, (bumper_y - (frame_h * 0.35)) / (frame_h * 0.60))
    x_center_factor = 1.0 - abs(bumper_x - frame_w * 0.5) / (frame_w * 0.5) * 0.3
    return max(0.01, min(1.0, y_factor * x_center_factor))


def _point_in_polygon(px: float, py: float, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), True) >= 0


# ── InfractionTracker ─────────────────────────────────────────────────


class InfractionTracker:
    """Tracks candidate vehicles frame-to-frame and fires infraction triggers."""

    def __init__(self, track_dist_threshold: float = 140.0) -> None:
        self._active: dict[str, dict] = {}  # key "inf_N" → infractor state
        self._counter: int = 0
        self._dist_threshold = track_dist_threshold
        self._snapshots: list[dict] = []  # mmrp_frame per triggered infractor

    # ── read-back ───────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def total_triggers(self) -> int:
        return len(self._snapshots)

    def get_snapshots(self) -> list[dict]:
        return list(self._snapshots)

    def get_active_infractors(self) -> dict[str, dict]:
        return dict(self._active)

    # ── per-frame update ────────────────────────────────────────────

    def process_detection(
        self,
        vehicle_center: tuple[int, int],     # (bumper_x, bumper_y)
        vehicle_area: float,
        h_frame: int,
        w_frame: int,
        polygon: np.ndarray,
        proximity_factor: float,
        has_plate_score: bool,
        frame_img: np.ndarray,
        bbox: tuple[int, int, int, int],     # x1, y1, x2, y2
        v_left: tuple[int, int],             # (x1, y2)
        v_right: tuple[int, int],            # (x2, y2)
        frame_index: int,
        plate_detector=None,
    ) -> Optional[dict]:
        """
        Process one vehicle detection.

        Returns:
            None  — no trigger this frame.
            dict  — trigger fired:
                track_id, snapshot, proximity_factor, num_frames, trigger_type
        """
        x1, y1, x2, y2 = bbox
        bx, by = vehicle_center

        # ── ASSOCIATE with existing tracker ─────────────────────────
        current_d = None
        is_new = True
        for _inf_id, data in self._active.items():
            last_center = data["center"]
            dist = math.hypot(bx - last_center[0], by - last_center[1])
            if dist < self._dist_threshold:
                current_d = data
                is_new = False
                break

        # ── 3-vertex polygon containment (V50) ──────────────────────
        in_polygon = (
            _point_in_polygon(bx, by, polygon)
            or _point_in_polygon(v_left[0], v_left[1], polygon)
            or _point_in_polygon(v_right[0], v_right[1], polygon)
        )

        # ── NEW infractor ───────────────────────────────────────────
        if (
            is_new
            and in_polygon
            and proximity_factor > 0.40
            and vehicle_area > 25000
        ):
            self._counter += 1
            inf_id = f"inf_{self._counter}"
            current_d = {
                "id": self._counter,
                "center": vehicle_center,
                "start_y": vehicle_center[1],
                "area_history": [],
                "mmrp_reached": False,
                "mmrp_frame": None,
                "best_pqi": -1.0,
                "async_sent": False,
            }
            self._active[inf_id] = current_d

        if current_d is None:
            return None  # not tracked, skip

        # ── UPDATE tracked state ────────────────────────────────────
        current_d["center"] = vehicle_center
        current_d["area_history"].append(vehicle_area)

        h, w = h_frame, w_frame  # shadows for crop math

        # ── PLATE QUICK CHECK score ─────────────────────────────────
        plate_score = 1.0 if has_plate_score else 0.0
        pqi = proximity_factor * (plate_score if plate_score > 0.1 else 0.03)

        # ── UPDATE best snapshot (LabForense V50) ───────────────────
        if pqi > current_d["best_pqi"]:
            current_d["best_pqi"] = pqi
            plate_stripped = None
            vehicle_ctx = None

            if plate_detector is not None:
                try:
                    tm_ctx = max(30, int(min(x2 - x1, y2 - y1) * 0.20))
                    ry1 = max(0, y1 - tm_ctx)
                    ry2 = min(h, y2 + tm_ctx)
                    rx1 = max(0, x1 - tm_ctx)
                    rx2 = min(w, x2 + tm_ctx)
                    vehicle_ctx = frame_img[ry1:ry2, rx1:rx2].copy()

                    p_det = plate_detector.detect_plates(vehicle_ctx, confidence=0.40)
                    if p_det:
                        px1, py1, px2, py2 = [int(v) for v in p_det[0]]
                        p_raw = vehicle_ctx[py1:py2, px1:px2].copy()
                        from src.core.processing.plate_processing import (
                            rectificar_perspectiva,
                        )

                        plate_stripped = rectificar_perspectiva(p_raw)
                except Exception:
                    pass

            bbox_margin = max(15, int(min(x2 - x1, y2 - y1) * 0.10))
            current_d["mmrp_frame"] = {
                "img": frame_img.copy(),
                "bbox": (
                    max(0, x1 - bbox_margin),
                    max(0, y1 - bbox_margin),
                    min(w, x2 + bbox_margin),
                    min(h, y2 + bbox_margin),
                ),
                "f": frame_index,
                "plate_stripped": plate_stripped,
                "vehicle_context": vehicle_ctx,
            }

        # ── MMRP peak detection ─────────────────────────────────────
        hist = current_d["area_history"]
        if not current_d["mmrp_reached"] and len(hist) >= 6:
            recent = hist[-5:]
            if sum(recent[-3:]) / 3 < (sum(recent[:3]) / 3) * 0.98:
                current_d["mmrp_reached"] = True

        # ── V50 TRIGGER ─────────────────────────────────────────────
        if not current_d["async_sent"]:
            num_f = len(hist)
            is_panic = proximity_factor >= 0.88 and plate_score > 0
            is_secure = num_f >= 3 and proximity_factor >= 0.85 and plate_score > 0
            is_peak_gold = (
                num_f >= 5
                and current_d["mmrp_reached"]
                and proximity_factor >= 0.78
            )
            is_heavy = num_f >= 22 and proximity_factor >= 0.75

            ready = is_panic or is_secure or is_peak_gold or is_heavy
            if not in_polygon and proximity_factor < 0.40:
                ready = False

            if ready:
                current_d["async_sent"] = True
                snap = (
                    current_d["mmrp_frame"]
                    if current_d["mmrp_frame"]
                    else {
                        "img": frame_img.copy(),
                        "bbox": bbox,
                        "f": frame_index,
                        "plate_stripped": None,
                        "vehicle_context": None,
                    }
                )
                trigger_type = (
                    "PEAK"
                    if is_peak_gold
                    else "PANIC"
                    if is_panic
                    else "PERSIST"
                )
                snap["track_id"] = current_d["id"]
                snap["trigger_type"] = trigger_type
                self._snapshots.append(snap)
                return {
                    "track_id": current_d["id"],
                    "snapshot": snap,
                    "proximity_factor": proximity_factor,
                    "num_frames": num_f,
                    "trigger_type": trigger_type,
                }

        return None
