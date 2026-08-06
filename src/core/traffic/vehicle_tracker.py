"""Small deterministic tracker used by the official video processor."""
from __future__ import annotations

import math


class CentroidVehicleTracker:
    def __init__(self, tolerance: float = 100.0, max_lost: int = 8):
        self.tolerance = tolerance
        self.max_lost = max_lost
        self._next_id = 0
        self._frame = 0
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[tuple[int, int, int, int, int, float]]) -> dict[int, dict]:
        self._frame += 1
        current = {}
        used: set[int] = set()
        for x1, y1, x2, y2, cls_id, confidence in detections:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            candidate = None
            distance = float("inf")
            for track_id, previous in self._tracks.items():
                if track_id in used:
                    continue
                d = math.dist(center, previous["center"])
                if d < self.tolerance and d < distance:
                    candidate, distance = track_id, d
            if candidate is None:
                candidate = self._next_id
                self._next_id += 1
                history = []
            else:
                history = list(self._tracks[candidate].get("history", []))
            history.append(center)
            history = history[-8:]
            current[candidate] = {
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "history": history,
                "class_id": cls_id,
                "confidence": confidence,
                "last_seen": self._frame,
            }
            used.add(candidate)
        for track_id, previous in self._tracks.items():
            if track_id not in current and self._frame - previous["last_seen"] <= self.max_lost:
                current[track_id] = previous
        self._tracks = current
        return current
