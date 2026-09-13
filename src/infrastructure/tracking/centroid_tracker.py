"""Tracker centroide/IoU ligero para vivo (i3-9100F).

Evita la CNN ReID de DeepSORT (1 embedding por detección en CPU) cuando
`INFRACTI_TRACKER=centroid` (default en vivo). Matching vectorizado con
numpy: O(T) por detección sin doble loop Python puro ni copias de historial.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.domain.entities import Vehicle


class CentroidTracker:
    """Asigna `track_id` por distancia euclídea al centroide previo."""

    def __init__(self, max_dist: float = 80.0, max_age: int = 30):
        self._max_dist2 = float(max_dist) ** 2
        self._max_age = int(max_age)
        self._next_id = 1
        # tid -> (cx, cy, last_seen_frame)
        self._tracks: dict[int, tuple[float, float, int]] = {}
        self._frame = 0

    def update(
        self, frame_bgr: np.ndarray, detections: Sequence[Vehicle]
    ) -> Sequence[Vehicle]:
        self._frame += 1
        if not detections:
            self._purge()
            return []

        centers = np.array([v.bbox.center for v in detections], dtype=np.float64)
        tids = list(self._tracks.keys())
        if tids:
            prev = np.array(
                [(self._tracks[t][0], self._tracks[t][1]) for t in tids],
                dtype=np.float64,
            )
            # Dist² detecciones x tracks, vectorizado.
            d2 = (
                (centers[:, None, 0] - prev[None, :, 0]) ** 2
                + (centers[:, None, 1] - prev[None, :, 1]) ** 2
            )
        else:
            d2 = np.empty((len(detections), 0), dtype=np.float64)

        used: set[int] = set()
        out: list[Vehicle] = []
        for i, v in enumerate(detections):
            best_id = None
            if d2.shape[1]:
                order = np.argsort(d2[i], kind="stable")
                for j in order:
                    tid = tids[int(j)]
                    if tid in used:
                        continue
                    if d2[i, int(j)] < self._max_dist2:
                        best_id = tid
                    break
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            cx, cy = centers[i]
            self._tracks[best_id] = (float(cx), float(cy), self._frame)
            used.add(best_id)
            v.track_id = best_id
            out.append(v)
        self._purge()
        return out

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame = 0

    def _purge(self) -> None:
        stale = [
            tid
            for tid, (_, _, seen) in self._tracks.items()
            if self._frame - seen > self._max_age
        ]
        for tid in stale:
            del self._tracks[tid]
