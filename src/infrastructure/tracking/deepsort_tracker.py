"""Adapter DeepSORT que implementa `TrackerPort`.

Usa `deep_sort_realtime` si está instalado; si no, recurre a un tracker
centroide simple para no bloquear el sistema. Esto se mantiene como
mecanismo de **multiplataforma**: en Linux/Windows funciona igual.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.exceptions import TrackerError
from src.core.logger import get_logger
from src.domain.entities import BoundingBox, Vehicle
from src.domain.interfaces import TrackerPort

log = get_logger("infra.tracker")


class DeepSortTracker(TrackerPort):
    def __init__(self, max_age: int = 30):
        self._impl = self._build(max_age)
        self._fallback_next_id = 1
        self._fallback_state: dict[int, tuple[float, float]] = {}

    # ─── Construcción con fallback ────────────────────────────────────────
    def _build(self, max_age: int):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore

            log.info("DeepSORT (deep_sort_realtime) cargado, max_age=%d", max_age)
            try:
                return DeepSort(max_age=max_age, embedder_gpu=False)
            except TypeError:
                return DeepSort(max_age=max_age)
        except Exception as e:  # pragma: no cover - fallback en runtime
            log.warning("deep_sort_realtime no disponible (%s). Usando centroide.", e)
            return None

    # ─── Tracking ─────────────────────────────────────────────────────────
    def update(
        self, frame_bgr: np.ndarray, detections: Sequence[Vehicle]
    ) -> Sequence[Vehicle]:
        if self._impl is None:
            return self._update_centroid(detections)

        try:
            ds_input = []
            for v in detections:
                x1, y1, x2, y2 = v.bbox.as_tuple()
                ds_input.append(([x1, y1, x2 - x1, y2 - y1], v.confidence, str(v.class_id)))
            tracks = self._impl.update_tracks(ds_input, frame=frame_bgr)
        except Exception as e:
            log.warning("DeepSORT update falló (%s). Volviendo a tracker centroide.", e)
            self._impl = None
            return self._update_centroid(detections)

        out: list[Vehicle] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = map(int, t.to_ltrb())
            try:
                cls_id = int(t.det_class) if t.det_class is not None else 2
            except Exception:
                cls_id = 2
            out.append(
                Vehicle(
                    bbox=BoundingBox(l, t_, r, b),
                    class_id=cls_id,
                    confidence=float(getattr(t, "det_conf", 0.0) or 0.0),
                    track_id=int(t.track_id),
                )
            )
        return out

    def reset(self) -> None:
        self._fallback_state.clear()
        self._fallback_next_id = 1
        if self._impl is not None:
            try:
                self._impl.delete_all_tracks()  # type: ignore[attr-defined]
            except Exception:
                # Reconstruimos si la librería no expone el método
                self._impl = self._build(30)

    # ─── Fallback centroide ───────────────────────────────────────────────
    def _update_centroid(self, detections: Sequence[Vehicle]) -> Sequence[Vehicle]:
        """Tracker mínimo: empareja por menor distancia al centroide previo."""
        out: list[Vehicle] = []
        used_ids: set[int] = set()
        for v in detections:
            cx, cy = v.bbox.center
            best_id, best_dist = None, float("inf")
            for tid, (px, py) in self._fallback_state.items():
                if tid in used_ids:
                    continue
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_dist and d < 80 ** 2:
                    best_id, best_dist = tid, d
            if best_id is None:
                best_id = self._fallback_next_id
                self._fallback_next_id += 1
            self._fallback_state[best_id] = (cx, cy)
            used_ids.add(best_id)
            v.track_id = best_id
            out.append(v)
        return out
