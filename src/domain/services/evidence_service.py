"""Servicio de evidencia: arma rutas y nombres de archivo.
La escritura física la hace la infraestructura (FrameExtractor / Recorder).
"""
from __future__ import annotations

import os
from datetime import datetime

from src.domain.entities import Vehicle, ViolationEvidence


class EvidenceService:
    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def build_paths(self, vehicle: Vehicle) -> ViolationEvidence:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plate = (vehicle.plate_text or "UNK").replace("-", "")
        track = vehicle.track_id if vehicle.track_id is not None else 0
        stem = f"{ts}_{track}_{plate}"
        image_path = os.path.join(self._base_dir, f"{stem}.jpg")
        video_path = os.path.join(self._base_dir, f"{stem}.mp4")
        return ViolationEvidence(image_path=image_path, video_path=video_path)
