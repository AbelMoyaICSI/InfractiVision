"""Pendientes sin placa: carro completo a la API solo si el crop es viable."""
from __future__ import annotations

import cv2
import numpy as np

from src.gui.preprocessing_dialog import FULL_CAR_MIN_WIDTH, split_viable_full_car


def _write(path, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
    return str(path)


def _pend(track_id, crop_path, w=0, h=0):
    return {
        "vehicle_id": track_id,
        "frame_index": 10,
        "timestamp_seconds": 1.0,
        "vehicle_class": "VEH",
        "crop_path": crop_path,
        "crop_w": w,
        "crop_h": h,
    }


def test_viable_pending_goes_to_review(tmp_path):
    big = _write(tmp_path / "v7_pending.jpg", 400, 300)
    rows, remaining = split_viable_full_car([_pend(7, big, 400, 300)], "vid.mp4")
    assert remaining == []
    assert len(rows) == 1
    row = rows[0]
    assert row.track_id == 7 and row.crop_path == big
    assert row.plate_text == ""
    assert row.metadata.get("full_car") is True
    assert "completo" in row.review_notes


def test_small_pending_stays_nie(tmp_path):
    small = _write(tmp_path / "v8_pending.jpg", 50, 40)
    rows, remaining = split_viable_full_car([_pend(8, small, 50, 40)], "vid.mp4")
    assert rows == []
    assert len(remaining) == 1


def test_missing_crop_stays_nie():
    rows, remaining = split_viable_full_car([_pend(9, "/no/existe.jpg")], "vid.mp4")
    assert rows == [] and len(remaining) == 1


def test_dimensions_probed_from_file_when_missing(tmp_path):
    big = _write(tmp_path / "v10_pending.jpg", FULL_CAR_MIN_WIDTH + 40, 200)
    rows, remaining = split_viable_full_car([_pend(10, big)], "vid.mp4")
    assert len(rows) == 1 and remaining == []


def test_mixed_pending_split(tmp_path):
    big = _write(tmp_path / "a.jpg", 400, 300)
    small = _write(tmp_path / "b.jpg", 60, 40)
    pending = [_pend(1, big, 400, 300), _pend(2, small, 60, 40), _pend(3, "/no.jpg")]
    rows, remaining = split_viable_full_car(pending, "vid.mp4")
    assert [r.track_id for r in rows] == [1]
    assert [p["vehicle_id"] for p in remaining] == [2, 3]
