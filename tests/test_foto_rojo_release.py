"""Salida de Foto Rojo: libera modelos IA + detiene hilos (sin fuga VRAM)."""
from __future__ import annotations

import collections
import queue


class _FakeDetector:
    def __init__(self):
        self.model = object()
        self.released = False

    def release(self):
        self.released = True
        self.model = None


def test_detectors_release_is_idempotent():
    from src.core.detection.plate_detector import PlateDetector
    from src.core.detection.vehicle_detector import VehicleDetector

    v = VehicleDetector.__new__(VehicleDetector)
    v.model, v.last_detections, v.last_frame_hash = object(), [1], 123
    v.release()
    v.release()
    assert v.model is None

    p = PlateDetector.__new__(PlateDetector)
    p.model, p._gamma_luts = object(), {0.5: [1]}
    p.release()
    p.release()
    assert p.model is None and p._gamma_luts == {}


def test_release_foto_rojo_models_resets_async_singleton():
    from src.core.processing import async_plate_processor as ap_mod
    from src.application.services.model_preloader import release_foto_rojo_models

    inst = ap_mod.AsyncPlateProcessor.__new__(ap_mod.AsyncPlateProcessor)
    inst.running = False
    inst.worker_thread = None
    inst.plate_detector = _FakeDetector()
    inst.upscaler = object()
    inst.processed_results = {7: "crop"}
    inst.pending_queue = queue.Queue()
    inst.pending_queue.put_nowait({"a": 1})
    ap_mod._processor_instance = inst

    release_foto_rojo_models()
    release_foto_rojo_models()  # idempotente
    assert ap_mod._processor_instance is None
    assert inst.plate_detector is None
    assert inst.processed_results == {}


def test_player_shutdown_releases_detectors_and_queues():
    from src.core.video.videoplayer_opencv import VideoPlayerOpenCV

    pl = VideoPlayerOpenCV.__new__(VideoPlayerOpenCV)
    pl._shutdown_done = False
    pl.running = True
    pl.plate_running = True
    pl._after_id = None
    pl._manual_scroll_timer = None
    pl._detect_worker_thread = None
    pl.plate_thread = None
    pl.cap = None
    pl.parent = type("P", (), {"after_cancel": staticmethod(lambda _i: None)})()
    vd, pd = _FakeDetector(), _FakeDetector()
    pl.vehicle_detector, pl.plate_detector = vd, pd
    pl._last_annotated_frame = object()
    pl._pending_timestamp = 1
    pl._pending_beeps = [1]
    pl.frame_history = collections.deque([1])
    pl._motion_tracks = {1: 2}
    pl._detect_in, pl._detect_out, pl.plate_queue = queue.Queue(), queue.Queue(), queue.Queue()

    pl.shutdown()
    pl.shutdown()  # idempotente
    assert vd.released and pd.released
    assert pl.vehicle_detector is None and pl.plate_detector is None
    assert pl._pending_beeps == []
