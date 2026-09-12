"""Precarga Foto Rojo (solo-detección) + validador Plate Recognizer."""
from __future__ import annotations


def test_has_plate_recognizer_token_env(monkeypatch):
    from src.infrastructure.ocr import cloud_plate_readers as cpr
    monkeypatch.setenv("PLATE_RECOGNIZER_API_TOKEN", "tok-123")
    assert cpr.has_plate_recognizer_token() is True
    monkeypatch.delenv("PLATE_RECOGNIZER_API_TOKEN", raising=False)
    monkeypatch.setattr(cpr.PlateRecognizerSnapshotReader, "_token_from_appdata",
                        staticmethod(lambda: None))
    assert cpr.has_plate_recognizer_token() is False
    monkeypatch.setattr(cpr.PlateRecognizerSnapshotReader, "_token_from_appdata",
                        staticmethod(lambda: "appdata-token"))
    assert cpr.has_plate_recognizer_token() is True


def test_preload_returns_dict_with_mocks(monkeypatch):
    import src.application.services.model_preloader as mp

    calls: list = []

    class FakeVD:
        using_gpu = False
        hardware_info = {}
        imgsz = 416

        def detect(self, img, conf=None, draw=False):
            calls.append(("vehicle", tuple(img.shape)))
            return []

    class FakePD:
        model = object()

        def detect(self, crop, conf=0.4, draw=False):
            calls.append(("plate", tuple(crop.shape)))
            return []

    monkeypatch.setattr("src.core.detection.vehicle_detector.VehicleDetector",
                        lambda *a, **k: FakeVD(), raising=False)
    import src.core.detection.plate_detector as pd_mod
    monkeypatch.setattr(pd_mod, "PlateDetector", lambda *a, **k: FakePD(), raising=False)
    monkeypatch.setattr(mp, "has_plate_token", lambda: False)
    import src.core.ocr.super_resolution as sr_mod
    monkeypatch.setattr(sr_mod, "get_upscaler", lambda: (_ for _ in ()).throw(RuntimeError("sin SR")))

    result = mp.preload_foto_rojo_models()
    assert result["plate_token_ok"] is False
    assert isinstance(result["vehicle_detector"], FakeVD)
    assert isinstance(result["plate_detector"], FakePD)
    assert "anpr_detector" not in result
    assert "lprnet_ok" not in result
    assert "elapsed_s" in result and "errors" in result
    # Warm-up: una inferencia dummy por detector (evita el lagazo en amarillo)
    assert ("vehicle", (416, 416, 3)) in calls
    assert ("plate", (320, 320, 3)) in calls
    assert set(result["warmup_s"]) >= {"vehicle", "plate"}


def test_preload_reports_errors_without_crash(monkeypatch):
    import src.application.services.model_preloader as mp

    def _boom(*a, **k):
        raise RuntimeError("sin pesos")

    monkeypatch.setattr("src.core.detection.vehicle_detector.VehicleDetector",
                        _boom, raising=False)
    import src.core.detection.plate_detector as pd_mod
    monkeypatch.setattr(pd_mod, "PlateDetector", _boom, raising=False)
    monkeypatch.setattr(mp, "has_plate_token", lambda: True)

    result = mp.preload_foto_rojo_models()
    assert result["vehicle_detector"] is None
    assert result["plate_detector"] is None
    assert len(result["errors"]) >= 2


def test_async_processor_is_lazy(monkeypatch):
    from types import SimpleNamespace
    from src.gui import preprocessing_dialog as pd_mod
    import src.core.processing.async_plate_processor as ap_mod

    calls: list = []

    class FakeAP:
        def start(self):
            calls.append("start")

        def update_semaphore_state(self, state):
            calls.append(("state", state))

    monkeypatch.setattr(ap_mod, "get_async_processor", lambda: FakeAP())

    self = SimpleNamespace(async_processor=None, _async_unavailable=False)
    # Solo lectura de estado: no crea nada
    assert pd_mod.PreprocessingDialog._get_async_processor(self, create=False) is None
    assert calls == []
    # Primer uso real: crea y arranca una sola vez
    ap = pd_mod.PreprocessingDialog._get_async_processor(self, create=True)
    assert isinstance(ap, FakeAP) and calls == ["start"]
    assert pd_mod.PreprocessingDialog._get_async_processor(self, create=True) is ap
    assert calls == ["start"]


def test_recognize_plate_use_case_is_detect_only():
    import numpy as np
    from src.application.use_cases.recognize_plate import RecognizePlateUseCase
    from src.domain.entities import BoundingBox, Vehicle

    class FakePlateDetector:
        def detect_plate(self, frame_bgr, vehicle_bbox):
            return BoundingBox(10, 10, 60, 30)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    vehicle = Vehicle(bbox=BoundingBox(0, 0, 80, 60), class_id=2, confidence=0.9,
                      track_id=7)
    out = RecognizePlateUseCase(FakePlateDetector(), ocr_reader=None).execute(frame, vehicle)
    assert out.plate_text is None  # en vivo no se lee texto
    assert out.extras["plate_bbox"] == (10, 10, 60, 30)
