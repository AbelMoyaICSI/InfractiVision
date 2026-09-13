from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.application.services.traffic_processing_planner import TrafficProcessingPlanner
from src.application.use_cases.process_violation_video import OfficialVideoProcessor
from src.infrastructure.configuration import VideoConfigRepository
from src.infrastructure.ocr.cloud_plate_readers import PlateRecognizerSnapshotReader
from src.infrastructure.reports import ReportRepository
from src.domain.entities.plate_evidence import PlateEvidence
from src.presentation.gui.plate_review_window import PlateReviewWindow


def test_video_config_repository_uses_existing_gui_files():
    root = Path(__file__).resolve().parents[1]
    config = VideoConfigRepository(root).require("VID2COLISEO.MOV")

    assert config.polygon
    assert config.green == 15
    assert config.yellow == 3
    assert config.red == 20
    assert config.pre_red_seconds == 0.5
    assert config.green_skip_rate == 60


def test_planner_skips_green_and_processes_half_second_before_red():
    planner = TrafficProcessingPlanner(10, 3, 15, fps=30, pre_red_seconds=0.5, green_skip_rate=60)

    assert planner.state_at(0) == "green"
    assert planner.should_detect(1) is False
    assert planner.should_detect(60) is False
    assert planner.should_display(60) is True
    assert planner.state_at(390) == "red"
    assert planner.should_detect(375) is True  # 12.5s, 0.5s before red

    configured_planner = TrafficProcessingPlanner(15, 3, 20, fps=30, pre_red_seconds=0.5, green_skip_rate=60)
    assert configured_planner.should_detect(1) is False
    assert configured_planner.should_detect(525) is True  # 17.5s, 0.5s before red


def test_plate_recognizer_reads_official_environment_variable(monkeypatch):
    monkeypatch.setenv("PLATE_RECOGNIZER_API_TOKEN", "token")
    reader = PlateRecognizerSnapshotReader()
    assert reader.token == "token"


def test_confirmed_infractor_stays_red_after_leaving_polygon():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = np.array([(20, 20), (80, 20), (80, 80), (20, 80)], dtype=np.int32)
    tracks = {
        3: {
            "bbox": (10, 10, 40, 40),
            "class_name": "CAR",
            "infractor_confirmed": True,
        }
    }
    result = OfficialVideoProcessor._draw(frame, polygon, tracks, "green", {}, 12)

    assert tuple(result[40, 25]) == (0, 0, 255)


def test_no_pending_state_unconfirmed_stays_green():
    """Nuevo flujo sin pending: solo el infractor confirmado va en rojo."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = np.array([(20, 20), (80, 20), (80, 80), (20, 80)], dtype=np.int32)
    tracks = {
        3: {
            "bbox": (10, 10, 40, 40),
            "class_name": "CAR",
            "infractor_confirmed": False,
            "pending_infractor": True,  # flag legacy: se ignora
        }
    }
    result = OfficialVideoProcessor._draw(frame, polygon, tracks, "green", {}, 12)

    assert tuple(result[40, 25]) == (0, 255, 0)


def test_report_excludes_evidence_without_plate_text(tmp_path):
    empty = PlateEvidence("video.mp4", 1, 10, 1.0, "CAR", 0.8, validated=True)
    valid = PlateEvidence("video.mp4", 2, 20, 2.0, "CAR", 0.9, plate_text="ABC123", validated=True)

    json_path, csv_path = ReportRepository().export_validated(tmp_path, [empty, valid])

    content = json_path.read_text(encoding="utf-8")
    assert "ABC123" in content
    assert '"vehicle_id": 1' not in content
    assert csv_path.exists()


def test_viable_plate_crop_rejects_tiny_crops():
    processor = OfficialVideoProcessor(Path.cwd())
    tiny = np.zeros((12, 26, 3), dtype=np.uint8)
    ok = np.zeros((35, 70, 3), dtype=np.uint8)

    assert processor._viable_plate_crop(tiny) is False
    assert processor._viable_plate_crop(ok) is True
    assert processor._viable_plate_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is False


def test_plate_crop_with_margin_pads_and_clamps_to_vehicle():
    processor = OfficialVideoProcessor(Path.cwd())
    vehicle = np.zeros((100, 120, 3), dtype=np.uint8)
    local = (40, 30, 80, 50)

    padded = processor._plate_crop_with_margin(vehicle, local)

    assert padded.shape[:2] == (40, 80)
    assert padded[10:30, 20:60].sum() == 0  # región original intacta

    corner = processor._plate_crop_with_margin(vehicle, (0, 0, 40, 20))
    assert corner.shape[:2] == (30, 60)
    assert corner[0, 0, 0] == vehicle[0, 0, 0]


def test_prepare_evidences_localizes_plate_once(tmp_path):
    """Post-proceso: 1 inferencia sobre el cuadrante -> recorte listo p/API."""
    import cv2 as _cv2

    from src.application.services.plate_review_preparer import (
        prepare_evidences_for_review,
    )

    car_path = tmp_path / "vid_v3_best.jpg"
    _cv2.imwrite(str(car_path), np.zeros((200, 200, 3), dtype=np.uint8))

    class FakePlateDetector:
        model = object()
        calls = 0

        def detect(self, quadrant, conf=0.40, draw=False):
            type(self).calls += 1
            h, w = quadrant.shape[:2]
            assert h == 100 and w == 200  # cuadrante = mitad inferior
            return [[60, 20, 160, 60, 0.9]]

    evidence = PlateEvidence("vid.mp4", 3, 10, 1.0, "CAR", 0.8, str(car_path))
    out = prepare_evidences_for_review([evidence], tmp_path, FakePlateDetector())

    assert FakePlateDetector.calls == 1
    assert out[0].crop_path.endswith("_plate.jpg")
    assert out[0].metadata.get("full_car") is False
    assert out[0].metadata.get("plate_bbox") == [60, 120, 160, 160]


def test_prepare_evidences_keeps_full_car_when_no_plate(tmp_path):
    """Sin placa localizada: el carro completo va a la API (full_car)."""
    import cv2 as _cv2

    from src.application.services.plate_review_preparer import (
        prepare_evidences_for_review,
    )

    car_path = tmp_path / "vid_v5_best.jpg"
    _cv2.imwrite(str(car_path), np.zeros((200, 200, 3), dtype=np.uint8))

    class FakePlateDetector:
        model = object()

        def detect(self, quadrant, conf=0.40, draw=False):
            return []

    evidence = PlateEvidence("vid.mp4", 5, 10, 1.0, "CAR", 0.8, str(car_path))
    out = prepare_evidences_for_review([evidence], tmp_path, FakePlateDetector())

    assert out[0].crop_path == str(car_path)
    assert out[0].metadata.get("full_car") is True


def test_direction_from_history_right_left_unknown():
    proc = OfficialVideoProcessor(Path.cwd())
    assert proc._direction_from_history([(0, 0), (10, 0), (30, 0)]) == "right"
    assert proc._direction_from_history([(30, 0), (10, 0), (0, 0)]) == "left"
    assert proc._direction_from_history([(0, 0), (5, 0), (8, 0)]) == "unknown"
    assert proc._direction_from_history([]) == "unknown"
    assert proc._direction_from_history([(1, 1)]) == "unknown"


def test_select_best_prefers_candidate_with_plate(tmp_path):
    """Revisa placa en CADA candidato: gana el 2do aunque tenga menos quality."""
    import cv2 as _cv2

    from src.application.services.plate_review_preparer import (
        select_best_with_plate,
    )

    c0 = tmp_path / "vid_v3_c10.jpg"
    c1 = tmp_path / "vid_v3_c20.jpg"
    _cv2.imwrite(str(c0), np.zeros((200, 200, 3), dtype=np.uint8))
    _cv2.imwrite(str(c1), np.zeros((200, 200, 3), dtype=np.uint8))

    class FakePlateDetector:
        model = object()
        seen = []

        def detect(self, quadrant, conf=0.40, draw=False):
            # 1er candidato (direction right -> cuadrante 100x100): sin placa.
            # 2do candidato (unknown -> cuadrante 100x200): con placa.
            type(self).seen.append(quadrant.shape[1])
            if quadrant.shape[1] == 100:
                return []
            return [[60, 20, 160, 60, 0.9]]

    evidence = PlateEvidence("vid.mp4", 3, 10, 1.0, "CAR", 0.9, str(c0), metadata={
        "candidate_crops": [
            {"path": str(c0), "quality": 0.9, "frame": 10,
             "timestamp_seconds": 1.0, "direction": "right"},
            {"path": str(c1), "quality": 0.5, "frame": 20,
             "timestamp_seconds": 2.0, "direction": "unknown"},
        ],
    })
    out = select_best_with_plate(evidence, FakePlateDetector(), tmp_path)

    assert FakePlateDetector.seen == [100, 200]
    assert out.metadata.get("fallback_by_quality") is False
    assert out.metadata.get("selected_candidate_frame") == 20
    assert out.frame_index == 20
    assert out.crop_path.endswith("_plate.jpg")


def test_select_best_falls_back_to_quality_without_plate(tmp_path):
    """Sin placa en ningun candidato: la de mayor calidad, carro completo."""
    import cv2 as _cv2

    from src.application.services.plate_review_preparer import (
        select_best_with_plate,
    )

    c0 = tmp_path / "vid_v4_c10.jpg"
    _cv2.imwrite(str(c0), np.zeros((200, 200, 3), dtype=np.uint8))

    class FakePlateDetector:
        model = object()

        def detect(self, quadrant, conf=0.40, draw=False):
            return []

    evidence = PlateEvidence("vid.mp4", 4, 10, 1.0, "CAR", 0.9, str(c0), metadata={
        "candidate_crops": [
            {"path": str(c0), "quality": 0.9, "frame": 10,
             "timestamp_seconds": 1.0, "direction": "left"},
        ],
    })
    out = select_best_with_plate(evidence, FakePlateDetector(), tmp_path)

    assert out.crop_path == str(c0)
    assert out.metadata.get("full_car") is True
    assert out.metadata.get("fallback_by_quality") is True


def test_live_saves_all_infractor_frames(tmp_path):
    """Sin recorte top-K: TODOS los frames del infractor van a disco."""
    import cv2 as _cv2

    from src.infrastructure.configuration.video_config_repository import VideoConfig

    vid = tmp_path / "t.mp4"
    writer = _cv2.VideoWriter(str(vid), _cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        _cv2.rectangle(frame, (40 + i * 4, 60), (150 + i * 4, 200), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    class FakeVeh:
        batch_size = 1
        device = "cpu"
        using_gpu = False

        def __init__(self):
            self.i = 0

        def detect(self, frame, conf=0.4, draw=False):
            x = 40 + self.i * 4
            self.i += 1
            return [[x, 60, x + 110, 200, 2, 0.9]]

    config = VideoConfig(video_name="t.mp4", polygon=((0, 0), (320, 0), (320, 240), (0, 240)),
                         green=0.1, yellow=0.1, red=10.0, green_skip_rate=1)
    processor = OfficialVideoProcessor(tmp_path, vehicle_detector=FakeVeh())
    out = processor.process(str(vid), config, tmp_path / "out", conf=0.4,
                            save_video=False, save_crops=True)

    assert out["infractor_count"] == 1
    item = out["evidence"][0]
    cands = item.get("candidate_crops") or []
    assert len(cands) > 5  # sin recorte: mucho mas que el viejo top-5
    assert all(Path(c["path"]).exists() for c in cands)
    assert Path(item["crop_path"]).exists()  # best.jpg copiado tras flush
    assert item.get("n_candidates") == len(cands)


def test_cleanup_removes_rejected_candidates_only(tmp_path):
    """Limpieza: borra no elegidos, conserva el recorte final."""
    from src.application.services.plate_review_preparer import (
        cleanup_rejected_candidates,
    )

    crops = tmp_path / "crops"
    crops.mkdir()
    chosen = crops / "vid_v3_plate.jpg"
    rejected = crops / "vid_v3_f10.jpg"
    other_track = crops / "vid_v4_f11.jpg"
    for path in (chosen, rejected, other_track):
        path.write_bytes(b"fake-jpg")

    evidences = [
        PlateEvidence("vid.mp4", 3, 10, 1.0, "CAR", 0.9, str(chosen), metadata={
            "candidate_crops": [
                {"path": str(chosen), "quality": 0.9, "frame": 10},
                {"path": str(rejected), "quality": 0.5, "frame": 12},
            ],
        }),
        PlateEvidence("vid.mp4", 4, 11, 1.1, "CAR", 0.8, str(other_track), metadata={
            "candidate_crops": [{"path": str(other_track), "quality": 0.8, "frame": 11}],
        }),
    ]
    summary = cleanup_rejected_candidates(evidences, crops)

    assert summary == {"removed": 1, "kept": 2, "errors": 0}
    assert chosen.exists() and other_track.exists()
    assert not rejected.exists()
    # Idempotente: segunda llamada no borra nada ni falla.
    summary2 = cleanup_rejected_candidates(evidences, crops)
    assert summary2 == {"removed": 0, "kept": 2, "errors": 0}


def test_cleanup_never_leaves_allowed_dir(tmp_path):
    """Seguridad: no borra fuera del dir oficial aunque el path lo pida."""
    from src.application.services.plate_review_preparer import (
        cleanup_rejected_candidates,
    )

    crops = tmp_path / "crops"
    crops.mkdir()
    chosen = crops / "vid_plate.jpg"
    chosen.write_bytes(b"fake-jpg")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"fake-jpg")

    evidences = [PlateEvidence("v.mp4", 1, 1, 0.1, "CAR", 0.5, str(chosen), metadata={
        "candidate_crops": [
            {"path": str(chosen), "quality": 0.9, "frame": 1},
            {"path": str(outside), "quality": 0.5, "frame": 2},
        ],
    })]
    summary = cleanup_rejected_candidates(evidences, crops)

    assert outside.exists()
    assert summary["removed"] == 0
    assert summary["errors"] == 1


def test_plate_review_poll_drains_worker_results():
    """El worker encola resultados y el poller de Tk los aplica (no `after` desde el hilo)."""
    window = PlateReviewWindow.__new__(PlateReviewWindow)
    window._results_queue = __import__("queue").Queue()
    seen = []
    window.window = SimpleNamespace(after=lambda _ms, _fn: None)
    window._show_result = lambda *args: seen.append(args)

    window._results_queue.put((2, "T5K479", 0.94, ""))
    window._poll_results()

    assert seen == [(2, "T5K479", 0.94, "")]
    assert window._results_queue.empty()
