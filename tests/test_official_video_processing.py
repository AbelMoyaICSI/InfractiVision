from pathlib import Path

import cv2
import numpy as np

from src.application.services.traffic_processing_planner import TrafficProcessingPlanner
from src.application.use_cases.process_violation_video import OfficialVideoProcessor
from src.infrastructure.configuration import VideoConfigRepository
from src.infrastructure.ocr.cloud_plate_readers import PlateRecognizerSnapshotReader
from src.infrastructure.reports import ReportRepository
from src.domain.entities.plate_evidence import PlateEvidence


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


def test_pending_infractor_stays_yellow_until_plate_confirms():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = np.array([(20, 20), (80, 20), (80, 80), (20, 80)], dtype=np.int32)
    tracks = {
        3: {
            "bbox": (10, 10, 40, 40),
            "class_name": "CAR",
            "infractor_confirmed": False,
            "pending_infractor": True,
        }
    }
    result = OfficialVideoProcessor._draw(frame, polygon, tracks, "green", {}, 12)

    assert tuple(result[40, 25]) == (0, 255, 255)


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
    ok = np.zeros((25, 50, 3), dtype=np.uint8)

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
