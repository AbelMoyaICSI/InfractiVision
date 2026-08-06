from .detector_interface import (
    VehicleDetectorPort,
    PlateDetectorPort,
    TrafficLightDetectorPort,
)
from .ocr_interface import OCRReaderPort
from .repository_interface import ViolationRepositoryPort
from .tracker_interface import TrackerPort
from .video_interface import VideoSourcePort, FrameExtractorPort, RecorderPort

__all__ = [
    "VehicleDetectorPort",
    "PlateDetectorPort",
    "TrafficLightDetectorPort",
    "OCRReaderPort",
    "ViolationRepositoryPort",
    "TrackerPort",
    "VideoSourcePort",
    "FrameExtractorPort",
    "RecorderPort",
]
