"""Value objects used by the official video-analysis workflow."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PlateEvidence:
    video_name: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    vehicle_class: str
    quality_score: float
    crop_path: str = ""
    plate_text: str = ""
    ocr_confidence: float = 0.0
    ocr_method: str = ""
    validated: bool = False
    review_notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video": self.video_name,
            "vehicle_id": self.track_id,
            "frame": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "vehicle_class": self.vehicle_class,
            "quality_score": round(self.quality_score, 4),
            "crop_path": self.crop_path,
            "plate": self.plate_text,
            "ocr_confidence": round(self.ocr_confidence, 4),
            "ocr_method": self.ocr_method,
            "validated": self.validated,
            "review_notes": self.review_notes,
            **self.metadata,
        }
