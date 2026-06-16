"""Composition Root: ÚNICO lugar donde se instancian implementaciones
concretas y se inyectan en los Casos de Uso.

Esto es el "core" de DI: cualquier capa superior recibe las dependencias
ya cableadas. Si mañana se cambia SQLite por MySQL, EasyOCR por LPRNet,
DeepSORT por ByteTrack, SOLO se modifica este archivo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config.settings import Settings, load_settings
from src.application.use_cases import (
    DetectRedLightUseCase,
    DetectVehicleUseCase,
    GenerateTicketUseCase,
    ProcessFrameUseCase,
    RecognizePlateUseCase,
)
from src.core.logger import get_logger
from src.domain.interfaces import (
    OCRReaderPort,
    TrafficLightDetectorPort,
    TrackerPort,
    ViolationRepositoryPort,
    VehicleDetectorPort,
    PlateDetectorPort,
)
from src.domain.services import EvidenceService, TrackingService, ViolationService

log = get_logger("composition_root")


@dataclass(frozen=True)
class Container:
    """Bolsa con todas las dependencias listas para usar."""
    settings: Settings
    process_frame: ProcessFrameUseCase
    repository: ViolationRepositoryPort
    detect_vehicle: DetectVehicleUseCase
    detect_red_light: DetectRedLightUseCase
    recognize_plate: RecognizePlateUseCase
    generate_ticket: GenerateTicketUseCase


# ─── Factories de infraestructura (con lazy load) ─────────────────────────

def _build_repository(settings: Settings) -> ViolationRepositoryPort:
    if settings.database.backend == "mysql" and settings.database.mysql_url:
        from src.infrastructure.database import MySQLViolationRepository
        return MySQLViolationRepository(settings.database.mysql_url)
    from src.infrastructure.database import SQLiteViolationRepository
    return SQLiteViolationRepository(settings.database.sqlite_path)


def _build_ocr(settings: Settings) -> OCRReaderPort:
    backend = settings.ocr.backend.lower()
    if backend == "easyocr":
        from src.infrastructure.ocr import EasyOCRReader
        return EasyOCRReader()
    if backend == "paddleocr":
        from src.infrastructure.ocr import PaddleOCRReader
        return PaddleOCRReader()
    from src.infrastructure.ocr import LPRNetReader
    return LPRNetReader(regional_context=settings.ocr.regional_context)


def _build_vehicle_detector(settings: Settings) -> VehicleDetectorPort:
    from src.infrastructure.ai import YoloVehicleDetector
    return YoloVehicleDetector(model_path=settings.models.yolo_vehicle)


def _build_plate_detector(settings: Settings) -> PlateDetectorPort:
    from src.infrastructure.ai import YoloPlateDetector
    return YoloPlateDetector(model_path=settings.models.yolo_plate)


def _build_tracker() -> TrackerPort:
    from src.infrastructure.tracking import DeepSortTracker
    return DeepSortTracker()


def _build_traffic_light(state_provider: Callable[[], str]) -> TrafficLightDetectorPort:
    from src.infrastructure.ai import VirtualTrafficLightDetector
    return VirtualTrafficLightDetector(state_provider)


def _load_stop_line(settings: Settings) -> tuple[tuple[float, float], tuple[float, float]]:
    """Lee `config/zones.json`. Si no existe, usa una línea horizontal por defecto."""
    cfg_path = Path(settings.config_files.zones)
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        sl = data["stop_line"]
        return (tuple(sl["p1"]), tuple(sl["p2"]))  # type: ignore[return-value]
    except Exception as e:
        log.warning("zones.json no usable (%s). Usando línea por defecto.", e)
        return ((0.0, 540.0), (1280.0, 540.0))


# ─── Factoría principal ───────────────────────────────────────────────────

def build_container(
    traffic_light_state_provider: Callable[[], str] | None = None,
) -> Container:
    """Cablea TODO el sistema y devuelve un `Container` listo para inyectar."""
    settings = load_settings()

    # Infraestructura
    repository = _build_repository(settings)
    ocr = _build_ocr(settings)
    vehicle_det = _build_vehicle_detector(settings)
    plate_det = _build_plate_detector(settings)
    tracker = _build_tracker()
    state_provider = traffic_light_state_provider or (lambda: "green")
    traffic_light_det = _build_traffic_light(state_provider)

    # Servicios de dominio
    stop_line = _load_stop_line(settings)
    violation_service = ViolationService(stop_line=stop_line)
    tracking_service = TrackingService(tracker)
    evidence_service = EvidenceService(base_dir=settings.storage.evidences)

    # Casos de uso
    from src.infrastructure.video import OpenCVFrameExtractor
    frame_extractor = OpenCVFrameExtractor()

    detect_vehicle_uc = DetectVehicleUseCase(vehicle_det)
    detect_red_light_uc = DetectRedLightUseCase(traffic_light_det)
    recognize_plate_uc = RecognizePlateUseCase(
        plate_det, ocr, min_confidence=settings.ocr.min_confidence
    )
    generate_ticket_uc = GenerateTicketUseCase(
        evidence_service=evidence_service,
        frame_extractor=frame_extractor,
        repository=repository,
    )
    process_frame_uc = ProcessFrameUseCase(
        detect_vehicle=detect_vehicle_uc,
        tracking=tracking_service,
        detect_red_light=detect_red_light_uc,
        violation_service=violation_service,
        recognize_plate=recognize_plate_uc,
        generate_ticket=generate_ticket_uc,
    )

    log.info("Container construido (DB=%s, OCR=%s)", settings.database.backend, settings.ocr.backend)
    return Container(
        settings=settings,
        process_frame=process_frame_uc,
        repository=repository,
        detect_vehicle=detect_vehicle_uc,
        detect_red_light=detect_red_light_uc,
        recognize_plate=recognize_plate_uc,
        generate_ticket=generate_ticket_uc,
    )
