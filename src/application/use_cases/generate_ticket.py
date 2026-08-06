"""Caso de uso: generar la papeleta (Violation persistida + evidencia)."""
from __future__ import annotations

import numpy as np

from src.core.logger import get_logger
from src.domain.entities import Vehicle, Violation, ViolationEvidence
from src.domain.interfaces import (
    FrameExtractorPort,
    ViolationRepositoryPort,
)
from src.domain.services import EvidenceService

log = get_logger("usecase.generate_ticket")


class GenerateTicketUseCase:
    def __init__(
        self,
        evidence_service: EvidenceService,
        frame_extractor: FrameExtractorPort,
        repository: ViolationRepositoryPort,
    ):
        self._evidence_service = evidence_service
        self._frame_extractor = frame_extractor
        self._repository = repository

    def execute(self, vehicle: Vehicle, frame_bgr: np.ndarray) -> Violation:
        if vehicle.track_id is None:
            raise ValueError("No se puede generar ticket sin track_id (DeepSORT)")

        evidence = self._evidence_service.build_paths(vehicle)
        saved_image = self._frame_extractor.extract(frame_bgr, evidence.image_path)
        evidence = ViolationEvidence(image_path=saved_image, video_path=evidence.video_path)

        violation = Violation(
            plate_text=vehicle.plate_text or "DESCONOCIDA",
            plate_confidence=vehicle.plate_confidence or 0.0,
            vehicle_class_id=vehicle.class_id,
            track_id=vehicle.track_id,
            evidence=evidence,
        )

        ticket_id = self._repository.save(violation)
        violation.ticket_number = ticket_id
        log.info(
            "Papeleta generada %s placa=%s track=%s",
            ticket_id, violation.plate_text, violation.track_id,
        )
        return violation
