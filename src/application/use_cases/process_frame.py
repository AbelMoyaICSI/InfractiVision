"""Caso de uso de alto nivel: procesa un frame de video y emite infracciones.

Sigue el pipeline funcional:
    1. Detecta vehículos (YOLOv8)
    2. Asigna track_id (DeepSORT)
    3. Lee estado del semáforo (hilo)
    4. Verifica cruce de línea
    5. Si cruce + ROJO  → captura evidencia → OCR → guarda BD → genera ticket
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.application.dto import FrameProcessingResultDTO, ViolationDTO
from src.application.use_cases.detect_red_light import DetectRedLightUseCase
from src.application.use_cases.detect_vehicle import DetectVehicleUseCase
from src.application.use_cases.generate_ticket import GenerateTicketUseCase
from src.application.use_cases.recognize_plate import RecognizePlateUseCase
from src.core.logger import get_logger
from src.domain.entities import TrafficLight, TrafficLightState, Vehicle
from src.domain.services import TrackingService, ViolationService

log = get_logger("usecase.process_frame")


class ProcessFrameUseCase:
    def __init__(
        self,
        detect_vehicle: DetectVehicleUseCase,
        tracking: TrackingService,
        detect_red_light: DetectRedLightUseCase,
        violation_service: ViolationService,
        recognize_plate: RecognizePlateUseCase,
        generate_ticket: GenerateTicketUseCase,
    ):
        self._detect_vehicle = detect_vehicle
        self._tracking = tracking
        self._detect_red_light = detect_red_light
        self._violation_service = violation_service
        self._recognize_plate = recognize_plate
        self._generate_ticket = generate_ticket
        # Para evitar emitir tickets duplicados del mismo track en el mismo ciclo de rojo
        self._already_ticketed: set[int] = set()

    def execute(self, frame_bgr: np.ndarray) -> FrameProcessingResultDTO:
        # 1-2. Detección + tracking
        detections = self._detect_vehicle.execute(frame_bgr)
        tracked: Sequence[Vehicle] = self._tracking.track(frame_bgr, detections)

        # 3. Semáforo
        light_state: TrafficLightState = self._detect_red_light.execute()
        light_entity = TrafficLight(state=light_state)

        # Reset de duplicados cuando deja de estar en rojo
        if not light_state.is_red:
            self._already_ticketed.clear()

        # 4. Verificación de cruce + 5/6. Regla de negocio
        new_violations: list[ViolationDTO] = []
        for vehicle in tracked:
            if vehicle.track_id is None:
                continue
            current = vehicle.bbox.bottom_center
            prev = self._tracking.previous_position(vehicle.track_id)
            if prev is not None:
                if self._violation_service.has_crossed_stop_line(prev, current):
                    vehicle.has_crossed_line = True
            self._tracking.remember_position(vehicle.track_id, current)

            if (
                self._violation_service.is_red_light_violation(vehicle, light_entity)
                and vehicle.track_id not in self._already_ticketed
            ):
                # 7-8. Recortar evidencia + OCR
                vehicle = self._recognize_plate.execute(frame_bgr, vehicle)
                # 9-10. Guardar evidencia + generar papeleta
                violation = self._generate_ticket.execute(vehicle, frame_bgr)
                self._already_ticketed.add(vehicle.track_id)
                new_violations.append(ViolationDTO.from_entity(violation))

        return FrameProcessingResultDTO(
            vehicle_count=len(tracked),
            traffic_light_state=light_state.value,
            new_violations=tuple(new_violations),
        )
