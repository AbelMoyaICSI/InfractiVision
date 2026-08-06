"""Detector de semáforo. Existen dos estrategias:

* `VirtualTrafficLightDetector` — lee el estado del panel virtual existente
  (`src.core.traffic_signal.semaphore.Semaforo`) que se ejecuta en un hilo Tk.
  Esto preserva el comportamiento actual del MVP.

Si en el futuro se entrena un modelo CV para detectar el color real del
semáforo en la escena, se agregará un nuevo adapter (`CVTrafficLightDetector`)
que también implementará el mismo `TrafficLightDetectorPort`.
"""
from __future__ import annotations

from typing import Callable

from src.core.logger import get_logger
from src.domain.entities import TrafficLightState
from src.domain.interfaces import TrafficLightDetectorPort

log = get_logger("infra.traffic_light")


class VirtualTrafficLightDetector(TrafficLightDetectorPort):
    """Adapter sobre el panel virtual `Semaforo` (Tk).

    Recibe un *callable* `state_provider` para no depender directamente de
    Tk en infraestructura — la GUI inyecta `lambda: self.semaforo.get_current_state()`.
    """

    def __init__(self, state_provider: Callable[[], str]):
        self._state_provider = state_provider

    def current_state(self) -> TrafficLightState:
        try:
            raw = (self._state_provider() or "green").lower()
        except Exception as e:
            log.warning("No se pudo leer el estado del semáforo: %s", e)
            return TrafficLightState.GREEN
        try:
            return TrafficLightState(raw)
        except ValueError:
            return TrafficLightState.GREEN
