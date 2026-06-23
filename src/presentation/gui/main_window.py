"""Ventana principal en la NUEVA arquitectura.

Recibe el `ProcessFrameUseCase` por DI y delega la navegación al
`AppManager` legacy mientras dura la migración. Esto evita reescribir
las 3 ventanas grandes de tk en una sola pasada.

La GUI legacy puede ahora **leer** los componentes Clean (caso de uso,
estado del semáforo) si se le pasan por kwargs. Si no se pasan, todo
sigue funcionando igual que antes (retro-compatible 100%).
"""
from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, MutableMapping

from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.application.use_cases import ProcessFrameUseCase

log = get_logger("presentation.main_window")


class MainWindow:
    """Composición raíz de la GUI principal."""

    def __init__(
        self,
        root: tk.Tk,
        process_frame_uc: "ProcessFrameUseCase",
        user_id: str | None = None,
        device_id: str | None = None,
        traffic_light_state: MutableMapping[str, str] | None = None,
    ):
        self._root = root
        self._process_frame_uc = process_frame_uc
        self._user_id = user_id
        self._device_id = device_id
        self._traffic_light_state = traffic_light_state
        self._build_legacy_app()

    def _build_legacy_app(self) -> None:
        # Mantenemos el AppManager existente para no romper la GUI actual.
        # Las pantallas internas (Foto Rojo, Gestión, etc.) seguirán funcionando.
        from src.gui.app_manager import AppManager

        self._app_manager = AppManager(
            self._root,
            user_id=self._user_id,
            device_id=self._device_id,
            process_frame_uc=self._process_frame_uc,
            traffic_light_state=self._traffic_light_state,
        )
        log.info("MainWindow inicializada (AppManager legacy embebido)")

    @property
    def process_frame_use_case(self) -> "ProcessFrameUseCase":
        """Expuesto para que las ventanas internas puedan procesar frames."""
        return self._process_frame_uc
