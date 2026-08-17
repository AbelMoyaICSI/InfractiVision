"""Ventana de visualización de infracciones.

Wrapper de la legacy `red_light_violation_window` que ahora consume DTOs
desde los casos de uso en lugar de tocar BD/YOLO directamente.
"""
from __future__ import annotations

import tkinter as tk
from typing import Sequence

from src.application.dto import ViolationDTO


class ViolationWindow:
    def __init__(self, parent: tk.Misc, violations: Sequence[ViolationDTO]):
        self._win = tk.Toplevel(parent)
        self._win.title("InfractiVision – Infracciones")
        self._render(violations)

    def _render(self, violations: Sequence[ViolationDTO]) -> None:
        if not violations:
            tk.Label(self._win, text="Sin infracciones recientes.").pack(padx=20, pady=20)
            return
        for v in violations:
            row = tk.Frame(self._win)
            row.pack(fill="x", padx=10, pady=4)
            tk.Label(
                row,
                text=f"{v.occurred_at:%Y-%m-%d %H:%M:%S} | {v.plate_text} ({v.plate_confidence:.2f}) | Ticket {v.ticket_number}",
                anchor="w",
            ).pack(fill="x")
