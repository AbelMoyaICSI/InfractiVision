# src/core/utils/timestamp.py

import time
from datetime import datetime


def format_time_sexagesimal(decimal_minutes: float) -> str:
    """Convierte minutos decimales a formato sexagesimal MM:SS.

    Ej: 1.5 -> "01:30". Valores negativos o inválidos devuelven "00:00".
    No altera la lógica matemática: es solo presentación UX junto al decimal.
    """
    try:
        total_seconds = int(round(float(decimal_minutes) * 60))
    except (TypeError, ValueError):
        return "00:00"
    if total_seconds < 0:
        total_seconds = 0
    mm, ss = divmod(total_seconds, 60)
    return f"{mm:02d}:{ss:02d}"


class TimestampUpdater:
    def __init__(self, label, root):
        self.label = label
        self.root = root
        self.running = False

    def start_timestamp(self):
        self.running = True
        self.update()

    def stop_timestamp(self):
        self.running = False

    def update(self):
        """Actualiza el timestamp a la hora actual"""
        if not self.running:
            return
            
        try:
            # Solo intentar actualizar el label si todavía existe
            if self.label.winfo_exists():
                now_str = time.strftime("%H:%M:%S")
                self.label.config(text=now_str)
                # Programar la siguiente actualización
                self.timer_id = self.label.after(1000, self.update)
            else:
                # Si el label ya no existe, detener el timer
                self.running = False
                self.timer_id = None
        except Exception as e:
            # Si hay cualquier error, detener las actualizaciones
            print(f"Error actualizando timestamp: {e}")
            self.running = False
            self.timer_id = None
