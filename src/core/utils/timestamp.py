# src/core/utils/timestamp.py

import time
from datetime import datetime

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