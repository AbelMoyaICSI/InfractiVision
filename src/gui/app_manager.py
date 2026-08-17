# src/gui/app_manager.py

import tkinter as tk
from src.gui.welcome_window import WelcomeFrame
from src.gui.red_light_violation_window import create_violation_window
from src.gui.infractions_management_window import create_infractions_window

class AppManager:
    """Centraliza la navegación entre pantallas GUI en una única ventana,
       sin intentar maximizar automáticamente ni restaurar estados problemáticos."""

    def __init__(self, root: tk.Tk, user_id: str = None, device_id: str = None,
                 process_frame_uc=None, traffic_light_state=None):
        self.user_id = user_id
        self.device_id = device_id
        self.process_frame_uc = process_frame_uc
        self.traffic_light_state = traffic_light_state

        self.root = root
        self.root.title("InfractiVision")
        # No se intenta maximizar automáticamente (evita segfault en Linux)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.show_welcome()

    def _clear_root(self):
        """Destruye todos los widgets en root y fuerza actualización."""
        for w in self.root.winfo_children():
            w.destroy()
        self.root.update_idletasks()

    def _on_closing(self):
        """Maneja el cierre de la aplicación principal."""
        self.root.quit()
        self.root.destroy()

    def show_welcome(self):
        """Pantalla de bienvenida."""
        self._clear_root()
        self.root.title("InfractiVision – Principal")
        frm = WelcomeFrame(self.root, self)
        frm.pack(fill="both", expand=True)

    def open_violation_window(self):
        """Pantalla de Foto Rojo."""
        self._clear_root()
        self.root.title("InfractiVision – Foto Rojo")
        create_violation_window(
            self.root,
            self.show_welcome,
            process_frame_uc=self.process_frame_uc,
            traffic_light_state=self.traffic_light_state,
        )

    def open_infractions_window(self):
        """Pantalla de Gestión de Infracciones."""
        self._clear_root()
        self.root.title("InfractiVision – Gestión de Infracciones")
        create_infractions_window(self.root, self.show_welcome)