import tkinter as tk
import os
from PIL import Image, ImageTk

class WelcomeFrame(tk.Frame):
    """Pantalla de inicio con acceso a otras ventanas."""

    def __init__(self, master, app_manager):
        super().__init__(master, bg="#273D86")
        self.app_manager = app_manager
        self._build()

    def _build(self):
        tk.Label(self, text="Bienvenido a InfractiVision", font=("Arial", 40, "bold"),
                 bg="#273D86", fg="white").pack(pady=(50, 20))

        # logo --------------------------------------
        img_path = os.path.join("img", "InfractiVision-logo.png")
        try:
            logo_img = Image.open(img_path)
            logo_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            tk.Label(self, image=self.logo_tk, bg="#273D86").pack(pady=(0, 20))
        except Exception:
            tk.Label(self, text="[Logo no encontrado]", bg="#273D86", fg="white").pack(pady=(0, 20))

        tk.Label(self, text="Selecciona la opción para continuar",
                 font=("Arial", 24), bg="#273D86", fg="white").pack(pady=(0, 40))

        btn_frame = tk.Frame(self, bg="#273D86")
        btn_frame.pack()
        tk.Button(btn_frame, text="Detección de Placas", font=("Arial", 18), width=18,
                  command=self.app_manager.open_violation_window).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Gestión de Infracciones", font=("Arial", 18), width=22,
                  command=self.app_manager.open_infractions_window).pack(side="left", padx=10)