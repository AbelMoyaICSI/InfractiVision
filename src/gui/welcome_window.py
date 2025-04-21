import tkinter as tk
import os
from PIL import Image, ImageTk

class WelcomeFrame(tk.Frame):
    def __init__(self, master, app_manager):
        super().__init__(master, bg="#273D86")
        self.app_manager = app_manager
        self.create_widgets()

    def create_widgets(self):
        # Título principal con letras grandes
        title_label = tk.Label(
            self,
            text="Bienvenido a InfractiVision",
            font=("Arial", 40, "bold"),
            bg="#273D86",
            fg="white"
        )
        title_label.pack(pady=(50, 20))

        # Cargar y redimensionar la imagen del logo de forma responsive
        image_path = os.path.join("img", "InfractiVision-logo.png")
        try:
            logo_image = Image.open(image_path)
            # Define el tamaño máximo permitido (ancho y alto)
            max_width = 300
            max_height = 300
            logo_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(logo_image)
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            self.logo_tk = None

        if self.logo_tk:
            logo_label = tk.Label(self, image=self.logo_tk, bg="#273D86")
            logo_label.pack(pady=(0, 20))
        else:
            logo_label = tk.Label(self, text="[No se pudo cargar la imagen]", bg="#273D86", fg="white")
            logo_label.pack(pady=(0, 20))

        # Subtítulo
        subtitle_label = tk.Label(
            self,
            text="Selecciona la opción para continuar",
            font=("Arial", 24),
            bg="#273D86",
            fg="white"
        )
        subtitle_label.pack(pady=(0, 40))

        # Contenedor para los botones
        btn_frame = tk.Frame(self, bg="#273D86")
        btn_frame.pack()

        # Botón para "Foto Rojo"
        btn_foto = tk.Button(
            btn_frame,
            text="Foto Rojo",
            font=("Arial", 18),
            width=15,
            command=self.app_manager.open_foto_rojo_window
        )
        btn_foto.pack(side="left", padx=10)

        # Botón para "Gestión de Infracciones"
        btn_gestion = tk.Button(
            btn_frame,
            text="Gestión de Infracciones",
            font=("Arial", 18),
            width=20,
            command=self.app_manager.open_gestion_infracciones_window
        )
        btn_gestion.pack(side="left", padx=10)
