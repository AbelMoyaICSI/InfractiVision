import cv2
import threading
import time
import queue
import tkinter as tk
import json
import os
import numpy as np
import torch
import psutil

from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from ultralytics import YOLO

from src.core.processing.plate_processing import process_plate

# Archivos de configuración
POLYGON_CONFIG_FILE = "config/polygon_config.json"
AVENUE_CONFIG_FILE = "config/avenue_config.json"
PRESETS_FILE = "config/time_presets.json"  # Para los tiempos del semáforo

class VideoPlayerOpenCV:
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo):
        self.parent = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label = timestamp_label
        self.semaforo = semaforo

        # Contenedor principal
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill="both", expand=True)

        # Botonera inferior
        self.btn_frame = tk.Frame(self.frame, bg='black')
        self.btn_frame.pack(side="bottom", fill="x", pady=5)

        self.load_button = tk.Button(
            self.btn_frame, text="CARGAR\nVIDEO", font=("Arial", 12),
            width=12, anchor="center", justify="center", command=self.select_video
        )
        self.load_button.pack(side="left", padx=5)

        # Botones renombrados a "ÁREA" en lugar de "POLÍGONO"
        self.save_poly_button = tk.Button(
            self.btn_frame, text="GUARDAR\nÁREA", font=("Arial", 12),
            width=12, anchor="center", justify="center", command=self.save_polygon
        )
        self.save_poly_button.pack(side="left", padx=5)

        self.delete_poly_button = tk.Button(
            self.btn_frame, text="BORRAR\nÁREA", font=("Arial", 12),
            width=12, anchor="center", justify="center", command=self.delete_polygon
        )
        self.delete_poly_button.pack(side="left", padx=5)

        self.btn_gestion_polys = tk.Button(
            self.btn_frame, text="GESTIONAR\nÁREAS", font=("Arial", 12),
            width=12, anchor="center", justify="center", command=self.gestionar_poligonos
        )
        self.btn_gestion_polys.pack(side="left", padx=5)

        self.btn_gestion_camaras = tk.Button(
            self.btn_frame, text="GESTIONAR\nCÁMARAS", font=("Arial", 12),
            width=12, anchor="center", justify="center", command=self.gestionar_camaras
        )
        self.btn_gestion_camaras.pack(side="left", padx=5)

        # Contenedor para video y panel lateral
        self.video_panel_container = tk.Frame(self.frame, bg='black')
        self.video_panel_container.pack(side="top", fill="both", expand=True)

        self.video_frame = tk.Frame(self.video_panel_container, bg='black', width=640, height=360)
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="black", bd=0, highlightthickness=0)
        self.video_label.pack(fill="both", expand=True)

        self.plates_frame = tk.Frame(self.video_panel_container, bg="gray", width=220)
        self.plates_frame.pack(side="right", fill="y")
        self.plates_frame.pack_propagate(False)

        self.plates_title = tk.Label(
            self.plates_frame, text="Placas Detectadas",
            bg="gray", fg="white", font=("Arial", 12, "bold")
        )
        self.plates_title.pack(pady=5)

        self.plates_canvas = tk.Canvas(self.plates_frame, bg="gray")
        self.plates_canvas.pack(side="left", fill="both", expand=True)

        self.plates_scrollbar = tk.Scrollbar(
            self.plates_frame, orient="vertical", command=self.plates_canvas.yview
        )
        self.plates_scrollbar.pack(side="right", fill="y")
        self.plates_canvas.configure(yscrollcommand=self.plates_scrollbar.set)

        self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="gray")
        self.plates_inner_frame.bind("<Configure>", self._on_plates_inner_configure)
        self.plates_canvas.create_window((0, 0), window=self.plates_inner_frame, anchor="nw")
        self.detected_plates_widgets = []

        # Configuración del timestamp
        self.timestamp_label.config(font=("Arial", 30, "bold"), bg="black", fg="yellow")
        self.timestamp_label.place(in_=self.video_label, x=50, y=10)
        
        # Label para el nombre de la avenida
        self.current_avenue = None
        self.avenue_label = tk.Label(
            self.video_frame, text="", font=("Arial", 20, "bold"),
            bg="black", fg="white", wraplength=300
        )
        self.avenue_label.place(relx=0.5, y=80, anchor="n")

        self.info_label = tk.Label(
            self.video_frame, text="...", bg="black", fg="white", font=("Arial", 11, "bold")
        )
        self.info_label.place(relx=0.98, y=10, anchor="ne")

        self.cap = None
        self.running = False
        self.orig_w = None
        self.orig_h = None

        # Variables para el área (antes "polígono")
        self.polygon_points = []
        self.have_polygon = False
        self.current_video_path = None

        self.plate_queue = queue.Queue()
        self.plate_running = True
        self.plate_thread = threading.Thread(target=self.plate_loop, daemon=True)
        self.plate_thread.start()

        self.last_time = time.time()
        self.fps_calc = 0.0
        self.using_gpu = torch.cuda.is_available()

        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(4)
        except:
            pass

        self.video_label.bind("<Button-1>", self.on_mouse_click_polygon)

    def _on_plates_inner_configure(self, event):
        self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))

    # Gestión de configuración de avenidas y tiempos
    def load_avenue_config(self):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return {}
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def save_avenue_config(self, data):
        with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_avenue_for_video(self, video_path):
        data = self.load_avenue_config()
        return data.get(video_path)

    def set_avenue_for_video(self, video_path, avenue_name):
        data = self.load_avenue_config()
        data[video_path] = avenue_name
        self.save_avenue_config(data)

    def load_time_presets(self):
        if not os.path.exists(PRESETS_FILE):
            return {}
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_time_presets(self, data):
        with open(PRESETS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def set_time_preset_for_video(self, video_path, times):
        presets = self.load_time_presets()
        presets[video_path] = times
        self.save_time_presets(presets)
        self.semaforo.cycle_durations = times
        self.semaforo.target_time = time.time() + times[self.semaforo.get_current_state()]

    def get_time_preset_for_video(self, video_path):
        presets = self.load_time_presets()
        return presets.get(video_path)

    def first_time_setup(self, video_path):
        if self.get_avenue_for_video(video_path) is not None and self.get_time_preset_for_video(video_path) is not None:
            messagebox.showinfo("Info", "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.", parent=self.parent)
            return
        setup_win = tk.Toplevel(self.parent)
        setup_win.title("Configuración Inicial del Video")

        tk.Label(setup_win, text="Nombre de la Avenida:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        avenue_entry = tk.Entry(setup_win, width=30)
        avenue_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(setup_win, text="Tiempo Verde (s):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        green_entry = tk.Entry(setup_win, width=10)
        green_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(setup_win, text="Tiempo Amarillo (s):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        yellow_entry = tk.Entry(setup_win, width=10)
        yellow_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(setup_win, text="Tiempo Rojo (s):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        red_entry = tk.Entry(setup_win, width=10)
        red_entry.grid(row=3, column=1, padx=5, pady=5)

        def save_setup():
            avenue = avenue_entry.get().strip()
            try:
                g = int(green_entry.get().strip())
                y = int(yellow_entry.get().strip())
                r = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser enteros.", parent=setup_win)
                return
            if not avenue:
                messagebox.showerror("Error", "Debe ingresar el nombre de la avenida.", parent=setup_win)
                return
            self.set_avenue_for_video(video_path, avenue)
            self.current_avenue = avenue
            self.avenue_label.config(text=avenue)
            times = {"green": g, "yellow": y, "red": r}
            self.set_time_preset_for_video(path, times)
            messagebox.showinfo("Éxito", "Configuración inicial guardada.", parent=setup_win)
            setup_win.destroy()

        tk.Button(setup_win, text="Guardar Configuración", command=save_setup).grid(row=4, column=0, columnspan=2, pady=10)
        setup_win.transient(self.parent)
        setup_win.grab_set()
        self.parent.wait_window(setup_win)

    # Resto de métodos (on_mouse_click_polygon, draw_polygon_on_np, save_polygon, load_polygon_for_video,
    # delete_polygon, gestionar_poligonos, select_video, load_video, stop_video, plate_loop,
    # update_frames, resize_and_letterbox, _safe_add_plate_to_panel, add_plate_to_panel,
    # gestionar_camaras, _cam_load, _cam_del): siguen idénticos a los arriba definidos.

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE): return
        try:
            with open(AVENUE_CONFIG_FILE, "r") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(AVENUE_CONFIG_FILE, "w") as fw:
                    json.dump(data, fw, indent=2)
        except:
            pass

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE): return
        try:
            with open(PRESETS_FILE, "r") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(PRESETS_FILE, "w") as fw:
                    json.dump(data, fw, indent=2)
        except:
            pass

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE): return
        try:
            with open(POLYGON_CONFIG_FILE, "r") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(POLYGON_CONFIG_FILE, "w") as fw:
                    json.dump(data, fw, indent=2)
        except:
            pass

# Fin del módulo VideoPlayerOpenCV
