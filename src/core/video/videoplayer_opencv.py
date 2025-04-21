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

from foto_rojo.plate_processing import process_plate

# Archivos de configuración
POLYGON_CONFIG_FILE = "polygon_config.json"
AVENUE_CONFIG_FILE = "avenue_config.json"
PRESETS_FILE = "time_presets.json"  # Para los tiempos del semáforo

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
        
        # Label para el nombre de la avenida, centrado debajo del timestamp
        self.current_avenue = None
        self.avenue_label = tk.Label(self.video_frame, text="", font=("Arial", 20, "bold"),
                                     bg="black", fg="white", wraplength=300)
        self.avenue_label.place(relx=0.5, y=80, anchor="n")

        self.info_label = tk.Label(self.video_frame, text="...", bg="black", fg="white", font=("Arial", 11, "bold"))
        self.info_label.place(relx=0.98, y=10, anchor="ne")

        self.cap = None
        self.running = False
        self.orig_w = None
        self.orig_h = None

        # Variables para el área (antes "polígono")
        self.polygon_points = []
        self.have_polygon = False  # Una vez guardada el área, no se permite modificarla hasta eliminarla.
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

    # ----- Funciones para gestionar el nombre de la avenida y los tiempos (preset) al cargar el video -----
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
        return data.get(video_path, None)

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
        self.cycle_durations = times
        self.target_time = time.time() + self.cycle_durations[self.semaforo.get_current_state()]

    def get_time_preset_for_video(self, video_path):
        presets = self.load_time_presets()
        return presets.get(video_path, None)

    def first_time_setup(self, video_path):
        # Si ya existe configuración para este video, se evita volver a pedirla.
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
                green_time = int(green_entry.get().strip())
                yellow_time = int(yellow_entry.get().strip())
                red_time = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser números enteros.", parent=setup_win)
                return
            if not avenue:
                messagebox.showerror("Error", "Debe ingresar el nombre de la avenida.", parent=setup_win)
                return
            self.set_avenue_for_video(video_path, avenue)
            self.current_avenue = avenue
            self.avenue_label.config(text=self.current_avenue)
            times = {"green": green_time, "yellow": yellow_time, "red": red_time}
            self.set_time_preset_for_video(video_path, times)
            messagebox.showinfo("Éxito", "Configuración inicial guardada.", parent=setup_win)
            setup_win.destroy()
        
        save_button = tk.Button(setup_win, text="Guardar Configuración", command=save_setup)
        save_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        setup_win.transient(self.parent)
        setup_win.grab_set()
        self.parent.wait_window(setup_win)

    # ----- Funciones de manejo del área (antes "polígono") -----
    def on_mouse_click_polygon(self, event):
        if self.have_polygon:
            return  # No se permite trazar si ya se guardó el área
        if self.orig_w is None or self.orig_h is None:
            return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return
        scale = min(wlbl / self.orig_w, hlbl / self.orig_h, 1.0)
        off_x = (wlbl - int(self.orig_w * scale)) // 2
        off_y = (hlbl - int(self.orig_h * scale)) // 2
        cx, cy = event.x, event.y
        x_rel = (cx - off_x) / scale
        y_rel = (cy - off_y) / scale
        self.polygon_points.append((int(x_rel), int(y_rel)))
        print("[DEBUG] Punto agregado al área:", (int(x_rel), int(y_rel)))

    def draw_polygon_on_np(self, bgr_img):
        if not self.polygon_points:
            return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2 or self.orig_w is None or self.orig_h is None:
            return
        scale = min(wlbl / self.orig_w, hlbl / self.orig_h, 1.0)
        off_x = (wlbl - int(self.orig_w * scale)) // 2
        off_y = (hlbl - int(self.orig_h * scale)) // 2
        pts_scaled = [(int(px * scale) + off_x, int(py * scale) + off_y) for (px, py) in self.polygon_points]
        n = len(pts_scaled)
        for i in range(n):
            x1, y1 = pts_scaled[i]
            x2, y2 = pts_scaled[(i + 1) % n]
            cv2.line(bgr_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def save_polygon(self):
        if not self.cap or not self.current_video_path:
            messagebox.showerror("Error", "No hay video cargado para asociar el área.")
            return
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Advertencia", "Área inválida (debe tener al menos 3 vértices).")
            return
        self.have_polygon = True  # Bloquea nuevos trazos hasta eliminar el área
        presets = {}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)
            except Exception as e:
                print(f"[ERROR] Cargando presets: {e}")
                presets = {}
        presets[self.current_video_path] = self.polygon_points
        with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
        if not getattr(self, "_area_saved_once", False):
            messagebox.showinfo("Éxito", "Área guardada para este video.")
            self._area_saved_once = True

    def load_polygon_for_video(self):
        self.have_polygon = False
        self.polygon_points = []
        if not self.current_video_path or not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if self.current_video_path in presets:
                self.polygon_points = presets[self.current_video_path]
                self.have_polygon = True
                print(f"[DEBUG] Área cargada para {self.current_video_path}")
        except Exception as e:
            print(f"[ERROR] Cargando área: {e}")

    def delete_polygon(self):
        if not self.current_video_path or not self.polygon_points:
            messagebox.showwarning("Advertencia", "No hay área guardada para este video.")
            return
        answer = messagebox.askyesno("Confirmar", "¿Borrar el área de este video?")
        if not answer:
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if self.current_video_path in presets:
                del presets[self.current_video_path]
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(presets, f, indent=2)
                if not getattr(self, "_area_deleted_once", False):
                    messagebox.showinfo("Éxito", "Área eliminada.")
                    self._area_deleted_once = True
                self.have_polygon = False
                self.polygon_points = []
            # Si hubiera ventana emergente de gestión, se cierra
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el área: {str(e)}")

    def gestionar_poligonos(self):
        if self.have_polygon:
            messagebox.showinfo("Información", "El área ya está cargada para este video.")
            return
        if not os.path.exists(POLYGON_CONFIG_FILE):
            messagebox.showinfo("Info", "No hay áreas guardadas.")
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception:
            presets = {}
        if not presets:
            messagebox.showinfo("Info", "No hay áreas guardadas.")
            return
        w = tk.Toplevel(self.parent)
        w.title("Áreas Guardadas")
        lb = tk.Listbox(w, width=80)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        for video_path, points in presets.items():
            lb.insert(tk.END, f"{video_path} - Vértices: {points}")
        tk.Button(w, text="Cerrar", command=w.destroy).pack(pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    # ----- Gestión de video -----
    def select_video(self):
        from tkinter import filedialog
        self.stop_video()
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        if path:
            self.load_video(path)

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return
        self.current_video_path = path
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el primer frame.")
            self.cap.release()
            self.cap = None
            return
        self.orig_h, self.orig_w, _ = frame.shape
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.video_fps = fps

        self.running = True
        self.load_polygon_for_video()

        stored_avenue = self.get_avenue_for_video(path)
        stored_times = self.get_time_preset_for_video(path)
        # Si no existe configuración para este video, se ejecuta el primer setup
        if stored_avenue is None or stored_times is None:
            self.first_time_setup(path)
        else:
            self.current_avenue = stored_avenue
            self.avenue_label.config(text=self.current_avenue)
            self.cycle_durations = stored_times
            self.target_time = time.time() + self.cycle_durations[self.semaforo.get_current_state()]
        self.avenue_label.config(text=self.current_avenue)

        # Mostrar mensaje solo si es la primera vez para este video
        if stored_avenue is None:
            messagebox.showinfo("Info", "Se cargó el video.")

        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()

        self.parent.after(0, self.update_frames)

    def first_time_setup(self, video_path):
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
                green_time = int(green_entry.get().strip())
                yellow_time = int(yellow_entry.get().strip())
                red_time = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser números enteros.", parent=setup_win)
                return
            if not avenue:
                messagebox.showerror("Error", "Debe ingresar el nombre de la avenida.", parent=setup_win)
                return
            self.set_avenue_for_video(video_path, avenue)
            self.current_avenue = avenue
            self.avenue_label.config(text=self.current_avenue)
            times = {"green": green_time, "yellow": yellow_time, "red": red_time}
            self.set_time_preset_for_video(video_path, times)
            messagebox.showinfo("Éxito", "Configuración inicial guardada.", parent=setup_win)
            setup_win.destroy()
        
        save_button = tk.Button(setup_win, text="Guardar Configuración", command=save_setup)
        save_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        setup_win.transient(self.parent)
        setup_win.grab_set()
        self.parent.wait_window(setup_win)

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # ----- Procesamiento de placas -----
    def plate_loop(self):
        while self.plate_running:
            try:
                frame, roi = self.plate_queue.get(timeout=1)
            except queue.Empty:
                continue
            bbox, plate_sr, ocr_text = process_plate(roi)
            if ocr_text:
                stamp = int(time.time())
                fname = f"plate_{ocr_text}_{stamp}.jpg"
                os.makedirs("data/output", exist_ok=True)
                cv2.imwrite(os.path.join("data", "output", fname), plate_sr)
                print("[DEBUG] Placa guardada:", fname)
                self._safe_add_plate_to_panel(plate_sr, ocr_text)
            self.plate_queue.task_done()

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.parent.after(30, self.update_frames)
            return

        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()

        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))

        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        self.info_label.lift()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def _safe_add_plate_to_panel(self, plate_bgr, plate_text):
        def add_later():
            self.add_plate_to_panel(plate_bgr, plate_text)
        self.parent.after(0, add_later)

    def add_plate_to_panel(self, plate_img_bgr, plate_text):
        plate_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(plate_rgb)
        max_w = 200
        ratio = pil_img.width / pil_img.height if pil_img.height > 0 else 1
        new_w = max_w
        new_h = int(new_w / ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(pil_img)
        f_item = tk.Frame(self.plates_inner_frame, bg="gray")
        f_item.pack(pady=5, fill="x")
        lbl_img = tk.Label(f_item, image=tk_img, bg="gray")
        lbl_img.image = tk_img
        lbl_img.pack(side="left", padx=5)
        lbl_txt = tk.Label(f_item, text=plate_text, bg="gray", fg="white", font=("Arial", 11, "bold"))
        lbl_txt.pack(side="left", padx=5)
        self.detected_plates_widgets.append((f_item, lbl_img, lbl_txt))
        if len(self.detected_plates_widgets) > 5:
            oldest = self.detected_plates_widgets.pop(0)
            oldest[0].destroy()

    def gestionar_camaras(self):
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")
        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        sb.config(command=lb.yview)
        video_folder = "videos"
        files = []
        if os.path.exists(video_folder):
            for f in os.listdir(video_folder):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    files.append(f)
        files.sort()
        for f in files:
            lb.insert(tk.END, f)
        fbtn = tk.Frame(w)
        fbtn.pack(fill="x")
        tk.Button(fbtn, text="Cargar", width=10,
                  command=lambda: [self._cam_load(lb, video_folder), w.destroy()]).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Borrar", width=10,
                  command=lambda: self._cam_del(lb, video_folder)).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Cerrar", width=10,
                  command=w.destroy).pack(side="left", padx=5, pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def _cam_load(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        self.stop_video()
        # Si el video ya está configurado (ya se cargó antes) se informa
        if self.get_avenue_for_video(path) is not None:
            messagebox.showinfo("Info", "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.")
            return
        self.load_video(path)
        if self.get_avenue_for_video(path) is None:
            messagebox.showinfo("Info", "Se cargó el video.")

    def _cam_del(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video para borrar.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fname}' de '{folder}'?"):
            return
        try:
            os.remove(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"Se eliminó el video '{fname}' y sus datos de configuración.")
            self.remove_video_data(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def remove_video_data(self, video_path):
        self.remove_avenue_data(video_path)
        self.remove_time_preset_data(video_path)
        self.remove_polygon_data(video_path)

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando datos de avenida para '{video_path}': {e}")

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r") as f:
                presets = json.load(f)
            if video_path in presets:
                del presets[video_path]
                with open(PRESETS_FILE, "w") as fw:
                    json.dump(presets, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando preset de tiempos para '{video_path}': {e}")

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            if video_path in polygons:
                del polygons[video_path]
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(polygons, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando área para '{video_path}': {e}")

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.parent.after(30, self.update_frames)
            return

        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()

        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))

        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        self.info_label.lift()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def _safe_add_plate_to_panel(self, plate_bgr, plate_text):
        def add_later():
            self.add_plate_to_panel(plate_bgr, plate_text)
        self.parent.after(0, add_later)

    def add_plate_to_panel(self, plate_img_bgr, plate_text):
        plate_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(plate_rgb)
        max_w = 200
        ratio = pil_img.width / pil_img.height if pil_img.height > 0 else 1
        new_w = max_w
        new_h = int(new_w / ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(pil_img)
        f_item = tk.Frame(self.plates_inner_frame, bg="gray")
        f_item.pack(pady=5, fill="x")
        lbl_img = tk.Label(f_item, image=tk_img, bg="gray")
        lbl_img.image = tk_img
        lbl_img.pack(side="left", padx=5)
        lbl_txt = tk.Label(f_item, text=plate_text, bg="gray", fg="white", font=("Arial", 11, "bold"))
        lbl_txt.pack(side="left", padx=5)
        self.detected_plates_widgets.append((f_item, lbl_img, lbl_txt))
        if len(self.detected_plates_widgets) > 5:
            oldest = self.detected_plates_widgets.pop(0)
            oldest[0].destroy()

    def gestionar_camaras(self):
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")
        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        sb.config(command=lb.yview)
        video_folder = "videos"
        files = []
        if os.path.exists(video_folder):
            for f in os.listdir(video_folder):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    files.append(f)
        files.sort()
        for f in files:
            lb.insert(tk.END, f)
        fbtn = tk.Frame(w)
        fbtn.pack(fill="x")
        tk.Button(fbtn, text="Cargar", width=10,
                  command=lambda: [self._cam_load(lb, video_folder), w.destroy()]).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Borrar", width=10,
                  command=lambda: self._cam_del(lb, video_folder)).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Cerrar", width=10,
                  command=w.destroy).pack(side="left", padx=5, pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def _cam_load(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        self.stop_video()
        # Si ya existe configuración para este video, se informa y se impide cargarlo desde "Cargar"
        if self.get_avenue_for_video(path) is not None:
            messagebox.showinfo("Info", "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.")
            return
        self.load_video(path)
        if self.get_avenue_for_video(path) is None:
            messagebox.showinfo("Info", "Se cargó el video.")

    def _cam_del(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video para borrar.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fname}' de '{folder}'?"):
            return
        try:
            os.remove(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"Se eliminó el video '{fname}' y sus datos de configuración.")
            self.remove_video_data(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def remove_video_data(self, video_path):
        self.remove_avenue_data(video_path)
        self.remove_time_preset_data(video_path)
        self.remove_polygon_data(video_path)

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando datos de avenida para '{video_path}': {e}")

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r") as f:
                presets = json.load(f)
            if video_path in presets:
                del presets[video_path]
                with open(PRESETS_FILE, "w") as fw:
                    json.dump(presets, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando preset de tiempos para '{video_path}': {e}")

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            if video_path in polygons:
                del polygons[video_path]
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(polygons, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando área para '{video_path}': {e}")

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.parent.after(30, self.update_frames)
            return

        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()

        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))

        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        self.info_label.lift()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def _safe_add_plate_to_panel(self, plate_bgr, plate_text):
        def add_later():
            self.add_plate_to_panel(plate_bgr, plate_text)
        self.parent.after(0, add_later)

    def add_plate_to_panel(self, plate_img_bgr, plate_text):
        plate_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(plate_rgb)
        max_w = 200
        ratio = pil_img.width / pil_img.height if pil_img.height > 0 else 1
        new_w = max_w
        new_h = int(new_w / ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(pil_img)
        f_item = tk.Frame(self.plates_inner_frame, bg="gray")
        f_item.pack(pady=5, fill="x")
        lbl_img = tk.Label(f_item, image=tk_img, bg="gray")
        lbl_img.image = tk_img
        lbl_img.pack(side="left", padx=5)
        lbl_txt = tk.Label(f_item, text=plate_text, bg="gray", fg="white", font=("Arial", 11, "bold"))
        lbl_txt.pack(side="left", padx=5)
        self.detected_plates_widgets.append((f_item, lbl_img, lbl_txt))
        if len(self.detected_plates_widgets) > 5:
            oldest = self.detected_plates_widgets.pop(0)
            oldest[0].destroy()

    def gestionar_camaras(self):
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")
        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        sb.config(command=lb.yview)
        video_folder = "videos"
        files = []
        if os.path.exists(video_folder):
            for f in os.listdir(video_folder):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    files.append(f)
        files.sort()
        for f in files:
            lb.insert(tk.END, f)
        fbtn = tk.Frame(w)
        fbtn.pack(fill="x")
        tk.Button(fbtn, text="Cargar", width=10,
                  command=lambda: [self._cam_load(lb, video_folder), w.destroy()]).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Borrar", width=10,
                  command=lambda: self._cam_del(lb, video_folder)).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Cerrar", width=10,
                  command=w.destroy).pack(side="left", padx=5, pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def _cam_load(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        self.stop_video()
        # Si ya existe configuración para este video, se informa y se impide cargarlo desde "Cargar"
        if self.get_avenue_for_video(path) is not None:
            messagebox.showinfo("Info", "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.")
            return
        self.load_video(path)
        if self.get_avenue_for_video(path) is None:
            messagebox.showinfo("Info", "Se cargó el video.")

    def _cam_del(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video para borrar.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fname}' de '{folder}'?"):
            return
        try:
            os.remove(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"Se eliminó el video '{fname}' y sus datos de configuración.")
            self.remove_video_data(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def remove_video_data(self, video_path):
        self.remove_avenue_data(video_path)
        self.remove_time_preset_data(video_path)
        self.remove_polygon_data(video_path)

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando datos de avenida para '{video_path}': {e}")

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r") as f:
                presets = json.load(f)
            if video_path in presets:
                del presets[video_path]
                with open(PRESETS_FILE, "w") as fw:
                    json.dump(presets, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando preset de tiempos para '{video_path}': {e}")

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            if video_path in polygons:
                del polygons[video_path]
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(polygons, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando área para '{video_path}': {e}")

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.parent.after(30, self.update_frames)
            return

        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()

        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))

        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        self.info_label.lift()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def _safe_add_plate_to_panel(self, plate_bgr, plate_text):
        def add_later():
            self.add_plate_to_panel(plate_bgr, plate_text)
        self.parent.after(0, add_later)

    def add_plate_to_panel(self, plate_img_bgr, plate_text):
        plate_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(plate_rgb)
        max_w = 200
        ratio = pil_img.width / pil_img.height if pil_img.height > 0 else 1
        new_w = max_w
        new_h = int(new_w / ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(pil_img)
        f_item = tk.Frame(self.plates_inner_frame, bg="gray")
        f_item.pack(pady=5, fill="x")
        lbl_img = tk.Label(f_item, image=tk_img, bg="gray")
        lbl_img.image = tk_img
        lbl_img.pack(side="left", padx=5)
        lbl_txt = tk.Label(f_item, text=plate_text, bg="gray", fg="white", font=("Arial", 11, "bold"))
        lbl_txt.pack(side="left", padx=5)
        self.detected_plates_widgets.append((f_item, lbl_img, lbl_txt))
        if len(self.detected_plates_widgets) > 5:
            oldest = self.detected_plates_widgets.pop(0)
            oldest[0].destroy()

    def gestionar_camaras(self):
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")
        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)
        sb.config(command=lb.yview)
        video_folder = "videos"
        files = []
        if os.path.exists(video_folder):
            for f in os.listdir(video_folder):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    files.append(f)
        files.sort()
        for f in files:
            lb.insert(tk.END, f)
        fbtn = tk.Frame(w)
        fbtn.pack(fill="x")
        tk.Button(fbtn, text="Cargar", width=10,
                  command=lambda: [self._cam_load(lb, video_folder), w.destroy()]).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Borrar", width=10,
                  command=lambda: self._cam_del(lb, video_folder)).pack(side="left", padx=5, pady=5)
        tk.Button(fbtn, text="Cerrar", width=10,
                  command=w.destroy).pack(side="left", padx=5, pady=5)
        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)

    def _cam_load(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        self.stop_video()
        self.load_video(path)
        if self.get_avenue_for_video(path) is None:
            messagebox.showinfo("Info", "Se cargó el video.")

    def _cam_del(self, lb, folder):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un video para borrar.")
            return
        fname = lb.get(sel[0])
        path = os.path.join(folder, fname)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fname}' de '{folder}'?"):
            return
        try:
            os.remove(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"Se eliminó el video '{fname}' y sus datos de configuración.")
            self.remove_video_data(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def remove_video_data(self, video_path):
        self.remove_avenue_data(video_path)
        self.remove_time_preset_data(video_path)
        self.remove_polygon_data(video_path)

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if video_path in data:
                del data[video_path]
                with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando datos de avenida para '{video_path}': {e}")

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r") as f:
                presets = json.load(f)
            if video_path in presets:
                del presets[video_path]
                with open(PRESETS_FILE, "w") as fw:
                    json.dump(presets, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando preset de tiempos para '{video_path}': {e}")

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            if video_path in polygons:
                del polygons[video_path]
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                    json.dump(polygons, fw, indent=2)
        except Exception as e:
            print(f"[ERROR] Borrando área para '{video_path}': {e}")

    def update_frames(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.parent.after(30, self.update_frames)
            return

        if self.polygon_points:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame.copy()

        if self.semaforo.get_current_state() == "red":
            if not self.plate_queue.full():
                self.plate_queue.put((frame.copy(), roi))

        bgr_img = self.resize_and_letterbox(frame)
        if self.polygon_points:
            self.draw_polygon_on_np(bgr_img)
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        self.info_label.lift()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.parent.after(10, self.update_frames)

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

# Fin del módulo VideoPlayerOpenCV
