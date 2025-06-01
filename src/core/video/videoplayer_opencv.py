# src/video/videoplayer_opencv.py

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

from src.core.detection.plate_detector import PlateDetector
from src.core.detection.vehicle_detector import VehicleDetector
from src.core.processing.plate_processing import process_plate

# Archivos de configuración
POLYGON_CONFIG_FILE = "config/polygon_config.json"
AVENUE_CONFIG_FILE  = "config/avenue_config.json"
PRESETS_FILE        = "config/time_presets.json"

from src.gui.preprocessing_dialog import PreprocessingDialog

class VideoPlayerOpenCV:
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo):
        self.parent            = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label   = timestamp_label
        self.semaforo          = semaforo

        self.yolo = YOLO('models/yolov8n.pt')      # peso pequeño, pre-entrenado en COCO
        self.CAR_CLASS_ID = 2               # en COCO, 'car' = 2
        self.CONF_THRESH   = 0.4

        # Variables para métricas
        self.detected_plates_widgets = []
        self.seen_plates = set()
        
        # Variables para métricas
        self.detection_start_time = time.time()
        self.registration_times = []
        self.plate_detection_history = {}

        # Configuración CUDA para mejor rendimiento
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True  # Optimiza operaciones para tamaños de imagen fijos
            torch.backends.cudnn.deterministic = False  # Permite optimizaciones no deterministas
            print("Usando GPU para aceleración")
        else:
            self.device = torch.device('cpu')
            print("GPU no disponible, usando CPU")

        # Directorio de vídeos
        self.video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # Contenedor principal
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill="both", expand=True)

        # Botonera inferior
        self.btn_frame = tk.Frame(self.frame, bg="black")
        self.btn_frame.pack(side="bottom", pady=12, anchor="w")

        btn_style = {
            "font": ("Arial", 12),
            "bg": "#34495e",
            "fg": "white",
            "activebackground": "#34495e",
            "activeforeground": "white",
            "bd": 0,
            "relief": "flat",
            "cursor": "hand2",
            "width": 36,
            "anchor": "center",
            "justify": "center"
        }

        self.load_button = tk.Button(
            self.btn_frame, text="CARGAR\nVIDEO",
            command=self.select_video,
            **btn_style
        )
        self.load_button.pack(side="left", padx=10)

        self.btn_gestion_camaras = tk.Button(
            self.btn_frame, text="PREPROCESAMIENTO\nDE VIDEO",
            command=self.gestionar_camaras,
            **btn_style
        )
        self.btn_gestion_camaras.pack(side="left", padx=10)

        # Panel vídeo + lateral
        self.video_panel_container = tk.Frame(self.frame, bg='black')
        self.video_panel_container.pack(side="top", fill="both", expand=True)

        self.video_frame = tk.Frame(
            self.video_panel_container, bg='black',
            width=640, height=360
        )
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(
            self.video_frame, bg="black", bd=0, highlightthickness=0
        )
        self.video_label.pack(fill="both", expand=True)

        # CORRECCIÓN: Eliminar código duplicado en la configuración del panel de placas
        self.plates_frame = tk.Frame(
            self.video_panel_container, bg="#34495e", width=320
        )
        self.plates_frame.pack(side="right", fill="y")
        self.plates_frame.pack_propagate(False)

        self.plates_title = tk.Label(
            self.plates_frame, text="Placas Detectadas",
            bg="#2c3e50", fg="white", font=("Arial", 16, "bold"),
            pady=10
        )
        self.plates_title.pack(fill="x")

        # Panel de métricas primero
        self._create_metrics_panel()

        # Configuración del canvas y scrollbar - IMPLEMENTACIÓN LIMPIA
        self.plates_canvas = tk.Canvas(
            self.plates_frame, bg="#ecf0f1", highlightthickness=0
        )
        self.plates_canvas.pack(side="left", fill="both", expand=True)

        self.plates_scrollbar = tk.Scrollbar(
            self.plates_frame, orient="vertical",
            command=self.plates_canvas.yview,
            bg="#7f8c8d", troughcolor="#bdc3c7", bd=0
        )
        self.plates_scrollbar.pack(side="right", fill="y")
        self.plates_canvas.configure(yscrollcommand=self.plates_scrollbar.set)

        # IMPORTANTE: Crear un solo frame interno para contener las cards
        self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="#ecf0f1")
        
        # CRÍTICO: Crear una sola ventana de canvas
        self.plates_canvas_window = self.plates_canvas.create_window(
            (0, 0), window=self.plates_inner_frame, anchor="nw",
            width=self.plates_canvas.winfo_width()
        )
        
        # Configurar eventos para actualizar correctamente el canvas
        self.plates_inner_frame.bind("<Configure>", self._on_plates_inner_configure)
        self.plates_canvas.bind("<Configure>", self._on_plates_canvas_configure)
        
        # Inicializar variables para las placas detectadas
        self.detected_plates_widgets = []
        self.seen_plates = set()

        # Timestamp y avenida
        self.timestamp_label.config(
            font=("Arial",30,"bold"), bg="black", fg="yellow"
        )
        self.timestamp_label.place(in_=self.video_label, x=50, y=10)

        self.current_avenue = None
        self.avenue_label = tk.Label(
            self.video_frame, text="", font=("Arial",20,"bold"),
            bg="black", fg="white", wraplength=300
        )
        self.avenue_label.place(relx=0.5, y=80, anchor="n")

        # Info CPU/FPS/RAM
        self.info_label = tk.Label(
            self.video_frame, text="...", bg="black",
            fg="white", font=("Arial",11,"bold")
        )
        self.info_label.place(relx=0.98, y=10, anchor="ne")

        # Estado
        self.cap                = None
        self.running            = False
        self.orig_w, self.orig_h= None, None
        self.polygon_points     = []
        self.have_polygon       = False
        self.current_video_path = None

        # Cola acotada de OCR
        self.plate_queue   = queue.Queue(maxsize=1)
        self.plate_running = True
        self.plate_thread  = threading.Thread(
            target=self.plate_loop, daemon=True
        )
        self.plate_thread.start()

        # Métricas
        self.last_time = time.time()
        self.fps_calc  = 0.0
        self.using_gpu = torch.cuda.is_available()

        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(4)
        except:
            pass

        self.video_label.bind(
            "<Button-1>", self.on_mouse_click_polygon
        )

    def _on_plates_inner_configure(self, event):
        self.plates_canvas.configure(
            scrollregion=self.plates_canvas.bbox("all")
        )

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
        return self.load_avenue_config().get(video_path)

    def set_avenue_for_video(self, video_path, avenue_name):
        cfg = self.load_avenue_config()
        cfg[video_path] = avenue_name
        self.save_avenue_config(cfg)

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

    def get_time_preset_for_video(self, video_path):
        return self.load_time_presets().get(video_path)

    def set_time_preset_for_video(self, video_path, times):
        presets = self.load_time_presets()
        presets[video_path] = times
        self.save_time_presets(presets)
        self.cycle_durations = times
        self.target_time     = time.time() + times[self.semaforo.get_current_state()]

    def first_time_setup(self, video_path):
        if ( self.get_avenue_for_video(video_path) is not None and
             self.get_time_preset_for_video(video_path) is not None ):
            messagebox.showinfo(
                "Info",
                "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.",
                parent=self.parent
            )
            return

        setup = tk.Toplevel(self.parent)
        setup.title("Configuración Inicial del Video")

        tk.Label(setup, text="Nombre de la Avenida:")\
          .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        avenue_entry = tk.Entry(setup, width=30)
        avenue_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Verde (s):")\
          .grid(row=1, column=0, sticky="w", padx=5, pady=5)
        green_entry = tk.Entry(setup, width=10)
        green_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Amarillo (s):")\
          .grid(row=2, column=0, sticky="w", padx=5, pady=5)
        yellow_entry = tk.Entry(setup, width=10)
        yellow_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Rojo (s):")\
          .grid(row=3, column=0, sticky="w", padx=5, pady=5)
        red_entry = tk.Entry(setup, width=10)
        red_entry.grid(row=3, column=1, padx=5, pady=5)

        def guardar():
            ave = avenue_entry.get().strip()
            try:
                g = int(green_entry.get().strip())
                y = int(yellow_entry.get().strip())
                r = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror(
                    "Error", "Los tiempos deben ser enteros.", parent=setup
                )
                return
            if not ave:
                messagebox.showerror(
                    "Error", "Debe ingresar nombre de avenida.", parent=setup
                )
                return
            self.set_avenue_for_video(video_path, ave)
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.set_time_preset_for_video(video_path, {"green":g,"yellow":y,"red":r})
            messagebox.showinfo("Éxito","Configuración guardada.",parent=setup)
            setup.destroy()

        tk.Button(setup, text="Guardar Configuración", command=guardar)\
          .grid(row=4, column=0, columnspan=2,pady=10)

        setup.transient(self.parent)
        setup.grab_set()
        self.parent.wait_window(setup)

    def on_mouse_click_polygon(self, event):
        if self.have_polygon or self.orig_w is None:
            return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl<2 or hlbl<2: return
        scale = min(wlbl/self.orig_w, hlbl/self.orig_h, 1.0)
        off_x = (wlbl - int(self.orig_w*scale))//2
        off_y = (hlbl - int(self.orig_h*scale))//2
        x_rel = (event.x - off_x)/scale
        y_rel = (event.y - off_y)/scale
        self.polygon_points.append((int(x_rel),int(y_rel)))

    def draw_polygon_on_np(self, img):
        if not self.polygon_points: return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl<2 or hlbl<2: return
        scale = min(wlbl/self.orig_w, hlbl/self.orig_h, 1.0)
        off_x=(wlbl-int(self.orig_w*scale))//2
        off_y=(hlbl-int(self.orig_h*scale))//2
        pts_scaled=[(int(px*scale)+off_x,int(py*scale)+off_y)
                    for px,py in self.polygon_points]
        for i in range(len(pts_scaled)):
            x1,y1=pts_scaled[i]
            x2,y2=pts_scaled[(i+1)%len(pts_scaled)]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    def save_polygon(self):
        if not self.cap or not self.current_video_path:
            messagebox.showerror("Error","No hay vídeo cargado.")
            return
        if len(self.polygon_points)<3:
            messagebox.showwarning("Advertencia","Al menos 3 vértices.")
            return
        self.have_polygon=True
        presets={}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                    presets=json.load(f)
            except: pass
        presets[self.current_video_path]=self.polygon_points
        with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
            json.dump(presets,f,indent=2)
        messagebox.showinfo("Éxito","Área guardada.")

    def load_polygon_for_video(self):
        self.have_polygon=False
        self.polygon_points=[]
        if not self.current_video_path or not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets=json.load(f)
            if self.current_video_path in presets:
                self.polygon_points=presets[self.current_video_path]
                self.have_polygon=True
        except: pass

    def delete_polygon(self):
        if not self.current_video_path or not self.polygon_points:
            messagebox.showwarning("Advertencia","No hay área.")
            return
        if not messagebox.askyesno("Confirmar","¿Borrar área?"):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets=json.load(f)
            presets.pop(self.current_video_path,None)
            with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
                json.dump(presets,f,indent=2)
            self.have_polygon=False
            self.polygon_points=[]
            messagebox.showinfo("Éxito","Área eliminada.")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def gestionar_poligonos(self):
        w = tk.Toplevel(self.parent)
        w.title("Áreas Guardadas")

        lb = tk.Listbox(w, width=80)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        # Cargar presets de áreas
        presets = {}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)
            except Exception:
                presets = {}

        # Poblar listbox
        for video_path, points in presets.items():
            lb.insert(tk.END, f"{video_path} → {points}")

        # Botón de cierre
        tk.Button(w, text="Cerrar", command=w.destroy).pack(pady=5)

        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)


    def select_video(self):
        """
        Permite seleccionar un video, configurar tiempos de semáforo y área restringida.
        """
        from tkinter import filedialog
        file = filedialog.askopenfilename(
            title="Seleccionar vídeo",
            filetypes=[("Vídeos","*.mp4 *.avi *.mov *.mkv"),("Todos","*.*")]
        )
        if not file:
            return
        
        fname = os.path.basename(file)
        dest = os.path.join(self.video_dir, fname)
        
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(file, dest)
        
        # Verificar si el video ya está configurado
        if (self.get_avenue_for_video(dest) is not None and 
            self.get_time_preset_for_video(dest) is not None and
            self.check_polygon_exists(dest)):
            
            # Video ya configurado, cargar directamente
            self._load_video_async(dest)
            return

        # Abrir vista previa del video para ayudar con la configuración
        cap_tmp = cv2.VideoCapture(dest)
        ret, preview_frame = cap_tmp.read()
        cap_tmp.release()
        
        if not ret:
            messagebox.showerror("Error", "No se pudo abrir el video para configuración.")
            return
        
        # Configuración inicial completa: pantalla combinada para semáforo y área
        self.setup_complete_video_config(dest, preview_frame)

    def check_polygon_exists(self, video_path):
        """Verifica si ya existe polígono definido para este video"""
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return False
        
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            return video_path in presets and len(presets[video_path]) >= 3
        except:
            return False

    def setup_complete_video_config(self, video_path, preview_frame):
        """
        Diálogo integrado para configuración completa de video:
        - Configuración de semáforo (tiempos)
        - Nombre de avenida
        - Franja horaria
        - Definición de área restringida
        - Todo en una misma ventana
        """
        setup = tk.Toplevel(self.parent)
        setup.title("Configuración Inicial del Video")
        setup.geometry("940x650")  # Un poco más alta para el nuevo campo
        setup.resizable(True, True)
        
        # Layouts principales
        main_frame = tk.Frame(setup)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Panel izquierdo - configuración de semáforo
        config_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        config_frame.pack(side="left", fill="both", padx=5, pady=5)
        
        # Título para configuración
        tk.Label(config_frame, text="Configuración del Semáforo", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Contenedor para entradas
        fields_frame = tk.Frame(config_frame)
        fields_frame.pack(fill="x", padx=20, pady=10)
        
        # Campos para avenida y tiempos
        tk.Label(fields_frame, text="Nombre de la Avenida:").grid(
            row=0, column=0, sticky="w", padx=5, pady=8)
        avenue_entry = tk.Entry(fields_frame, width=30)
        avenue_entry.grid(row=0, column=1, padx=5, pady=8)
        
        # NUEVO: Campo para franja horaria
        tk.Label(fields_frame, text="Franja Horaria:").grid(
            row=1, column=0, sticky="w", padx=5, pady=8)
        time_slot_entry = tk.Entry(fields_frame, width=30)
        time_slot_entry.grid(row=1, column=1, padx=5, pady=8)
        time_slot_entry.insert(0, "7:00 - 19:00")  # Valor predeterminado
        
        # Tiempos del semáforo (indices ajustados)
        tk.Label(fields_frame, text="Tiempo Verde (s):").grid(
            row=2, column=0, sticky="w", padx=5, pady=8)
        green_entry = tk.Entry(fields_frame, width=10)
        green_entry.grid(row=2, column=1, padx=5, pady=8)
        green_entry.insert(0, "30")  # Valor predeterminado
        
        tk.Label(fields_frame, text="Tiempo Amarillo (s):").grid(
            row=3, column=0, sticky="w", padx=5, pady=8)
        yellow_entry = tk.Entry(fields_frame, width=10)
        yellow_entry.grid(row=3, column=1, padx=5, pady=8)
        yellow_entry.insert(0, "5")  # Valor predeterminado
        
        tk.Label(fields_frame, text="Tiempo Rojo (s):").grid(
            row=4, column=0, sticky="w", padx=5, pady=8)
        red_entry = tk.Entry(fields_frame, width=10)
        red_entry.grid(row=4, column=1, padx=5, pady=8)
        red_entry.insert(0, "30")  # Valor predeterminado
        
        # Panel derecho - previsualización y área restringida
        preview_frame_container = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        preview_frame_container.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Título para área restringida
        tk.Label(preview_frame_container, text="Definición de Área Restringida", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Instrucciones para el usuario
        instructions = tk.Label(preview_frame_container, 
                                text="Haga clic en la imagen para definir los vértices del área restringida.\n"
                                    "Se requieren al menos 3 puntos para definir un área válida.",
                                wraplength=450)
        instructions.pack(pady=5)
        
        # Preparar imagen para visualización
        h, w = preview_frame.shape[:2]
        max_preview_w, max_preview_h = 500, 350
        
        scale = min(max_preview_w/w, max_preview_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        preview_resized = cv2.resize(preview_frame, (new_w, new_h))
        preview_rgb = cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB)
        
        # Canvas para dibujar los puntos del polígono
        canvas = tk.Canvas(preview_frame_container, width=new_w, height=new_h, 
                        highlightthickness=1, highlightbackground="gray")
        canvas.pack(pady=10)
        
        # Mostrar imagen en canvas
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(preview_rgb))
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk
        
        # Variables para polígono
        polygon_points = []
        polygon_canvas_items = []
        
        # Estado del polígono
        status_var = tk.StringVar()
        status_var.set("Estado: No se ha definido área restringida")
        status_label = tk.Label(preview_frame_container, textvariable=status_var, fg="red")
        status_label.pack(pady=5)
        
        def on_canvas_click(event):
            """Maneja clicks en el canvas para crear polígono"""
            # Convertir a escala original
            x_real = int(event.x / scale)
            y_real = int(event.y / scale)
            
            # Añadir punto al polígono
            polygon_points.append((x_real, y_real))
            
            # Dibujar punto en canvas
            point_id = canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, 
                                        fill="red", outline="white", tags="polygon")
            polygon_canvas_items.append(point_id)
            
            # Dibujar línea al punto anterior
            if len(polygon_points) > 1:
                prev_x = polygon_points[-2][0] * scale
                prev_y = polygon_points[-2][1] * scale
                line_id = canvas.create_line(prev_x, prev_y, event.x, event.y, 
                                        fill="yellow", width=2, tags="polygon")
                polygon_canvas_items.append(line_id)
                
                # Si hay suficientes puntos, añadir línea temporal de cierre
                if len(polygon_points) > 2:
                    # Borrar línea de cierre anterior si existe
                    canvas.delete("closing_line")
                    
                    # Dibujar nueva línea de cierre
                    first_x = polygon_points[0][0] * scale
                    first_y = polygon_points[0][1] * scale
                    close_id = canvas.create_line(event.x, event.y, first_x, first_y, 
                                            fill="yellow", width=2, dash=(5,2), 
                                            tags=("polygon", "closing_line"))
                    polygon_canvas_items.append(close_id)
            
            # Actualizar estado
            if len(polygon_points) >= 3:
                status_var.set(f"Estado: Área definida con {len(polygon_points)} puntos")
                status_label.config(fg="green")
            else:
                status_var.set(f"Estado: Definiendo área ({len(polygon_points)}/3 puntos mínimos)")
        
        def clear_polygon():
            """Limpia todos los puntos del polígono"""
            polygon_points.clear()
            for item_id in polygon_canvas_items:
                canvas.delete(item_id)
            polygon_canvas_items.clear()
            status_var.set("Estado: No se ha definido área restringida")
            status_label.config(fg="red")
        
        # Enlazar eventos
        canvas.bind("<Button-1>", on_canvas_click)
        
        # Botón para limpiar polígono
        clear_button = tk.Button(preview_frame_container, text="Borrar Puntos", 
                            command=clear_polygon)
        clear_button.pack(pady=5)
        
        # Panel inferior con botones de acción
        button_frame = tk.Frame(setup)
        button_frame.pack(fill="x", pady=15)
        
        def guardar_configuracion():
            """Guarda la configuración completa"""
            # Validar campos de semáforo
            ave = avenue_entry.get().strip()
            time_slot = time_slot_entry.get().strip()  # Nuevo campo
            
            try:
                g = int(green_entry.get().strip())
                y = int(yellow_entry.get().strip())
                r = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser números enteros", parent=setup)
                return
            
            if not ave:
                messagebox.showerror("Error", "Debe ingresar un nombre para la avenida", parent=setup)
                return
            
            if not time_slot:
                messagebox.showerror("Error", "Debe ingresar una franja horaria", parent=setup)
                return
            
            # Validar polígono
            if len(polygon_points) < 3:
                resp = messagebox.askyesno("Advertencia", 
                                "No se ha definido un área restringida válida.\n"
                                "¿Desea continuar sin definir un área?", 
                                parent=setup)
                if not resp:
                    return
            
            # Guardar configuración del semáforo
            self.set_avenue_for_video(video_path, ave)
            self.current_avenue = ave 
            self.avenue_label.config(text=ave)
            
            # Guardar configuración con franja horaria incluida
            self.set_time_preset_for_video(video_path, {
                "green": g, 
                "yellow": y, 
                "red": r,
                "time_slot": time_slot  # Guardar franja horaria
            })
            
            # Guardar polígono si existe
            if len(polygon_points) >= 3:
                self.polygon_points = polygon_points
                self.have_polygon = True
                
                # Guardar en archivo de configuración
                presets = {}
                if os.path.exists(POLYGON_CONFIG_FILE):
                    try:
                        with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                            presets = json.load(f)
                    except:
                        pass
                presets[video_path] = polygon_points
                with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(presets, f, indent=2)
            
            # Cerrar diálogo y cargar video
            setup.destroy()
            self._load_video_async(video_path)
        
        # Botones finales
        guardar_btn = tk.Button(button_frame, text="Guardar y Cargar Video", 
                            command=guardar_configuracion,
                            bg="#4CAF50", fg="white", font=("Arial", 11))
        guardar_btn.pack(side="right", padx=10)
        
        cancelar_btn = tk.Button(button_frame, text="Cancelar", 
                            command=setup.destroy,
                            bg="#f44336", fg="white", font=("Arial", 11))
        cancelar_btn.pack(side="right", padx=10)
        
        # Hacer la ventana modal
        setup.transient(self.parent)
        setup.grab_set()
        setup.wait_window()

    def _load_video_async(self, path):
        cap_tmp = cv2.VideoCapture(path)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            self.parent.after(0, lambda: messagebox.showerror("Error", "No se pudo leer el vídeo."))
            return
        self.parent.after(0, lambda: self._finish_loading_video(path, frame))

    def _finish_loading_video(self, path, first_frame):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_video_path = path
        h, w = first_frame.shape[:2]
        self.orig_h, self.orig_w = h, w
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
        self.running = True
        self.load_polygon_for_video()
        self.clear_detected_plates()
        
        # Configurar y activar semáforo
        self.semaforo.current_state = "green"
        
        ave = self.get_avenue_for_video(path)
        times = self.get_time_preset_for_video(path)
        if ave is None or times is None:
            self.first_time_setup(path)
        else:
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.cycle_durations = times
            
            # Actualizar el semáforo con la configuración
            self.semaforo.cycle_durations = {
                "green": times["green"],
                "yellow": times["yellow"],
                "red": times["red"]
            }
            self.semaforo.target_time = time.time() + self.semaforo.cycle_durations[self.semaforo.current_state]
            
            # Activar el semáforo
            self.semaforo.activate_semaphore()
            
        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()
        self.update_frames()

    def load_video(self, path):
        """
        Carga un video y realiza el análisis de infracciones sin reproducirlo por completo
        """
        def on_preprocessing_complete(success, infractions=None):
            """Función que se ejecuta cuando finaliza el preprocesamiento"""
            if success:
                # No cargar el video completo, solo mostrar el mensaje de éxito
                print(f"Análisis completado: {len(infractions) if infractions else 0} infracciones detectadas")
                
                # Si queremos cargar la primera imagen del video como vista previa
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Mostrar el primer frame como vista previa estática
                    frame_with_poly = frame.copy()
                    if self.polygon_points:
                        pts = np.array(self.polygon_points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame_with_poly, [pts], True, (0, 0, 255), 2)
                    
                    # Mostrar en la interfaz
                    bgr_img = self.resize_and_letterbox(frame_with_poly)
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk
                    
                    # Cargar datos del video
                    self.current_video_path = path
                    h, w = frame.shape[:2]
                    self.orig_h, self.orig_w = h, w
                    
                    # Establecer información del video
                    ave = self.get_avenue_for_video(path)
                    if ave:
                        self.current_avenue = ave
                        self.avenue_label.config(text=ave)
            else:
                messagebox.showinfo("Procesamiento cancelado", "El análisis del video fue cancelado.")
        
        # Iniciar el diálogo de preprocesamiento
        PreprocessingDialog(self.parent, path, self, on_preprocessing_complete)

    def stop_video(self):
        self.running = False
        if hasattr(self, "_after_id") and self._after_id:
            self.parent.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Desactivar el semáforo cuando se detiene el video
        self.semaforo.deactivate_semaphore()

    def plate_loop(self):
        while self.plate_running:
            try:
                # Ahora recibimos también el flag de noche y el timestamp
                if hasattr(self, 'plate_queue') and not self.plate_queue.empty():
                    try:
                        frame, roi, is_night, timestamp = self.plate_queue.get(timeout=1)
                    except ValueError:
                        # Para compatibilidad con código que no envía timestamp
                        try:
                            frame, roi, is_night = self.plate_queue.get(timeout=1)
                            timestamp = None
                        except ValueError:
                            # Para compatibilidad con código que solo envía frame y roi
                            frame, roi = self.plate_queue.get(timeout=1)
                            is_night = False
                            timestamp = None
                else:
                    time.sleep(0.1)
                    continue
                    
                # Pasar el flag de noche para que process_plate pueda aplicar
                # tratamientos específicos para la noche
                start_process = time.time()
                bbox, plate_sr, ocr_text = process_plate(roi, is_night)
                end_process = time.time()
                
                if ocr_text:
                    # Registrar tiempo de proceso
                    process_time = end_process - start_process
                    if not hasattr(self, "registration_times"):
                        self.registration_times = []
                    self.registration_times.append(process_time)
                    
                    # Crear directorios para placas y autos
                    plates_dir = os.path.join("data", "output", "placas")
                    vehicles_dir = os.path.join("data", "output", "autos")
                    os.makedirs(plates_dir, exist_ok=True)
                    os.makedirs(vehicles_dir, exist_ok=True)
                    
                    # Guardar la imagen del vehículo
                    vehicle_filename = f"vehicle_{ocr_text}_{int(time.time())}.jpg"
                    vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
                    cv2.imwrite(vehicle_path, roi)
                    
                    # Si no tenemos historial de detección, lo creamos
                    if not hasattr(self, "plate_detection_history"):
                        self.plate_detection_history = {}
                    
                    # Guardar referencia a la imagen del vehículo
                    if ocr_text not in self.plate_detection_history:
                        self.plate_detection_history[ocr_text] = {}
                    
                    # Guardar imagen del vehículo en el historial
                    self.plate_detection_history[ocr_text]["vehicle_img"] = roi.copy()
                    
                    # Pasar el timestamp al método de añadir placa
                    self._safe_add_plate_to_panel(plate_sr, ocr_text, timestamp)
                    
                self.plate_queue.task_done()
                
            except queue.Empty:
                # Si la cola está vacía, esperar un poco
                time.sleep(0.1)
            except Exception as e:
                print(f"Error en plate_loop: {e}")
                time.sleep(0.5)  # Evitar bucle rápido en caso de error

    def detect_and_draw_cars(self, frame):
        """
        Detecta vehículos en el frame con soporte mejorado para condiciones nocturnas.
        Todos los vehículos serán marcados en verde sin importar su tipo.
        """
        # Detectar condición nocturna
        is_night = self._is_night_scene(frame)
        
        # Reducir resolución para procesamiento
        proc_scale = 0.5  # Procesar a la mitad de resolución
        h, w = frame.shape[:2]
        proc_w, proc_h = int(w * proc_scale), int(h * proc_scale)
        
        # Redimensionar frame para procesamiento
        small_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        
        # Pre-procesamiento específico para escenas nocturnas
        if is_night:
            # Aumentar brillo y contraste para mejorar detección nocturna
            small_frame = self._enhance_night_visibility(small_frame)
        
        # Detección en el frame pequeño
        car_detections = []
        
        try:
            # 1. Inicializar detector de vehículos si no existe
            if not hasattr(self, 'vehicle_detector'):
                self.vehicle_detector = VehicleDetector(model_path="models/yolov8n.pt")
            
            # 2. Ajustar umbral de confianza según condiciones de luz
            confidence_threshold = 0.25 if is_night else 0.4  # Más permisivo en la noche
            
            # 3. Detectar vehículos en frame
            detections = self.vehicle_detector.detect(small_frame, conf=confidence_threshold, draw=False)
            
            # 4. Copiar frame solo si hay detecciones (ahorra memoria)
            frame_with_cars = None
            
            # Escalar detecciones al tamaño original
            scale_factor = 1.0 / proc_scale
            for detection in detections:
                # Desempaquetar valores
                x1, y1, x2, y2, cls_id = detection
                    
                # Solo procesar vehículos (clase 2=car, 5=bus, 7=truck)
                if cls_id in [2, 5, 7]:  
                    # Escalar coordenadas a tamaño original
                    x1s, y1s = int(x1 * scale_factor), int(y1 * scale_factor)
                    x2s, y2s = int(x2 * scale_factor), int(y2 * scale_factor)
                    
                    # Crear copia del frame solo cuando sea necesario
                    if frame_with_cars is None:
                        frame_with_cars = frame.copy()
                    
                    # MODIFICACIÓN: Usar color verde para todos los vehículos
                    box_color = (0, 255, 0)  # Verde para todos los tipos de vehículos
                    
                    # Dibujar rectángulo
                    cv2.rectangle(frame_with_cars, (x1s, y1s), (x2s, y2s), box_color, 2)
                    
                    # Etiquetas según la clase
                    label = "CAR" if cls_id == 2 else "BUS" if cls_id == 5 else "TRUCK"
                    
                    # Dibujar texto con fondo para mejor visibilidad
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame_with_cars, 
                                (x1s, y1s - text_size[1] - 10), 
                                (x1s + text_size[0], y1s), 
                                box_color, -1)
                    cv2.putText(frame_with_cars, label,
                                (x1s, y1s - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 2)
                    
                    # Añadir a las detecciones con formato consistente de 6 valores
                    car_detections.append((x1s, y1s, x2s, y2s, cls_id, label))
            
            # Si no hubo detecciones, devolver frame original
            if frame_with_cars is None:
                frame_with_cars = frame
                
        except Exception as e:
            print(f"Error al detectar vehículos: {str(e)}")
            import traceback
            traceback.print_exc()
            frame_with_cars = frame
        
        return frame_with_cars, car_detections, is_night

    # Añadir estas funciones a la clase VideoPlayerOpenCV
    def _is_night_scene(self, frame):
        """Determina si el frame corresponde a una escena nocturna"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular brillo promedio
        avg_brightness = cv2.mean(gray)[0]
        
        # Si el brillo promedio es bajo, consideramos que es una escena nocturna
        return avg_brightness < 70  # Umbral ajustable según tus vídeos

    def _enhance_night_visibility(self, frame):
        """Mejora la visibilidad en escenas nocturnas"""
        # Convertir a LAB para trabajar con el canal de luminosidad
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L para mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Fusionar canales de nuevo
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convertir de vuelta a BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Aumentar ganancia para mayor visibilidad
        return cv2.convertScaleAbs(enhanced_bgr, alpha=1.3, beta=30)
    
    def is_vehicle_in_polygon(self, car_box, polygon_points):
        """
        Determina si un vehículo está dentro del polígono de infracción.
        """
        if not polygon_points or len(polygon_points) < 3:
            return False
        
        # Extraer correctamente las coordenadas del vehículo
        # car_box puede tener 5 o 6 valores (x1,y1,x2,y2,cls_id) o (x1,y1,x2,y2,cls_id,label)
        x1, y1, x2, y2 = car_box[0], car_box[1], car_box[2], car_box[3]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Resto del código sin cambios
        polygon = np.array(polygon_points, np.int32)
        
        if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
            return True
        
        front_x = (x1 + x2*3) // 4
        front_y = center_y
        rear_x = (x1*3 + x2) // 4
        rear_y = center_y
        
        if cv2.pointPolygonTest(polygon, (front_x, front_y), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon, (rear_x, rear_y), False) >= 0:
            return True
        
        if cv2.pointPolygonTest(polygon, (x1, y1), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon, (x2, y1), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon, (x1, y2), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon, (x2, y2), False) >= 0:
            return True
        
        return False

    def is_vehicle_in_polygon_night(self, car_box, polygon_points):
        """
        Versión adaptada para la noche - más permisiva.
        """
        if not polygon_points or len(polygon_points) < 3:
            return False
        
        # Extraer correctamente las coordenadas del vehículo
        # car_box puede tener 5 o 6 valores (x1,y1,x2,y2,cls_id) o (x1,y1,x2,y2,cls_id,label)
        x1, y1, x2, y2 = car_box[0], car_box[1], car_box[2], car_box[3]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Resto del código sin cambios
        polygon = np.array(polygon_points, np.int32)
        
        if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
            return True
        
        width = x2 - x1
        height = y2 - y1
        
        check_points = [
            (x1 + width//4, y1 + height//4),
            (center_x, y1 + height//4),
            (x2 - width//4, y1 + height//4),
            (x1 + width//4, center_y),
            (center_x, center_y),
            (x2 - width//4, center_y),
            (x1 + width//4, y2 - height//4),
            (center_x, y2 - height//4),
            (x2 - width//4, y2 - height//4),
            (x1 + width//4, y2),
            (center_x, y2),
            (x2 - width//4, y2),
        ]
        
        for point in check_points:
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return True
        
        return False

    def update_frames(self):
        """
        Actualiza los frames del video y detecta infracciones con soporte mejorado para noche.
        """
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000/30), self.update_frames)
            return

        # Usar nuestra función mejorada para detectar y dibujar vehículos
        # Ahora capturamos los tres valores devueltos
        frame_with_cars, car_detections, is_night = self.detect_and_draw_cars(frame)
        
        # Si hay un polígono definido, dibujarlo con color adaptado para visibilidad nocturna
        if self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape(-1, 1, 2)
            # Color más brillante en la noche para mejor visibilidad
            poly_color = (0, 220, 255) if is_night else (0, 0, 255)  # Amarillo vs Rojo
            cv2.polylines(frame_with_cars, [pts], True, poly_color, 2)

        # Procesar placas si está en rojo (mejorado)
        current_state = self.semaforo.get_current_state()
        
        # Agregar información visual del estado del semáforo en el frame
        # Texto con fondo para mejor visibilidad especialmente en la noche
        semaforo_text = f"Semaforo: {current_state.upper()}"
        
        # Color según estado
        if current_state == "red":
            text_color = (0, 0, 255)  # Rojo
            bg_color = (255, 255, 255)  # Fondo blanco
        elif current_state == "yellow":
            text_color = (0, 255, 255)  # Amarillo
            bg_color = (0, 0, 0)  # Fondo negro
        else:  # green
            text_color = (0, 255, 0)  # Verde
            bg_color = (0, 0, 0)  # Fondo negro
        
        # Añadir texto con fondo para mejor visibilidad
        text_size = cv2.getTextSize(semaforo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
        cv2.rectangle(frame_with_cars, 
                    (5, 5), 
                    (text_size[0] + 20, 40), 
                    bg_color, -1)
        cv2.putText(frame_with_cars, semaforo_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    text_color, 3)
        
        # Indicador de modo nocturno si es el caso
        if is_night:
            cv2.putText(frame_with_cars, "MODO NOCTURNO", 
                        (frame_with_cars.shape[1] - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 255), 2)
        
        # Si el semáforo está en ROJO, buscar infracciones (lógica mejorada)
        if current_state == "red" and not self.plate_queue.full():
            # Verificar si hay vehículos dentro de la zona del polígono
            for car_detection in car_detections:
                if self.polygon_points and len(self.polygon_points) >= 3:
                    # Umbral más permisivo para detección nocturna
                    if is_night:
                        # Verificar con criterios más flexibles para la noche
                        in_polygon = self.is_vehicle_in_polygon_night(car_detection, self.polygon_points)
                    else:
                        # Verificación normal para día
                        in_polygon = self.is_vehicle_in_polygon(car_detection, self.polygon_points)
                        
                    if in_polygon:
                        # Crear una región de interés alrededor del vehículo
                        x1, y1, x2, y2 = car_detection[0], car_detection[1], car_detection[2], car_detection[3]
                        
                        # Ampliar más el área para capturar la placa (especialmente en la noche)
                        height = y2 - y1
                        width = x2 - x1
                        
                        # Mayor expansión en modo nocturno
                        expand_factor = 0.15 if is_night else 0.1
                        
                        y1_extended = max(0, y1 - int(height * expand_factor))
                        y2_extended = min(frame.shape[0], y2 + int(height * expand_factor))
                        x1_extended = max(0, x1 - int(width * expand_factor))
                        x2_extended = min(frame.shape[1], x2 + int(width * expand_factor))
                        
                        # Recortar el área del vehículo para procesamiento de placa
                        car_roi = frame[y1_extended:y2_extended, x1_extended:x2_extended].copy()

                        # Recortar el área del vehículo para procesamiento de placa
                        vehicle_roi = frame_with_cars[y1_extended:y2_extended, x1_extended:x2_extended]
                        
                        # Dibujar caja roja para indicar infracción
                        cv2.rectangle(frame_with_cars, 
                                    (x1_extended, y1_extended), 
                                    (x2_extended, y2_extended), 
                                    (0, 0, 255), 3)

                        # MODIFICACIÓN: Obtener el timestamp del video actual
                        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        current_time = current_frame / self.video_fps

                        # Pasar el recorte del vehículo para OCR - incluir timestamp
                        if not self.plate_queue.full():
                            self.plate_queue.put((frame_with_cars, vehicle_roi, is_night, current_time))
                        
                        # En la noche, mejorar el roi antes de procesarlo
                        if is_night:
                            car_roi = self._enhance_night_visibility(car_roi)
                        
                        # Procesar placa en esta área (poner en cola para procesamiento)
                        if car_roi.size > 0:  # Verificar que el ROI no esté vacío
                            try:
                                # Poner en la cola para procesamiento
                                if not self.plate_queue.full():
                                    # Pasar el flag de noche al procesador de placas
                                    self.plate_queue.put((frame.copy(), car_roi, is_night))
                                
                                # Dibujar un rectángulo rojo alrededor del vehículo infractor
                                # Color más brillante en la noche para mejor visibilidad
                                infraction_color = (0, 0, 255)  # Rojo
                                
                                cv2.rectangle(frame_with_cars, (x1, y1), (x2, y2), infraction_color, 3)
                                
                                # Texto con fondo para mejor visibilidad
                                cv2.putText(frame_with_cars, "INFRACCION", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, infraction_color, 2)
                            except Exception as e:
                                print(f"Error al procesar infracción: {e}")
        
        # Mostrar el frame anotado
        bgr_img = self.resize_and_letterbox(frame_with_cars)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        
        # Métricas y siguiente frame
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        mode = "NOCHE" if is_night else "DÍA"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB | {mode}"
        self.info_label.config(text=info_text)
        
        # Asegurarse que las etiquetas estén visibles
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.info_label.lift()
        
        self._after_id = self.parent.after(10, self.update_frames)

    def _safe_add_plate_to_panel(self, plate_img, plate_text, timestamp=None):
        """
        Añade una placa detectada al panel lateral con diseño de card.
        Guarda las imágenes en carpetas separadas de placas y autos.
        También guarda la infracción en el archivo JSON centralizado.
        """
        # Verificaciones básicas
        if plate_img is None or not isinstance(plate_text, str):
            print(f"Error: Datos de placa inválidos - img: {plate_img is not None}, text: {plate_text}")
            return
        
        # Crear las carpetas necesarias
        plates_dir = os.path.join("data", "output", "placas")
        vehicles_dir = os.path.join("data", "output", "autos")
        os.makedirs(plates_dir, exist_ok=True)
        os.makedirs(vehicles_dir, exist_ok=True)
        
        # Verificar si ya existe esta placa (para evitar duplicados)
        plate_filename = f"plate_{plate_text}.jpg"
        vehicle_filename = f"vehicle_{plate_text}.jpg"
        plate_path = os.path.join(plates_dir, plate_filename)
        vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
        
        # Determinar si es escena nocturna para aplicar tratamientos específicos
        is_night = False
        if hasattr(self, '_is_night_scene'):
            try:
                # Si no tenemos el frame completo, usamos la imagen de la placa
                is_night = self._is_night_scene(plate_img) 
            except:
                # Si falla, asumimos valor por defecto
                pass
        
        # Importar la función para mejorar las imágenes de placas
        try:
            from src.core.processing.resolution_process import enhance_plate_image
            
            # Aplicar super-resolución y guardar la placa si no existe
            if not os.path.exists(plate_path):
                # Mejorar la placa con super-resolución
                enhanced_plate = enhance_plate_image(plate_img, is_night, plate_path)
        except Exception as e:
            print(f"Error al mejorar la placa con super-resolución: {e}")
            # En caso de error, intentar guardar la placa original
            if not os.path.exists(plate_path):
                cv2.imwrite(plate_path, plate_img)
        
        # Intentar obtener imagen del vehículo completo
        vehicle_img = None
        
        # Si tenemos información de detección del vehículo en el historial
        if hasattr(self, "plate_detection_history") and plate_text in self.plate_detection_history:
            if "vehicle_img" in self.plate_detection_history[plate_text]:
                vehicle_img = self.plate_detection_history[plate_text]["vehicle_img"]
                # Solo guardar si no existe
                if not os.path.exists(vehicle_path):
                    cv2.imwrite(vehicle_path, vehicle_img)
        
        # Registrar tiempo actual como tiempo de registro
        current_registration_time = time.time()
        
        # Estimar tiempo de detección basado en timestamp del video
        detection_time = None
        if timestamp is not None:
            # Si tenemos la marca de tiempo del video, calcular aproximadamente
            detection_time = self.detection_start_time + timestamp
        
        # Función para ejecutar en el hilo principal de Tkinter
        def _add():
            try:
                # IMPORTANTE: Verificar duplicados en el panel
                # Si la placa ya está en el panel, no añadir de nuevo
                for widget in self.detected_plates_widgets:
                    if isinstance(widget, dict) and widget.get("plate_text") == plate_text:
                        print(f"Placa {plate_text} ya existe en el panel - no duplicando")
                        return
                
                print(f"Creando card para placa: {plate_text}")
                
                # CRÍTICO: Verificar que el panel interno existe
                if not hasattr(self, "plates_inner_frame") or self.plates_inner_frame is None:
                    print("ERROR: El frame interno no existe")
                    # Crear el frame interno si no existe
                    self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="#ecf0f1")
                    self.plates_canvas_window = self.plates_canvas.create_window(
                        (0, 0), window=self.plates_inner_frame, anchor="nw"
                    )
                
                # CARD PRINCIPAL - Contenedor con borde
                card_frame = tk.Frame(
                    self.plates_inner_frame,
                    bg="#ffffff",
                    relief=tk.RAISED,
                    bd=1,
                    padx=8,
                    pady=8
                )
                card_frame.pack(fill="x", padx=8, pady=5)
                
                # LAYOUT: Dos columnas (izquierda info, derecha imagen)
                info_frame = tk.Frame(card_frame, bg="#ffffff")
                info_frame.pack(side="left", fill="both", expand=True)
                
                img_frame = tk.Frame(card_frame, bg="#ffffff", width=120, height=90)
                img_frame.pack(side="right", padx=(5,0), pady=5)
                img_frame.pack_propagate(False)  # Mantener tamaño fijo
                
                # COLUMNA IZQUIERDA: Información de la placa
                plate_label = tk.Label(
                    info_frame,
                    text=f"Placa: {plate_text}",
                    font=("Arial", 12, "bold"),
                    bg="#ffffff",
                    fg="#333333",
                    anchor="w",
                    justify="left"
                )
                plate_label.pack(fill="x", pady=(0, 5), anchor="w")
                
                # Timestamp si disponible
                if timestamp is not None:
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    msecs = int((timestamp % 1) * 1000)
                    time_str = f"{mins:02d}:{secs:02d}.{msecs:03d}"
                    
                    time_label = tk.Label(
                        info_frame,
                        text=f"Tiempo: {time_str}",
                        font=("Arial", 10),
                        bg="#ffffff",
                        fg="#666666",
                        anchor="w",
                        justify="left"
                    )
                    time_label.pack(fill="x", anchor="w")
                
                # COLUMNA DERECHA: Imagen del vehículo en lugar de la placa
                try:
                    # Priorizar imagen del vehículo si existe
                    display_img = vehicle_img if vehicle_img is not None else plate_img
                    
                    # Redimensionar imagen para mantener proporción y tamaño adecuado
                    h, w = display_img.shape[:2]
                    max_width, max_height = 110, 80
                    
                    # Escalar preservando proporción
                    scale = min(max_width / w, max_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    # Redimensionar y convertir para tkinter
                    resized = cv2.resize(display_img, (new_w, new_h))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # Crear y posicionar label con imagen
                    img_label = tk.Label(img_frame, image=img_tk, bg="#eeeeee", bd=1, relief="solid")
                    img_label.image = img_tk  # Mantener referencia
                    img_label.place(relx=0.5, rely=0.5, anchor="center")
                    
                except Exception as img_err:
                    print(f"Error al procesar imagen: {img_err}")
                    # Placeholder si falla la imagen
                    img_label = tk.Label(img_frame, text="Sin imagen", bg="#eeeeee", fg="#999999")
                    img_label.place(relx=0.5, rely=0.5, anchor="center")
                
                # Registrar en lista de placas detectadas
                plate_data = {
                    "container": card_frame,
                    "plate_text": plate_text,
                    "timestamp": timestamp,
                    "plate_path": plate_path,
                    "vehicle_path": vehicle_path if os.path.exists(vehicle_path) else None
                }
                self.detected_plates_widgets.append(plate_data)
                
                # CRÍTICO: Actualizar el historial de detección con tiempos
                if not hasattr(self, "plate_detection_history"):
                    self.plate_detection_history = {}
                    
                if plate_text in self.plate_detection_history:
                    # Actualizar registro existente
                    self.plate_detection_history[plate_text]["last_detection"] = timestamp
                    self.plate_detection_history[plate_text]["registration_time"] = current_registration_time
                    
                    # Actualizar tiempos para métricas
                    if detection_time is not None and "detection_time" not in self.plate_detection_history[plate_text]:
                        self.plate_detection_history[plate_text]["detection_time"] = detection_time
                    
                    # Calcular tiempo de procesamiento si no existe
                    if detection_time is not None:
                        proc_time = current_registration_time - detection_time
                        self.plate_detection_history[plate_text]["processing_time"] = proc_time
                        
                        # Añadir a los tiempos de registro para estadísticas
                        if not hasattr(self, "registration_times"):
                            self.registration_times = []
                        self.registration_times.append(proc_time)
                    
                    # Almacenar las rutas de los archivos
                    self.plate_detection_history[plate_text]["plate_path"] = plate_path
                    if os.path.exists(vehicle_path):
                        self.plate_detection_history[plate_text]["vehicle_path"] = vehicle_path
                else:
                    # Crear nuevo registro
                    new_record = {
                        "count": 1,
                        "first_detection": timestamp,
                        "last_detection": timestamp,
                        "plate_path": plate_path,
                        "vehicle_path": vehicle_path if os.path.exists(vehicle_path) else None,
                        "registration_time": current_registration_time
                    }
                    
                    # Añadir tiempo de detección si está disponible
                    if detection_time is not None:
                        new_record["detection_time"] = detection_time
                        
                        # Calcular y guardar tiempo de procesamiento
                        proc_time = current_registration_time - detection_time
                        new_record["processing_time"] = proc_time
                        
                        # Añadir a los tiempos de registro para estadísticas
                        if not hasattr(self, "registration_times"):
                            self.registration_times = []
                        self.registration_times.append(proc_time)
                    
                    self.plate_detection_history[plate_text] = new_record
                    
                # Registrar como ya procesada globalmente
                if not hasattr(self, "processed_plates"):
                    self.processed_plates = set()
                self.processed_plates.add(plate_text)
                
                # Actualizar indicadores de rendimiento
                if hasattr(self, "_update_metrics_panel"):
                    self._update_metrics_panel()
                
                # CRÍTICO: Actualizar región de desplazamiento y vista
                self.plates_inner_frame.update_idletasks()
                self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
                self.plates_canvas.yview_moveto(1.0)  # Mostrar la última placa añadida
                
                print(f"Card añadido exitosamente: {plate_text}")
                
            except Exception as e:
                print(f"ERROR al añadir placa: {e}")
                import traceback
                traceback.print_exc()
        
        # Ejecutar en el hilo principal de tkinter con pequeño retraso
        if hasattr(self, "parent") and self.parent:
            self.parent.after(50, _add)  # 50ms de retraso para asegurar que la UI esté lista
        else:
            print("Error: No se puede acceder al widget principal")

    def _create_metrics_panel(self):
        """Crea el panel de métricas de rendimiento"""
        # Panel de métricas con estilo moderno
        self.metrics_frame = tk.Frame(self.plates_frame, bg="#34495e")
        self.metrics_frame.pack(side="top", fill="x", pady=0, after=self.plates_title)
        
        # Título con mejor estilo
        metrics_title = tk.Label(self.metrics_frame, text="Indicadores de Rendimiento",
                            bg="#34495e", fg="white", font=("Arial", 12, "bold"),
                            pady=5)
        metrics_title.pack(fill="x")
        
        # Contenedor para indicadores con estilo moderno
        indicators_container = tk.Frame(self.metrics_frame, bg="#34495e", padx=10, pady=5)
        indicators_container.pack(fill="x")
        
        # Crear etiquetas para indicadores
        self.ti_label = tk.Label(indicators_container, bg="#34495e", fg="white", 
                            font=("Arial", 10), anchor="w")
        self.ti_label.pack(fill="x", pady=2)
        
        self.tr_label = tk.Label(indicators_container, bg="#34495e", fg="white", 
                            font=("Arial", 10), anchor="w")
        self.tr_label.pack(fill="x", pady=2)
        
        self.ir_label = tk.Label(indicators_container, bg="#34495e", fg="white", 
                            font=("Arial", 10), anchor="w")
        self.ir_label.pack(fill="x", pady=2)
        
        # Separador estético
        separator = tk.Frame(self.metrics_frame, height=2, bg="#ecf0f1")
        separator.pack(fill="x", padx=10, pady=5)
        
        # Inicializar con valores en cero
        self._update_metrics_panel()

    def _update_metrics_panel(self):
        """Actualiza los valores de los indicadores de rendimiento"""
        if hasattr(self, "ti_label") and hasattr(self, "tr_label") and hasattr(self, "ir_label"):
            # Calcular métricas
            ti = self._calculate_infraction_rate()
            tr = self._calculate_registration_time()
            ir = self._calculate_reincidence_index()
            
            # Actualizar etiquetas con formato más atractivo
            self.ti_label.config(text=f"TI: {ti:.2f} infracciones")
            self.tr_label.config(text=f"TR: {tr:.2f} segundos")
            self.ir_label.config(text=f"IR: {ir:.1f}%")

    def clear_detected_plates(self):
        """Limpia todas las placas detectadas del panel lateral"""
        try:
            # Verificar que existe la lista de widgets
            if not hasattr(self, 'detected_plates_widgets'):
                self.detected_plates_widgets = []
                return
            
            # Eliminar todos los widgets de placas
            for plate_widget in self.detected_plates_widgets:
                try:
                    if isinstance(plate_widget, dict) and 'container' in plate_widget:
                        plate_widget['container'].destroy()
                except Exception as widget_err:
                    print(f"Error al destruir widget: {widget_err}")
            
            # Limpiar listas y conjuntos
            self.detected_plates_widgets = []
            
            if hasattr(self, 'seen_plates'):
                self.seen_plates = set()
            
            # Reiniciar métricas
            if hasattr(self, "plate_detection_history"):
                self.plate_detection_history = {}
            
            if hasattr(self, "registration_times"):
                self.registration_times = []
            
            # Actualizar panel de métricas
            if hasattr(self, "_update_metrics_panel"):
                self._update_metrics_panel()
            
            # Forzar actualización del canvas
            if hasattr(self, "plates_inner_frame") and hasattr(self, "plates_canvas"):
                self.plates_inner_frame.update_idletasks()
                self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
        
        except Exception as e:
            print(f"Error al limpiar placas: {e}")
            import traceback
            traceback.print_exc()


    def gestionar_camaras(self):
        """
        Abre un diálogo para elegir un vídeo existente, y al 'Cargar'
        reinicia completamente el estado de Foto Rojo y carga el nuevo vídeo.
        """
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")

        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        for f in sorted(os.listdir(self.video_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                lb.insert(tk.END, f)

        btn_frame = tk.Frame(w)
        btn_frame.pack(fill="x", pady=5)

        def on_cargar():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un vídeo.")
                return
            fn   = lb.get(sel[0])
            path = os.path.join(self.video_dir, fn)
            w.destroy()

            # 1) Detener y limpiar todo el estado actual
            self.stop_video()
            self.clear_detected_plates()
            self.semaforo.current_state = "green"
            self.semaforo.show_state()

            # 2) Maximizar la ventana principal nuevamente
            main_win = self.parent.winfo_toplevel()
            main_win.deiconify()
            # main_win.state("zoomed")

            # 3) Cargar el nuevo vídeo con preprocesamiento
            self.load_video(path)

        tk.Button(btn_frame, text="Cargar",  width=10, command=on_cargar).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Borrar",  width=10, command=lambda: self._cam_del(lb)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cerrar",  width=10, command=w.destroy).pack(side="left", padx=5)

        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)


    def _on_plates_canvas_configure(self, event):
        """Actualiza el ancho del frame interno cuando cambia el tamaño del canvas"""
        width = event.width
        try:
            # Actualizar el ancho de la ventana del canvas
            self.plates_canvas.itemconfig(self.plates_canvas_window, width=width)
            
            # Forzar actualización
            self.plates_canvas.update()
            
            print(f"Canvas redimensionado: {width}px de ancho")
        except Exception as e:
            print(f"Error en _on_plates_canvas_configure: {e}")

    def _on_plates_inner_configure(self, event):
        """Actualiza la región scrollable cuando cambia el contenido del frame interno"""
        try:
            # Actualizar la región de desplazamiento
            self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
            
            # Forzar actualización del canvas
            self.plates_canvas.update()
        except Exception as e:
            print(f"Error en _on_plates_inner_configure: {e}")

    def _cam_load_async(self, path):
        cap_tmp = cv2.VideoCapture(path)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            self.parent.after(0, lambda: messagebox.showerror("Error", "No se pudo leer el vídeo."))
            return

        # Ahora volvemos al hilo principal:
        self.parent.after(0, lambda: (
            self.stop_video(),
            self.load_video(path)
        ))


    def _cam_load(self, lb):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia","Seleccione un vídeo.")
            return
        path = os.path.join(self.video_dir, lb.get(sel[0]))
        self.stop_video()
        self.load_video(path)

    def _cam_del(self, lb):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un vídeo para borrar.")
            return
        fn = lb.get(sel[0])
        path = os.path.join(self.video_dir, fn)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fn}'?"):
            return
        if path == self.current_video_path:
            self.running = False
            if hasattr(self, "_after_id") and self._after_id:
                self.parent.after_cancel(self._after_id)
                self._after_id = None
            if self.cap:
                self.cap.release()
                self.cap = None
            for item in self.detected_plates_widgets:
                item[0].destroy()
            self.detected_plates_widgets.clear()
            self.video_label.config(image="")
            self.current_video_path = None
        try:
            os.remove(path)
            self.remove_avenue_data(path)
            self.remove_time_preset_data(path)
            self.remove_polygon_data(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"'{fn}' y datos borrados.")
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
            data.pop(video_path, None)
            with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                json.dump(data, fw, indent=2)
        except:
            pass

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            presets.pop(video_path, None)
            with open(PRESETS_FILE, "w", encoding="utf-8") as fw:
                json.dump(presets, fw, indent=2)
        except:
            pass

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            polygons.pop(video_path, None)
            with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                json.dump(polygons, fw, indent=2)
        except:
            pass

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

    def start_processed_video(self, path):
        """
        Inicia la reproducción optimizada del video después de que ha sido procesado.
        Solo muestra detección de vehículos sin procesar placas para optimizar recursos.
        """
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, first_frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el vídeo procesado.")
            return
        
        self.current_video_path = path
        h, w = first_frame.shape[:2]
        self.orig_h, self.orig_w = h, w
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
        self.running = True
        
        # Cargar configuraciones procesadas
        self.load_polygon_for_video()
        
        # Cargar configuraciones del video
        ave = self.get_avenue_for_video(path)
        times = self.get_time_preset_for_video(path)
        
        if ave is not None and times is not None:
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.cycle_durations = times
            
            # Configurar el semáforo con esta configuración
            self.semaforo.cycle_durations = {
                "green": times["green"],
                "yellow": times["yellow"],
                "red": times["red"]
            }
            
            # Inicializar el semáforo en verde como punto de partida
            self.semaforo.current_state = "green"
            self.semaforo.target_time = time.time() + self.semaforo.cycle_durations[self.semaforo.current_state]
            
            # Activar el semáforo
            self.semaforo.activate_semaphore()
        
        # Iniciar reloj
        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()
        
        # Activar modo de reproducción optimizada (solo detección de vehículos)
        self.optimization_mode = "post_processing"
        
        # Mostrar mensaje de inicio de reproducción optimizada
        print("Iniciando reproducción optimizada (solo detección de vehículos)")
        
        # Iniciar reproducción inmediatamente
        self.update_frames_optimized()

    def update_frames_optimized(self):
        """
        Versión optimizada de update_frames que sólo detecta vehículos sin procesar placas.
        Se usa después del preprocesamiento para mostrar el video de forma más eficiente.
        """
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000/30), self.update_frames_optimized)
            return

        # Detectar si es escena nocturna para optimizaciones
        is_night = self._is_night_scene(frame)
        
        # Procesamiento optimizado: sólo detectamos vehículos, sin procesar placas
        try:
            # Reducir resolución para procesamiento más rápido
            proc_scale = 0.5
            h, w = frame.shape[:2]
            proc_w, proc_h = int(w * proc_scale), int(h * proc_scale)
            
            # Redimensionar frame para procesamiento
            small_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            
            # Pre-procesamiento específico para escenas nocturnas (más ligero)
            if is_night:
                # Usar conversión rápida en lugar de CLAHE completo
                small_frame = cv2.convertScaleAbs(small_frame, alpha=1.3, beta=30)
            
            # Ajustar umbral de confianza
            confidence_threshold = 0.25 if is_night else 0.4
            
            # Detectar vehículos (optimizado)
            if hasattr(self, 'vehicle_detector'):
                detections = self.vehicle_detector.detect(
                    small_frame, 
                    conf=confidence_threshold,
                    draw=False
                )
                
                # Escalar detecciones al tamaño original
                frame_with_cars = frame.copy()
                scale_factor = 1.0 / proc_scale
                
                # Dibujar polígono de área si existe
                if self.polygon_points:
                    pts = np.array(self.polygon_points, np.int32).reshape(-1, 1, 2)
                    poly_color = (0, 220, 255) if is_night else (0, 0, 255)
                    cv2.polylines(frame_with_cars, [pts], True, poly_color, 2)
                
                # Dibujar vehículos detectados
                for detection in detections:
                    # Extraer coordenadas y clase
                    x1, y1, x2, y2, cls_id = detection[:5]
                    
                    # Solo procesar vehículos (coches, buses, camiones)
                    if cls_id in [2, 5, 7]:
                        # Escalar coordenadas a tamaño original
                        x1s, y1s = int(x1 * scale_factor), int(y1 * scale_factor)
                        x2s, y2s = int(x2 * scale_factor), int(y2 * scale_factor)
                        
                        # Color según si está en zona restringida
                        in_polygon = False
                        if self.polygon_points and len(self.polygon_points) >= 3:
                            if is_night:
                                in_polygon = self.is_vehicle_in_polygon_night((x1s, y1s, x2s, y2s), self.polygon_points)
                            else:
                                in_polygon = self.is_vehicle_in_polygon((x1s, y1s, x2s, y2s), self.polygon_points)
                        
                        # Color según estado (en área + semáforo rojo = rojo, en área = amarillo, fuera de área = verde)
                        if in_polygon and self.semaforo.get_current_state() == "red":
                            box_color = (0, 0, 255)  # Rojo para infracciones
                        elif in_polygon:
                            box_color = (0, 255, 255)  # Amarillo para vehículos en área permitida
                        else:
                            box_color = (0, 255, 0)  # Verde para vehículos fuera del área
                        
                        # Dibujar rectángulo
                        cv2.rectangle(frame_with_cars, (x1s, y1s), (x2s, y2s), box_color, 2)
                        
                        # Etiquetas según la clase
                        label = "CAR" if cls_id == 2 else "BUS" if cls_id == 5 else "TRUCK"
                        
                        # Dibujar texto con fondo para mejor visibilidad
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame_with_cars, 
                                    (x1s, y1s - text_size[1] - 10), 
                                    (x1s + text_size[0], y1s), 
                                    box_color, -1)
                        cv2.putText(frame_with_cars, label,
                                    (x1s, y1s - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 0), 2)
            else:
                frame_with_cars = frame
                    
        except Exception as e:
            print(f"Error al detectar vehículos: {str(e)}")
            frame_with_cars = frame
        
        # Mostrar información del estado del semáforo
        current_state = self.semaforo.get_current_state()
        semaforo_text = f"Semaforo: {current_state.upper()}"
        
        # Color según estado
        if current_state == "red":
            text_color = (0, 0, 255)  # Rojo
            bg_color = (255, 255, 255)  # Fondo blanco
        elif current_state == "yellow":
            text_color = (0, 255, 255)  # Amarillo
            bg_color = (0, 0, 0)  # Fondo negro
        else:  # green
            text_color = (0, 255, 0)  # Verde
            bg_color = (0, 0, 0)  # Fondo negro
        
        # Mostrar estado del semáforo
        text_size = cv2.getTextSize(semaforo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
        cv2.rectangle(frame_with_cars, 
                    (5, 5), 
                    (text_size[0] + 20, 40), 
                    bg_color, -1)
        cv2.putText(frame_with_cars, semaforo_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    text_color, 3)
        
        # Indicador de modo optimizado
        cv2.putText(frame_with_cars, "MODO OPTIMIZADO", 
                    (frame_with_cars.shape[1] - 250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 165, 255), 2)
        
        # Indicador de modo nocturno si es el caso
        if is_night:
            cv2.putText(frame_with_cars, "MODO NOCTURNO", 
                        (frame_with_cars.shape[1] - 250, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 255), 2)
        
        # Mostrar el frame anotado
        bgr_img = self.resize_and_letterbox(frame_with_cars)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        
        # Métricas y siguiente frame
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps

        # Actualizar métricas
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        mode = "NOCHE" if is_night else "DÍA"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB | {mode} | OPTIMIZADO"
        self.info_label.config(text=info_text)
        
        # Asegurarse que las etiquetas estén visibles
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.info_label.lift()
        
        self._after_id = self.parent.after(10, self.update_frames_optimized)

    def _calculate_infraction_rate(self):
        """Calcula la Tasa de Infracciones: infracciones detectadas"""
        # CORRECCIÓN: Devolver directamente el número exacto de infracciones, no una tasa
        
        if hasattr(self, "plate_detection_history"):
            # Usar el número exacto de elementos en el historial de detecciones
            return len(self.plate_detection_history)
        
        # Si no hay historial, contar los widgets en el panel
        if hasattr(self, "detected_plates_widgets"):
            return len(self.detected_plates_widgets)
        
        # Si no hay datos disponibles
        return 0

    def _calculate_registration_time(self):
        """
        Calcula el Tiempo de Registro: tiempo promedio entre detección y registro en el sistema.
        El tiempo se mide desde que se detecta una infracción hasta que se completa su procesamiento.
        """
        if not hasattr(self, "plate_detection_history") or not self.plate_detection_history:
            return 0.0
        
        # Obtener tiempos de registro de todas las placas detectadas
        registration_times = []
        
        for plate_id, data in self.plate_detection_history.items():
            # Verificar que tengamos los datos necesarios
            if "processing_time" in data and data["processing_time"] > 0:
                # Si ya tenemos el tiempo calculado previamente y es positivo
                registration_times.append(data["processing_time"])
                
            elif "detection_time" in data and "registration_time" in data:
                # Calcular la diferencia entre detección y registro
                proc_time = data["registration_time"] - data["detection_time"]
                
                # Asegurar que el tiempo sea positivo (corregir posibles errores de sincronización)
                if proc_time > 0:
                    registration_times.append(proc_time)
                    # Guardar para futuras consultas
                    data["processing_time"] = proc_time
        
        # Si no hay datos de procesamiento válidos, intentar usar los tiempos guardados
        if not registration_times and hasattr(self, "registration_times") and self.registration_times:
            # Filtrar solo valores positivos
            valid_times = [t for t in self.registration_times if t > 0]
            if valid_times:
                registration_times = valid_times
        
        # Si aún no hay datos válidos, devolver un valor predeterminado positivo
        if not registration_times:
            return 0.0
        
        # Calcular el promedio (evitar dividir por cero)
        avg_time = sum(registration_times) / len(registration_times)
        
        # Asegurar que el resultado sea positivo (mínimo 0.01 segundos)
        return max(0.01, avg_time)

    def _calculate_reincidence_index(self):
        """Calcula el Índice de Reincidencia: % de placas con más de una detección"""
        if not hasattr(self, "plate_detection_history"):
            return 0
            
        # Contar placas con más de una detección
        reincident_plates = sum(1 for plate_data in self.plate_detection_history.values() 
                            if plate_data.get("count", 1) > 1)  # Usar .get() con valor predeterminado
        
        # Calcular índice como porcentaje
        total_plates = len(self.plate_detection_history)
        if total_plates > 0:
            return (reincident_plates / total_plates) * 100
        return 0

# Fin del módulo VideoPlayerOpenCV