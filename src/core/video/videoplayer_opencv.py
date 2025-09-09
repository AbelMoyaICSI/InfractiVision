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
from src.path_helper import resource_path

# Archivos de configuración
POLYGON_CONFIG_FILE = resource_path("config/polygon_config.json")
AVENUE_CONFIG_FILE  = resource_path("config/avenue_config.json")
PRESETS_FILE        = resource_path("config/time_presets.json")

from src.gui.preprocessing_dialog import PreprocessingDialog

class VideoPlayerOpenCV:
    def get_video_key(self, video_path):
        """Extrae solo el nombre del archivo para usar como clave en configs"""
        return os.path.basename(video_path)
    
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo):
        self.parent            = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label   = timestamp_label
        self.semaforo          = semaforo

        self.yolo = YOLO(resource_path('models/yolov8n.pt'))      # peso pequeño, pre-entrenado en COCO
        self.CAR_CLASS_ID = 2               # en COCO, 'car' = 2
        self.CONF_THRESH   = 0.4

        # Variables para métricas
        self.detected_plates_widgets = []
        self.seen_plates = set()
        
        # Variables para métricas
        self.detection_start_time = time.time()
        self.registration_times = []
        self.plate_detection_history = {}

        # Detección mejorada de hardware GPU/CPU
        self.detect_hardware()
        self.configure_hardware_settings()

                # Directorio de vídeos
        self.video_dir = resource_path("videos")
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
            self.btn_frame, text="CONFIGURACIÓN\nDE VIDEOS",
            command=self.select_video_visual,
            **btn_style
        )
        self.load_button.pack(side="left", padx=10)

        self.btn_preprocesar = tk.Button(
            self.btn_frame, text="INICIAR PROCESAMIENTO\nDE INFRACCIONES",
            command=self.iniciar_preprocesamiento,
            **btn_style
        )
        self.btn_preprocesar.pack(side="left", padx=10)

        # Los botones de limpieza y gestión ahora están integrados en el selector visual

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
        
        # Label para mostrar video actual
        self.current_video_label = tk.Label(
            self.video_frame, text="Ningún video cargado", font=("Arial",12),
            bg="black", fg="yellow", wraplength=300
        )
        self.current_video_label.place(relx=0.5, y=110, anchor="n")
        
        # Label para mostrar información de sistema
        self.system_info_label = tk.Label(
            self.video_frame, text="", font=("Arial",10),
            bg="black", fg="#bdc3c7", wraplength=300
        )
        self.system_info_label.place(relx=0.5, y=140, anchor="n")
        
        # Actualizar información del sistema
        self.update_system_info()

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
        config = self.load_avenue_config()
        video_key = self.get_video_key(video_path)
        # Verificar que la configuración exista y no esté vacía
        if video_key in config and config[video_key] and config[video_key].strip():
            return config[video_key]
        return None

    def set_avenue_for_video(self, video_path, avenue_name):
        cfg = self.load_avenue_config()
        cfg[self.get_video_key(video_path)] = avenue_name
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
        presets = self.load_time_presets()
        video_key = self.get_video_key(video_path)
        # Verificar que la configuración exista y tenga los campos necesarios
        if video_key in presets and presets[video_key]:
            config = presets[video_key]
            # Verificar que tenga al menos los tiempos básicos
            if isinstance(config, dict) and 'green' in config and 'yellow' in config and 'red' in config:
                return config
        return None

    def set_time_preset_for_video(self, video_path, times):
        presets = self.load_time_presets()
        presets[self.get_video_key(video_path)] = times
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
        
        # Configurar icono
        icon_path = resource_path("img/icon.ico")
        if os.path.exists(icon_path):
            setup.iconbitmap(icon_path)

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
        presets[self.get_video_key(self.current_video_path)]=self.polygon_points
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
            # Usar solo el nombre del archivo como clave
            video_key = self.get_video_key(self.current_video_path)
            if video_key in presets:
                self.polygon_points=presets[video_key]
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
            presets.pop(self.get_video_key(self.current_video_path),None)
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


    def select_video_visual(self):
        """
        Selector visual moderno de videos con miniaturas y metadatos completos
        """
        try:
            from src.gui.video_selector_window import show_video_selector
            
            def on_video_selected(video_path, force_config=False):
                """Callback cuando se selecciona un video del selector visual"""
                if video_path:
                    # Usar la función existente de setup pero adaptada
                    self._setup_selected_video(video_path, force_config)
            
            # Mostrar selector visual
            selected_video = show_video_selector(
                parent=self.parent,
                video_dir=self.video_dir,
                on_video_selected=on_video_selected
            )
            
        except ImportError as e:
            print(f"Error importando selector visual: {e}")
            # Fallback al selector original
            self.select_video_classic()
        except Exception as e:
            print(f"Error en selector visual: {e}")
            messagebox.showerror("Error", f"Error en selector visual: {str(e)}")
            # Fallback al selector original
            self.select_video_classic()

    def select_video_classic(self):
        """
        Selector clásico de videos (función original como backup)
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
        
        # NUEVA LÓGICA: SIEMPRE permitir cargar el video
        # Si ya está configurado, cargar directamente
        # Si no está configurado, abrir diálogo de configuración
        
        # Verificar si el video ya está configurado COMPLETAMENTE
        has_avenue = self.get_avenue_for_video(dest) is not None
        has_times = self.get_time_preset_for_video(dest) is not None
        has_polygon = self.check_polygon_exists(dest)
        
        if has_avenue and has_times and has_polygon:
            # Video ya configurado completamente, cargar directamente
            self._load_video_async(dest)
            return
        elif has_avenue and has_times:
            # Video tiene configuración básica pero no polígono - cargar y permitir configurar polígono
            self._load_video_async(dest)
            return
        else:
            # Video NO configurado - abrir diálogo de configuración
            # Abrir vista previa del video para ayudar con la configuración
            cap_tmp = cv2.VideoCapture(dest)
            ret, preview_frame = cap_tmp.read()
            cap_tmp.release()
            
            if not ret:
                messagebox.showerror("Error", "No se pudo abrir el video para configuración.")
                return
            
            # Configuración inicial completa: pantalla combinada para semáforo y área
            self.setup_complete_video_config(dest, preview_frame)

    def _setup_selected_video(self, video_path, force_config=False):
        """
        Configurar video seleccionado desde el selector visual
        """
        try:
            fname = os.path.basename(video_path)
            dest = os.path.join(self.video_dir, fname)
            
            # Copiar archivo si no existe en directorio de videos
            if not os.path.exists(dest):
                import shutil
                shutil.copy2(video_path, dest)
                print(f"Video copiado a: {dest}")
            
            # Verificar si ya tiene configuración completa
            has_polygon = self.check_polygon_exists(dest)
            has_semaphore = self.get_time_preset_for_video(fname) is not None
            has_avenue = self.get_avenue_for_video(fname) is not None
            
            if has_polygon and has_semaphore and has_avenue and not force_config:
                # Video ya completamente configurado, cargar directamente
                messagebox.showinfo(
                    "Video configurado", 
                    f"El video '{fname}' ya está completamente configurado.\n¡Cargando automáticamente!",
                    parent=self.parent
                )
                self._load_video_async(dest)
            else:
                # Video necesita configuración o se fuerza configuración
                if force_config:
                    # Configuración forzada desde botón "Configurar"
                    response = True
                    message_title = "Configuración forzada"
                else:
                    # Video necesita configuración, mostrar ventana de setup
                    missing_items = []
                    if not has_polygon:
                        missing_items.append("• Área restrictiva (polígono)")
                    if not has_semaphore:
                        missing_items.append("• Tiempos de semáforo")
                    if not has_avenue:
                        missing_items.append("• Ubicación/avenida")
                    
                    response = messagebox.askyesno(
                        "Configuración incompleta",
                        f"El video '{fname}' necesita configuración:\n\n" + 
                        "\n".join(missing_items) + 
                        "\n\n¿Desea configurarlo ahora?",
                        parent=self.parent
                    )
                
                if response:
                    # Obtener frame de preview para la configuración
                    cap = cv2.VideoCapture(dest)
                    ret, preview_frame = cap.read()
                    cap.release()
                    
                    if ret:
                        self.setup_complete_video_config(dest, preview_frame)
                    else:
                        messagebox.showerror("Error", "No se pudo leer el video para configuración.", parent=self.parent)
                        
        except Exception as e:
            messagebox.showerror("Error", f"Error configurando video: {str(e)}", parent=self.parent)

    def check_polygon_exists(self, video_path):
        """Verifica si ya existe polígono definido para este video"""
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return False
        
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            video_key = self.get_video_key(video_path)
            # Ser más permisivo: solo verificar que exista la clave, no necesariamente 3+ puntos
            return video_key in presets
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
        
        # Configurar icono
        icon_path = resource_path("img/icon.ico")
        if os.path.exists(icon_path):
            setup.iconbitmap(icon_path)
        setup.geometry("1150x700")  # MÁS ANCHA: 940→1150, MÁS ALTA: 650→700
        setup.resizable(True, True)
        
        # Centrar ventana en pantalla
        setup.update_idletasks()
        screen_width = setup.winfo_screenwidth()
        screen_height = setup.winfo_screenheight()
        x = (screen_width - 1150) // 2
        y = (screen_height - 700) // 2
        setup.geometry(f"1150x700+{x}+{y}")
        
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
        
        # MEJORA: Selector visual para franja horaria (layout compacto)
        tk.Label(fields_frame, text="Franja Horaria:").grid(
            row=1, column=0, sticky="nw", padx=5, pady=8)
        
        # Frame para selectores de hora (layout vertical compacto)
        time_frame = tk.Frame(fields_frame)
        time_frame.grid(row=1, column=1, sticky="w", padx=5, pady=4)
        
        # Variables para horarios (STRING para evitar problema formato Spinbox)
        var_start_h = tk.StringVar(value="7")
        var_start_m = tk.StringVar(value="0")
        var_start_ampm = tk.StringVar(value="AM")
        var_end_h = tk.StringVar(value="7")
        var_end_m = tk.StringVar(value="0")
        var_end_ampm = tk.StringVar(value="PM")
        
        # MEJORA: AUTOCOMPLETAR - Cargar valores existentes automáticamente
        existing_avenue = self.get_avenue_for_video(video_path)
        existing_times = self.get_time_preset_for_video(video_path)
        
        # AUTOCOMPLETAR AVENIDA
        if existing_avenue:
            avenue_entry.delete(0, tk.END)
            avenue_entry.insert(0, existing_avenue)
            print(f"✅ AUTOCOMPLETADO: Avenida '{existing_avenue}'")
            
        # AUTOCOMPLETAR TIEMPOS Y HORARIOS
        if existing_times:
            var_green.set(existing_times.get("green", 30))
            var_yellow.set(existing_times.get("yellow", 5))
            var_red.set(existing_times.get("red", 30))
            print(f"✅ AUTOCOMPLETADO: Tiempos semáforo Verde:{var_green.get()}s, Amarillo:{var_yellow.get()}s, Rojo:{var_red.get()}s")
            
            # AUTOCOMPLETAR HORARIOS - Convertir de 24h a 12h AM/PM
            time_slot = existing_times.get("time_slot", "7:00 - 19:00")
            try:
                start_str, end_str = time_slot.split(" - ")
                start_h_24, start_m = map(int, start_str.split(":"))
                end_h_24, end_m = map(int, end_str.split(":"))
                
                # Función para convertir 24h a 12h AM/PM
                if start_h_24 == 0:
                    var_start_h.set(12)
                    var_start_ampm.set("AM")
                elif start_h_24 < 12:
                    var_start_h.set(start_h_24)
                    var_start_ampm.set("AM")
                elif start_h_24 == 12:
                    var_start_h.set(12)
                    var_start_ampm.set("PM")
                else:
                    var_start_h.set(start_h_24 - 12)
                    var_start_ampm.set("PM")
                
                # Convertir hora final de 24h a 12h
                if end_h_24 == 0:
                    var_end_h.set(12)
                    var_end_ampm.set("AM")
                elif end_h_24 < 12:
                    var_end_h.set(end_h_24)
                    var_end_ampm.set("AM")
                elif end_h_24 == 12:
                    var_end_h.set(12)
                    var_end_ampm.set("PM")
                else:
                    var_end_h.set(end_h_24 - 12)
                    var_end_ampm.set("PM")
                
                var_start_m.set(start_m)
                var_end_m.set(end_m)
                
                print(f"✅ AUTOCOMPLETADO: Horarios {var_start_h.get()}:{var_start_m.get():02d} {var_start_ampm.get()} - {var_end_h.get()}:{var_end_m.get():02d} {var_end_ampm.get()}")
            except:
                # Si hay error en formato, usar valores por defecto
                pass
        
        # MEJORA: Layout compacto - Hora inicio con formato 00:00
        tk.Label(time_frame, text="Desde:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w")
        
        # Spinbox para horas (01-12) SIN formato para evitar problemas
        spin_start_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_start_h,
                                 font=("Arial", 9), justify="center")
        spin_start_h.grid(row=0, column=1, padx=2)
        tk.Label(time_frame, text=":").grid(row=0, column=2)
        
        # Spinbox para minutos (00-59) SIN formato para evitar problemas
        spin_start_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_start_m,
                                 font=("Arial", 9), justify="center", increment=15)
        spin_start_m.grid(row=0, column=3, padx=2)
        
        # Selector AM/PM para inicio más compacto
        from tkinter import ttk
        combo_start_ampm = ttk.Combobox(time_frame, textvariable=var_start_ampm, 
                                       values=["AM", "PM"], width=4, state="readonly")
        combo_start_ampm.grid(row=0, column=4, padx=2)
        
        # MEJORA: Layout compacto - Hora fin en fila 1 con formato 00:00
        tk.Label(time_frame, text="Hasta:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w", pady=(5,0))
        
        spin_end_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_end_h,
                               font=("Arial", 9), justify="center")
        spin_end_h.grid(row=1, column=1, padx=2, pady=(5,0))
        tk.Label(time_frame, text=":").grid(row=1, column=2, pady=(5,0))
        
        spin_end_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_end_m,
                               font=("Arial", 9), justify="center", increment=15)
        spin_end_m.grid(row=1, column=3, padx=2, pady=(5,0))
        
        # Selector AM/PM para fin más compacto
        combo_end_ampm = ttk.Combobox(time_frame, textvariable=var_end_ampm, 
                                     values=["AM", "PM"], width=4, state="readonly")
        combo_end_ampm.grid(row=1, column=4, padx=2, pady=(5,0))
        
        # MEJORA: Tiempos del semáforo con Spinbox
        tk.Label(fields_frame, text="Tiempo Verde (s):").grid(
            row=2, column=0, sticky="w", padx=5, pady=8)
        var_green = tk.IntVar(value=30)
        green_spin = tk.Spinbox(fields_frame, from_=1, to=300, width=8, textvariable=var_green,
                               font=("Arial", 10), justify="center", buttonbackground="#4CAF50")
        green_spin.grid(row=2, column=1, sticky="w", padx=5, pady=8)
        
        tk.Label(fields_frame, text="Tiempo Amarillo (s):").grid(
            row=3, column=0, sticky="w", padx=5, pady=8)
        var_yellow = tk.IntVar(value=5)
        yellow_spin = tk.Spinbox(fields_frame, from_=1, to=10, width=8, textvariable=var_yellow,
                                font=("Arial", 10), justify="center", buttonbackground="#FFC107")
        yellow_spin.grid(row=3, column=1, sticky="w", padx=5, pady=8)
        
        tk.Label(fields_frame, text="Tiempo Rojo (s):").grid(
            row=4, column=0, sticky="w", padx=5, pady=8)
        var_red = tk.IntVar(value=30)
        red_spin = tk.Spinbox(fields_frame, from_=1, to=300, width=8, textvariable=var_red,
                             font=("Arial", 10), justify="center", buttonbackground="#F44336")
        red_spin.grid(row=4, column=1, sticky="w", padx=5, pady=8)
        
        # Panel derecho - previsualización y área restringida (MÁS ESPACIO)
        preview_frame_container = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        preview_frame_container.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Configurar pesos para que el panel derecho sea más grande
        main_frame.columnconfigure(1, weight=3)  # Panel derecho más grande
        main_frame.columnconfigure(0, weight=1)  # Panel izquierdo más pequeño
        
        # Título para área restringida
        tk.Label(preview_frame_container, text="Definición de Área Restringida", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Instrucciones para el usuario
        instructions = tk.Label(preview_frame_container, 
                                text="Haga clic en la imagen para definir los vértices del área restringida.\n"
                                    "Se requieren al menos 3 puntos para definir un área válida.",
                                wraplength=450)
        instructions.pack(pady=5)
        
        # Preparar imagen para visualización (MÁS GRANDE)
        h, w = preview_frame.shape[:2]
        max_preview_w, max_preview_h = 650, 450  # Aumentado de 500x350 a 650x450
        
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
            
            # MEJORA: Construir time_slot desde selectores 12h y convertir a 24h
            def convert_12h_to_24h(hour_12, ampm):
                """Convierte hora de formato 12h a 24h"""
                # Convertir a entero base 10 para evitar problema con octales
                if isinstance(hour_12, str):
                    hour_12 = int(hour_12, 10)
                elif isinstance(hour_12, float):
                    hour_12 = int(hour_12)
                    
                if ampm == "AM":
                    if hour_12 == 12:
                        return 0
                    else:
                        return hour_12
                else:  # PM
                    if hour_12 == 12:
                        return 12
                    else:
                        return hour_12 + 12
            
            start_h_24 = convert_12h_to_24h(var_start_h.get(), var_start_ampm.get())
            end_h_24 = convert_12h_to_24h(var_end_h.get(), var_end_ampm.get())
            
            # Convertir minutos también a entero base 10
            start_m = int(str(var_start_m.get()), 10) if isinstance(var_start_m.get(), str) else int(var_start_m.get())
            end_m = int(str(var_end_m.get()), 10) if isinstance(var_end_m.get(), str) else int(var_end_m.get())
            
            start_time = f"{start_h_24:02d}:{start_m:02d}"
            end_time = f"{end_h_24:02d}:{end_m:02d}"
            time_slot = f"{start_time} - {end_time}"
            
            # MEJORA: Obtener valores de Spinbox
            g = var_green.get()
            y = var_yellow.get()
            r = var_red.get()
            
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
                presets[self.get_video_key(video_path)] = polygon_points
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
        # Actualizar indicador visual
        self.current_video_label.config(text=f"📹 {os.path.basename(path)}")
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
        """
        Versión completamente inactiva que simplemente vacía la cola sin procesar nada.
        No detecta placas, ya que esto lo maneja exclusivamente el preprocesamiento.
        """
        while self.plate_running:
            try:
                # Simplemente vaciar la cola sin procesar los datos
                if hasattr(self, 'plate_queue') and not self.plate_queue.empty():
                    try:
                        self.plate_queue.get_nowait()  # Eliminar datos sin procesarlos
                        self.plate_queue.task_done()
                    except Exception:
                        pass
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error en plate_loop: {e}")
                time.sleep(0.5)

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
                self.vehicle_detector = VehicleDetector(model_path=resource_path("models/yolov8n.pt"))
            
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
        self.current_video_label.lift()
        self.system_info_label.lift()
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
        plates_dir = resource_path("data/output/placas")
        vehicles_dir = resource_path("data/output/autos")
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
        """Crea el panel de métricas de rendimiento (oculto)"""
        # Crear pero no mostrar el panel
        self.metrics_frame = tk.Frame(self.plates_frame, bg="#34495e")
        # NO hacer pack() del frame para que no se muestre
        
        # Creamos las referencias a las etiquetas pero no las mostramos
        self.ti_label = tk.Label(self.metrics_frame, bg="#34495e", fg="white", font=("Arial", 10))
        self.tr_label = tk.Label(self.metrics_frame, bg="#34495e", fg="white", font=("Arial", 10))
        self.ir_label = tk.Label(self.metrics_frame, bg="#34495e", fg="white", font=("Arial", 10))

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




    def iniciar_preprocesamiento(self):
        """
        Inicia el procesamiento de infracciones del video actualmente cargado
        """
        if not self.current_video_path:
            messagebox.showwarning(
                "Advertencia", 
                "Primero debe seleccionar un video usando 'CONFIGURACIÓN DE VIDEOS'.",
                parent=self.parent
            )
            return
        
        if not os.path.exists(self.current_video_path):
            messagebox.showerror(
                "Error", 
                "El video cargado ya no existe en el sistema.",
                parent=self.parent
            )
            return
        
        # Verificar configuración básica
        video_key = self.get_video_key(self.current_video_path)
        has_avenue = self.get_avenue_for_video(self.current_video_path) is not None
        has_times = self.get_time_preset_for_video(self.current_video_path) is not None
        has_polygon = self.check_polygon_exists(self.current_video_path)
        
        print(f"🔍 VERIFICACIÓN PROCESAMIENTO:")
        print(f"   📹 Video: {video_key}")
        print(f"   🛣️ Avenida: {'✅' if has_avenue else '❌'}")
        print(f"   ⏱️ Tiempos: {'✅' if has_times else '❌'}")
        print(f"   📐 Polígono: {'✅' if has_polygon else '❌'}")
        
        if not (has_avenue and has_times and has_polygon):
            # Mensaje simple - no permitir procesamiento
            messagebox.showwarning(
                "Configuración Incompleta",
                "El video no está completamente configurado.\n\n"
                f"Estado actual:\n"
                f"• Avenida: {'✅' if has_avenue else '❌'}\n"
                f"• Tiempos de semáforo: {'✅' if has_times else '❌'}\n"
                f"• Área restrictiva: {'✅' if has_polygon else '❌'}\n\n"
                "Use 'CONFIGURACIÓN DE VIDEOS' para completar la configuración.",
                parent=self.parent
            )
            return
        
        # Todo configurado - Iniciar procesamiento directamente
        print("✅ Video completamente configurado. Iniciando procesamiento...")
        try:
            self.load_video(self.current_video_path)
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Error iniciando procesamiento: {str(e)}",
                parent=self.parent
            )

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
            # Actualizar indicador visual
            self.current_video_label.config(text="Ningún video cargado")
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
            presets.pop(self.get_video_key(video_path), None)
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
            polygons.pop(self.get_video_key(video_path), None)
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
        self.current_video_label.lift()
        self.system_info_label.lift()
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

    # ===== FUNCIONES DE LIMPIEZA Y GESTIÓN =====
    
    def limpiar_configuracion_video(self):
        """Limpia solo la configuración del video actual"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            messagebox.showwarning("Advertencia", "No hay video cargado para limpiar.")
            return
        
        # Confirmar acción
        video_name = os.path.basename(self.current_video_path)
        respuesta = messagebox.askyesno(
            "Confirmar Limpieza",
            f"¿Estás seguro de que quieres limpiar la configuración del video:\n'{video_name}'?\n\n"
            "Esto eliminará:\n"
            "• Área restrictiva (polígono)\n"
            "• Tiempos del semáforo\n"
            "• Nombre de la avenida\n\n"
            "El video permanecerá, pero necesitarás reconfigurarlo."
        )
        
        if not respuesta:
            return
        
        try:
            video_key = self.get_video_key(self.current_video_path)
            
            # Limpiar polígono
            if os.path.exists(POLYGON_CONFIG_FILE):
                with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)
                if video_key in presets:
                    del presets[video_key]
                    with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(presets, f, indent=2)
            
            # Limpiar avenida
            if os.path.exists(AVENUE_CONFIG_FILE):
                with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if video_key in cfg:
                    del cfg[video_key]
                    with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
            
            # Limpiar tiempos
            if os.path.exists(PRESETS_FILE):
                with open(PRESETS_FILE, "r") as f:
                    presets = json.load(f)
                if video_key in presets:
                    del presets[video_key]
                    with open(PRESETS_FILE, "w") as f:
                        json.dump(presets, f, indent=2)
            
            # Resetear estado interno
            self.have_polygon = False
            self.polygon_points = []
            self.current_avenue = None
            self.avenue_label.config(text="")
            
            messagebox.showinfo(
                "Limpieza Completada",
                f"La configuración del video '{video_name}' ha sido limpiada.\n"
                "Ahora puedes reconfigurarlo desde cero."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error al limpiar la configuración:\n{str(e)}"
            )

    def limpiar_todas_configuraciones(self):
        """Limpia todas las configuraciones de todos los videos"""
        respuesta = messagebox.askyesno(
            "Confirmar Limpieza Total",
            "¿Estás seguro de que quieres limpiar TODAS las configuraciones?\n\n"
            "Esto eliminará:\n"
            "• Todas las áreas restrictivas\n"
            "• Todos los tiempos de semáforo\n"
            "• Todos los nombres de avenidas\n"
            "• Todos los indicadores de rendimiento\n"
            "• Todas las infracciones registradas\n\n"
            "La aplicación se reseteará completamente."
        )
        
        if not respuesta:
            return
        
        try:
            # Limpiar todos los archivos de configuración
            config_files = [
                POLYGON_CONFIG_FILE,
                AVENUE_CONFIG_FILE,
                PRESETS_FILE,
                resource_path("data/indicadores_rendimiento.json"),
                resource_path("data/infracciones.json")
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    # Crear archivo vacío
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump({}, f, indent=2)
            
            # Limpiar carpeta de outputs
            output_dirs = [resource_path("data/output/placas"), resource_path("data/output/autos")]
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    for file in os.listdir(output_dir):
                        if file.endswith(('.jpg', '.png')):
                            os.remove(os.path.join(output_dir, file))
            
            # Resetear estado interno
            self.have_polygon = False
            self.polygon_points = []
            self.current_avenue = None
            self.avenue_label.config(text="")
            
            # Si hay video cargado, detenerlo
            if hasattr(self, 'current_video_path') and self.current_video_path:
                self.stop_video()
                self.current_video_path = None
            # Actualizar indicador visual
            self.current_video_label.config(text="Ningún video cargado")
            
            messagebox.showinfo(
                "Limpieza Total Completada",
                "Todas las configuraciones han sido limpiadas.\n"
                "La aplicación ha sido reseteada completamente."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error al limpiar todas las configuraciones:\n{str(e)}"
            )

    def eliminar_video_y_config(self):
        """Elimina el video actual y su configuración"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            messagebox.showwarning("Advertencia", "No hay video cargado para eliminar.")
            return
        
        # Confirmar acción
        video_name = os.path.basename(self.current_video_path)
        respuesta = messagebox.askyesno(
            "Confirmar Eliminación",
            f"¿Estás seguro de que quieres ELIMINAR el video:\n'{video_name}'?\n\n"
            "Esto eliminará:\n"
            "• El archivo de video\n"
            "• Toda su configuración\n"
            "• Todas las imágenes generadas\n\n"
            "Esta acción NO se puede deshacer."
        )
        
        if not respuesta:
            return
        
        try:
            video_key = self.get_video_key(self.current_video_path)
            
            # Detener video si está reproduciéndose
            if hasattr(self, 'is_playing') and self.is_playing:
                self.stop_video()
            
            # Eliminar configuración
            self.limpiar_configuracion_video()
            
            # Eliminar archivo de video
            if os.path.exists(self.current_video_path):
                os.remove(self.current_video_path)
            
            # Eliminar imágenes generadas para este video
            video_base = os.path.splitext(video_name)[0]
            output_dirs = [resource_path("data/output/placas"), resource_path("data/output/autos")]
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    for file in os.listdir(output_dir):
                        if file.startswith(video_base):
                            os.remove(os.path.join(output_dir, file))
            
            # Resetear estado
            self.current_video_path = None
            # Actualizar indicador visual
            self.current_video_label.config(text="Ningún video cargado")
            self.video_label.config(image="")
            
            messagebox.showinfo(
                "Video Eliminado",
                f"El video '{video_name}' y toda su configuración han sido eliminados."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error al eliminar el video:\n{str(e)}"
            )
    
    def detect_hardware(self):
        """Detección mejorada de hardware disponible"""
        import subprocess
        
        self.gpu_info = {
            'cuda_available': False,
            'gpu_name': None,
            'gpu_memory': None,
            'gpu_count': 0
        }
        
        # Verificar CUDA con PyTorch
        if torch.cuda.is_available():
            self.gpu_info['cuda_available'] = True
            self.gpu_info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(self.gpu_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                
                if i == 0:  # GPU principal
                    self.gpu_info['gpu_name'] = gpu_name
                    self.gpu_info['gpu_memory'] = gpu_memory
                
                print(f"🔍 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Detección adicional de GPU con wmic (Windows)
        try:
            if not self.gpu_info['cuda_available']:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if line and ('NVIDIA' in line.upper() or 'RADEON' in line.upper() or 'GTX' in line.upper()):
                            self.gpu_info['gpu_name'] = line
                            print(f"🔍 GPU detectada (sin CUDA): {line}")
                            break
        except:
            pass
    
    def configure_hardware_settings(self):
        """Configurar ajustes según hardware detectado"""
        if self.gpu_info['cuda_available']:
            self.device = torch.device('cuda')
            self.using_gpu = True
            
            # Optimizaciones CUDA
            torch.backends.cudnn.benchmark = True  
            torch.backends.cudnn.deterministic = False
            
            # Configuración GPU según memoria disponible
            gpu_memory = self.gpu_info.get('gpu_memory', 0)
            if gpu_memory >= 6:  # GPU alta gama
                self.gpu_imgsz = 640
                self.gpu_conf_threshold = 0.25
                self.gpu_batch_size = 8
                performance_level = "ULTRA"
            elif gpu_memory >= 4:  # GPU media
                self.gpu_imgsz = 512
                self.gpu_conf_threshold = 0.3
                self.gpu_batch_size = 4
                performance_level = "ALTA"
            else:  # GPU básica
                self.gpu_imgsz = 416
                self.gpu_conf_threshold = 0.35
                self.gpu_batch_size = 2
                performance_level = "MEDIA"
            
            print(f"🚀 GPU CONFIGURADA: {self.gpu_info['gpu_name']}")
            print(f"   💫 Nivel: {performance_level} ({gpu_memory:.1f}GB)")
            print(f"   ⚙️ Resolución: {self.gpu_imgsz}px | Lotes: {self.gpu_batch_size}")
            
        else:
            self.device = torch.device('cpu')
            self.using_gpu = False
            
            # Configuración CPU optimizada
            self.cpu_imgsz = 320
            self.cpu_conf_threshold = 0.5
            self.cpu_batch_size = 1
            self.cpu_skip_frames = 2
            
            gpu_name = self.gpu_info.get('gpu_name', 'No detectada')
            print(f"💻 MODO CPU OPTIMIZADO")
            print(f"   🔍 GPU: {gpu_name}")
            print(f"   ⚙️ Resolución: {self.cpu_imgsz}px | Skip: {self.cpu_skip_frames} frames")
    
    def check_internet_connection(self):
        """Verificar conexión a Internet"""
        import urllib.request
        
        try:
            urllib.request.urlopen('http://www.google.com', timeout=3)
            return True
        except:
            return False
    
    def update_system_info(self):
        """Actualizar información del sistema en la interfaz"""
        try:
            # Información de GPU/CPU
            if hasattr(self, 'gpu_info') and self.gpu_info['gpu_name']:
                if self.gpu_info['cuda_available']:
                    gpu_text = f"🚀 {self.gpu_info['gpu_name'][:20]}..."
                else:
                    gpu_text = f"🔍 {self.gpu_info['gpu_name'][:20]}... (sin CUDA)"
            else:
                gpu_text = "💻 Solo CPU"
            
            # Información de Internet
            has_internet = self.check_internet_connection()
            internet_text = "🌐 Conectado" if has_internet else "🔌 Sin Internet"
            
            # Combinar información
            system_text = f"{gpu_text} | {internet_text}"
            self.system_info_label.config(text=system_text)
            
        except Exception as e:
            self.system_info_label.config(text="🔧 Sistema: Detectando...")
            print(f"Error actualizando info del sistema: {e}")

# Fin del módulo VideoPlayerOpenCV