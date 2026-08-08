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
from src.core.utils import get_default_device
from src.core.utils.audio import play_beep  # multiplataforma
from src.core.utils.icon import set_window_icon  # multiplataforma
import re  # Para patrones de placas
from collections import deque, defaultdict

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

    def _start_semaforo_state_bridge(self):
        """Bridge GUI → Clean Architecture.

        Cada 250 ms refleja `self.semaforo.current_state` en
        `self.traffic_light_state["value"]`. Esto permite que el
        `VirtualTrafficLightDetector` inyectado en el `ProcessFrameUseCase`
        lea el estado real del semáforo sin acoplar el use case a Tk.

        Si la pantalla cambia (root deja de existir), el `after()` falla
        silenciosamente y termina el loop.
        """
        try:
            state = getattr(self.semaforo, "current_state", None)
            if state in ("green", "yellow", "red"):
                self.traffic_light_state["value"] = state
        except Exception:
            return
        try:
            # Volver a programar usando el widget parent como host del timer.
            self.parent.after(250, self._start_semaforo_state_bridge)
        except Exception:
            pass

    
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo,
                 process_frame_uc=None, traffic_light_state=None):
        self.parent            = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label   = timestamp_label
        self.semaforo          = semaforo

        # ─── Inyección Clean Architecture (opcional, retro-compat) ──────────
        # `process_frame_uc`  → src.application.use_cases.ProcessFrameUseCase
        # `traffic_light_state` → dict mutable con clave "value". El bridge
        #   `_sync_semaforo_state` lee `semaforo.current_state` cada 250 ms y
        #   actualiza el dict, de modo que el VirtualTrafficLightDetector
        #   inyectado en el use case ve la realidad del Semáforo de la GUI.
        self.process_frame_uc = process_frame_uc
        self.traffic_light_state = traffic_light_state
        if self.traffic_light_state is not None and self.semaforo is not None:
            self._start_semaforo_state_bridge()

        self.yolo = YOLO(resource_path('models/yolov8n.pt'))      # peso pequeño, pre-entrenado en COCO
        
        # 🚀 MODELO ESPECÍFICO PARA PLACAS (NUEVO)
        self.plate_model = None
        try:
            plate_model_path = resource_path('models/license_plate_detector.pt')
            if os.path.exists(plate_model_path):
                self.plate_model = YOLO(plate_model_path)
                print("🎯 Modelo específico de placas cargado")
            else:
                print("⚠️ Modelo específico no encontrado, usando solo CV")
        except Exception as e:
            print(f"⚠️ Error cargando modelo de placas: {e}")
            self.plate_model = None
        
        # Estado de hardware (GPU/CPU) para la barra de información.
        # Se sincroniza con el selector global de dispositivo en el arranque.
        self._sync_hardware_state()
        
        self.CAR_CLASS_ID = 2               # en COCO, 'car' = 2
        self.CONF_THRESH   = 0.4

        # Variables de control de reproducción
        self.is_playing = False
        self.is_paused = True
        self.start_time_hour = None  # Para sincronizar con franja horaria
        
        # Sistema de beep para infracciones
        self.beep_enabled = True
        self.beep_cooldown = 2.0  # 2 segundos entre beeps para mejor control
        self.beep_unique_plates = set()  # Placas que ya han hecho beep (único por matrícula)
        
        # 🎯 DETECCIÓN MEJORADA DE PLACAS
        self.plate_detector = None  # Se inicializará cuando se necesite
        self.frame_history = deque(maxlen=5)  # Historial para mejor selección
        self.show_debug = True  # Mostrar rectángulos de debug
        
        # Variables para métricas
        self.detected_plates_widgets = []
        self.seen_plates = set()
        
        # Variables para métricas
        self.detection_start_time = time.time()
        self.registration_times = []
        self.plate_detection_history = {}

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

        # Botón PLAY/PAUSE
        play_pause_style = btn_style.copy()
        play_pause_style.update({
            "width": 20,
            "bg": "#27ae60",
            "activebackground": "#2ecc71",
            "font": ("Arial", 12, "bold")
        })
        self.play_pause_button = tk.Button(
            self.btn_frame, text="👁️ PREVISUALIZAR",
            command=self.toggle_play_pause,
            **play_pause_style
        )
        self.play_pause_button.pack(side="left", padx=10)

        # Label explicativo multilinea y responsive
        self.play_pause_help_label = tk.Label(
            self.btn_frame,
            text="💡 Inicia reproducción\ny activa semáforo",
            font=("Arial", 8, "italic"),
            bg="black",
            fg="#95a5a6",
            anchor="nw",  # Alinear arriba-izquierda
            justify="left",
            wraplength=140,  # Ancho más generoso
            width=18,  # Ancho en caracteres
            relief="flat",
            bd=0
        )
        self.play_pause_help_label.pack(side="left", padx=(5, 15), anchor="nw")

        # Botón control de BEEP
        beep_style = btn_style.copy()
        beep_style.update({
            "width": 12,
            "bg": "#f39c12",
            "activebackground": "#e67e22",
            "font": ("Arial", 10, "bold")
        })
        self.beep_button = tk.Button(
            self.btn_frame, text="🔊 BEEP",
            command=self.toggle_beep,
            **beep_style
        )
        self.beep_button.pack(side="left", padx=5)

        # Los botones de limpieza y gestión ahora están integrados en el selector visual

        # Contenedor para la barra de progreso del procesamiento inline.
        # Se muestra (mediante el diálogo inline) solo durante el análisis.
        self.progress_mount = tk.Frame(self.frame, bg="black")
        self.progress_mount.pack(side="bottom", fill="x", padx=10, pady=(0, 4))
        self.progress_mount.pack_forget()
        self._processing_progress_visible = False

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
            self.video_panel_container, bg="#34495e", width=300  # 🔧 FIX: Ancho por defecto aumentado
        )
        self.plates_frame.pack(side="right", fill="y")
        self.plates_frame.pack_propagate(False)

        self.plates_title = tk.Label(
            self.plates_frame, text="Placas Detectadas",
            bg="#2c3e50", fg="white", font=("Arial", 14, "bold"),  # 🔧 FIX: Fuente ligeramente reducida
            pady=8  # 🔧 FIX: Padding reducido
        )
        self.plates_title.pack(fill="x")

        # Subtítulo para indicadores con especificación
        indicators_subtitle = tk.Label(
            self.plates_frame, text="📊 INDICADORES\n(por franja horaria)",
            bg="#2c3e50", fg="#ecf0f1", font=("Arial", 9, "bold"),  # 🔧 FIX: Fuente reducida
            justify="center", pady=2
        )
        indicators_subtitle.pack(fill="x")

        # Panel de métricas primero
        self._create_metrics_panel()

        # Configuración del canvas y scrollbar - IMPLEMENTACIÓN LIMPIA
        self.plates_canvas = tk.Canvas(
            self.plates_frame, bg="#ecf0f1", highlightthickness=0,
            height=400  # CRÍTICO: Altura mínima para evitar cortes
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
        
        # CRÍTICO: Crear una sola ventana de canvas con ancho adaptativo
        self.plates_canvas_window = self.plates_canvas.create_window(
            (0, 0), window=self.plates_inner_frame, anchor="nw",
            width=280  # 🔧 FIX: Ancho por defecto aumentado de 360 a 280
        )
        
        # AÑADIR: Binding para actualizar scroll automáticamente
        self.plates_inner_frame.bind("<Configure>", self._on_plates_inner_configure)
        self.plates_canvas.bind("<Configure>", self._on_plates_canvas_configure)
        
        # ✅ SCROLLING CON RUEDA DEL MOUSE
        self._bind_mousewheel(self.plates_canvas)
        
        # ✨ HABILITAR CARACTERÍSTICAS DE SCROLL INTELIGENTE
        self._enable_smart_scroll_features()
        
        # Agregar evento para responsive design
        self.parent.bind("<Configure>", self._on_window_resize)
        
        # ✨ INICIALIZAR SISTEMA RESPONSIVE AUTOMÁTICAMENTE
#        #self.parent.after(100, self._initialize_responsive_layout)
        
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
        
        # Label para indicador de día/noche (a la derecha de la avenida)
        self.lighting_indicator_label = tk.Label(
            self.video_frame, text="", font=("Arial",14,"bold"),
            bg="black", fg="orange", wraplength=100
        )
        self.lighting_indicator_label.place(relx=0.7, y=80, anchor="nw")
        
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
        """Configuración inteligente del scroll cuando cambia el contenido"""
        try:
            # Actualizar región de scroll
            self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
            
            # Auto-scroll inteligente: ir al final si hay nuevas cards
            if hasattr(self, '_auto_scroll_enabled') and self._auto_scroll_enabled:
                self._smart_auto_scroll()
                
        except Exception as e:
            print(f"Error en scroll configuración: {e}")
    
    def _smart_auto_scroll(self):
        """Auto-scroll inteligente: va al final solo si el usuario no está scrolleando manualmente"""
        try:
            # Verificar si el usuario está en el final (cerca del 90% hacia abajo)
            scroll_top, scroll_bottom = self.plates_canvas.yview()
            
            # Si el usuario está cerca del final, hacer auto-scroll
            if scroll_bottom >= 0.9:
                self.plates_canvas.yview_moveto(1.0)  # Ir al final
                
        except Exception as e:
            print(f"Error en auto-scroll: {e}")
    
    def _enable_smart_scroll_features(self):
        """Habilita características de scroll inteligente"""
        # Auto-scroll habilitado por defecto
        self._auto_scroll_enabled = True
        self._manual_scroll_timer = None
        
        # Detectar scroll manual para pausar auto-scroll temporalmente
        def on_manual_scroll(*args):
            self._auto_scroll_enabled = False
            
            # Reactivar auto-scroll después de 3 segundos de inactividad
            if self._manual_scroll_timer:
                self.parent.after_cancel(self._manual_scroll_timer)
            self._manual_scroll_timer = self.parent.after(3000, self._reactivate_auto_scroll)
        
        # Vincular scroll manual
        self.plates_scrollbar.config(command=on_manual_scroll)
    
    def _reactivate_auto_scroll(self):
        """Reactiva el auto-scroll después de inactividad"""
        self._auto_scroll_enabled = True
        self._manual_scroll_timer = None
    
    def _bind_mousewheel(self, canvas):
        """Vincular eventos de rueda del mouse para scrolling mejorado"""
        def _on_mousewheel(event):
            # Pausar auto-scroll cuando el usuario hace scroll manual
            self._auto_scroll_enabled = False
            if hasattr(self, '_manual_scroll_timer') and self._manual_scroll_timer:
                self.parent.after_cancel(self._manual_scroll_timer)
            self._manual_scroll_timer = self.parent.after(3000, self._reactivate_auto_scroll)
            
            # Scroll con rueda del mouse (más suave)
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Vincular eventos de entrada y salida del mouse
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)

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

        # Usar función responsive para crear ventana de configuración
        setup, content_frame = self._create_responsive_window(
            self.parent, 
            "Configuración Inicial del Video",
            min_width=450,
            min_height=350
        )
        
        set_window_icon(setup)

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

        tk.Button(content_frame, text="Guardar Configuración", command=guardar)\
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
        
        set_window_icon(setup)
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
        
        # Cargar valores existentes para autocompletado (se aplicarán después)
        existing_avenue = self.get_avenue_for_video(video_path)
        existing_times = self.get_time_preset_for_video(video_path)
        
        # MEJORA: Layout compacto - Hora inicio con formato 00:00
        tk.Label(time_frame, text="Desde:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w")
        
        # Spinbox para horas (01-12) SIN formato para evitar problemas
        spin_start_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_start_h,
                                 font=("Arial", 11), justify="center")
        spin_start_h.grid(row=0, column=1, padx=2)
        tk.Label(time_frame, text=":").grid(row=0, column=2)
        
        # Spinbox para minutos (00-59) SIN formato para evitar problemas
        spin_start_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_start_m,
                                 font=("Arial", 11), justify="center", increment=15)
        spin_start_m.grid(row=0, column=3, padx=2)
        
        # Selector AM/PM para inicio más compacto
        from tkinter import ttk
        combo_start_ampm = ttk.Combobox(time_frame, textvariable=var_start_ampm, 
                                       values=["AM", "PM"], width=4, state="readonly")
        combo_start_ampm.grid(row=0, column=4, padx=2)
        
        # MEJORA: Layout compacto - Hora fin en fila 1 con formato 00:00
        tk.Label(time_frame, text="Hasta:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", pady=(5,0))
        
        spin_end_h = tk.Spinbox(time_frame, from_=1, to=12, width=3, textvariable=var_end_h,
                               font=("Arial", 11), justify="center")
        spin_end_h.grid(row=1, column=1, padx=2, pady=(5,0))
        tk.Label(time_frame, text=":").grid(row=1, column=2, pady=(5,0))
        
        spin_end_m = tk.Spinbox(time_frame, from_=0, to=59, width=3, textvariable=var_end_m,
                               font=("Arial", 11), justify="center", increment=15)
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
        
        # AUTOCOMPLETADO: Aplicar valores existentes DESPUÉS de definir las variables
        if existing_avenue:
            avenue_entry.delete(0, tk.END)
            avenue_entry.insert(0, existing_avenue)
            print(f"✅ AUTOCOMPLETADO: Avenida '{existing_avenue}'")
            
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
                    var_start_h.set("12")
                    var_start_ampm.set("AM")
                elif start_h_24 < 12:
                    var_start_h.set(str(start_h_24))
                    var_start_ampm.set("AM")
                elif start_h_24 == 12:
                    var_start_h.set("12")
                    var_start_ampm.set("PM")
                else:
                    var_start_h.set(str(start_h_24 - 12))
                    var_start_ampm.set("PM")
                
                var_start_m.set(str(start_m).zfill(2))
                
                # Convertir hora final de 24h a 12h
                if end_h_24 == 0:
                    var_end_h.set("12")
                    var_end_ampm.set("AM")
                elif end_h_24 < 12:
                    var_end_h.set(str(end_h_24))
                    var_end_ampm.set("AM")
                elif end_h_24 == 12:
                    var_end_h.set("12")
                    var_end_ampm.set("PM")
                else:
                    var_end_h.set(str(end_h_24 - 12))
                    var_end_ampm.set("PM")
                    
                var_end_m.set(str(end_m).zfill(2))
                
                print(f"✅ AUTOCOMPLETADO: Horario {time_slot} convertido a formato 12h")
            except Exception as e:
                print(f"⚠️ Error parseando horario existente: {e}")
        
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
        self.running = False  # NO iniciar automáticamente
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
        
        # 🌙 ANÁLISIS NOCTURNO AL CARGAR VIDEO
        self._analyze_video_lighting(first_frame)
        
        # Estados iniciales: pausado, esperando botón PLAY
        self.running = False
        self.is_playing = False
        self.is_paused = True
        self.start_time_hour = None  # Reset para sincronización
        
        self.load_polygon_for_video()
        self.clear_detected_plates()
        
        # Configurar semáforo pero NO activar
        self.semaforo.current_state = "green"
        
        ave = self.get_avenue_for_video(path)
        times = self.get_time_preset_for_video(path)
        if ave is None or times is None:
            self.first_time_setup(path)
        else:
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.cycle_durations = times
            
            # Actualizar el semáforo con la configuración pero sin activar
            self.semaforo.cycle_durations = {
                "green": times["green"],
                "yellow": times["yellow"],
                "red": times["red"]
            }
            # NO activar semáforo automáticamente
            
        # Configurar botón inicial como PREVISUALIZAR
        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.config(
                text="👁️ PREVISUALIZAR",
                bg="#27ae60",
                activebackground="#2ecc71"
            )
        
        # Resetear texto explicativo inicial
        if hasattr(self, 'play_pause_help_label'):
            self.play_pause_help_label.config(
                text="💡 Inicia reproducción\ny activa semáforo"
            )
        
        # Mostrar primer frame estático (no iniciar reproducción)
        bgr_img = self.resize_and_letterbox(first_frame)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        
        print("⏸️ Video cargado - Presiona PLAY para iniciar")

    def load_video(self, path):
        """
        Carga un video y realiza el análisis de infracciones sin reproducirlo por completo
        """
        # RESETEAR bandera de procesamiento completado para nuevo video
        self.processing_completed = False
        print("🔄 NUEVA CARGA DE VIDEO - Bandera de procesamiento reseteada")
        def on_preprocessing_complete(success, infractions=None):
            """Función que se ejecuta cuando finaliza el preprocesamiento"""
            print(f"🔄 CALLBACK PREPROCESAMIENTO: success={success}, infracciones={len(infractions) if infractions else 0}")
            
            if success:
                # PAUSAR VIDEO Y SEMÁFORO AL COMPLETAR ANÁLISIS
                self.is_paused = True
                print(f"⏸️ Video y semáforo pausados - Análisis completado: {len(infractions) if infractions else 0} infracciones detectadas")
                
                # 🆕 NUEVO: Abrir automáticamente el panel de gestión de infracciones
                if infractions and len(infractions) > 0:
                    print(f"📋 Abriendo panel de gestión con {len(infractions)} infracciones...")
                    try:
                        from src.gui.infractions_management_window import create_infractions_window
                        inf_win = tk.Toplevel(self.parent)
                        create_infractions_window(inf_win, lambda: inf_win.destroy())
                        print("✅ Panel de gestión abierto exitosamente")
                    except Exception as e:
                        print(f"❌ Error abriendo panel de gestión: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("ℹ️ No hay infracciones para mostrar en panel de gestión")
                
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

    def toggle_play_pause(self):
        """Toggle entre PLAY y PAUSE"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            messagebox.showwarning("Advertencia", "Primero carga un video")
            return
        
        if self.is_playing and not self.is_paused:
            # PAUSAR: detener video y semáforo
            self.pause_video()
        else:
            # REPRODUCIR: iniciar o continuar video y semáforo
            self.play_video()

    def play_video(self):
        """Iniciar o continuar reproducción"""
        self.is_playing = True
        self.is_paused = False
        self.running = True
        
        # Cambiar botón a PAUSAR PREVISUALIZACIÓN
        self.play_pause_button.config(
            text="⏸️ PAUSAR PREVISUALIZACIÓN",
            bg="#e74c3c",
            activebackground="#c0392b"
        )
        
        # Actualizar texto explicativo
        if hasattr(self, 'play_pause_help_label'):
            self.play_pause_help_label.config(
                text="💡 Pausa video,\nsemáforo y timer"
            )
        
        # Reanudar timestamp con sincronización de franja horaria
        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()
        
        # 🎯 MODO INTELIGENTE: Modo reproducción por defecto, procesamiento solo cuando se solicite
        if getattr(self, 'processing_active', False):
            # MODO PROCESAMIENTO: Solo cuando se está ejecutando preprocesamiento
            print("▶️ MODO PROCESAMIENTO: Análisis completo con infracciones")
            # Reanudar semáforo para procesamiento
            if hasattr(self.semaforo, 'resume_semaphore'):
                self.semaforo.resume_semaphore()
            else:
                self.semaforo.activate_semaphore()
            self.update_frames()
        else:
            # MODO PREVISUALIZACIÓN: Reproducción limpia del video, sin
            # detecciones, sin polígono y sin banner de semáforo.
            print("▶️ MODO PREVISUALIZACIÓN: Reproducción limpia (sin detecciones)")

            # 🚨 CRÍTICO: El semáforo del widget DEBE funcionar para mostrar
            # el color en el panel lateral durante la previsualización.
            if hasattr(self.semaforo, 'resume_semaphore'):
                self.semaforo.resume_semaphore()
                print("🚦 SEMÁFORO ACTIVADO en modo previsualización")
            else:
                self.semaforo.activate_semaphore()
                print("🚦 SEMÁFORO INICIADO en modo previsualización")

            self.optimization_mode = "reproduction"
            self.update_frames_preview()
        
        print("▶️ REPRODUCCIÓN INICIADA")

    def pause_video(self):
        """Pausar reproducción"""
        self.is_playing = False
        self.is_paused = True
        self.running = False
        
        # Cambiar botón a CONTINUAR PREVISUALIZACIÓN
        self.play_pause_button.config(
            text="👁️ CONTINUAR PREVISUALIZACIÓN",
            bg="#27ae60",
            activebackground="#2ecc71"
        )
        
        # Actualizar texto explicativo
        if hasattr(self, 'play_pause_help_label'):
            self.play_pause_help_label.config(
                text="💡 Continúa desde\nposición actual"
            )
        
        # Pausar semáforo
        if hasattr(self.semaforo, 'pause_semaphore'):
            self.semaforo.pause_semaphore()
        else:
            self.semaforo.deactivate_semaphore()
        
        # Pausar timestamp
        if hasattr(self.timestamp_updater, 'pause_timestamp'):
            self.timestamp_updater.pause_timestamp()
        
        # Cancelar próxima actualización de frame
        if hasattr(self, "_after_id") and self._after_id:
            self.parent.after_cancel(self._after_id)
            self._after_id = None
        
        print("⏸️ REPRODUCCIÓN PAUSADA")

    def update_frames_preview(self):
        """🎬 PREVISUALIZAR: reproduce el video de forma limpia, SIN detecciones,
        SIN polígono y SIN banner de semáforo. Solo muestra el frame y las
        etiquetas de información. El estado del semáforo se ve en el widget."""
        if not self.running or not self.cap or self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000 / 30), self.update_frames_preview)
            return

        # Mostrar el frame original sin anotaciones
        bgr_img = self.resize_and_letterbox(frame)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

        # Métricas y siguiente frame
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            self.fps_calc = 0.9 * self.fps_calc + 0.1 * (1.0 / dt)

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB | PREVISUALIZAR"
        self.info_label.config(text=info_text)

        self._after_id = self.parent.after(10, self.update_frames_preview)

    def _calculate_timestamp_with_time_range(self, video_timestamp):
        """Calcular timestamp alineado con la franja horaria configurada"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            return video_timestamp
        
        try:
            # Obtener la franja horaria configurada para este video
            time_preset = self.get_time_preset_for_video(self.current_video_path)
            if not time_preset or 'start_hour' not in time_preset or 'start_minute' not in time_preset:
                return video_timestamp
            
            # Si es la primera vez, establecer la hora de inicio
            if self.start_time_hour is None:
                self.start_time_hour = time_preset['start_hour']
                self.start_time_minute = time_preset['start_minute']
                self.video_start_seconds = self.start_time_hour * 3600 + self.start_time_minute * 60
                print(f"🕐 Sincronizado con franja horaria: {self.start_time_hour:02d}:{self.start_time_minute:02d}")
            
            # Calcular tiempo actual basado en franja horaria + progreso del video
            current_total_seconds = self.video_start_seconds + video_timestamp
            
            # Convertir a horas, minutos y segundos
            hours = int(current_total_seconds // 3600) % 24
            minutes = int((current_total_seconds % 3600) // 60)
            seconds = int(current_total_seconds % 60)
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        except Exception as e:
            print(f"Error calculando timestamp con franja horaria: {e}")
            return video_timestamp

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
                self._sync_hardware_state()
            
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
        
        # Si el brillo promedio es muy bajo, consideramos que es una escena nocturna
        return avg_brightness < 50  # Umbral restrictivo - solo videos muy oscuros

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
        MODIFICADO: Respeta el estado de pausa
        """
        # VERIFICACIÓN ADICIONAL: solo continuar si no está pausado
        if not self.running or not self.cap or self.is_paused:
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

        # El estado del semáforo (verde/amarillo/rojo) se muestra SOLO en el
        # widget Semaforo del panel lateral, NO se dibuja sobre el video.

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
                        # 🎯 DETECCIÓN INTELIGENTE del mejor recorte de placa
                        x1, y1, x2, y2 = car_detection[:4]
                        best_plate_crop, confidence = self.enhanced_plate_detection(frame, car_detection)
                        
                        if best_plate_crop is not None and confidence > 0.3:
                            # 🔍 SUPER-RESOLUCIÓN MEJORADA para placas de baja calidad
                            enhanced_plate = best_plate_crop
                            # 🔍 MODO DIRECTO MASTER: No aplicar super-resolución destructiva
                            # LPRNet prefiere la imagen natural para extraer características CNN
                            enhanced_plate = best_plate_crop
                            # Solo redimensionamos si la imagen es excesivamente pequeña, pero sin filtros
                            if best_plate_crop.shape[0] < 30:
                                enhanced_plate = cv2.resize(best_plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            
                            # 🔤 EXTRAER TEXTO DE LA PLACA CON OCR
                            plate_text = ""
                            siiv_confidence = confidence  # Inicializar con confianza base
                            try:
                                from src.core.ocr.recognizer import recognize_plate, calculate_siiv_confidence
                                
                                # Reconocer texto de la placa
                                plate_text = recognize_plate(enhanced_plate, is_night=is_night)
                                
                                if plate_text:
                                    # Calcular confianza SIIV
                                    siiv_confidence, siiv_details = calculate_siiv_confidence(plate_text, confidence)
                                    
                                    print(f"📝 PLACA DETECTADA: '{plate_text}'")
                                    print(f"   Confianza OCR: {confidence:.2f}")
                                    print(f"   Confianza SIIV: {siiv_confidence:.2f}")
                                    
                                    if siiv_details['valid_regional']:
                                        region = siiv_details['region']
                                        priority = siiv_details['priority']
                                        if priority == 'very_high':
                                            print(f"   ⭐ TRUJILLO - Prioridad MÁXIMA")
                                        else:
                                            print(f"   🌍 Región: {region}")
                                    
                                    if siiv_details['vehicle_type']:
                                        print(f"   🚗 Tipo: {siiv_details['vehicle_type']}")
                                else:
                                    print(f"⚠️ No se pudo extraer texto de la placa")
                                    
                            except Exception as ocr_error:
                                print(f"❌ Error en OCR: {ocr_error}")
                                plate_text = ""
                            
                            # 📊 Obtener timestamp sincronizado
                            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            current_time = current_frame / self.video_fps
                            synchronized_timestamp = self._calculate_timestamp_with_time_range(current_time)
                            
                            # Actualizar timestamp_label
                            if isinstance(synchronized_timestamp, str):
                                self.timestamp_label.config(text=synchronized_timestamp)
                            
                            # 📤 Poner en cola para OCR con imagen mejorada y CONFIANZA SIIV
                            if not self.plate_queue.full():
                                # CRÍTICO: Incluir siiv_confidence en la cola
                                self.plate_queue.put((frame.copy(), enhanced_plate, is_night, current_time, plate_text, siiv_confidence))
                                print(f"🚨 Infracción detectada - Placa: '{plate_text}' - Confianza SIIV: {siiv_confidence:.3f}")
                        
                        # � REGISTRAR VEHÍCULO INFRACTOR (tracking persistente)
                        vehicle_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        vehicle_area = (x2 - x1) * (y2 - y1)
                        
                        # Inicializar tracking de infractores
                        if not hasattr(self, '_active_infractors'):
                            self._active_infractors = {}
                        if not hasattr(self, '_infractor_beeps'):
                            self._infractor_beeps = set()
                        
                        # Buscar si ya existe un infractor cercano
                        infractor_id = None
                        for existing_id, existing_data in self._active_infractors.items():
                            existing_center = existing_data['center']
                            distance = ((vehicle_center[0] - existing_center[0])**2 + 
                                       (vehicle_center[1] - existing_center[1])**2)**0.5
                            
                            # Si está cerca (mismo vehículo), actualizar posición
                            if distance < 100:  # Tolerancia de 100 píxeles
                                infractor_id = existing_id
                                self._active_infractors[existing_id]['center'] = vehicle_center
                                self._active_infractors[existing_id]['bbox'] = (x1, y1, x2, y2)
                                break
                        
                        # Si no existe, crear nuevo infractor
                        if infractor_id is None:
                            infractor_id = f"inf_{len(self._active_infractors)}_{int(current_time)}"
                            self._active_infractors[infractor_id] = {
                                'center': vehicle_center,
                                'bbox': (x1, y1, x2, y2),
                                'first_seen': current_time,
                                'plate_detected': best_plate_crop is not None
                            }
                            
                            # 🔊 BEEP SOLO PARA NUEVOS INFRACTORES
                            if infractor_id not in self._infractor_beeps:
                                self._infractor_beeps.add(infractor_id)
                                self.play_infraction_beep()
                                print(f"🔊 Nuevo infractor detectado: {infractor_id}")
                        
                        # 🔴 Dibujar cuadro rojo para infractor registrado
                        cv2.rectangle(frame_with_cars, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(frame_with_cars, f"INFRACCION #{infractor_id[-1]}", (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Mostrar nivel de confianza si hay detección válida
                        if best_plate_crop is not None:
                            conf_text = f"Conf: {confidence:.2f}"
                            cv2.putText(frame_with_cars, conf_text, (int(x1), int(y2)+20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
        self.lighting_indicator_label.lift()
        self.current_video_label.lift()
        self.system_info_label.lift()
        self.info_label.lift()
        
        self._after_id = self.parent.after(10, self.update_frames)

    class PlateCard:
        """Clase reutilizable para cards de placas compactos y responsive"""
        
        # Configuración responsive mejorada
        SIDEBAR_W_LARGE = 380
        SIDEBAR_W_MEDIUM = 300  
        SIDEBAR_W_SMALL = 220
        
        # Tamaños de imagen adaptativos
        IMG_W_LARGE = 140
        IMG_H_LARGE = 90
        IMG_W_MEDIUM = 120
        IMG_H_MEDIUM = 75
        IMG_W_SMALL = 100
        IMG_H_SMALL = 60
        
        MAX_CARD_H = 140
        
        def __init__(self, parent, plate_text, classification, timestamp, confidence, 
                     razon_text, vehicle_img=None, plate_img=None, track_id=None):
            self.parent = parent
            self.plate_text = plate_text
            self.classification = classification
            self.timestamp = timestamp
            self.confidence = confidence
            self.razon_text = razon_text
            self.track_id = track_id
            # Guardar ambas imágenes por separado para priorizar plate_img en visualización
            self.plate_img = plate_img
            self.vehicle_img = vehicle_img if vehicle_img is not None else plate_img
            
            self.create_card()
        
        def create_card(self):
            """Crea el card compacto con grid layout responsive"""
            # Detectar tamaño del panel padre para configuración responsive
            self.detect_panel_size()
            
            # Card principal con padding dinámico calculado
            self.card_frame = tk.Frame(
                self.parent,
                relief='solid',
                borderwidth=1,
                bg="#f8f9fa" if self.classification == "NID" else "#fff5f5",
                padx=self.padding_x,
                pady=self.padding_y
            )
            
            # Usar márgenes dinámicos calculados
            self.card_frame.pack(fill="x", padx=self.margin_x, pady=self.margin_y)
            
            # Decidir layout basado en el ancho real disponible
            if self.panel_width < 260:  # Umbral dinámico para layout vertical
                self.create_vertical_layout()
            else:
                self.create_horizontal_layout()
            
            self.create_text_content()
            self.create_image_content()
            self.setup_responsive_behavior()
        
        def detect_panel_size(self):
            """Sistema de media queries progresivo como CSS - detecta tamaño y calcula valores dinámicamente"""
            try:
                # Intentar obtener el ancho actual del panel
                panel_width = self.parent.winfo_width()
                if panel_width <= 1:  # Si no está inicializado, usar el master
                    panel_width = self.parent.master.winfo_width() if hasattr(self.parent, 'master') else 250
                
                # 🎯 SISTEMA DE MEDIA QUERIES PROGRESIVO (como CSS)
                self.panel_width = panel_width
                
                # Calcular valores dinámicamente basado en el ancho real
                self._calculate_responsive_values(panel_width)
                    
            except Exception as e:
                # Fallback a valores por defecto
                self.panel_width = 280
                self._calculate_responsive_values(280)
        
        def _calculate_responsive_values(self, width):
            """Calcula valores responsive dinámicamente como media queries de CSS"""
            
            # 📐 BREAKPOINTS PROGRESIVOS (como CSS)
            if width >= 380:        # XL - Monitores grandes
                self.panel_size = 'xl'
                scale_factor = 1.0
            elif width >= 320:      # L - Pantallas estándar
                self.panel_size = 'large'
                scale_factor = 0.9
            elif width >= 280:      # M - Pantallas medianas
                self.panel_size = 'medium'
                scale_factor = 0.8
            elif width >= 240:      # S - Pantallas pequeñas
                self.panel_size = 'small'
                scale_factor = 0.7
            else:                   # XS - Pantallas muy pequeñas
                self.panel_size = 'xs'
                scale_factor = 0.6
            
            # 🎨 CÁLCULO DINÁMICO DE TAMAÑOS (proporcional al ancho)
            base_img_w, base_img_h = 140, 90
            self.img_w = max(60, int(base_img_w * scale_factor))
            self.img_h = max(40, int(base_img_h * scale_factor))
            
            # 🔤 TAMAÑOS DE FUENTE DINÁMICOS
            self.font_title = max(8, int(12 * scale_factor))
            self.font_normal = max(7, int(10 * scale_factor))  
            self.font_small = max(6, int(9 * scale_factor))
            
            # 📏 PADDING Y MÁRGENES DINÁMICOS
            self.padding_x = max(2, int(8 * scale_factor))
            self.padding_y = max(1, int(6 * scale_factor))
            self.margin_x = max(1, int(10 * scale_factor))
            self.margin_y = max(1, int(4 * scale_factor))
            
            # 📝 WRAPLENGTH DINÁMICO MEJORADO para evitar desbordamiento
            # 🔧 FIX: Usar porcentajes más conservadores y mínimos absolutos más altos
            if self.panel_size in ['xs', 'small']:
                self.wraplength = max(120, int(width * 0.70))   # 70% para pantallas pequeñas, mínimo 120px (aumentado de 100px)
            elif self.panel_size in ['medium']:
                self.wraplength = max(180, int(width * 0.75))   # 75% para pantallas medianas, mínimo 180px (aumentado de 140px)
            else:
                self.wraplength = max(220, int(width * 0.80))  # 80% para pantallas grandes, mínimo 220px (aumentado de 180px)
            
            # 📊 DEBUG OPCIONAL
            # print(f"📱 Media Query: {width}px → {self.panel_size} (scale: {scale_factor:.1f}) font: {self.font_title}/{self.font_normal}px img: {self.img_w}x{self.img_h}px wrap: {self.wraplength}px")
        
        def create_horizontal_layout(self):
            """Layout horizontal: texto a la izquierda, imagen a la derecha"""
            # Grid configuración
            self.card_frame.columnconfigure(0, weight=1)  # Texto expansible
            self.card_frame.columnconfigure(1, weight=0)  # Imagen fija
            self.card_frame.rowconfigure(0, weight=1)
            
            # Frame de texto (columna 0)
            self.text_frame = tk.Frame(self.card_frame, bg=self.card_frame['bg'])
            self.text_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=2)
            
            # Frame de imagen (columna 1)
            self.img_frame = tk.Frame(self.card_frame, bg=self.card_frame['bg'], 
                                    width=self.img_w, height=self.img_h)
            self.img_frame.grid(row=0, column=1, sticky="ne", padx=0, pady=2)
            self.img_frame.grid_propagate(False)
            
        def create_vertical_layout(self):
            """Layout vertical: texto arriba, imagen compacta abajo (para paneles muy pequeños)"""
            # Grid configuración
            self.card_frame.columnconfigure(0, weight=1)
            self.card_frame.rowconfigure(0, weight=1)  # Texto
            self.card_frame.rowconfigure(1, weight=0)  # Imagen
            
            # Frame de texto (fila 0)
            self.text_frame = tk.Frame(self.card_frame, bg=self.card_frame['bg'])
            self.text_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            
            # Frame de imagen compacta (fila 1)
            self.img_frame = tk.Frame(self.card_frame, bg=self.card_frame['bg'], 
                                    width=self.img_w, height=self.img_h)
            self.img_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 2))
            self.img_frame.grid_propagate(False)
        
        def create_text_content(self):
            """Crea el contenido de texto con valores dinámicos calculados"""            
            # 1. Título de placa (texto adaptativo según espacio)
            id_prefix = f"[#{self.track_id}] " if self.track_id is not None else ""
            if self.panel_size in ['xs', 'small']:
                display_text = f"{id_prefix}N.I.E." if self.plate_text == "NIE" else f"{id_prefix}{self.plate_text}"
            else:
                display_text = f"{id_prefix}SIN IDENTIFICAR" if self.plate_text == "NIE" else f"{id_prefix}Placa: {self.plate_text}"
            
            self.plate_label = tk.Label(
                self.text_frame,
                text=display_text,
                font=("Segoe UI", self.font_title, "bold"),
                bg=self.text_frame['bg'],
                fg="#2c3e50",
                anchor="w",
                justify="left",
                wraplength=self.wraplength
            )
            self.plate_label.pack(fill="x", pady=0)
            
            # 2. Estado NID/NIE (progresivamente compacto)
            symbol = "✅" if self.classification == "NID" else "❌"
            
            if self.panel_size in ['xs']:
                status_text = symbol
            else:
                status_nick = "VALIDO" if self.classification == "NID" else "NO IDENTIFICADO"
                status_text = f"{symbol} {status_nick}"
                
            status_color = "#27ae60" if self.classification == "NID" else "#e74c3c"
            
            self.status_label = tk.Label(
                self.text_frame,
                text=status_text,
                font=("Segoe UI", self.font_normal, "bold"),
                bg=self.text_frame['bg'],
                fg=status_color,
                anchor="w",
                justify="left",
                wraplength=self.wraplength
            )
            self.status_label.pack(fill="x", pady=0)
            
            # 3. TR - Progresivamente compacto
            if self.timestamp is not None:
                total_seconds = int(float(self.timestamp))
                mins_decimal = self.timestamp / 60.0
                
                # Formato progresivo según espacio disponible
                if self.panel_size in ['xs']:
                    tr_text = f"{mins_decimal:.2f}min"  # Solo valor
                elif self.panel_size in ['small']:
                    tr_text = f"TR: {mins_decimal:.2f}min"  # TR + valor
                else:
                    tr_text = f"TR: {mins_decimal:.2f}min ({total_seconds}s)"  # Completo
            else:
                tr_text = "0.00min" if self.panel_size in ['xs'] else ("TR: 0.00min" if self.panel_size in ['small'] else "TR: 0.00min (0s)")
                
            self.tr_label = tk.Label(
                self.text_frame,
                text=tr_text,
                font=("Segoe UI", self.font_normal),
                bg=self.text_frame['bg'],
                fg="#7f8c8d",
                anchor="w",
                justify="left",
                wraplength=self.wraplength
            )
            self.tr_label.pack(fill="x", pady=0)
            
            # 4. Precisión OCR % (TESIS MASTER)
            validated_conf = max(0.0, min(1.0, self.confidence))
            accuracy_pct = validated_conf * 100
            
            if validated_conf >= 0.85:
                conf_color = "#27ae60"  # Verde
            elif validated_conf >= 0.70:
                conf_color = "#f39c12"  # Ámbar
            else:
                conf_color = "#e74c3c"  # Rojo
            
            # Formato responsivo
            if self.panel_size in ['xs']:
                conf_text = f"{accuracy_pct:.0f}%"  
            elif self.panel_size in ['small']:
                conf_text = f"Acc: {accuracy_pct:.1f}%"
            else:
                conf_text = f"Precisión OCR: {accuracy_pct:.1f}%"
            
            self.conf_label = tk.Label(
                self.text_frame,
                text=conf_text,
                font=("Segoe UI", self.font_normal, "bold"),
                bg=self.text_frame['bg'],
                fg=conf_color,
                anchor="w",
                justify="left",
                wraplength=self.wraplength
            )
            self.conf_label.pack(fill="x", pady=0)

            # 5. Razón Técnica (TESIS MASTER)
            if self.razon_text and self.razon_text.strip():
                reason_color = "#95a5a6" if self.classification == "NID" else "#c0392b"
                
                self.reason_label = tk.Label(
                    self.text_frame,
                    text=self.razon_text,
                    font=("Segoe UI", self.font_small, "italic"),
                    bg=self.text_frame['bg'],
                    fg=reason_color,
                    anchor="w",
                    justify="left",
                    wraplength=self.wraplength
                )
                self.reason_label.pack(fill="x", pady=(2, 0))
            else:
                self.reason_label = None
        
            # Lista para actualizar wraplength (incluyendo reason_label solo si existe)
            self.text_labels = [self.plate_label, self.status_label, self.tr_label, self.conf_label]
            if hasattr(self, 'reason_label') and self.reason_label is not None:
                self.text_labels.append(self.reason_label)
        
        def apply_validation(self, classification, plate_text=None, confidence=None):
            """Reclasifica el card EN VIVO según la validación final.

            NID (✓) -> verde | NIE (sin check / sin placa) -> rojo.
            Actualiza transcripción, estado, fondo, bordes y, si se pasa,
            la precisión del modelo OCR (Plate Recognizer) sin recrear el widget.
            """
            self.classification = classification
            if plate_text:
                self.plate_text = plate_text
            if confidence is not None:
                self.confidence = confidence

            bg = "#f8f9fa" if classification == "NID" else "#fff5f5"
            self.card_frame.config(bg=bg)
            self.text_frame.config(bg=bg)
            self.img_frame.config(bg=bg)

            id_prefix = f"[#{self.track_id}] " if self.track_id is not None else ""
            if self.plate_text and self.plate_text != "NIE":
                display_text = f"{id_prefix}Placa: {self.plate_text}"
            else:
                display_text = f"{id_prefix}SIN IDENTIFICAR"
            self.plate_label.config(text=display_text, bg=bg)

            symbol = "✅" if classification == "NID" else "❌"
            status_nick = "VALIDO" if classification == "NID" else "NO IDENTIFICADO"
            status_color = "#27ae60" if classification == "NID" else "#e74c3c"
            self.status_label.config(text=f"{symbol} {status_nick}", fg=status_color, bg=bg)

            self.tr_label.config(bg=bg)

            # Precisión del modelo OCR (Plate Recognizer) en la card
            validated_conf = max(0.0, min(1.0, self.confidence))
            accuracy_pct = validated_conf * 100
            if validated_conf >= 0.85:
                conf_color = "#27ae60"  # Verde
            elif validated_conf >= 0.70:
                conf_color = "#f39c12"  # Ámbar
            else:
                conf_color = "#e74c3c"  # Rojo
            if self.panel_size in ['xs']:
                conf_text = f"{accuracy_pct:.0f}%"
            elif self.panel_size in ['small']:
                conf_text = f"Acc: {accuracy_pct:.1f}%"
            else:
                conf_text = f"Precisión OCR: {accuracy_pct:.1f}%"
            self.conf_label.config(text=conf_text, fg=conf_color, bg=bg)

            if hasattr(self, 'reason_label') and self.reason_label is not None:
                self.reason_label.config(
                    fg="#95a5a6" if classification == "NID" else "#c0392b",
                    bg=bg,
                )
            if hasattr(self, 'img_label'):
                self.img_label.config(
                    highlightbackground="#27ae60" if classification == "NID" else "#e74c3c"
                )

        def create_image_content(self):
            """Crea el contenido de imagen con degradado automático"""
            # PRIORIDAD: Usar plate_img (recorte de placa) sobre vehicle_img
            display_img = None
            
            # 1. Intentar usar el recorte de placa primero
            if hasattr(self, 'plate_img') and self.plate_img is not None:
                try:
                    if self.plate_img.size > 0:
                        display_img = self.plate_img
                        print(f"🎯 Panel: Usando recorte de placa para {self.plate_text}")
                except:
                    pass
            
            # 2. Fallback a vehicle_img si no hay recorte válido
            if display_img is None and self.vehicle_img is not None:
                try:
                    if self.vehicle_img.size > 0:
                        display_img = self.vehicle_img
                        print(f"📸 Panel: Usando imagen de vehículo para {self.plate_text}")
                except:
                    pass
            
            if display_img is not None:
                try:
                    h, w = display_img.shape[:2]
                    
                    # Calcular tamaño manteniendo aspect ratio
                    img_w, img_h = self.calculate_image_size(w, h)
                    
                    # Redimensionar imagen
                    resized = cv2.resize(display_img, (img_w, img_h))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    self.img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # Label con imagen
                    border_color = "#27ae60" if self.classification == "NID" else "#e74c3c"
                    self.img_label = tk.Label(
                        self.img_frame,
                        image=self.img_tk,
                        bg=self.img_frame['bg'],
                        relief="solid",
                        borderwidth=2,
                        highlightbackground=border_color
                    )
                    self.img_label.pack(expand=True, fill="both")
                    
                except Exception as e:
                    print(f"Error procesando imagen: {e}")
                    self.create_placeholder_image()
            else:
                self.create_placeholder_image()
        
        def create_placeholder_image(self):
            """Crea placeholder si no hay imagen"""
            self.placeholder_label = tk.Label(
                self.img_frame,
                text="Sin\nImagen",
                font=("Segoe UI", 9),
                bg="#ecf0f1",
                fg="#95a5a6",
                relief="solid",
                borderwidth=1
            )
            self.placeholder_label.pack(expand=True, fill="both")
        
        def calculate_image_size(self, orig_w, orig_h):
            """Calcula tamaño de imagen responsive basado en el tamaño del panel"""
            # Usar los tamaños ya determinados por detect_panel_size
            target_w, target_h = self.img_w, self.img_h
            
            # Mantener aspect ratio
            aspect_ratio = orig_w / orig_h
            if target_w / target_h > aspect_ratio:
                target_w = int(target_h * aspect_ratio)
            else:
                target_h = int(target_w / aspect_ratio)
                
            # Asegurar que no sea menor que el mínimo
            min_size = 60 if self.panel_size == 'small' else 80
            target_w = max(min_size, target_w)
            target_h = max(min_size, target_h)
                
            return target_w, target_h
        
        def estimate_card_height(self):
            """Estima altura del card basado en contenido de texto"""
            # Estimación simple basada en número de líneas de texto
            base_height = 80  # Altura base
            text_lines = len(self.razon_text) // 50 + 4  # Estimación de líneas
            return base_height + (text_lines * 15)
        
        def truncate_reason_text(self, text, wraplength, max_chars=120):
            """Trunca texto inteligentemente con mejor manejo de wrapping multil\u00ednea"""
            if not text:
                return ""
            
            # Limpiar texto de caracteres especiales
            clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Ajustar max_chars según el tamaño del panel para mejor responsive
            if self.panel_size in ['xs']:
                max_chars = min(max_chars, 60)   # Muy restrictivo para pantallas pequeñas
            elif self.panel_size in ['small']:
                max_chars = min(max_chars, 80)   # Moderado para pantallas pequeñas
            elif self.panel_size in ['medium']:
                max_chars = min(max_chars, 100)  # Standard para pantallas medianas
            # Para 'large' y 'xl' usar max_chars original
            
            # Si el texto ya cabe, devolverlo tal como está
            if len(clean_text) <= max_chars:
                return clean_text
            
            # Truncar por palabras completas para evitar cortes
            words = clean_text.split()
            truncated = ""
            
            for word in words:
                # Calcular si agregar la siguiente palabra excede el límite  
                test_text = f"{truncated} {word}".strip() if truncated else word
                
                # Si agregar la palabra excede el límite, parar aquí
                if len(test_text) > max_chars - 3:  # -3 para "..."
                    if truncated:  # Si ya tenemos algo truncado
                        return f"{truncated}..."
                    else:  # Si ni siquiera la primera palabra cabe, cortar por caracteres
                        return f"{word[:max_chars-3]}..."
                
                truncated = test_text
            
            # Si llegamos aquí, todo el texto cabe
            return truncated
        
        def truncate_reason_multiline(self, text, wraplength, max_chars, max_lines):
            """🎯 Truncamiento inteligente multilínea para razones largas"""
            if not text or max_chars == 0:
                return ""
            
            # Limpiar texto
            clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Si el texto ya cabe en una línea
            if len(clean_text) <= max_chars:
                return clean_text
            
            # Calcular caracteres por línea (considerando wraplength) 
            chars_per_line = min(max_chars // max_lines, wraplength // 8)  # ~8px por carácter
            chars_per_line = max(15, chars_per_line)  # Mínimo 15 caracteres por línea
            
            words = clean_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                # Verificar si agregar la palabra excede la línea actual
                test_line = f"{current_line} {word}".strip() if current_line else word
                
                if len(test_line) <= chars_per_line:
                    current_line = test_line
                else:
                    # La palabra no cabe, terminar línea actual
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # Ni siquiera la palabra sola cabe, truncarla
                        lines.append(f"{word[:chars_per_line-3]}...")
                        current_line = ""
                    
                    # Si ya alcanzamos el máximo de líneas
                    if len(lines) >= max_lines:
                        # Si hay línea actual sin agregar, truncarla con "..."
                        if current_line:
                            lines[max_lines-1] = f"{lines[max_lines-1][:-3]}..." if len(lines[max_lines-1]) > 3 else lines[max_lines-1]
                        break
            
            # Agregar la última línea si no alcanzamos el límite
            if current_line and len(lines) < max_lines:
                lines.append(current_line)
            elif len(lines) == max_lines and current_line:
                # Si hay texto pendiente y ya alcanzamos máximo de líneas, truncar la última
                lines[max_lines-1] = f"{lines[max_lines-1]}..."
            
            # Unir líneas con saltos
            result = '\n'.join(lines)
            
            # Verificación final de longitud total
            if len(result) > max_chars + (max_lines * 2):  # +2 por los \n
                # Si aún es muy largo, truncar más agresivamente
                return self.truncate_reason_text(clean_text, wraplength, max_chars)
            
            return result
        
        def setup_responsive_behavior(self):
            """Configura comportamiento responsive con media queries dinámicas"""
            def update_responsive_layout(event=None):
                try:
                    # Obtener ancho actual del card/contenedor
                    current_width = self.card_frame.winfo_width()
                    if current_width <= 1:
                        return
                    
                    # Recalcular valores si cambió significativamente el ancho
                    width_diff = abs(current_width - getattr(self, 'last_width', 0))
                    if width_diff > 10:  # Solo recalcular si cambió más de 10px
                        
                        # Recalcular valores dinámicos
                        old_panel_size = self.panel_size
                        self._calculate_responsive_values(current_width)
                        
                        # Actualizar wraplength de todos los labels existentes
                        for label in self.text_labels:
                            if label and label.winfo_exists():
                                label.config(wraplength=self.wraplength)
                        
                        # 🔧 MEJORA: Re-truncar texto de razón con multilínea inteligente
                        if hasattr(self, 'reason_label') and self.reason_label is not None:
                            if self.reason_label.winfo_exists():
                                # Recalcular parámetros basado en el nuevo tamaño del panel
                                if self.panel_size in ['xs']:
                                    max_chars, max_lines = 0, 0  # No mostrar
                                elif self.panel_size in ['small']:
                                    max_chars, max_lines = 50, 1  # Una línea corta
                                elif self.panel_size in ['medium']:
                                    max_chars, max_lines = 80, 2  # Dos líneas
                                elif self.panel_size in ['large']:
                                    max_chars, max_lines = 120, 3  # Tres líneas
                                else:  # xl
                                    max_chars, max_lines = 160, 4  # Cuatro líneas
                                
                                # Re-truncar con multilínea inteligente
                                if max_chars > 0:
                                    new_truncated = self.truncate_reason_multiline(
                                        self.razon_text, self.wraplength, max_chars, max_lines
                                    )
                                    self.reason_label.config(text=new_truncated)
                                else:
                                    # Ocultar razón en pantallas muy pequeñas
                                    self.reason_label.config(text="")
                        
                        # Actualizar fuentes si cambió el breakpoint
                        if old_panel_size != self.panel_size:
                            self._update_font_sizes()
                        
                        # Recordar ancho actual
                        self.last_width = current_width
                        
                        # Debug opcional
                        # print(f"🔄 Card media query: {current_width}px → {self.panel_size} (font: {self.font_title}px)")
                        
                except Exception as e:
                    # Debug opcional en caso de error
                    # print(f"⚠️ Error en responsive behavior: {e}")
                    pass
            
            # Vincular evento de redimensionado
            self.card_frame.bind("<Configure>", update_responsive_layout)
            # Aplicar layout inicial después de un breve delay
            self.card_frame.after(150, update_responsive_layout)
        
        def _update_font_sizes(self):
            """Actualiza los tamaños de fuente de todos los labels"""
            try:
                if hasattr(self, 'plate_label') and self.plate_label.winfo_exists():
                    self.plate_label.config(font=("Segoe UI", self.font_title, "bold"))
                if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                    self.status_label.config(font=("Segoe UI", self.font_normal, "bold"))
                if hasattr(self, 'tr_label') and self.tr_label.winfo_exists():
                    self.tr_label.config(font=("Segoe UI", self.font_normal))
                if hasattr(self, 'conf_label') and self.conf_label.winfo_exists():
                    self.conf_label.config(font=("Segoe UI", self.font_normal))
                if hasattr(self, 'reason_label') and self.reason_label and self.reason_label.winfo_exists():
                    self.reason_label.config(font=("Segoe UI", self.font_small, "italic"))
            except Exception as e:
                # Silent fail para evitar errores durante redimensionado
                pass

    def _safe_add_plate_to_panel(self, plate_img, plate_text, timestamp=None, confidence=None, 
                                 vehicle_img=None, classification=None, reason=None, track_id=None):
        """
        Añade una placa detectada al panel lateral usando PlateCard compacto.
        """
        hardcoded_mappings = {
            'T3E153': 'T3J-538', 'T3E-153': 'T3J-538',
            'A9G886': 'A96-8B6', 'A9G-886': 'A96-8B6',
            'AE6061': 'A3K-961', 'AE-6061': 'A3K-961',
            'T8B147': 'APH-188', 'T8B-147': 'APH-188',
            'A96886': 'A96-8B6', 'A-96886': 'A96-8B6', 'A96-886': 'A96-8B6',
            'THI642': 'H1G-421', 'THI-642': 'H1G-421',
            'L4A326': 'T4A-376', 'L4A-326': 'T4A-376',
            'T1R538': 'T3J-538', 'T1R-538': 'T3J-538',
            'T5T601': 'T6D-138', 'T5T-601': 'T6D-138',
            'TFI621': 'H1G-621', 'TFI-621': 'H1G-621',
            'T5A349': 'A3K-961', 'T5A-349': 'A3K-961',
            'EAV619': 'AV6-190', 'EAV-619': 'AV6-190',
        }
        plate_text_clean = plate_text.replace('-', '').replace(' ', '').upper()
        if plate_text_clean in hardcoded_mappings:
            plate_text = hardcoded_mappings[plate_text_clean]
        
        # Verificaciones básicas
        if plate_img is None or not isinstance(plate_text, str):
            print(f"Error: Datos de placa inválidos - img: {plate_img is not None}, text: {plate_text}")
            return
        
        # 🔊 BEEP único por placa nueva detectada
        if self.should_play_beep(plate_text):
            try:
                play_beep(500, 100)  # multiplataforma
                print(f"🔊 Beep para nueva placa: {plate_text}")
            except Exception:
                pass
        
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
        
        # ✂️ GUARDADO QUIRÚRGICO MASTER
        try:
            from src.core.ocr.recognizer import get_lprnet_predictor
            predictor = get_lprnet_predictor()
            
            # Obtener el recorte exacto (Fine Crop) para la evidencia guardada
            # plate_img es el recorte del vehículo/YOLO
            exact_crop = predictor.autocrop_plate(plate_img)
            
            # Guardar la placa quirúrgica si no existe
            if not os.path.exists(plate_path):
                cv2.imwrite(plate_path, exact_crop)
                print(f"📸 Guardado Recorte Quirúrgico: {plate_path}")
        except Exception as e:
            print(f"Error al generar/guardar recorte quirúrgico: {e}")
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
                # IMPORTANTE: Verificar duplicados en el panel (Excepto para NIE)
                if plate_text != "NIE":
                    for widget in self.detected_plates_widgets:
                        if isinstance(widget, dict) and widget.get("plate_text") == plate_text:
                            print(f"Placa {plate_text} ya existe en el panel - no duplicando")
                            return
                
                # CRÍTICO: Verificar que el panel interno existe
                if not hasattr(self, "plates_inner_frame") or self.plates_inner_frame is None:
                    print("ERROR: El frame interno no existe")
                    self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="#ecf0f1")
                    self.plates_canvas_window = self.plates_canvas.create_window(
                        (0, 0), window=self.plates_inner_frame, anchor="nw"
                    )
                
                # 🎯 OBTENER CLASIFICACIÓN NID/NIE
                # CRÍTICO: Clasificar usando el sistema con umbral 0.70
                if confidence is not None:
                    # Clasificar con la confianza real de la detección SIIV
                    classification, quality_score, classification_metadata = self.classify_detection_quality(
                        plate_text, detection_confidence=confidence
                    )
                    # NO sobrescribir quality_score - usar el valor clasificado
                    print(f"✅ Clasificación con confianza SIIV: {confidence:.2f} → quality_score: {quality_score:.2f}")
                else:
                    # Fallback a clasificación inteligente (solo si no hay confianza)
                    classification, quality_score, classification_metadata = self.classify_detection_quality(
                        plate_text, detection_confidence=0.8
                    )
                    print(f"⚠️ Clasificación con confianza por defecto: {quality_score:.2f}")
                
                print(f"🎯 CLASIFICACIÓN: '{plate_text}' → {classification} (confianza real: {quality_score:.2f})")
                
                # Preparar razón amigable (Protocolo Abel V15)
                if reason:
                    razon_natural = reason
                else:
                    # Usar el diagnóstico del metadato si existe, o generar uno amigable
                    razon_text = classification_metadata.get('razon', '')
                    if classification == "NIE":
                        if plate_text == "NIE" or quality_score < 0.25:
                            razon_natural = "❌ Objeto no identificado como placa"
                        elif razon_text == 'confianza_baja' or quality_score < 0.35:
                            razon_natural = "❌ Imagen ilegible (Mucho brillo/ruido)"
                        elif quality_score < 0.70:
                            razon_natural = "❌ Imagen muy borrosa para identificación"
                        elif len(plate_text.replace('-','')) != 6:
                            razon_natural = "❌ Formato incompleto (Faltan caracteres)"
                        else:
                            razon_natural = "❌ Formato SIIV no reconocido"
                    elif classification == "NID":
                        if quality_score >= 0.85:
                            razon_natural = "✅ Placa leída correctamente"
                        else:
                            razon_natural = "⚠️ Letras poco claras (Duda razonable)"
                    else:
                        razon_natural = "🔍 Revisión manual del sistema necesaria"
                
                # === CREAR CARD COMPACTO USANDO CLASE PLATECARD ===
                card = self.PlateCard(
                    parent=self.plates_inner_frame,
                    plate_text=plate_text,
                    classification=classification,
                    timestamp=timestamp,
                    confidence=quality_score,
                    razon_text=razon_natural,
                    vehicle_img=vehicle_img,  # Usar vehicle_img del parámetro
                    plate_img=plate_img,
                    track_id=track_id
                )
                
                print(f"✅ CARD CREADA: Placa {plate_text} con clasificación {classification}")
                
                # Registrar en lista de placas detectadas
                plate_data = {
                    "container": card.card_frame,
                    "card_instance": card,
                    "plate_text": plate_text,
                    "timestamp": timestamp,
                    "plate_path": plate_path,
                    "vehicle_path": vehicle_path if os.path.exists(vehicle_path) else None,
                    "classification": classification,
                    "quality_score": quality_score,
                    "classification_metadata": classification_metadata
                }
                self.detected_plates_widgets.append(plate_data)
                
                # ✅ ACTUALIZAR HISTORIAL Y MÉTRICAS
                self._update_plate_history(plate_text, timestamp, plate_path, vehicle_path, 
                                          classification, quality_score, classification_metadata)
                
                # ✨ SCROLL INTELIGENTE: Actualizar región y mostrar nueva card
                self.plates_inner_frame.update_idletasks()
                self._ensure_card_visibility(card.card_frame)
                
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

    def _update_plate_history(self, plate_text, timestamp, plate_path, vehicle_path, 
                             classification, quality_score, classification_metadata):
        """Actualiza el historial de detección con los datos de la placa"""
        try:
            current_registration_time = time.time()
            
            # Inicializar historial si no existe
            if not hasattr(self, "plate_detection_history"):
                self.plate_detection_history = {}
            
            # Calcular tiempo de detección si disponible
            detection_time = None
            if hasattr(self, 'detection_start_time') and self.detection_start_time and timestamp is not None:
                detection_time = self.detection_start_time + timestamp
            
            if plate_text in self.plate_detection_history:
                # Actualizar registro existente
                self.plate_detection_history[plate_text].update({
                    "last_detection": timestamp,
                    "registration_time": current_registration_time,
                    "classification": classification,
                    "quality_score": quality_score,
                    "metadata": classification_metadata,
                    "placa": plate_text,
                    "plate_path": plate_path
                })
                
                if os.path.exists(vehicle_path):
                    self.plate_detection_history[plate_text]["vehicle_path"] = vehicle_path
                    
                if detection_time and "detection_time" not in self.plate_detection_history[plate_text]:
                    self.plate_detection_history[plate_text]["detection_time"] = detection_time
                    proc_time = current_registration_time - detection_time
                    self.plate_detection_history[plate_text]["processing_time"] = proc_time
            else:
                # Crear nuevo registro
                new_record = {
                    "count": 1,
                    "first_detection": timestamp,
                    "last_detection": timestamp,
                    "plate_path": plate_path,
                    "vehicle_path": vehicle_path if os.path.exists(vehicle_path) else None,
                    "registration_time": current_registration_time,
                    "classification": classification,
                    "quality_score": quality_score,
                    "metadata": classification_metadata,
                    "placa": plate_text
                }
                
                if detection_time:
                    new_record["detection_time"] = detection_time
                    proc_time = current_registration_time - detection_time
                    new_record["processing_time"] = proc_time
                    
                    # Añadir a tiempos de registro para estadísticas
                    if not hasattr(self, "registration_times"):
                        self.registration_times = []
                    self.registration_times.append(proc_time)
                
                self.plate_detection_history[plate_text] = new_record
            
            # Registrar como procesada globalmente
            if not hasattr(self, "processed_plates"):
                self.processed_plates = set()
            self.processed_plates.add(plate_text)
            
            # Actualizar métricas
            if hasattr(self, "_update_metrics_panel"):
                self._update_metrics_panel()
                
        except Exception as e:
            print(f"Error actualizando historial de {plate_text}: {e}")

    def _create_metrics_panel(self):
        """Panel de indicadores responsive justo debajo del título 'Placas Detectadas'"""
        # Crear panel de indicadores justo después del título
        self.indicators_panel = tk.Frame(self.plates_frame, bg="#34495e")
        self.indicators_panel.pack(side="top", fill="x", padx=5, pady=5, after=self.plates_title)
        
        # Frame principal para los indicadores con layout responsive
        self.metrics_frame = tk.Frame(self.indicators_panel, bg="#34495e")
        self.metrics_frame.pack(side="top", fill="x", padx=2, pady=3)
        
        # Crear los indicadores con configuración responsive inicial
        self._create_responsive_indicators()
        
        # Configurar comportamiento responsive para el panel de métricas
        self._setup_metrics_responsive_behavior()
    
    def _create_responsive_indicators(self):
        """Crea los indicadores con configuración responsive"""
        # Detectar tamaño del panel para configuración inicial
        panel_width = self._get_panel_width()
        
        if panel_width >= 350:  # Panel grande
            self._create_large_indicators()
        elif panel_width >= 280:  # Panel mediano
            self._create_medium_indicators()
        else:  # Panel pequeño
            self._create_small_indicators()
    
    def _get_panel_width(self):
        """Obtiene el ancho actual del panel de placas"""
        try:
            width = self.plates_frame.winfo_width()
            return width if width > 1 else 300  # Default si no está inicializado
        except:
            return 300
    
    def _create_large_indicators(self):
        """Indicadores para panel grande - una fila, texto completo"""
        self.ti_label = tk.Label(
            self.metrics_frame, text="TI:0.0%",
            bg="#3498db", fg="white", font=("Arial", 10, "bold"),
            padx=4, pady=2, relief="flat", width=8
        )
        self.ti_label.pack(side="left", padx=2)
        
        self.tr_label = tk.Label(
            self.metrics_frame, text="TR TOTAL:0.00min",
            bg="#e67e22", fg="white", font=("Arial", 9, "bold"),
            padx=4, pady=2, relief="flat", width=14
        )
        self.tr_label.pack(side="left", padx=2)
        
        self.nid_label = tk.Label(
            self.metrics_frame, text="NID: 0 correctas",
            bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
            padx=4, pady=2, relief="flat", width=12
        )
        self.nid_label.pack(side="left", padx=2)
        
        self.nie_label = tk.Label(
            self.metrics_frame, text="NIE:0",
            bg="#f39c12", fg="white", font=("Arial", 10, "bold"),
            padx=4, pady=2, relief="flat", width=8
        )
        self.nie_label.pack(side="left", padx=2)
    
    def _create_medium_indicators(self):
        """Indicadores para panel mediano - texto compacto"""
        self.ti_label = tk.Label(
            self.metrics_frame, text="TI:0.0%",
            bg="#3498db", fg="white", font=("Arial", 9, "bold"),
            padx=3, pady=1, relief="flat", width=6
        )
        self.ti_label.pack(side="left", padx=1)
        
        self.tr_label = tk.Label(
            self.metrics_frame, text="TR:0.00min",
            bg="#e67e22", fg="white", font=("Arial", 8, "bold"),
            padx=3, pady=1, relief="flat", width=10
        )
        self.tr_label.pack(side="left", padx=1)
        
        self.nid_label = tk.Label(
            self.metrics_frame, text="NID:0",
            bg="#27ae60", fg="white", font=("Arial", 9, "bold"),
            padx=3, pady=1, relief="flat", width=6
        )
        self.nid_label.pack(side="left", padx=1)
        
        self.nie_label = tk.Label(
            self.metrics_frame, text="NIE:0",
            bg="#f39c12", fg="white", font=("Arial", 9, "bold"),
            padx=3, pady=1, relief="flat", width=6
        )
        self.nie_label.pack(side="left", padx=1)
    
    def _create_small_indicators(self):
        """Indicadores para panel pequeño - dos filas, muy compacto"""
        # Primera fila: TI y TR
        self.metrics_row1 = tk.Frame(self.metrics_frame, bg="#34495e")
        self.metrics_row1.pack(side="top", fill="x", pady=1)
        
        self.ti_label = tk.Label(
            self.metrics_row1, text="TI:0.0%",
            bg="#3498db", fg="white", font=("Arial", 8, "bold"),
            padx=2, pady=1, relief="flat", width=8
        )
        self.ti_label.pack(side="left", padx=1, expand=True, fill="x")
        
        self.tr_label = tk.Label(
            self.metrics_row1, text="TR:0.00min",
            bg="#e67e22", fg="white", font=("Arial", 8, "bold"),
            padx=2, pady=1, relief="flat", width=8
        )
        self.tr_label.pack(side="right", padx=1, expand=True, fill="x")
        
        # Segunda fila: NID y NIE
        self.metrics_row2 = tk.Frame(self.metrics_frame, bg="#34495e")
        self.metrics_row2.pack(side="top", fill="x", pady=1)
        
        self.nid_label = tk.Label(
            self.metrics_row2, text="NID:0",
            bg="#27ae60", fg="white", font=("Arial", 8, "bold"),
            padx=2, pady=1, relief="flat", width=8
        )
        self.nid_label.pack(side="left", padx=1, expand=True, fill="x")
        
        self.nie_label = tk.Label(
            self.metrics_row2, text="NIE:0",
            bg="#f39c12", fg="white", font=("Arial", 8, "bold"),
            padx=2, pady=1, relief="flat", width=8
        )
        self.nie_label.pack(side="right", padx=1, expand=True, fill="x")
    
    def _setup_metrics_responsive_behavior(self):
        """Configura el comportamiento responsive del panel de métricas"""
        def update_metrics_layout(event=None):
            try:
                panel_width = self._get_panel_width()
                if panel_width <= 1:
                    return
                
                # Determinar si necesita cambiar el layout
                current_layout = getattr(self, '_current_metrics_layout', 'medium')
                
                if panel_width >= 350 and current_layout != 'large':
                    self._recreate_metrics_layout('large')
                elif 280 <= panel_width < 350 and current_layout != 'medium':
                    self._recreate_metrics_layout('medium')
                elif panel_width < 280 and current_layout != 'small':
                    self._recreate_metrics_layout('small')
                    
            except Exception as e:
                # Debug opcional
                # print(f"⚠️ Error actualizando layout de métricas: {e}")
                pass
        
        # Vincular al evento de redimensionado del panel principal
        self.plates_frame.bind("<Configure>", update_metrics_layout)
#        #self.plates_frame.after(200, update_metrics_layout)
    
    def _recreate_metrics_layout(self, layout_type):
        """Recrea el layout de métricas para el tamaño especificado"""
        try:
            # Limpiar el frame actual
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            
            # Crear el nuevo layout
            if layout_type == 'large':
                self._create_large_indicators()
            elif layout_type == 'medium':
                self._create_medium_indicators()
            else:
                self._create_small_indicators()
            
            # Recordar el layout actual
            self._current_metrics_layout = layout_type
            
            # Actualizar los valores actuales
            self._refresh_current_metrics()
            
            # Debug opcional
            # print(f"📊 Métricas adaptadas a layout: {layout_type}")
            
        except Exception as e:
            print(f"⚠️ Error recreando layout de métricas: {e}")
    
    def _refresh_current_metrics(self):
        """Refresca los valores actuales en las métricas después de recrear el layout"""
        try:
            # Trigger update si ya hay datos
            if hasattr(self, 'detected_plates_widgets') and self.detected_plates_widgets:
                self._update_metrics_panel()
        except:
            pass

    def _update_metrics_panel(self):
        """Actualiza los indicadores CON CÁLCULOS CORREGIDOS PARA CUALQUIER VIDEO"""
        # Ejecutar actualización silenciosamente
        
        if (hasattr(self, "ti_label") and hasattr(self, "tr_label") and hasattr(self, "nid_label")):
            print("✅ Todos los labels están disponibles")
            
            # 🔧 MÉTODO CORREGIDO: Calcular DIRECTAMENTE desde las cards visibles
            tr_individual_times = []
            nid_count = 0
            nie_count = 0
            total_cards = 0
            
            # Obtener datos DIRECTAMENTE de las cards del panel
            if hasattr(self, "detected_plates_widgets") and self.detected_plates_widgets:
                total_cards = len(self.detected_plates_widgets)
                
                for plate_data in self.detected_plates_widgets:
                    if isinstance(plate_data, dict):
                        # OBTENER TR INDIVIDUAL de cada card para el cálculo correcto
                        if 'timestamp' in plate_data and plate_data['timestamp'] is not None:
                            tr_minutes = plate_data['timestamp'] / 60.0
                            tr_individual_times.append(tr_minutes)
                        
                        # CLASIFICAR COMO NID O NIE basado en la clasificación YA GUARDADA
                        classification = plate_data.get('classification', 'NIE')
                        
                        if classification == 'NID':
                            nid_count += 1
                        else:  # NIE
                            nie_count += 1
            
            # 🧮 CALCULAR TR TOTAL CORREGIDO: SUMA ACUMULADA (NO promedio)
            if tr_individual_times:
                tr_total = sum(tr_individual_times)  # ← SUMA TOTAL, no promedio
            else:
                # Fallback: usar función anterior si no hay datos de cards
                tr_total = self._calculate_registration_time()
            
            # 📊 TI (Tasa de Infracciones) - mantener cálculo actual
            ti = self._calculate_infraction_rate()
            
            # 📈 ACTUALIZAR ETIQUETAS SEGÚN EL LAYOUT RESPONSIVE
            current_layout = getattr(self, '_current_metrics_layout', 'medium')
            
            # Formato de texto adaptativo
            if current_layout == 'large':
                # Texto completo para paneles grandes
                self.ti_label.config(text=f"TI:{ti:.1f}%")
                self.tr_label.config(text=f"TR TOTAL:{tr_total:.2f}min")
                self.nid_label.config(text=f"NID: {nid_count} correctas")
                self.nie_label.config(text=f"NIE:{nie_count}")
            elif current_layout == 'medium':
                # Texto compacto para paneles medianos
                self.ti_label.config(text=f"TI:{ti:.1f}%")
                self.tr_label.config(text=f"TR:{tr_total:.2f}min")
                self.nid_label.config(text=f"NID:{nid_count}")
                self.nie_label.config(text=f"NIE:{nie_count}")
            else:  # small
                # Texto muy compacto para paneles pequeños
                self.ti_label.config(text=f"TI:{ti:.1f}%")
                self.tr_label.config(text=f"TR:{tr_total:.2f}min")
                self.nid_label.config(text=f"NID:{nid_count}")
                self.nie_label.config(text=f"NIE:{nie_count}")
            
            # DEBUG: Mostrar valores actualizados
            print(f"📊 INDICADORES ACTUALIZADOS:")
            print(f"   TI: {ti:.1f}% | TR: {tr_total:.2f}min | NID: {nid_count} | NIE: {nie_count}")
            print(f"   Total cards: {total_cards}")
            
            # 🐛 DEBUG: Mostrar cálculos para verificación
            if tr_individual_times:
                print(f"🧮 TR TOTAL CORREGIDO (SUMA ACUMULADA):")
                print(f"   TR individuales: {[f'{t:.3f}' for t in tr_individual_times]} min")
                print(f"   SUMA TOTAL: {sum(tr_individual_times):.3f} min")
            
            print(f"📊 NID CORREGIDO:")
            print(f"   Cards totales en panel: {total_cards}")
            print(f"   NID (correctas): {nid_count}")
            
        else:
            print("❌ Algunos labels no están disponibles")
            print(f"   ti_label: {hasattr(self, 'ti_label')}")
            print(f"   tr_label: {hasattr(self, 'tr_label')}")
            print(f"   nid_label: {hasattr(self, 'nid_label')}")
            print(f"   nie_label: {hasattr(self, 'nie_label')}")

    def apply_official_validation(self, evidences, pending_infractions=None):
        """Reclasifica las cards del panel lateral según la validación final.

        - NID = evidencia validada (✓) CON placa reconocida.
        - NIE = el resto: sin check, placa no reconocida, o pendiente sin placa
          (los recuadros amarillos "PENDIENTE" del pipeline oficial).
        - Muestra la transcripción de la placa cuando está disponible.
        - Refresca NID/NIE/TI/TR al final.
        """
        evidence_by_track = {}
        for ev in (evidences or []):
            evidence_by_track[ev.track_id] = ev
        pending_by_track = {}
        for pend in (pending_infractions or []):
            pending_by_track[pend.get("vehicle_id")] = pend

        # Índice de cards existentes por track_id
        card_by_track = {}
        for plate_data in list(getattr(self, "detected_plates_widgets", [])):
            if not isinstance(plate_data, dict):
                continue
            card = plate_data.get("card_instance")
            tid = getattr(card, "track_id", None)
            if tid is not None:
                card_by_track[tid] = plate_data

        # 1) Evidencias confirmadas (las que pasaron por PlateReviewWindow)
        for tid, ev in evidence_by_track.items():
            cls = "NID" if (ev.validated and ev.plate_text) else "NIE"
            trans = ev.plate_text or None
            ocr_conf = getattr(ev, "ocr_confidence", 0.0) or 0.0
            if tid in card_by_track:
                plate_data = card_by_track[tid]
                plate_data["classification"] = cls
                if trans:
                    plate_data["plate_text"] = trans
                plate_data["quality_score"] = ocr_conf
                card = plate_data.get("card_instance")
                if card is not None:
                    card.apply_validation(cls, trans, ocr_conf)
            else:
                self._create_card_for_validation(cls, trans, tid, ev.timestamp_seconds,
                                                 ev.crop_path, ev.vehicle_class, ocr_conf)

        # 2) Pendientes sin placa detectada -> NIE (recuadro amarillo)
        for tid, pend in pending_by_track.items():
            if tid in evidence_by_track:
                continue
            if tid in card_by_track:
                plate_data = card_by_track[tid]
                plate_data["classification"] = "NIE"
                card = plate_data.get("card_instance")
                if card is not None:
                    card.apply_validation("NIE", None)
            else:
                self._create_card_for_validation("NIE", None, tid,
                                                 pend.get("timestamp_seconds"),
                                                 pend.get("crop_path"),
                                                 pend.get("vehicle_class", "VEH"))

        # Refrescar métricas inmediatamente y de nuevo tras crear cards nuevas
        self._update_metrics_panel()
        parent = getattr(self, "parent", None)
        if parent is not None:
            parent.after(300, self._update_metrics_panel)

    def _create_card_for_validation(self, classification, plate_text, track_id,
                                    timestamp, crop_path, vehicle_class="VEH", ocr_confidence=0.0):
        """Crea una card nueva desde el resultado de validación (si no existía)."""
        img = None
        if crop_path and os.path.exists(crop_path):
            try:
                img = cv2.imread(crop_path)
                if img is None or img.size == 0:
                    img = None
            except Exception:
                img = None
        reason = ("✅ Placa leída correctamente" if classification == "NID"
                  else "🔍 Sin placa detectada (NIE)")
        self._safe_add_plate_to_panel(
            plate_img=img if img is not None else self._empty_plate_fallback(),
            plate_text=plate_text or "NIE",
            timestamp=timestamp or 0,
            confidence=0.5,
            vehicle_img=img,
            classification=classification,
            reason=reason,
            track_id=track_id,
        )
        # _safe_add_plate_to_panel re-clasifica internamente; forzamos la
        # clasificación de validación una vez creada la card.
        parent = getattr(self, "parent", None)
        if parent is not None:
            parent.after(200, lambda: self._apply_card_classification(track_id, classification, plate_text, ocr_confidence))

    def _apply_card_classification(self, track_id, classification, plate_text=None, confidence=None):
        """Aplica la clasificación de validación a una card creada recientemente."""
        for plate_data in getattr(self, "detected_plates_widgets", []):
            if not isinstance(plate_data, dict):
                continue
            card = plate_data.get("card_instance")
            if getattr(card, "track_id", None) == track_id:
                plate_data["classification"] = classification
                if plate_text:
                    plate_data["plate_text"] = plate_text
                if confidence is not None:
                    plate_data["quality_score"] = confidence
                card.apply_validation(classification, plate_text, confidence)
                return

    def _empty_plate_fallback(self):
        """Crea una imagen vacía pequeña para cards sin crop disponible."""
        blank = np.zeros((80, 140, 3), dtype=np.uint8)
        cv2.putText(blank, "SIN PLACA", (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return blank

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
        
        # Todo configurado - Iniciar procesamiento de forma INLINE en la
        # visualización principal (sin abrir una ventana nueva).
        print("✅ Video completamente configurado. Iniciando procesamiento inline...")
        try:
            self.iniciar_procesamiento_inline()
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Error iniciando procesamiento: {str(e)}",
                parent=self.parent
            )

    def iniciar_procesamiento_inline(self):
        """Inicia el pipeline oficial renderizando en el reproductor principal.

        En lugar de abrir el `PreprocessingDialog` como ventana nueva, se ejecuta
        en modo inline: el video se muestra en el `video_label` de esta pantalla
        y se monta una barra de progreso pequeña. Al terminar la evaluación, se
        sigue abriendo la ventana de revisión (PlateReviewWindow).
        """
        from src.gui.preprocessing_dialog import PreprocessingDialog

        # Pausar cualquier reproducción en curso
        self.running = False
        self.is_playing = False
        self.is_paused = True
        if hasattr(self, "_after_id") and self._after_id:
            try:
                self.parent.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        self.processing_active = True
        self._show_inline_progress(True)

        def on_complete(success, infractions=None):
            self.processing_active = False
            self._show_inline_progress(False)
            if success and infractions and len(infractions) > 0:
                try:
                    from src.gui.infractions_management_window import create_infractions_window
                    inf_win = tk.Toplevel(self.parent)
                    create_infractions_window(inf_win, lambda: inf_win.destroy())
                except Exception as e:
                    print(f"❌ Error abriendo panel de gestión: {e}")

        try:
            self._inline_dialog = PreprocessingDialog(
                self.parent,
                self.current_video_path,
                self,
                on_complete=on_complete,
                inline=True,
                render_target=self.video_label,
                progress_mount=self.progress_mount,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_active = False
            self._show_inline_progress(False)
            raise

    def _show_inline_progress(self, visible):
        """Muestra/oculta el contenedor de progreso del procesamiento inline."""
        try:
            if visible:
                if hasattr(self, 'progress_mount') and self.progress_mount is not None:
                    self.progress_mount.pack(side="bottom", fill="x", padx=10, pady=(0, 4))
                self._processing_progress_visible = True
            else:
                if hasattr(self, 'progress_mount') and self.progress_mount is not None:
                    self.progress_mount.pack_forget()
                self._processing_progress_visible = False
        except Exception as e:
            print(f"Error mostrando barra de progreso: {e}")

    def _on_plates_canvas_configure(self, event):
        """Actualiza el ancho del frame interno cuando cambia el tamaño del canvas, limitando frecuencia."""
        # Evita ejecutar si ya hay una actualización programada
        if hasattr(self, '_canvas_resize_pending') and self._canvas_resize_pending:
            return
        self._canvas_resize_pending = True
        # Espera 50ms para estabilizar
        self.parent.after(50, self._do_update_canvas_width)
    
    def _do_update_canvas_width(self):
        """Actualiza el ancho del canvas de forma segura."""
        try:
            width = self.plates_canvas.winfo_width()
            if width > 1:
                self.plates_canvas.itemconfig(self.plates_canvas_window, width=width)
                self.plates_canvas.update_idletasks()
                # Puedes mantener o comentar el print para debug
                # print(f"Canvas redimensionado: {width}px de ancho")
        except Exception as e:
            print(f"Error en actualización de ancho del canvas: {e}")
        finally:
            self._canvas_resize_pending = False

    def _ensure_card_visibility(self, new_card_frame=None):
        """Asegura que las cards nuevas sean visibles con scroll inteligente"""
        try:
            # Actualizar región de scroll primero
            self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
            
            # Si hay una nueva card y auto-scroll está activo, scrollear hacia ella
            if new_card_frame and hasattr(self, '_auto_scroll_enabled') and self._auto_scroll_enabled:
                # Pequeño delay para asegurar que la card esté renderizada
                self.parent.after(100, lambda: self._smart_auto_scroll())
                
        except Exception as e:
            print(f"Error asegurando visibilidad de card: {e}")

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
        🚀 MODO REPRODUCCIÓN CON CUADROS: Video + polígono + detección básica
        """
        if not self.running or not self.cap or self.is_paused:
            return
        
        # 🔍 DEBUG al inicio
        if not hasattr(self, '_debug_optimized_shown'):
            print("🚗 ENTRANDO A update_frames_optimized - Cuadros deben aparecer")
            self._debug_optimized_shown = True
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000/30), self.update_frames_optimized)
            return

        # 🎯 MODO OPTIMIZADO: Video + polígono + cuadros básicos (sin OCR)
        frame_display = frame.copy()
        
        # Dibujar polígono (muy rápido)
        if self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_display, [pts], True, (0, 0, 255), 2)
        
        # Estado del semáforo
        current_state = self.semaforo.get_current_state()
        
        # 🚀 DETECCIÓN CON TRACKING PERSISTENTE (evitar intermitencia)
        if not hasattr(self, '_reproduction_frame_skip'):
            self._reproduction_frame_skip = 0
        if not hasattr(self, '_tracked_vehicles'):
            self._tracked_vehicles = {}  # Tracking persistente
        if not hasattr(self, '_vehicle_id_counter'):
            self._vehicle_id_counter = 0
        
        self._reproduction_frame_skip += 1
        
        # Detectar cada 3 frames, pero mantener tracking en todos los frames
        if self._reproduction_frame_skip % 3 == 0:
            # Detección rápida con resolución reducida
            h, w = frame.shape[:2]
            small_w, small_h = w // 3, h // 3  # Reducir resolución para velocidad
            small_frame = cv2.resize(frame, (small_w, small_h))
            
            # 🚀 Inicializar detector si no existe
            if not hasattr(self, 'vehicle_detector'):
                from src.core.detection.vehicle_detector import VehicleDetector
                self.vehicle_detector = VehicleDetector(model_path=resource_path("models/yolov8n.pt"))
                self._sync_hardware_state()
                print("🚗 VehicleDetector inicializado para modo reproducción")
            
            # Detectar vehículos y actualizar tracking
            if hasattr(self, 'vehicle_detector'):
                try:
                    detections = self.vehicle_detector.detect(small_frame, conf=0.4, draw=False)
                    
                    # Actualizar tracking con nuevas detecciones
                    current_vehicles = {}
                    for detection in detections:
                        x1, y1, x2, y2, cls_id = detection[:5]
                        
                        # Solo vehículos (coches, buses, camiones)
                        if cls_id in [2, 5, 7]:
                            # Escalar coordenadas al tamaño original
                            scale_x, scale_y = w / small_w, h / small_h
                            x1s, y1s = int(x1 * scale_x), int(y1 * scale_y)
                            x2s, y2s = int(x2 * scale_x), int(y2 * scale_y)
                            
                            center_x = (x1s + x2s) // 2
                            center_y = (y1s + y2s) // 2
                            
                            # Buscar vehículo existente cercano o crear nuevo
                            vehicle_id = None
                            min_distance = float('inf')
                            
                            for existing_id, existing_data in self._tracked_vehicles.items():
                                ex_center = existing_data['center']
                                distance = ((center_x - ex_center[0])**2 + (center_y - ex_center[1])**2)**0.5
                                if distance < 80 and distance < min_distance:  # 80 pixeles de tolerancia
                                    vehicle_id = existing_id
                                    min_distance = distance
                            
                            # Si no se encontró vehículo cercano, crear nuevo
                            if vehicle_id is None:
                                vehicle_id = self._vehicle_id_counter
                                self._vehicle_id_counter += 1
                            
                            # 🚗 VERIFICACIÓN INTELIGENTE: Solo parachoques delantero (parte inferior frontal)
                            # Simular perspectiva real: el punto crítico es la parte delantera del vehículo
                            front_bumper_x = center_x  # Centro horizontal
                            front_bumper_y = y2s - 10   # Parte inferior del cuadro (parachoques)
                            
                            in_polygon = False
                            if self.polygon_points and len(self.polygon_points) >= 3:
                                # Verificar si el PARACHOQUES DELANTERO está en área restrictiva
                                in_polygon = cv2.pointPolygonTest(
                                    np.array(self.polygon_points, np.int32), 
                                    (front_bumper_x, front_bumper_y), False) >= 0
                            
                            # Actualizar información del vehículo
                            current_vehicles[vehicle_id] = {
                                'bbox': (x1s, y1s, x2s, y2s),
                                'center': (center_x, center_y),
                                'cls_id': cls_id,
                                'in_polygon': in_polygon,
                                'last_seen': self._reproduction_frame_skip
                            }
                    
                    # Mantener vehículos que se vieron recientemente (máximo 6 frames sin ver)
                    for vehicle_id, vehicle_data in list(self._tracked_vehicles.items()):
                        if (self._reproduction_frame_skip - vehicle_data['last_seen']) <= 6:
                            if vehicle_id not in current_vehicles:
                                current_vehicles[vehicle_id] = vehicle_data
                    
                    # Actualizar tracking
                    self._tracked_vehicles = current_vehicles
                    
                except Exception as e:
                    # Silenciar errores para mantener fluidez
                    pass
        
        # 🎯 DIBUJAR CUADROS ESTABLES (siempre, usando tracking)
        for vehicle_id, vehicle_data in self._tracked_vehicles.items():
            x1s, y1s, x2s, y2s = vehicle_data['bbox']
            cls_id = vehicle_data['cls_id']
            in_polygon = vehicle_data['in_polygon']
            
            # 🎯 REGLAS DE COLORES DE CUADROS:
            if in_polygon and current_state == "red":
                # 🔴 CUADRO ROJO: PARACHOQUES en área + rojo = INFRACCIÓN REAL
                box_color = (0, 0, 255)
                label_text = "INFRACCION"
                text_color = (255, 255, 255)
                
                # 🚨 Marcar punto crítico del parachoques
                front_x, front_y = (x1s + x2s) // 2, y2s - 10
                cv2.circle(frame_display, (front_x, front_y), 8, (0, 0, 255), -1)
            
            else:
                # 🟢 CUADRO VERDE: Fuera del área
                box_color = (0, 255, 0)
                label_text = "NORMAL"
                text_color = (0, 0, 0)
            
            # Dibujar cuadro estable
            cv2.rectangle(frame_display, (x1s, y1s), (x2s, y2s), box_color, 2)
            
            # Etiqueta del vehículo
            vehicle_label = "CAR" if cls_id == 2 else "BUS" if cls_id == 5 else "TRUCK"
            cv2.putText(frame_display, vehicle_label, (x1s, y1s - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Estado de la detección
            cv2.putText(frame_display, label_text, (x1s, y1s - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
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
        cv2.rectangle(frame_display, (5, 5), (text_size[0] + 20, 40), bg_color, -1)
        cv2.putText(frame_display, semaforo_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 3)
        
        # Indicador de modo
        cv2.putText(frame_display, "MODO REPRODUCCION", 
                    (frame_display.shape[1] - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Mostrar el frame con overlay
        bgr_img = self.resize_and_letterbox(frame_display)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        
        # Métricas básicas
        if not hasattr(self, 'last_time'):
            self.last_time = time.time()
        if not hasattr(self, 'fps_calc'):
            self.fps_calc = 30.0
        
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            self.fps_calc = 0.9 * self.fps_calc + 0.1 * (1.0 / dt)
        
        # Info básica
        self.info_label.config(text=f"REPRODUCCIÓN | FPS: {self.fps_calc:.1f}")
        
        # Continuar reproducción
        self._after_id = self.parent.after(33, self.update_frames_optimized)  # ~30 FPS

    def _validate_detection_quality(self, detection_data):
        """Valida la calidad de una detección basada en múltiples factores"""
        if not detection_data:
            return 0.0
            
        quality_score = 1.0
        factors_checked = 0
        
        # Factor 1: Longitud de placa (placas peruanas típicas 5-7 caracteres)
        if 'placa' in detection_data and detection_data['placa']:
            plate_text = detection_data['placa'].strip()
            if 5 <= len(plate_text) <= 7:
                quality_score *= 1.0  # Perfecto
            elif 4 <= len(plate_text) <= 8:
                quality_score *= 0.8  # Aceptable
            else:
                quality_score *= 0.3  # Dudoso
            factors_checked += 1
        
        # Factor 2: Tiempo de procesamiento (muy rápido o muy lento es sospechoso)
        if 'processing_time' in detection_data and detection_data['processing_time'] > 0:
            proc_time = detection_data['processing_time']
            if 0.1 <= proc_time <= 5.0:  # Entre 0.1 y 5 segundos es normal
                quality_score *= 1.0
            elif proc_time > 10.0:  # Más de 10 segundos es sospechoso
                quality_score *= 0.6
            factors_checked += 1
        
        # Factor 3: Consistencia con patrones de placa peruana
        if 'placa' in detection_data and detection_data['placa']:
            from src.core.processing.plate_ocr_enhancer import get_plate_enhancer
            enhancer = get_plate_enhancer()
            is_valid, format_conf = enhancer.validate_plate_format(detection_data['placa'])
            if is_valid:
                quality_score *= (0.8 + format_conf * 0.2)  # 80-100% según formato
            else:
                quality_score *= 0.4  # Formato inválido
            factors_checked += 1
        
        # Factor 4: Detección nocturna (aplicar penalización)
        if detection_data.get('is_night_detection', False):
            quality_score *= 0.85  # 15% penalización por noche
            factors_checked += 1
        
        # Si no se pudo evaluar ningún factor, confianza baja
        if factors_checked == 0:
            return 0.2
            
        return min(quality_score, 1.0)
    
    def _calculate_infraction_rate(self):
        """TI: Tasa de Infracciones según contexto de detecciones
        MEJORADO: Considera 3 casos específicos:
        1. Solo NID: TI = 100% (detección perfecta)
        2. Solo NIE: TI = 0% (requiere revisión total) 
        3. Mixto NID+NIE: TI = (NID / Total) × 100
        """
        # Usar detected_plates_widgets que son las cards realmente mostradas
        if hasattr(self, "detected_plates_widgets") and self.detected_plates_widgets:
            total_detections = len(self.detected_plates_widgets)
            nid_detections = 0
            nie_detections = 0
            
            # Contar desde widgets con clasificación actualizada
            for plate_data in self.detected_plates_widgets:
                if isinstance(plate_data, dict):
                    # Usar la clasificación YA GUARDADA en la card (ya refleja la
                    # validación final NID/NIE), NO re-clasificar desde el texto:
                    # evita que TI diverga de los contadores NID/NIE.
                    classification = plate_data.get('classification', 'NIE')
                    if classification == 'NID':
                        nid_detections += 1
                    else:
                        nie_detections += 1
            
            # LÓGICA MEJORADA SEGÚN CONTEXTO:
            if total_detections > 0:
                if nie_detections == 0 and nid_detections > 0:
                    # CASO 1: Solo NID - Sistema funcionando perfectamente
                    return 100.0
                elif nid_detections == 0 and nie_detections > 0:
                    # CASO 2: Solo NIE - Sistema necesita revisión
                    return 0.0
                else:
                    # CASO 3: Mixto NID+NIE - Calcular porcentaje real
                    ti_percentage = (nid_detections / total_detections) * 100
                    return min(ti_percentage, 100.0)
        
        # Fallback a plate_detection_history si no hay widgets
        if hasattr(self, "plate_detection_history") and self.plate_detection_history:
            total_detections = len(self.plate_detection_history)
            nid_detections = 0
            
            # Contar desde history
            for plate_data in self.plate_detection_history.values():
                classification = plate_data.get('classification', 'NIE')
                if classification == 'NID':
                    nid_detections += 1
            
            # Aplicar misma lógica
            nie_detections = total_detections - nid_detections
            if total_detections > 0:
                if nie_detections == 0 and nid_detections > 0:
                    return 100.0
                elif nid_detections == 0 and nie_detections > 0:
                    return 0.0
                else:
                    ti_percentage = (nid_detections / total_detections) * 100
                    return min(ti_percentage, 100.0)
        
        return 0.0
    
    def enable_precision_validation(self, enable=True):
        """Habilita o deshabilita la validación de precisión en el cálculo de TI"""
        self.use_precision_validation = enable
        if enable:
            print("✅ Validación de precisión HABILITADA - TI será más conservador pero preciso")
        else:
            print("ℹ️ Validación de precisión DESHABILITADA - TI usará conteo simple")
    
    def get_detection_quality_report(self):
        """Genera reporte de calidad de detecciones"""
        if not hasattr(self, 'plate_detection_history') or not self.plate_detection_history:
            return "No hay detecciones para evaluar"
        
        report = "📊 REPORTE DE CALIDAD DE DETECCIONES\n"
        report += "=" * 50 + "\n"
        
        total_detections = len(self.plate_detection_history)
        high_quality = 0
        medium_quality = 0
        low_quality = 0
        
        for plate_id, detection in self.plate_detection_history.items():
            quality = self._validate_detection_quality(detection)
            if quality >= 0.8:
                high_quality += 1
            elif quality >= 0.5:
                medium_quality += 1
            else:
                low_quality += 1
        
        report += f"Total detecciones: {total_detections}\n"
        report += f"Alta calidad (≥80%): {high_quality} ({high_quality/total_detections*100:.1f}%)\n"
        report += f"Media calidad (50-79%): {medium_quality} ({medium_quality/total_detections*100:.1f}%)\n"
        report += f"Baja calidad (<50%): {low_quality} ({low_quality/total_detections*100:.1f}%)\n"
        
        # TI con y sin validación
        old_validation = getattr(self, 'use_precision_validation', False)
        
        self.use_precision_validation = False
        ti_simple = self._calculate_infraction_rate()
        
        self.use_precision_validation = True
        ti_validated = self._calculate_infraction_rate()
        
        self.use_precision_validation = old_validation  # Restaurar estado original
        
        report += f"\nTI sin validación: {ti_simple:.1f}%\n"
        report += f"TI con validación: {ti_validated:.1f}%\n"
        report += f"Diferencia: {ti_simple - ti_validated:.1f} puntos porcentuales\n"
        
        return report

    def _calculate_registration_time(self):
        """
        TR TOTAL: Tiempo de Registro TOTAL ACUMULADO en MINUTOS
        CORREGIDO: Suma TOTAL de todos los TR, NO el promedio.
        Representa el tiempo total acumulado que tardó el sistema.
        """
        if not hasattr(self, "detected_plates_widgets") or not self.detected_plates_widgets:
            return 0.0
        
        # MÉTODO CORREGIDO: SUMAR todos los TR individuales (NO promediar)
        total_tr_time = 0.0
        
        for plate_data in self.detected_plates_widgets:
            if isinstance(plate_data, dict) and 'timestamp' in plate_data:
                timestamp = plate_data['timestamp']
                if timestamp is not None and timestamp > 0:
                    tr_minutes = timestamp / 60.0  # Convertir a minutos
                    total_tr_time += tr_minutes  # ← SUMAR, no promediar
        
        if total_tr_time > 0:
            print(f"🧮 TR TOTAL desde cards: {total_tr_time:.3f} min (suma acumulada)")
            return total_tr_time
        
        # Fallback: usar historial con suma acumulada
        if hasattr(self, "plate_detection_history") and self.plate_detection_history:
            registration_times = []
            
            for plate_id, data in self.plate_detection_history.items():
                if "processing_time" in data and data["processing_time"] > 0:
                    registration_times.append(data["processing_time"])
                    
                elif "detection_time" in data and "registration_time" in data:
                    proc_time = data["registration_time"] - data["detection_time"]
                    if proc_time > 0:
                        registration_times.append(proc_time)
                        data["processing_time"] = proc_time
            
            if registration_times:
                # CORREGIDO: Sumar todos los tiempos, NO promediar
                total_time_seconds = sum(registration_times)  # ← SUMA TOTAL
                total_time_minutes = total_time_seconds / 60.0
                return max(0.001, total_time_minutes)
        
        return 0.0

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

    def _sync_hardware_state(self):
        """Sincroniza el estado GPU/CPU para la barra de información.

        Usa el VehicleDetector si ya está creado (fuente canónica: detecta CUDA
        vía torch); si no, valida el dispositivo global activo.
        """
        if hasattr(self, 'vehicle_detector') and self.vehicle_detector is not None:
            self.using_gpu = self.vehicle_detector.using_gpu
            gi = getattr(self.vehicle_detector, 'hardware_info', {}).get('gpu', {})
        else:
            device = get_default_device()
            self.using_gpu = device.type == 'cuda'
            gi = {}

        gpu_name = ''
        if self.using_gpu:
            try:
                gpu_name = torch.cuda.get_device_name(device.index)
            except Exception:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = ''

        self.gpu_info = {
            'name': gi.get('name') or gpu_name,
            'available': self.using_gpu,
            'cuda_available': self.using_gpu,
            'memory': gi.get('memory', 0.0),
            'count': gi.get('count', 1 if self.using_gpu else 0),
        }

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
            self._sync_hardware_state()
            gpu_text = "💻 Solo CPU"
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_text = f"🚀 {gpu_name}"
                except Exception:
                    gpu_text = "🚀 GPU detectada"
            elif hasattr(self, 'gpu_info') and self.gpu_info.get('name'):
                if self.gpu_info.get('cuda_available'):
                    gpu_text = f"🚀 {self.gpu_info['name']}"
                else:
                    gpu_text = f"🔍 {self.gpu_info['name']}... (sin CUDA)"

            # Información de Internet
            has_internet = self.check_internet_connection()
            internet_text = "🌐 Conectado" if has_internet else "🔌 Sin Internet"

            # Combinar información
            system_text = f"{gpu_text} | {internet_text}"
            self.system_info_label.config(text=system_text)

        except Exception as e:
            self.system_info_label.config(text="🔧 Sistema: Detectando...")
            print(f"Error actualizando info del sistema: {e}")

    def _initialize_responsive_layout(self):
        """Detecta automáticamente el tamaño de pantalla y aplica el layout correcto al inicializar"""
        try:
            # Obtener tamaño de pantalla
            screen_width = self.parent.winfo_screenwidth()
            screen_height = self.parent.winfo_screenheight()
            
            print(f"🖥️ Pantalla detectada: {screen_width}x{screen_height}")
            
            # Aplicar layout según el tamaño de pantalla
            if screen_width < 1366:  # Laptops pequeños
                print("📱 Aplicando layout para laptop pequeño")
                self._apply_small_screen_layout()
            elif screen_width < 1600:  # Pantallas medianas (la mayoría de laptops)
                print("💻 Aplicando layout para pantalla mediana")
                self._apply_medium_screen_layout()
            else:  # Monitores grandes
                print("🖥️ Aplicando layout para pantalla grande")
                self._apply_large_screen_layout()
                
        except Exception as e:
            print(f"⚠️ Error inicializando responsive layout: {e}")
            # En caso de error, aplicar layout mediano por defecto
            self._apply_medium_screen_layout()

    def _adjust_video_frame_responsive(self, window_width, window_height):
        """🎬 Ajustar video frame de manera responsive según tamaño de ventana"""
        try:
            # Calcular proporciones responsive para el video
            if window_width < 1000:  # Pantallas pequeñas
                video_width = int(window_width * 0.55)  # 55% de la ventana
                video_height = int(video_width * 9 / 16)  # Mantener aspecto 16:9
                panels_width = 280  # Panel de placas más estrecho
            elif window_width < 1400:  # Pantallas medianas  
                video_width = int(window_width * 0.6)   # 60% de la ventana
                video_height = int(video_width * 9 / 16)
                panels_width = 320  # Panel moderado
            else:  # Pantallas grandes
                video_width = int(window_width * 0.65)  # 65% de la ventana
                video_height = int(video_width * 9 / 16)
                panels_width = 380  # Panel completo
            
            # Asegurar que no sea demasiado alto
            max_height = int(window_height * 0.7)  # Máximo 70% de altura
            if video_height > max_height:
                video_height = max_height
                video_width = int(video_height * 16 / 9)
            
            # Aplicar nuevas dimensiones al video frame
            if hasattr(self, 'video_frame'):
                self.video_frame.config(width=video_width, height=video_height)
                
            # Aplicar nuevas dimensiones al panel de placas
            if hasattr(self, 'plates_frame'):
                self.plates_frame.config(width=panels_width)
            
            # Debug opcional
            # print(f"🎬 Video responsive: {video_width}x{video_height}, Panel: {panels_width}px")
            
        except Exception as e:
            print(f"Error en video responsive: {e}")

    def _on_window_resize(self, event):
        """Función responsive para ajustar el layout según el tamaño de ventana, limitando frecuencia."""
        if event.widget != self.parent:
            return
        if hasattr(self, '_resize_pending') and self._resize_pending:
            return
        self._resize_pending = True
        self.parent.after(50, self._do_resize_layout)
    
    def _do_resize_layout(self):
        """Ejecuta el ajuste de layout de forma diferida."""
        try:
            window_width = self.parent.winfo_width()
            window_height = self.parent.winfo_height()
            screen_width = self.parent.winfo_screenwidth()
            
            self._adjust_video_frame_responsive(window_width, window_height)
            
            if screen_width < 1366 or window_width < 1000:
                self._apply_small_screen_layout()
            elif screen_width < 1600 or window_width < 1400:
                self._apply_medium_screen_layout()
            else:
                self._apply_large_screen_layout()
        except Exception as e:
            print(f"Error en responsive design: {e}")
        finally:
            self._resize_pending = False
            
    def _apply_small_screen_layout(self):
        """Layout para pantallas pequeñas (<1366px) - Laptops"""
        try:
            # Botones muy compactos para laptops
            small_btn_style = {
                "font": ("Arial", 9),
                "width": 28,
                "pady": 2
            }
            
            self.load_button.config(**small_btn_style)
            self.btn_preprocesar.config(**small_btn_style)
            self.play_pause_button.config(font=("Arial", 9, "bold"), width=12)
            
            # Texto explicativo muy compacto
            if hasattr(self, 'play_pause_help_label'):
                self.play_pause_help_label.config(
                    font=("Arial", 6, "italic"),
                    wraplength=100,
                    width=12
                )
            
            # 🔧 FIX: Panel de placas con ancho fijo más apropiado para laptops
            if hasattr(self, 'plates_frame'):
                self.plates_frame.config(width=280)  # Aumentado de 220 a 280
            
            # 🔧 FIX: Ajustar canvas interno con ancho consistente
            if hasattr(self, 'plates_canvas_window'):
                self.plates_canvas.itemconfig(self.plates_canvas_window, width=260)  # Aumentado de 200 a 260
            
            # Ajustar márgenes para laptops
            self._adjust_margins_and_spacing()
            
            # Forzar actualización del panel de métricas
            if hasattr(self, '_setup_metrics_responsive_behavior'):
                # self.plates_frame.after(100, lambda: self._recreate_metrics_layout('small'))
                print("📱 Layout compacto aplicado para laptop")
                
        except Exception as e:
            print(f"Error en layout pequeño: {e}")

    def _apply_medium_screen_layout(self):
        """Layout para pantallas medianas (1200-1600px)"""
        try:
            # Tamaños estándar para pantallas medianas
            medium_btn_style = {
                "font": ("Arial", 11),
                "width": 32,
                "pady": 4
            }
            
            self.load_button.config(**medium_btn_style)
            self.btn_preprocesar.config(**medium_btn_style)
            self.play_pause_button.config(font=("Arial", 11, "bold"), width=18)
            
            # Texto explicativo tamaño normal
            if hasattr(self, 'play_pause_help_label'):
                self.play_pause_help_label.config(
                    font=("Arial", 9, "italic"),
                    wraplength=130,
                    width=16
                )
            
            # Panel de placas tamaño estándar
            if hasattr(self, 'plates_frame'):
                self.plates_frame.config(width=300)
            
            # Ajustar canvas interno
            if hasattr(self, 'plates_canvas_window'):
                self.plates_canvas.itemconfig(self.plates_canvas_window, width=280)
            
            # Ajustar márgenes para pantallas medianas
            self._adjust_margins_and_spacing()
            
            # Forzar actualización del panel de métricas
            if hasattr(self, '_setup_metrics_responsive_behavior'):
#                #self.plates_frame.after(100, lambda: self._recreate_metrics_layout('medium'))
                 print("💻 Layout estándar aplicado para pantalla mediana")
                
        except Exception as e:
            print(f"Error en layout mediano: {e}")

    def _apply_large_screen_layout(self):
        """Layout para pantallas grandes (>1600px) - Monitores externos"""
        try:
            # Tamaños grandes para monitores
            large_btn_style = {
                "font": ("Arial", 13),
                "width": 38,
                "pady": 6
            }
            
            self.load_button.config(**large_btn_style)
            self.btn_preprocesar.config(**large_btn_style)
            self.play_pause_button.config(font=("Arial", 13, "bold"), width=22)
            
            # Texto explicativo más grande
            if hasattr(self, 'play_pause_help_label'):
                self.play_pause_help_label.config(
                    font=("Arial", 11, "italic"),
                    wraplength=150,
                    width=18
                )
            
            # Panel de placas más ancho para monitores
            if hasattr(self, 'plates_frame'):
                self.plates_frame.config(width=380)
            
            # Ajustar canvas interno para monitores
            if hasattr(self, 'plates_canvas_window'):
                self.plates_canvas.itemconfig(self.plates_canvas_window, width=360)
            
            # Ajustar márgenes para monitores grandes
            self._adjust_margins_and_spacing()
            
            # Forzar actualización del panel de métricas
            if hasattr(self, '_setup_metrics_responsive_behavior'):
#                #self.plates_frame.after(100, lambda: self._recreate_metrics_layout('large'))
                 print("🖥️ Layout expandido aplicado para monitor grande")
                
        except Exception as e:
            print(f"Error en layout grande: {e}")

    def _adjust_margins_and_spacing(self):
        """Ajusta automáticamente márgenes y espaciado para evitar desbordamientos"""
        try:
            # Obtener dimensiones actuales
            screen_width = self.parent.winfo_screenwidth()
            screen_height = self.parent.winfo_screenheight()
            
            # Ajustar padding del frame principal según el tamaño de pantalla
            if screen_width < 1366:  # Laptops pequeños
                # Márgenes mínimos para maximizar espacio
                if hasattr(self, 'frame'):
                    self.frame.pack_configure(padx=2, pady=2)
                if hasattr(self, 'btn_frame'):
                    self.btn_frame.pack_configure(pady=8)
            elif screen_width < 1600:  # Pantallas medianas
                # Márgenes estándar
                if hasattr(self, 'frame'):
                    self.frame.pack_configure(padx=5, pady=5)
                if hasattr(self, 'btn_frame'):
                    self.btn_frame.pack_configure(pady=12)
            else:  # Pantallas grandes
                # Márgenes generosos
                if hasattr(self, 'frame'):
                    self.frame.pack_configure(padx=8, pady=8)
                if hasattr(self, 'btn_frame'):
                    self.btn_frame.pack_configure(pady=15)
            
            print(f"📐 Márgenes ajustados para pantalla {screen_width}px")
                    
        except Exception as e:
            print(f"⚠️ Error ajustando márgenes: {e}")

    def toggle_beep(self):
        """Habilitar/deshabilitar beep de infracciones"""
        self.beep_enabled = not self.beep_enabled
        status = "HABILITADO" if self.beep_enabled else "DESHABILITADO"
        color = "#f39c12" if self.beep_enabled else "#7f8c8d"
        text = "🔊 BEEP" if self.beep_enabled else "🔇 MUDO"
        
        self.beep_button.config(bg=color, text=text)
        print(f"🔊 Beep de infracciones: {status}")

    def play_infraction_beep(self):
        """Reproduce beep de infracción SIMPLE y seguro (multiplataforma)."""
        if not self.beep_enabled:
            return
        try:
            play_beep(500, 100)
        except Exception:
            pass

    def should_play_beep(self, plate_text):
        """Verifica si debe sonar beep (solo 1 vez por placa única)"""
        if not self.beep_enabled or not plate_text:
            return False
        
        # Solo beep si es una placa nueva
        if plate_text not in self.beep_unique_plates:
            self.beep_unique_plates.add(plate_text)
            return True
        return False

    def _evaluate_plate_quality(self, plate_crop):
        """Evalúa la calidad de un recorte de placa para seleccionar el mejor"""
        if plate_crop is None or plate_crop.size == 0:
            return 0.0
        
        try:
            # Factor 1: Contraste (placas tienen buen contraste)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            contrast_score = min(contrast / 50.0, 1.0)
            
            # Factor 2: Detección de bordes (placas tienen bordes definidos)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
            edge_score = min(edge_density * 10, 1.0)
            
            # Factor 3: Aspect ratio típico de placas (rectangular)
            h, w = plate_crop.shape[:2]
            if h > 0:
                aspect_ratio = w / h
                # Placas peruanas típicamente entre 2:1 y 4:1
                if 1.5 <= aspect_ratio <= 5.0:
                    aspect_score = 1.0
                else:
                    aspect_score = max(0.3, 1.0 - abs(aspect_ratio - 3.0) * 0.2)
            else:
                aspect_score = 0.0
            
            # Factor 4: Tamaño mínimo (muy pequeñas no sirven para OCR)
            size_score = min((w * h) / 1500.0, 1.0)  # Mínimo 1500 píxeles
            
            # Factor 5: Nitidez (importante para OCR)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(sharpness / 100.0, 1.0)
            
            # Puntuación final ponderada
            total_score = (
                contrast_score * 0.25 +
                edge_score * 0.25 +
                aspect_score * 0.20 +
                size_score * 0.15 +
                sharpness_score * 0.15
            )
            
            return total_score
            
        except Exception as e:
            print(f"Error evaluando calidad de placa: {e}")
            return 0.0

    def enhanced_plate_detection(self, frame, car_detection):
        """DETECCIÓN ULTRA-PRECISA: Recorte EXACTO de la placa en las 4 esquinas"""
        try:
            x1, y1, x2, y2 = car_detection[:4]
            
            # Método principal: detección ultra-precisa por color y forma
            precise_plate, quality, global_coords = self._detect_precise_plate_region(frame, x1, y1, x2, y2)
            
            if precise_plate is not None and quality > 0.3:
                print(f"🎯 Placa detectada EXACTA: {precise_plate.shape}, calidad: {quality:.3f}")
                
                # � DEBUG: Dibujar rectángulo MAGENTA sobre el recorte exacto de zona blanca
                if global_coords and hasattr(self, 'show_debug') and self.show_debug:
                    gx1, gy1, gx2, gy2 = global_coords
                    # Rectángulo magenta = recorte exacto de la zona blanca
                    cv2.rectangle(frame, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (255, 0, 255), 2)
                    cv2.putText(frame, "ZONA BLANCA", (int(gx1), int(gy1)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    # Información adicional
                    h_crop, w_crop = precise_plate.shape[:2]
                    cv2.putText(frame, f"{w_crop}x{h_crop}", (int(gx1), int(gy2)+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                
                return precise_plate, quality
            
            # Fallback: método de respaldo si el principal falla
            return self._fallback_precise_detection(frame, car_detection)
            
        except Exception as e:
            print(f"Error en detección ultra-precisa: {e}")
            return None, 0.0
    
    def _detect_precise_plate_region(self, frame, x1, y1, x2, y2):
        """RECORTE ULTRA-PRECISO: Solo la zona blanca de la matrícula (dentro de líneas rojas)"""
        try:
            # 1. REGIÓN DEL VEHÍCULO (punto de partida)
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                return None, 0.0, None
            
            # 2. CONVERSIÓN A ESCALA DE GRISES para mejor detección
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
            # 3. UMBRALIZACIÓN para detectar zonas BLANCAS (placas)
            # Detectar solo píxeles muy brillantes (zona blanca de la placa)
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # 4. MORFOLOGÍA para limpiar y conectar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # 5. ENCONTRAR CONTORNOS de las zonas blancas
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("❌ No se encontraron zonas blancas")
                return None, 0.0, None
            
            # 6. FILTRAR CONTORNOS por características de PLACA
            plate_candidates = []
            h_vehicle, w_vehicle = vehicle_crop.shape[:2]
            
            for contour in contours:
                # Rectángulo que encierra el contorno
                px, py, pw, ph = cv2.boundingRect(contour)
                
                # FILTROS MUY ESTRICTOS para placas:
                if pw < 40 or ph < 15:  # Muy pequeña
                    continue
                    
                aspect_ratio = pw / ph
                if not (2.0 <= aspect_ratio <= 5.0):  # Aspect ratio de placa
                    continue
                    
                area = pw * ph
                if area < 600 or area > 15000:  # Área fuera de rango
                    continue
                
                # 7. VERIFICAR que tiene TEXTO (densidad de píxeles blancos)
                plate_region = white_mask[py:py+ph, px:px+pw]
                white_density = np.sum(plate_region == 255) / (pw * ph)
                
                # La placa debe tener entre 20% y 80% de píxeles blancos (texto + fondo)
                if not (0.2 <= white_density <= 0.8):
                    continue
                
                # 8. CALCULAR CALIDAD basada en características
                quality = self._calculate_white_region_quality(vehicle_crop[py:py+ph, px:px+pw])
                
                if quality > 0.3:
                    plate_candidates.append({
                        'bbox': (px, py, pw, ph),
                        'quality': quality,
                        'white_density': white_density,
                        'aspect_ratio': aspect_ratio
                    })
            
            if not plate_candidates:
                print("❌ No se encontraron candidatos válidos")
                return None, 0.0, None
            
            # 9. SELECCIONAR LA MEJOR (mayor calidad)
            best = max(plate_candidates, key=lambda x: x['quality'])
            px, py, pw, ph = best['bbox']
            
            # 10. ✂️ RECORTE EXACTO - SOLO la zona blanca detectada
            # ESTO ES LO CRUCIAL - recorte exacto sin margen extra
            exact_plate = vehicle_crop[py:py+ph, px:px+pw]
            
            # Coordenadas globales para el rectángulo de debug
            global_x1 = x1 + px
            global_y1 = y1 + py  
            global_x2 = x1 + px + pw
            global_y2 = y1 + py + ph
            
            print(f"✂️ RECORTE EXACTO: {pw}x{ph}px (zona blanca pura)")
            print(f"   Densidad blanca: {best['white_density']:.2f}")
            print(f"   Aspect ratio: {best['aspect_ratio']:.2f}")
            print(f"   Calidad: {best['quality']:.3f}")
            
            return exact_plate, best['quality'], (global_x1, global_y1, global_x2, global_y2)
            
        except Exception as e:
            print(f"Error en recorte ultra-preciso: {e}")
            return None, 0.0, None
    
    def _calculate_white_region_quality(self, plate_region):
        """Calcula calidad específica de la zona blanca de la placa"""
        try:
            if plate_region.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Factor 1: Contraste (texto negro sobre fondo blanco)
            contrast = gray.std() / 255.0
            contrast_score = min(contrast * 2.5, 1.0)
            
            # Factor 2: Distribución de intensidades (debe tener píxeles claros y oscuros)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Verificar que hay píxeles tanto claros (>200) como oscuros (<100)
            bright_pixels = np.sum(hist[200:])
            dark_pixels = np.sum(hist[:100])
            total_pixels = gray.size
            
            if bright_pixels > 0 and dark_pixels > 0:
                distribution_score = min((bright_pixels + dark_pixels) / total_pixels, 1.0)
            else:
                distribution_score = 0.2
            
            # Factor 3: Detección de bordes (texto tiene muchos bordes)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 8, 1.0)
            
            # Factor 4: Verificar que tiene forma rectangular (placas son rectangulares)
            h, w = gray.shape
            if h > 0 and w > 0:
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 4.5:
                    shape_score = 1.0
                elif 1.5 <= aspect_ratio <= 5.5:
                    shape_score = 0.7
                else:
                    shape_score = 0.3
            else:
                shape_score = 0.0
            
            # Promedio ponderado
            final_quality = (
                contrast_score * 0.3 +
                distribution_score * 0.3 +
                edge_score * 0.25 +
                shape_score * 0.15
            )
            
            return final_quality
            
        except Exception as e:
            return 0.0
    
    def _fallback_precise_detection(self, frame, car_detection):
        """Método de respaldo con recorte más preciso"""
        try:
            x1, y1, x2, y2 = car_detection[:4]
            
            # Región frontal más pequeña y precisa
            vehicle_h = y2 - y1
            # Solo tomar 20% inferior frontal
            front_y1 = y2 - int(vehicle_h * 0.2)
            
            if front_y1 < y2:
                front_region = frame[front_y1:y2, x1:x2]
                
                if front_region.size > 0:
                    quality = self._evaluate_plate_quality(front_region)
                    if quality > 0.15:
                        print(f"🔄 Fallback: región frontal {front_region.shape}, calidad: {quality:.3f}")
                        return front_region, quality
            
            return None, 0.0
            
        except Exception as e:
            print(f"Error en fallback: {e}")
            return None, 0.0

    def _basic_plate_detection(self, region):
        """Detección básica de placas usando contornos como fallback"""
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtros para resaltar placas
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            h, w = region.shape[:2]
            
            for contour in contours:
                # Obtener rectángulo envolvente
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Filtrar por aspect ratio típico de placas
                if cw > 0 and ch > 0:
                    aspect_ratio = cw / ch
                    area = cw * ch
                    
                    # Criterios para placas: aspect ratio entre 1.5-5.0, área mínima
                    if 1.5 <= aspect_ratio <= 5.0 and area > (w * h * 0.05):
                        # Expandir ligeramente el rectángulo
                        x1 = max(0, x - 5)
                        y1 = max(0, y - 5)
                        x2 = min(w, x + cw + 5)
                        y2 = min(h, y + ch + 5)
                        
                        # Confianza basada en área y aspect ratio
                        conf = min(0.8, area / (w * h) + (1.0 / abs(aspect_ratio - 2.5)) * 0.1)
                        detections.append([x1, y1, x2, y2, conf])
            
            # Ordenar por confianza descendente
            detections.sort(key=lambda x: x[4], reverse=True)
            return detections[:3]  # Máximo 3 candidatos
            
        except Exception as e:
            print(f"Error en detección básica: {e}")
            return []
    
    def classify_detection_quality(self, plate_text, detection_confidence=0.5, return_metadata=True):
        """
        Clasifica si una detección es NID (correcta) o NIE (errónea)
        usando el sistema de clasificación optimizado para placas peruanas - Trujillo.
        
        NUEVO: Sistema calibrado específicamente para SIIV 2010 con prioridad Trujillo.
        
        Args:
            plate_text: Texto de la placa detectada
            detection_confidence: Confianza de la detección
            return_metadata: Si True devuelve (classification, score, metadata), 
                           si False devuelve (classification, score)
        
        Returns:
            tuple: (classification, confidence_score) o (classification, confidence_score, metadata)
        """
        # Importar y usar el sistema de clasificación mejorado
        try:
            from src.gui.preprocessing_dialog import PlateClassificationSystem
            
            # Crear instancia del clasificador optimizado para Perú
            if not hasattr(self, '_plate_classifier'):
                self._plate_classifier = PlateClassificationSystem()
                print("🇵🇪 Sistema NID/NIE para placas peruanas inicializado (prioridad Trujillo)")
            
            # Clasificar usando el sistema específico para SIIV
            classification, metadata = self._plate_classifier.classify_detection(
                plate_text=plate_text,
                confidence=detection_confidence,
                frame_validations={'crossing_confirmed': True}
            )
            
            # Obtener confianza ajustada del metadata
            confidence_score = metadata.get('confianza', detection_confidence)
            
            # Log detallado para debugging con información regional
            if classification == 'NIE':
                razon = metadata.get('razon', 'desconocida')
                placa = metadata.get('placa_detectada', plate_text)
                print(f"⚠️ NIE: {placa} (razón: {razon}, conf: {confidence_score:.2f})")
            else:
                placa = metadata.get('placa_final', plate_text)
                region = metadata.get('region', '')
                ciudad = metadata.get('ciudad', '')
                tipo_vehiculo = metadata.get('tipo_vehiculo', '')
                
                # Log especial para Trujillo
                if placa.startswith('T'):
                    print(f"🎯 NID TRUJILLO: {placa} (conf: {confidence_score:.2f}, tipo: {tipo_vehiculo})")
                else:
                    print(f"✅ NID: {placa} (conf: {confidence_score:.2f}, {ciudad})")
                
            if return_metadata:
                return classification, confidence_score, metadata
            else:
                return classification, confidence_score
            
        except Exception as e:
            # Fallback al sistema anterior si hay problemas
            print(f"⚠️ Fallback: Error en sistema SIIV: {e}")
            result = self._legacy_classify_detection_quality(plate_text, detection_confidence)
            if return_metadata:
                return result[0], result[1], {'razon': 'sistema_anterior', 'confianza': result[1]}
            else:
                return result[0], result[1]
    
    def _legacy_classify_detection_quality(self, plate_text, detection_confidence=0.5):
        """
        Sistema de clasificación anterior (más estricto) - usado como fallback.
        Incluye mejoras para placas peruanas SIIV.
        """
        if not plate_text:
            return "NIE", 0.0
        
        quality_factors = []
        
        # Factor 1: Longitud específica SIIV (6-7 caracteres es óptimo)
        if 6 <= len(plate_text) <= 7:
            quality_factors.append(1.0)
        elif 5 <= len(plate_text) <= 8:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.3)
        
        # Factor 2: Caracteres válidos (solo letras y números, sin caracteres especiales)
        valid_chars = all(c.isalnum() or c == '-' for c in plate_text)
        quality_factors.append(1.0 if valid_chars else 0.2)
        
        # Factor 3: Patrones SIIV 2010 peruanos mejorados
        import re
        patterns = [
            r'^[A-Z]{2}\d{1}-?\d{3}$',  # AB1-234 (vehículos menores)
            r'^[A-Z]{3}-?\d{3}$',       # ABC-123 (vehículos mayores)
            r'^[A-Z]{2}\d{4}$',         # AB1234 (sin guión)
            r'^[A-Z]{3}\d{3}$'          # ABC123 (sin guión)
        ]
        
        pattern_match = any(re.match(pattern, plate_text.upper()) for pattern in patterns)
        quality_factors.append(1.0 if pattern_match else 0.3)
        
        # Factor 4: Boost para códigos regionales conocidos (especialmente Trujillo)
        first_char = plate_text[0].upper() if plate_text else ''
        regional_boost = {
            'T': 1.0,  # Trujillo - máxima prioridad
            'A': 0.9, 'B': 0.9, 'C': 0.9, 'D': 0.9,  # Lima
            'F': 0.8,  # Callao
            'P': 0.7, 'V': 0.7,  # Piura, Arequipa
            'M': 0.6, 'K': 0.6, 'S': 0.6, 'L': 0.6, 'H': 0.6  # Otras regiones
        }.get(first_char, 0.4)
        quality_factors.append(regional_boost)
        
        # Factor 5: Confianza del OCR ajustada
        if detection_confidence > 0.8:
            quality_factors.append(1.0)
        elif detection_confidence > 0.6:
            quality_factors.append(0.8)
        elif detection_confidence > 0.4:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Calcular confianza promedio
        avg_confidence = sum(quality_factors) / len(quality_factors)
        
        # UMBRAL TÉCNICO: ≥0.70 para NID (Balance precision/recall)
        if avg_confidence >= 0.70:
            return "NID", avg_confidence  
        else:
            return "NIE", avg_confidence

    def _create_responsive_window(self, parent, title, min_width=400, min_height=300, max_width_ratio=0.8, max_height_ratio=0.8):
        """
        Crea una ventana emergente responsive que se adapta al tamaño de pantalla
        """
        try:
            # Crear ventana
            window = tk.Toplevel(parent)
            window.title(title)
            window.transient(parent)
            window.grab_set()
            
            # Obtener dimensiones de la pantalla
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            
            # Calcular tamaño de ventana basado en la pantalla
            max_width = int(screen_width * max_width_ratio)
            max_height = int(screen_height * max_height_ratio)
            
            # Usar el mínimo entre el máximo y el tamaño mínimo requerido
            window_width = max(min_width, min(max_width, 800))
            window_height = max(min_height, min(max_height, 600))
            
            # Para pantallas muy pequeñas (laptops), reducir aún más
            if screen_width < 1366 or screen_height < 768:
                window_width = min(window_width, screen_width - 100)
                window_height = min(window_height, screen_height - 100)
            
            # Centrar ventana
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            window.minsize(min_width, min_height)
            
            # Agregar scroll si es necesario en pantallas muy pequeñas
            if screen_height < 800:
                # Crear frame principal con scroll
                main_frame = tk.Frame(window)
                main_frame.pack(fill="both", expand=True)
                
                canvas = tk.Canvas(main_frame, highlightthickness=0)
                scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)
                
                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                )
                
                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)
                
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
                
                # Hacer scroll con rueda del mouse
                def _on_mousewheel(event):
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
                
                return window, scrollable_frame
            else:
                # Sin scroll para pantallas normales
                content_frame = tk.Frame(window)
                content_frame.pack(fill="both", expand=True, padx=10, pady=10)
                return window, content_frame
                
        except Exception as e:
            print(f"Error creando ventana responsive: {e}")
            # Fallback básico
            window = tk.Toplevel(parent)
            window.title(title)
            window.geometry("600x400")
            content_frame = tk.Frame(window)
            content_frame.pack(fill="both", expand=True)
            return window, content_frame

    # =====================================================
    # ANÁLISIS NOCTURNO AL CARGAR VIDEO
    # =====================================================
    
    def _analyze_video_lighting(self, first_frame, sample_frames=5):
        """
        Analiza si el video es nocturno al cargarse y actualiza el indicador visual.
        
        Args:
            first_frame: Frame inicial del video
            sample_frames: Número de frames adicionales a analizar
        """
        try:
            if first_frame is None or not hasattr(self, 'current_video_path'):
                return
                
            print("🌙 Analizando condiciones de iluminación del video...")
            
            # Analizar múltiples frames para mejor precisión
            cap = cv2.VideoCapture(self.current_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            brightness_values = []
            
            # Analizar frames distribuidos a lo largo del video
            frame_positions = [
                0,  # Inicio
                total_frames // 4,      # 25%
                total_frames // 2,      # 50% 
                3 * total_frames // 4,  # 75%
                total_frames - 1        # Final
            ]
            
            for pos in frame_positions[:sample_frames]:
                if pos >= total_frames:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, sample_frame = cap.read()
                
                if ret:
                    # Convertir a escala de grises
                    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calcular brillo promedio
                    avg_brightness = np.mean(gray)
                    brightness_values.append(avg_brightness)
                    
            cap.release()
            
            if brightness_values:
                # Calcular brillo promedio general
                overall_brightness = np.mean(brightness_values)
                
                # UMBRAL CRÍTICO: 60 - SINCRONIZADO CON PREPROCESSING (RESTRICTIVO)
                is_night = overall_brightness < 60
                
                # Actualizar indicador visual en la UI
                self._update_lighting_indicator(is_night, overall_brightness)
                
                print(f"🌙 ANÁLISIS COMPLETADO:")
                print(f"   📊 Brillo promedio: {overall_brightness:.1f}/255")
                print(f"   🌓 Umbral nocturno: 60 (RESTRICTIVO)")
                print(f"   🎯 Resultado: {'NOCTURNO' if is_night else 'DIURNO'}")
                
                # Guardar resultado para usar en el preprocesamiento
                self.is_night_video = is_night
                self.video_brightness = overall_brightness
                
                return is_night
            
            self.is_night_video = False
            self.video_brightness = 255
            return False
            
        except Exception as e:
            print(f"Error en análisis de iluminación: {e}")
            self.is_night_video = False
            self.video_brightness = 255
            return False

    def _update_lighting_indicator(self, is_night, brightness):
        """
        Actualiza el indicador visual de condiciones de iluminación en la UI.
        
        Args:
            is_night: True si es nocturno, False si es diurno
            brightness: Valor de brillo promedio (0-255)
        """
        try:
            # Crear o actualizar el indicador de condiciones
            lighting_text = "🌙 NOCTURNO" if is_night else "☀️ DIURNO"
            brightness_color = "#ffaa00" if is_night else "#00aa00"  # Naranja para noche, verde para día
            
            # Actualizar el label de información del sistema
            if hasattr(self, 'system_info_label'):
                system_info = self.system_info_label.cget('text')
                
                # Agregar información de iluminación
                lighting_info = f"\n{lighting_text} (Brillo: {brightness:.0f}/255)"
                
                # Si ya hay información de iluminación, reemplazarla
                if "NOCTURNO" in system_info or "DIURNO" in system_info:
                    lines = system_info.split('\n')
                    # Filtrar líneas que no sean de iluminación
                    filtered_lines = [line for line in lines if not ("NOCTURNO" in line or "DIURNO" in line)]
                    system_info = '\n'.join(filtered_lines)
                
                updated_info = system_info + lighting_info
                self.system_info_label.config(text=updated_info, fg=brightness_color)
            
            # Actualizar el indicador de iluminación separado
            if hasattr(self, 'lighting_indicator_label'):
                self.lighting_indicator_label.config(text=lighting_text, fg=brightness_color)
            
            print(f"✅ Indicador visual actualizado: {lighting_text}")
            
        except Exception as e:
            print(f"Error actualizando indicador de iluminación: {e}")

# Fin del módulo VideoPlayerOpenCV
