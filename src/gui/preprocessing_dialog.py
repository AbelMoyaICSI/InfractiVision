import os
import json
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import re
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.infractions_management_window import generate_performance_indicators_json
from src.path_helper import resource_path
from src.core.utils.icon import set_window_icon

# --- Clases extraídas a sus capas Clean Architecture (Fase 2) ---
from src.infrastructure.ocr.plate_corrector import SmartPlateCorrector
from src.application.services.metrics_calculator import ThesisMetricsCalculator
from src.application.services.processing_planner import IntelligentTrafficOptimizer
from src.domain.services.intelligent_tracker import IntelligentVehicleTracker
from src.domain.services.plate_classification import PlateClassificationSystem

# --- Popups extraídos a su Mixin (Fase 3) ---
from src.presentation.gui.popups.preprocessing_popups import PreprocessingPopupsMixin
from src.application.use_cases.process_violation_video import OfficialVideoProcessor
from src.domain.entities.plate_evidence import PlateEvidence
from src.presentation.gui.plate_review_window import PlateReviewWindow




class PreprocessingDialog(PreprocessingPopupsMixin):
    """
    Diálogo que muestra una barra de progreso mientras se procesa un video.
    Analiza las infracciones y detecta las placas sin reproducir el video completo.
    """
    """
    Diálogo que muestra una barra de progreso mientras se procesa un video.
    Analiza las infracciones y detecta las placas sin reproducir el video completo.
    """

    # Atributo estático para almacenar tiempos de procesamiento
    recorded_processing_times = []
    
    # NUEVO: Variable para controlar ventanas emergentes nocturnas
    _night_popup_active = False
    
    @staticmethod
    def generar_config_id(semaforo=None, cycle_durations=None):
        """
        Genera un ID único para la configuración del semáforo basado en sus tiempos.
        
        Args:
            semaforo: Objeto semáforo con atributos green_duration, yellow_duration, red_duration (opcional)
            cycle_durations: Dict con tiempos {'green': X, 'yellow': Y, 'red': Z} (opcional)
        
        Returns:
            str: ID en formato "verde-amarillo-rojo" (ej: "10-3-15") o "sin-configurar"
        
        Examples:
            >>> generar_config_id(semaforo)  # semaforo con 10s verde, 3s amarillo, 15s rojo
            "10-3-15"
            >>> generar_config_id(cycle_durations={'green': 12, 'yellow': 3, 'red': 12})
            "12-3-12"
        """
        try:
            # Método 1: Intentar desde el objeto semáforo
            if semaforo and hasattr(semaforo, 'green_duration') and hasattr(semaforo, 'yellow_duration') and hasattr(semaforo, 'red_duration'):
                green = int(semaforo.green_duration)
                yellow = int(semaforo.yellow_duration)
                red = int(semaforo.red_duration)
                if green > 0 and yellow > 0 and red > 0:  # Validar que no sean 0
                    return f"{green}-{yellow}-{red}"
            
            # Método 2: Intentar desde cycle_durations
            if cycle_durations and isinstance(cycle_durations, dict):
                green = int(cycle_durations.get('green', 0))
                yellow = int(cycle_durations.get('yellow', 0))
                red = int(cycle_durations.get('red', 0))
                if green > 0 and yellow > 0 and red > 0:  # Validar que no sean 0
                    return f"{green}-{yellow}-{red}"
            
            return "sin-configurar"
        except Exception as e:
            print(f"⚠️ Error generando config_id: {e}")
            return "sin-configurar"
    
    def __init__(self, parent, video_path, player_instance, on_complete=None,
                 inline=False, render_target=None, progress_mount=None):
        """
        Inicializa el diálogo de preprocesamiento.
        
        Args:
            parent: Widget padre
            video_path: Ruta del video a procesar
            player_instance: Instancia del VideoPlayerOpenCV para acceder a sus métodos
            on_complete: Función a llamar cuando se complete el procesamiento
            inline: Si es True, NO abre una ventana nueva: renderiza en el
                video_label del reproductor principal (render_target) y monta
                una barra de progreso pequeña en progress_mount.
            render_target: video_label del reproductor principal (modo inline).
            progress_mount: contenedor (Frame) donde montar la barra de progreso.
        """

        self.parent = parent
        self.video_path = video_path
        self.player = player_instance
        self.on_complete = on_complete
        self.canceled = False
        self.processing_paused = False  # NUEVO: Control de pausa para ventanas emergentes
        
        # Pausar el video durante el preprocesamiento para evitar lags
        if hasattr(self.player, 'running'):
            self.player_was_running = self.player.running
            self.player.running = False
        else:
            self.player_was_running = False
        self.progress_value = 0
        self.current_frame = None
        self.detected_infractions = []
        self._official_infraction_ids = set()
        self._pending_infractions = []
        self._official_infraction_count = 0
        self._meta_timer_ready = False
        self.processed_frames = 0
        self.total_frames = 0
        self.result_queue = queue.Queue()
        
        # 🌙 DETECCIÓN DE ESCENA NOCTURNA (inicializar antes de usar)
        self.is_night = False  # Se actualizará cuando se procese el primer frame
        
        # 🔴 VISUALES PERSISTENTES E INFRAESTRUCTURA DE MONITOR
        self.visual_feedback_items = []      # Lista de {'type', 'pos', 'frame_expiry', 'bbox'}
        self.feedback_lock = threading.Lock()
        self.last_plate_crop = None          # Para el monitor lateral al ladito
        self.plate_monitor_ready = False     # Flag de UI desactivada
        self.detected_plates_global = set()  # Registro único de placas
        self.plate_registry_lock = threading.Lock()
        
        # 🚀 INFRAESTRUCTURA DE ANÁLISIS ASÍNCRONO (FASE 2)
        self.analysis_queue = queue.Queue()
        self.analysis_active = False
        self.analysis_worker_thread = None
        self.completed_analysis_count = 0
        self.analysis_results_lock = threading.Lock()
        # 🚀 SISTEMA DE VISUALIZACIÓN FLUIDA (NO AFECTA PROCESAMIENTO)
        try:
            self.display_buffer = deque(maxlen=90)  # Buffer circular para 3 segundos a 30fps
            self.display_active = True
            self.display_thread = None
            self.display_lock = threading.Lock()
            self.last_display_frame = None
            self.display_fps = 30  # FPS fluidos para visualización
            # Cola thread-safe para pasar frames del hilo de visualización al
            # hilo principal de Tk (Tk NO es thread-safe: llamarlo desde un
            # hilo de fondo provoca SIGSEGV).
            self._display_queue = queue.Queue(maxsize=1)
            # Cola genérica de UI: los hilos de fondo encolan callables y el
            # hilo principal los ejecuta (único que puede tocar Tk).
            self._ui_queue = queue.Queue()
            self.frame_interpolation = True  # Activar interpolación suave
            self.display_enabled = True  # Flag para habilitar/deshabilitar sistema fluido
            
            # 🕰️ TR REAL DEL USUARIO - TIEMPO PERCIBIDO
            self.user_tr_start_time = None
            self.user_tr_segments = []  # Lista de tiempos por segmento
            self.current_segment_start = None
            self.total_user_time = 0
            self.visual_acceleration_active = False
            
            # ⚡ ACELERACIÓN VISUAL MEJORADA
            self.visual_speed_multiplier = 1.0  # Multiplicador de velocidad visual
            self.target_visual_speed = 1.0
            
            # 🧠 SISTEMA DE CORRECCIÓN INTELIGENTE DE PLACAS
            try:
                self.smart_corrector = SmartPlateCorrector()
                print("🧠 Sistema de corrección inteligente inicializado")
            except Exception as e:
                print(f"⚠️ Error inicializando corrector: {e}")
                self.smart_corrector = None
            self.speed_transition_rate = 0.1  # Suavizado de cambios de velocidad
            
        except Exception as e:
            print(f"⚠️ Error inicializando sistema fluido: {e}")
            self.display_enabled = False  # Deshabilitar si hay problemas
        
        # Variables de configuración de optimización (necesarias para _get_skip_rate_for_frame)
        self.green_skip_rate = 3    # Skip x3 durante fase VERDE (más evidente)
        self.fast_skip_rate = 2     # Skip x2 durante fast-scan (amarillo)
        
        # Definir rutas de configuración usando resource_path para PyInstaller
        self.POLYGON_CONFIG_FILE = resource_path("config/polygon_config.json")
        self.AVENUE_CONFIG_FILE = resource_path("config/avenue_config.json")
        self.PRESETS_FILE = resource_path("config/time_presets.json")

        # Add this line to track start time
        self.processing_start_time = time.time()

        # Reiniciar el marcador del reproductor para medir la duración total
        # del procesamiento del video (numerador del TR en Foto Rojo).
        if self.player is not None and hasattr(self.player, "detection_start_time"):
            self.player.detection_start_time = self.processing_start_time
        
        self.plate_classifier = PlateClassificationSystem()
        self.metrics_calculator = ThesisMetricsCalculator()
        self.smart_corrector = SmartPlateCorrector()
        
        # 🚀 PIPELINE ASÍNCRONO: Procesa durante VERDE/AMARILLO (Idea de Abel 2026)
        try:
            from src.core.processing.async_plate_processor import get_async_processor
            self.async_processor = get_async_processor()
            self.async_processor.start()
            print("🚀 Pipeline Asíncrono: Activado (procesa en intervalos vacíos)")
        except Exception as e:
            self.async_processor = None
            print(f"⚠️ Pipeline Asíncrono no disponible: {e}")
        
        print("🧠 Sistema de clasificación NID/NIE inicializado con umbrales balanceados")

        
        # Reset class variable for this instance
        if len(PreprocessingDialog.recorded_processing_times) > 100:  # Limit history
            PreprocessingDialog.recorded_processing_times = []
        
        # Cargar configuración del video si existe
        self.load_video_config()
        
        # Crear ventana de diálogo (o modo inline en el reproductor principal)
        self.inline = inline
        self._render_target = render_target
        self._progress_mount = progress_mount
        self._inline_widgets = []

        if inline:
            # MODO INLINE: NO se abre ventana nueva. Se renderiza directamente
            # en el video_label del reproductor principal y se monta una barra
            # de progreso pequeña en el contenedor de progreso.
            self.dialog = parent.winfo_toplevel()
            self._after_ids = []
            self.video_label = render_target if render_target is not None else None
            self._setup_inline_ui(progress_mount)
        else:
            self.dialog = tk.Toplevel(parent)
            self.dialog.title("Análisis de infracciones")
            self._after_ids = []   # <-- NUEVA LÍNEA

            set_window_icon(self.dialog)
            self.dialog.geometry("1050x800")
            self.dialog.resizable(False, False)

            # Centrar ventana
            self.dialog.update_idletasks()
            width, height = 1050, 850
            x = (self.dialog.winfo_screenwidth() - width) // 2
            y = (self.dialog.winfo_screenheight() - height) // 2
            self.dialog.geometry(f"{width}x{height}+{x}+{y}")

            self.dialog.grab_set()  # Modal
            self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)  # Manejar cierre
            # Unir hilos de fondo al destruirse (evita Abort/SIGSEGV al salir)
            self.dialog.bind("<Destroy>", self._on_dialog_destroy)

            # Configurar el layout
            self._setup_ui()

        # Pump de UI: enruta callables de hilos de fondo al hilo principal de Tk
        self._start_ui_pump()

        # Precargar modelos en un hilo separado para evitar bloquear la UI
        self.preload_thread = threading.Thread(target=self._preload_models, daemon=True)
        self.preload_thread.start()

        # Programar actualizaciones periódicas de la UI
        self._schedule_ui_update()

        # 🚀 INICIALIZAR VISUALIZACIÓN FLUIDA (DIFERIDA PARA EVITAR ERRORES)
        self._safe_after(1000, self._start_smooth_display_thread_safe)

    def _safe_after(self, ms, func, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            try:
                after_id = self.dialog.after(ms, func, *args, **kwargs)
                self._after_ids.append(after_id)
                return after_id
            except tk.TclError:
                return None
        # Llamado desde un hilo de fondo: enrutar al pump de UI (hilo principal)
        self._ui_call(self._safe_after, ms, func, *args, **kwargs)
        return None

    def _safe_after_idle(self, func, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            try:
                after_id = self.dialog.after_idle(func, *args, **kwargs)
                self._after_ids.append(after_id)
                return after_id
            except tk.TclError:
                return None
        self._ui_call(self._safe_after_idle, func, *args, **kwargs)
        return None

    # ─── Pump de UI: ejecuta en el hilo principal los callables encolados ──

    def _start_ui_pump(self):
        try:
            self.dialog.after(50, self._pump_ui_queue)
        except tk.TclError:
            pass

    def _ui_call(self, fn, *args, **kwargs):
        """Encola un callable para el hilo principal de Tk. Seguro desde
        cualquier hilo; nunca toca Tk aquí."""
        try:
            self._ui_queue.put_nowait((fn, args, kwargs))
        except Exception:
            pass

    def _pump_ui_queue(self):
        """[HILO PRINCIPAL] Ejecuta los callables de UI encolados por hilos."""
        try:
            while True:
                fn, args, kwargs = self._ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    print(f"Error en UI encolado: {e}")
        except queue.Empty:
            pass
        if getattr(self, 'display_active', False):
            try:
                self.dialog.after(50, self._pump_ui_queue)
            except tk.TclError:
                pass

    def _cancel_all_after(self):
        for after_id in self._after_ids:
            try:
                self.dialog.after_cancel(after_id)
            except:
                pass
        self._after_ids.clear()
    
    def _preload_models(self):
        """Precarga los modelos de IA antes de procesar el video"""
        try:
            self._ui_call(self.phase_label.config, text="Preparando modelos de IA...")
            self._ui_call(self.details_label.config, text="Inicializando detectores...")
            
            # Inicializar detectores si son necesarios
            if not hasattr(self.player, 'vehicle_detector'):
                from src.core.detection.vehicle_detector import VehicleDetector
                self.player.vehicle_detector = VehicleDetector(model_path=resource_path("models/yolov8n.pt"))
                
            # Inicializar el detector ANPR para placas
            if not hasattr(self.player, 'anpr_detector'):
                from src.core.detection.anpr import ANPR
                self.player.anpr_detector = ANPR(languages=['es', 'en'])
                
            # Mantener el detector de placas anterior como fallback
            if not hasattr(self.player, 'plate_detector'):
                from src.core.detection.plate_detector import PlateDetector
                self.player.plate_detector = PlateDetector()
            
            # Una vez cargados los modelos, iniciar procesamiento del video
            self.process_thread = threading.Thread(target=self._process_video, daemon=True)
            self.process_thread.start()
        except Exception as e:
            self._safe_after(0, lambda msg=str(e): self._show_error(f"Error cargando modelos: {msg}"))
    
    def load_video_config(self):
        """Carga la configuración del video (polígono, semáforo, etc.)"""
        self.polygon_points = []
        self.cycle_durations = None
        self.current_avenue = None
        
        try:
            # Cargar todas las configuraciones de una vez
            configs = {}
            config_files = {
                'polygon': self.POLYGON_CONFIG_FILE,
                'presets': self.PRESETS_FILE,
                'avenue': self.AVENUE_CONFIG_FILE
            }
            
            for key, path in config_files.items():
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            configs[key] = json.load(f)
                    except Exception as e:
                        print(f"Error al cargar {key}: {e}")
                        configs[key] = {}
                else:
                    configs[key] = {}
            
            # Extraer datos específicos para este video usando solo el nombre del archivo
            video_key = os.path.basename(self.video_path)
            
            if video_key in configs.get('polygon', {}):
                self.polygon_points = configs['polygon'][video_key]
                
            if video_key in configs.get('presets', {}):
                self.cycle_durations = configs['presets'][video_key]
                
            if video_key in configs.get('avenue', {}):
                self.current_avenue = configs['avenue'][video_key]
            
            # Validación final
            valid_polygon = self.polygon_points and len(self.polygon_points) >= 3
            valid_times = (self.cycle_durations and 
                          isinstance(self.cycle_durations, dict) and
                          'green' in self.cycle_durations and
                          'yellow' in self.cycle_durations and
                          'red' in self.cycle_durations)
                
        except Exception as e:
            print(f"❌ Error en load_video_config: {e}")
            import traceback
            traceback.print_exc()
    
    def update_synchronized_semaphore(self):
        """Actualizar semáforo sincronizado con el estado principal (MEJORADO)"""
        try:
            if hasattr(self.player, 'semaforo') and self.player.semaforo and self.player.semaforo.active:
                current_state = self.player.semaforo.get_current_state()
                
                # Calcular tiempo restante con mayor precisión
                time_left = 0
                if hasattr(self.player.semaforo, 'target_time'):
                    time_diff = self.player.semaforo.target_time - time.time()
                    time_left = max(0, int(time_diff))
                
                # Resetear todas las luces a estado apagado
                self.semaphore_canvas.itemconfig(self.red_light, fill="#400000")
                self.semaphore_canvas.itemconfig(self.yellow_light, fill="#404000") 
                self.semaphore_canvas.itemconfig(self.green_light, fill="#004000")
                
                # Encender luz correspondiente con colores exactos
                if current_state == "red":
                    self.semaphore_canvas.itemconfig(self.red_light, fill="red")
                    self.state_label.config(text="ROJO", fg="red")
                elif current_state == "yellow":
                    self.semaphore_canvas.itemconfig(self.yellow_light, fill="yellow")
                    self.state_label.config(text="AMARILLO", fg="orange")
                elif current_state == "green":
                    self.semaphore_canvas.itemconfig(self.green_light, fill="green") 
                    self.state_label.config(text="VERDE", fg="green")
                
                # Actualizar tiempo con más precisión
                self.time_label.config(text=f"{time_left}s")
                
            else:
                # Semáforo no activo - mostrar estado inactivo
                self.semaphore_canvas.itemconfig(self.red_light, fill="grey")
                self.semaphore_canvas.itemconfig(self.yellow_light, fill="grey")
                self.semaphore_canvas.itemconfig(self.green_light, fill="grey")
                self.state_label.config(text="INACTIVO", fg="gray")
                self.time_label.config(text="-- s")
                
        except Exception as e:
            print(f"Error actualizando semáforo sincronizado: {e}")
            
        # Programar próxima actualización con MISMA frecuencia que semáforo principal (50ms)
        if hasattr(self, 'semaphore_frame') and self.semaphore_frame.winfo_exists():
            self.semaphore_frame.after(50, self.update_synchronized_semaphore)
    
    def _setup_ui(self):
        """Configura la interfaz de usuario del diálogo"""
        # Contenedor principal con padding
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="Analizando video para detección de infracciones", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Frame contenedor para video y semáforo
        video_container = ttk.Frame(main_frame)
        video_container.pack(pady=(0, 20), fill="x")
        
        # Frame para la visualización del video (AUMENTADO A 950x540 para mejor visión)
        self.video_frame = ttk.Frame(video_container, width=950, height=540, relief="groove", borderwidth=2)
        self.video_frame.pack(side="left", padx=5)
        self.video_frame.pack_propagate(False)
        
        # Label para mostrar el frame actual (Contenedor con Monitor)
        display_container = ttk.Frame(self.video_frame)
        display_container.pack(fill="both", expand=True)
        
        self.video_label = ttk.Label(display_container)
        self.video_label.pack(side="left", fill="both", expand=True)
        
        # Monitor lateral eliminado a petición del usuario (Fase 1 limpia)
        self.plate_monitor_ready = False
        
        # Monitor lateral listo
        self.plate_monitor_ready = True
        
        # Información de procesamiento
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.pack(fill="x", pady=(0, 10))
        
        # Etiqueta para mostrar la fase actual
        self.phase_label = ttk.Label(
            self.info_frame, 
            text="Preparando análisis...", 
            font=("Arial", 14, "bold")
        )
        self.phase_label.pack(anchor="w")
        
        # Etiqueta para mostrar detalles del procesamiento
        self.details_label = ttk.Label(
            self.info_frame, 
            text="",
            font=("Arial", 12)
        )
        self.details_label.pack(anchor="w")
        
        # Frame para la barra de progreso
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=10)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100, 
            length=760,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x")
        
        # Frame para etiquetas de estado debajo de la barra
        labels_under_progress = ttk.Frame(progress_frame)
        labels_under_progress.pack(fill="x", pady=(5, 0))
        
        # Etiqueta de infracciones (Izquierda)
        self.infractions_counter_label = ttk.Label(
            labels_under_progress, 
            text="Infracciones: 0", 
            font=("Arial", 11, "bold"),
            foreground="#e74c3c"
        )
        self.infractions_counter_label.pack(side="left")
        
        # Etiqueta de porcentaje (Derecha)
        self.percentage_label = ttk.Label(
            labels_under_progress, 
            text="0%", 
            font=("Arial", 12, "bold")
        )
        self.percentage_label.pack(side="right")
        
        # Contador de INFRACCIONES (NID/NIE - Estilo Tesis)
        self.stats_frame = ttk.LabelFrame(main_frame, text=" 📊 Métricas de Procesamiento (Tesis) ")
        self.stats_frame.pack(fill="x", pady=10, padx=5)
        
        # Grid para métricas
        metrics_inner = ttk.Frame(self.stats_frame)
        metrics_inner.pack(pady=5, padx=10, fill="x")
        
        self.nid_label = ttk.Label(metrics_inner, text="NID: 0", font=("Arial", 11, "bold"), foreground="#27ae60")
        self.nid_label.pack(side="left", expand=True)
        
        self.nie_label = ttk.Label(metrics_inner, text="NIE: 0", font=("Arial", 11, "bold"), foreground="#e67e22")
        self.nie_label.pack(side="left", expand=True)
        
        self.v_count_label = ttk.Label(metrics_inner, text="Vehículos: 0", font=("Arial", 11, "bold"), foreground="#2980b9")
        self.v_count_label.pack(side="left", expand=True)
        
        # NUEVO: Panel de contadores detallados alineado con la tesis
        self.stats_panel = ttk.Frame(main_frame)
        self.stats_panel.pack(fill="x", pady=5)
        
        # Frame para botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        # Botón de cancelar
        self.cancel_button = ttk.Button(
            button_frame, 
            text="Cancelar", 
            command=self.on_cancel
        )
        self.cancel_button.pack(side="right")

    def _setup_inline_ui(self, progress_mount):
        """UI compacta para el modo inline: barra de progreso pequeña en main.

        Usa los MISMOS nombres de atributo que `_setup_ui` para que la lógica
        existente (`_process_results_queue`, `_schedule_ui_update`, etc.)
        funcione sin cambios.
        """
        mount = progress_mount if progress_mount is not None else self.parent
        frame = ttk.Frame(mount)
        frame.pack(fill="x", expand=False)
        self._inline_widgets.append(frame)

        self.progress_var = tk.DoubleVar(value=0)

        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100, length=760,
            mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=(4, 2))

        status_row = ttk.Frame(frame)
        status_row.pack(fill="x")
        self.phase_label = ttk.Label(
            status_row, text="Preparando análisis...",
            font=("Arial", 10, "bold")
        )
        self.phase_label.pack(side="left")
        self.percentage_label = ttk.Label(
            status_row, text="0%", font=("Arial", 10, "bold")
        )
        self.percentage_label.pack(side="right")

        self.details_label = ttk.Label(
            frame, text="", font=("Arial", 9)
        )
        self.details_label.pack(fill="x", pady=1)

        counter_row = ttk.Frame(frame)
        counter_row.pack(fill="x")
        self.infractions_counter_label = ttk.Label(
            counter_row, text="Infracciones: 0",
            font=("Arial", 9, "bold"), foreground="#e74c3c"
        )
        self.infractions_counter_label.pack(side="left")
        self.nid_label = ttk.Label(
            counter_row, text="NID: 0", font=("Arial", 9, "bold"),
            foreground="#27ae60")
        self.nid_label.pack(side="left", padx=(10, 0))
        self.nie_label = ttk.Label(
            counter_row, text="NIE: 0", font=("Arial", 9, "bold"),
            foreground="#e67e22")
        self.nie_label.pack(side="left", padx=(10, 0))
        self.v_count_label = ttk.Label(
            counter_row, text="Vehículos: 0", font=("Arial", 9, "bold"),
            foreground="#2980b9")
        self.v_count_label.pack(side="left", padx=(10, 0))

        self.cancel_button = ttk.Button(
            frame, text="Cancelar", command=self.on_cancel
        )
        self.cancel_button.pack(anchor="e", pady=(2, 2))

        # Stubs para la API de métricas/Tesis (no se usan en modo inline)
        self.stats_frame = frame
        self.stats_panel = frame

    def _sync_semaforo_state(self, state):
        """Refleja el estado del semáforo en el widget principal (modo inline).

        En lugar de dibujar verde/amarillo/rojo sobre el video, se enciende la
        luz correspondiente en el Semaforo de la pantalla (y se activa si está
        inactivo durante el procesamiento).
        """
        try:
            semaforo = getattr(self.player, "semaforo", None)
            if semaforo is None:
                return
            if state in ("green", "yellow", "red"):
                if not semaforo.active:
                    semaforo.active = True
                semaforo.start_meta_timer()
                semaforo.current_state = state
                semaforo.show_state()
                semaforo.set_state_display()
        except Exception as e:
            print(f"Error sincronizando semáforo inline: {e}")

    def _inline_progress_show(self, visible):
        """Muestra/oculta los widgets de progreso inline."""
        for w in self._inline_widgets:
            try:
                if visible:
                    w.pack(fill="x", expand=False)
                else:
                    w.pack_forget()
            except Exception:
                pass

    def _schedule_ui_update(self):
        """Programa actualizaciones periódicas de la interfaz"""
        if not self.canceled:
            try:
                # Actualizar barra de progreso con animación suave
                self.progress_var.set(self.progress_value)
                self.percentage_label.config(text=f"{int(self.progress_value)}%")
                
                # Actualizar contadores de Tesis (NID/NIE/Vehículos)
                total_inf = len(self.detected_infractions)
                if self._official_infraction_count:
                    total_inf = max(total_inf, self._official_infraction_count)
                nid_count = 0
                for inf in self.detected_infractions:
                    if inf.get('clasificacion') == 'NIE':
                        continue
                    nid_count += 1
                
                # SINCRONIZACIÓN DE CONTADORES: Reflejar estado real en todo momento
                self.nid_label.config(text=f"✅ NID: {nid_count}")
                self.nie_label.config(text=f"⚠️ NIE: {total_inf - nid_count}")
                # Vehículos muestra el total de infracciones registradas (NID + NIE)
                self.v_count_label.config(text=f"🚗 Vehículos: {total_inf}")
                
                # Actualizar label principal de infracciones para que coincida con el total
                if self.progress_value >= 100:
                    status_text = "FINALIZADO"
                    self.infractions_counter_label.config(text=f"Total: {total_inf}", foreground="#27ae60")
                else:
                    self.infractions_counter_label.config(text=f"Infracciones: {total_inf}")
                
                # Procesar cualquier resultado pendiente de los hilos de trabajo
                self._process_results_queue()

                # Forzar actualización de la interfaz
                self.dialog.update_idletasks()

                # Programar próxima actualización (más frecuente para que sea fluido).
                # En modo inline, al llegar a 100% el procesamiento terminó y no
                # hace falta seguir actualizando la UI de la ventana principal.
                if getattr(self, "inline", False) and self.progress_value >= 100:
                    return
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self._safe_after(50, self._schedule_ui_update)
            except Exception as e:
                print(f"Error en actualización de UI: {e}")
                # Seguir intentando actualizar la UI
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self._safe_after(100, self._schedule_ui_update)

    
    def _process_results_queue(self):
        """Procesa los resultados en la cola sin bloquear la interfaz"""
        try:
            # Procesar solo un número limitado de elementos por ciclo para evitar bloqueos
            max_items_per_cycle = 5
            items_processed = 0
            
            while items_processed < max_items_per_cycle:
                # Obtener un elemento sin bloquear
                result = self.result_queue.get_nowait()
                items_processed += 1
                
                # Verificar el tipo de resultado
                if isinstance(result, tuple) and len(result) == 2:
                    result_type, data = result

                    if result_type == "official_frame":
                        self.current_frame = data["frame"]
                        self._update_video_frame(data["frame"])
                        # En modo inline: reflejar el estado en el widget Semaforo
                        if getattr(self, "inline", False):
                            # Reiniciar temporizador una vez por corrida (hilo principal)
                            if not self._meta_timer_ready:
                                self._meta_timer_ready = True
                                if getattr(self.player, "semaforo", None) is not None:
                                    self.player.semaforo.reset_execution_timer()
                            self._sync_semaforo_state(data.get("state"))
                        total = max(1, data.get("total_frames", 1))
                        self.progress_value = min(99.9, data.get("frame_index", 0) / total * 100)
                        mode = "análisis completo" if data.get("processed") else "verde: skip x60"
                        self.details_label.config(text=f"{data.get('state', '').upper()} | {mode} | frame {data.get('frame_index', 0)}/{total}")
                        continue

                    if result_type == "official_infraction":
                        track_id = data["track_id"]
                        if track_id not in self._official_infraction_ids:
                            self._official_infraction_ids.add(track_id)
                            self._official_infraction_count = len(self._official_infraction_ids)
                            self.detected_infractions.append({
                                "vehicle_id": track_id,
                                "frame": data["frame_index"],
                                "timestamp_seconds": data["timestamp_seconds"],
                                "vehicle_class": data.get("vehicle_class", "VEH"),
                                "clasificacion": "NID",
                                "official_detection": True,
                            })
                            self.infractions_counter_label.config(
                                text=f"Infracciones: {self._official_infraction_count}",
                                foreground="#e74c3c",
                            )
                            self.v_count_label.config(text=f"🚗 Vehículos: {self._official_infraction_count}")
                        continue

                    if result_type == "official_complete":
                        payload = data
                        self.progress_value = 100
                        self.total_frames = int(payload.get("frames", 0))
                        self.detected_infractions = payload.get("evidence", [])
                        self._official_infraction_count = int(payload.get("infractor_count", self._official_infraction_count))
                        self.phase_label.config(text="Análisis completado: revisar placas")
                        self.details_label.config(text=f"{self._official_infraction_count} infractores detectados | {len(self.detected_infractions)} mejores frames listos para OCR")
                        if getattr(self, "inline", False):
                            self._inline_progress_show(False)
                        # Congelar temporizador y semáforo al terminar el procesamiento
                        if getattr(self.player, "semaforo", None) is not None:
                            self.player.semaforo.deactivate_semaphore()
                        self._open_official_review(payload)
                        continue

                    if result_type == "official_error":
                        self._show_error(data)
                        continue
                    
                    if result_type == "frame_update":
                        # data: (display, segment_id, processed, total_frames, abs_f)
                        frame, segment_id, processed_in_segment, segment_length, absolute_frame = data
                        
                        # Actualizar el frame actual y mostrarlo inmediatamente
                        self.current_frame = frame
                        self._update_video_frame(frame)
                        
                        # Actualizar información de progreso para este segmento
                        segment_progress = (processed_in_segment / segment_length) * 100 if segment_length > 0 else 0
                        
                        # Actualizar progreso global considerando segmentos completados
                        base_progress = (self.completed_segments / self.total_segments) * 100
                        segment_part = (1 / self.total_segments) * (segment_progress / 100) * 100
                        self.progress_value = min(base_progress + segment_part, 99.9)  # No llegar a 100% hasta terminar
                        
                        # ⚡ ACTUALIZAR ACELERACIÓN VISUAL EN TIEMPO REAL
                        curr_state = self._get_semaphore_state_for_frame(absolute_frame)
                        self._update_visual_acceleration(curr_state, absolute_frame)
                        
                        # Actualizar texto de progreso SIN CONTADOR (se actualiza en segment_complete)
                        self.details_label.config(text=f"Procesando segmento {segment_id+1}/{self.total_segments} | Frame {absolute_frame}/{segment_length}")
                    
                    # Estos msg tipos de monitor lateral fueron eliminados para Fase 1 limpia
                    elif result_type in ["plate_monitor_status", "plate_monitor_update"]:
                        pass

                    elif result_type == "phase2_result":
                        # data: {'index', 'plate_text', 'confidence', 'plate_crop', 'vehicle_img', 'infraction'}
                        self._display_phase2_result(data)
                    
                    elif result_type == "phase2_skip":
                        self._phase2_index += 1
                        self._phase2_processing = False
                        print(f"⏭️ Infracción descartada, pasando a la siguiente...")
                        self._safe_after(50, self._run_phase2_analysis)

                    elif result_type == "segment_complete":
                        segment_id, infractions = data
                        # Añadir las infracciones detectadas
                        previous_count = len(self.detected_infractions)
                        self.detected_infractions.extend(infractions)
                        new_count = len(self.detected_infractions)
                        
                        # 🔊 BEEP MOVIDO A FINAL DEL PROCESAMIENTO (evitar duplicados por mismo vehículo)
                        # Los beeps ahora se reproducen después del filtrado de vehículos únicos
                        
                        # Actualizar contador de segmentos completados
                        self.completed_segments += 1
                        # Actualizar progreso CON CONTADOR SINCRONIZADO
                        base_progress = (self.completed_segments / self.total_segments) * 100
                        self.progress_value = base_progress
                        current_infractions = len(self.detected_infractions)
                        self.details_label.config(text=f"Completado: {self.completed_segments}/{self.total_segments} segmentos | 🚗 {current_infractions} infracciones detectadas")
                        
                        # Mostrar último frame con infracciones si hay alguna
                        if infractions and not self.canceled:
                            try:
                                # Cargar y mostrar el frame con la infracción detectada
                                temp_cap = cv2.VideoCapture(self.video_path)
                                temp_cap.set(cv2.CAP_PROP_POS_FRAMES, infractions[0]['frame'])
                                ret, demo_frame = temp_cap.read()
                                if ret:
                                    # Calcular estado real del semáforo para el frame de infracción
                                    infraction_semaphore_state = self._get_semaphore_state_for_frame(infractions[0]['frame'])
                                    skip_rate_for_frame = self._get_skip_rate_for_frame(infractions[0]['frame'])  # Debería ser 1 (rojo)
                                    
                                    # Dibujar información en el frame
                                    self._draw_mini_semaphore(demo_frame, infraction_semaphore_state, 0, self.fps, self.is_night, skip_rate_for_frame)
                                    cv2.rectangle(demo_frame, (10, 50), (300, 80), (0, 0, 0), -1)
                                    cv2.putText(demo_frame, f"Placa: {infractions[0]['plate']}", (15, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    # Actualizar inmediatamente
                                    self.current_frame = demo_frame
                                    self._update_video_frame(demo_frame)
                                temp_cap.release()
                            except Exception as e:
                                print(f"Error mostrando frame de infracción: {e}")
                    
        except queue.Empty:
            # Cola vacía, no hay problema
            pass
        except Exception as e:
            # Manejar cualquier otra excepción sin interrumpir el flujo
            print(f"Error procesando cola: {e}")
    
    def _start_smooth_display_thread_safe(self):
        """🚀 Inicialización segura del thread de visualización fluida"""
        try:
            if not hasattr(self, 'video_label') or not hasattr(self, 'dialog'):
                print("⚠️ UI no está completamente inicializada, posponiendo visualización fluida")
                return
                
            self._start_smooth_display_thread()
        except Exception as e:
            print(f"⚠️ Error iniciando visualización fluida: {e}")
            print("📹 Continuando con visualización estándar")
    
    def _start_smooth_display_thread(self):
        """🚀 Inicia thread separado para visualización fluida (NO AFECTA PROCESAMIENTO)"""
        def smooth_display_loop():
            base_frame_time = 1.0 / self.display_fps  # 33ms para 30 FPS base
            
            while getattr(self, 'display_active', False) and hasattr(self, 'dialog'):
                try:
                    start_time = time.time()
                    
                    # Obtener frame más reciente del buffer
                    with self.display_lock:
                        if len(self.display_buffer) > 0:
                            current_frame = self.display_buffer[-1]  # Frame más reciente
                        else:
                            current_frame = self.last_display_frame
                    
                    # Encolar para el hilo principal de Tk. Este hilo de fondo
                    # NO debe tocar Tk (winfo_exists/after/config): Tkinter no
                    # es thread-safe y provoca SIGSEGV.
                    if current_frame is not None and hasattr(self, 'video_label'):
                        try:
                            self._display_queue.put_nowait(current_frame)
                        except queue.Full:
                            # Descartar el frame viejo y publicar el más reciente
                            try:
                                self._display_queue.get_nowait()
                            except queue.Empty:
                                pass
                            self._display_queue.put_nowait(current_frame)
                    
                    # 🚀 Control de FPS fluido CON ACELERACIÓN VISUAL
                    # Aplicar factor de aceleración visual: mayor velocidad = menor tiempo entre frames
                    accelerated_frame_time = base_frame_time / getattr(self, 'visual_speed_multiplier', 1.0)
                    
                    elapsed = time.time() - start_time
                    sleep_time = max(0, accelerated_frame_time - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"Error en visualización fluida: {e}")
                    time.sleep(0.1)  # Evitar bucle infinito en errores
        
        # Crear y iniciar thread de visualización
        try:
            self.display_thread = threading.Thread(target=smooth_display_loop, daemon=True)
            self.display_thread.start()
            print("🚀 Thread de visualización fluida iniciado (30 FPS)")
        except Exception as e:
            print(f"⚠️ Error creando thread de visualización: {e}")
            print("📹 Continuando sin visualización fluida")

        # Poller en el hilo principal: drena la cola y actualiza el label.
        # Se programa con `after` DESDE el hilo principal (seguro para Tk).
        self._schedule_display_poller()

    def _schedule_display_poller(self):
        """Programa el poller de la cola de visualización en el hilo principal."""
        try:
            self.dialog.after(16, self._poll_display_queue)
        except tk.TclError:
            pass

    def _poll_display_queue(self):
        """[HILO PRINCIPAL] Drena `_display_queue` y muestra el frame más reciente."""
        if not getattr(self, 'display_active', False):
            return
        try:
            if not hasattr(self, 'dialog') or not self.dialog.winfo_exists():
                return
        except tk.TclError:
            return
        frame = None
        try:
            frame = self._display_queue.get_nowait()
            while True:
                try:
                    candidate = self._display_queue.get_nowait()
                    frame = candidate
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        if frame is not None and hasattr(self, 'video_label'):
            try:
                self._display_frame_immediate(frame)
            except tk.TclError:
                return
        self._schedule_display_poller()
    
    def _display_frame_immediate(self, frame):
        """🎬 Muestra frame inmediatamente sin procesamiento pesado"""
        if frame is None or not hasattr(self, 'video_label'):
            return
            
        # Verificaciones de seguridad
        if not hasattr(self, 'dialog') or not self.dialog.winfo_exists():
            return
            
        try:
            # Redimensionar frame (optimizado para velocidad)
            h, w = frame.shape[:2]
            max_w, max_h = 640, 360
            
            ratio = min(max_w/w, max_h/h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            # Usar INTER_LINEAR para mejor calidad sin sacrificar mucho rendimiento
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 🌟 MEJORA: Aplicar suavizado ligero para mejor calidad visual
            if hasattr(self, 'frame_interpolation') and self.frame_interpolation:
                kernel = np.array([[0, -0.1, 0], [-0.1, 1.4, -0.1], [0, -0.1, 0]], dtype=np.float32)
                resized = cv2.filter2D(resized, -1, kernel)
                resized = np.clip(resized, 0, 255).astype(np.uint8)
            
            # Convertir a RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Crear imagen para Tkinter
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Actualizar UI inmediatamente
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            # Actualizar semáforo (no pesado). En modo inline el semáforo se
            # sincroniza con el widget principal desde el estado del pipeline.
            if not getattr(self, "inline", False):
                self.update_synchronized_semaphore()
            
        except Exception as e:
            print(f"Error mostrando frame fluido: {e}")
    
    def _update_visual_acceleration(self, semaphore_state, frame_index):
        """⚡ Actualiza la aceleración visual según el estado del semáforo"""
        
        # Determinar velocidad objetivo según el estado
        if semaphore_state == "green":
            self.target_visual_speed = 3.0  # 3x más rápido visualmente (Estricto)
            self.visual_acceleration_active = True
            status = "🟢 VERDE - Acelerando x3"
        elif semaphore_state == "yellow":
            self.target_visual_speed = 1.5  # 1.5x más rápido visualmente (Parcial)
            self.visual_acceleration_active = True
            status = "🟡 AMARILLO - Acelerando x1.5"
        elif semaphore_state == "red":
            self.target_visual_speed = 1.0  # Velocidad normal para análisis
            self.visual_acceleration_active = False
            status = "🔴 ROJO - Velocidad normal (detectando)"
        else:
            self.target_visual_speed = 1.0
            self.visual_acceleration_active = False
            status = "⚪ Desconocido - Velocidad normal"
        
        # Suavizar transición de velocidad
        speed_diff = self.target_visual_speed - self.visual_speed_multiplier
        self.visual_speed_multiplier += speed_diff * self.speed_transition_rate
        
        # Mostrar en terminal cada 30 frames para no saturar
        if frame_index % 30 == 0:
            print(f"📺 VISUALIZACIÓN: {status} | Velocidad actual: {self.visual_speed_multiplier:.1f}x")
    
    def get_total_user_tr(self):
        """📈 Obtiene el TR total desde perspectiva del usuario"""
        if self.user_tr_start_time is None:
            return 0
            
        current_time = time.time()
        return current_time - self.user_tr_start_time
    
    def _add_frame_to_buffer(self, frame):
        """📦 Añade frame al buffer de visualización fluida (thread-safe)"""
        if frame is None or not hasattr(self, 'display_enabled') or not self.display_enabled:
            return
            
        try:
            # Hacer copia del frame para evitar problemas de referencia
            frame_copy = frame.copy()
            
            with self.display_lock:
                self.display_buffer.append(frame_copy)
                self.last_display_frame = frame_copy
                
        except Exception as e:
            print(f"Error añadiendo frame al buffer: {e}")
            # Deshabilitar sistema fluido si hay errores continuos
            self.display_enabled = False
    
    # _update_monitor_image eliminado (Fase 1 limpia)

    def _deep_analyze_infraction(self, frame, infraction, absolute_frame, segment_id):
        """Realiza el análisis pesado de una infracción detectada (Fase 2)"""
        try:
            # Recortar el vehículo con MARGEN EXTRA (10%) para mejorar detección de bordes
            car_bbox = infraction['bbox']
            cx1, cy1, cx2, cy2 = [int(v) for v in car_bbox]
            vh, vw = frame.shape[:2]
            mw, mh = int((cx2-cx1)*0.1), int((cy2-cy1)*0.1)
            x1, y1 = max(0, cx1-mw), max(0, cy1-mh)
            x2, y2 = min(vw, cx2+mw), min(vh, cy2+mh+mh) # Un poco más de margen abajo por la placa
            vehicle_roi = frame[y1:y2, x1:x2].copy()
            
            # Verificar si existe el detector ANPR
            has_anpr = hasattr(self.player, 'anpr_detector') and self.player.anpr_detector is not None
            
            # 🚀 LLAMADA CORREGIDA: Pasar vehicle_roi y absolute_frame
            # Retorna: (plate_text, plate_img, confidence)
            p_text, p_img, p_conf = self._extract_plate_from_vehicle(vehicle_roi, has_anpr, absolute_frame)
            
            plate_img = p_img
            if plate_img is not None:
                # Determinar texto y confianza (Fase 2 asíncrona)
                if p_text and len(p_text.replace('-', '')) >= 4:
                    plate_text, confidence = p_text, p_conf
                else:
                    plate_text, confidence = self._perform_smart_ocr(plate_img)
                    
                if plate_text and len(plate_text) >= 3:
                     plate_text = self._normalize_plate(plate_text)
                     with self.plate_registry_lock:
                         is_duplicate = False
                         plate_variations = self.smart_corrector.generate_variations(plate_text) if self.smart_corrector else [plate_text]
                         for var in plate_variations:
                             if var in self.detected_plates_global:
                                 is_duplicate = True
                                 break
                         # Notificación visual desactivada para Fase 1 limpia
                         pass
                         
                         if not is_duplicate:
                             for var in plate_variations:
                                 self.detected_plates_global.add(var)
                             
                             # Registrar feedback visual en el frame central
                             with self.feedback_lock:
                                 self.visual_feedback_items.append({
                                     'type': 'infraction', 
                                     'pos': infraction['crossing_point'], 
                                     'bbox': infraction['bbox'], 
                                     'expiry': absolute_frame + 30
                                 })
                             
                             # 🚀 CREAR REGISTRO OFICIAL USANDO EL MÉTODO ESTÁNDAR
                             inf_id = self._create_infraction_record(
                                 plate_text=plate_text,
                                 plate_img=plate_img,
                                 vehicle_img=vehicle_roi, # Usar ROI del vehículo, no el frame completo
                                 frame_index=absolute_frame,
                                 fps=self.fps,
                                 bbox=infraction['bbox'],
                                 track_id=infraction.get('track_id', 0),
                                 confidence=confidence
                             )
                             
                             if inf_id:
                                 with self.analysis_results_lock:
                                     self.detected_infractions.append(inf_id)
        except Exception as e:
            print(f"Error en _deep_analyze_infraction: {e}")
    def _update_video_frame(self, frame):
        """📺 VERSIÓN MEJORADA: Actualiza frame Y añade al buffer fluido"""
        if frame is None:
            return
            
        # 🚀 NUEVA: Añadir al buffer de visualización fluida
        self._add_frame_to_buffer(frame)
        
        # 📊 MANTENER: Lógica original para compatibilidad (SIN AFECTAR PROCESAMIENTO)
        try:
            # Solo actualizar ocasionalmente la UI desde aquí si el hilo fluido no está activo
            if not getattr(self, 'display_thread', None) or not self.display_thread.is_alive():
                self._display_frame_immediate(frame)
        except Exception as e:
            print(f"Error en actualización de frame: {e}")
    
    def is_vehicle_in_polygon(self, bbox, polygon_points, is_night=False):
        """Determina si un vehículo está dentro del polígono restrictivo con optimización de cálculos"""
        if not polygon_points or len(polygon_points) < 3:
            return False
            
        x1, y1, x2, y2 = bbox
        
        # Precomputar el polígono numpy una sola vez y reutilizarlo
        if not hasattr(self, '_np_polygon') or self._np_polygon is None:
            self._np_polygon = np.array(polygon_points, np.int32)
            
            # Para escenas nocturnas, precomputar también el polígono expandido
            if is_night and not hasattr(self, '_np_expanded_polygon'):
                center = np.mean(self._np_polygon, axis=0).astype(int)
                # Expandir 10%
                expanded_polygon = []
                for point in self._np_polygon:
                    vector = point - center
                    expanded_point = center + vector * 1.1
                    expanded_polygon.append(expanded_point)
                self._np_expanded_polygon = np.array(expanded_polygon, np.int32)
        
        # En modo nocturno, usar enfoque más permisivo con polígono expandido
        if is_night:
            # Usar puntos estratégicos para detección nocturna
            check_points = [
                ((x1+x2)//2, y2),        # Punto inferior central (ruedas)
                ((x1+x2)//2, (y1+y2)//2) # Centro
            ]
            
            # Solo verificar puntos clave, no todos
            for point in check_points:
                if cv2.pointPolygonTest(self._np_expanded_polygon, point, False) >= 0:
                    return True
            return False
        else:
            # En modo diurno usar solo el centro inferior (ruedas)
            center_x = (x1 + x2) // 2
            center_y = y2  # Borde inferior
            
            # Comprobar si el punto está dentro del polígono
            return cv2.pointPolygonTest(self._np_polygon, (center_x, center_y), False) >= 0
    
    def _process_video(self):
        """Procesa el video utilizando multithreading para detección de infracciones"""
        try:
            # The official pipeline owns the complete video-processing flow.
            # The legacy implementation remains below for compatibility with
            # older callers, but normal GUI execution uses this path.
            self._process_video_official()
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._safe_after(0, lambda msg=str(e): self._show_error(msg))

    def _process_video_official(self):
        """Run the shared core use case and feed Tk through the result queue."""
        if not self.polygon_points or not self.cycle_durations:
            raise ValueError("Este video no está configurado con polígono y semáforo")

        from pathlib import Path
        from src.infrastructure.configuration import VideoConfigRepository

        project_root = Path(__file__).resolve().parents[2]
        config = VideoConfigRepository(project_root).require(Path(self.video_path).name)
        output_dir = Path(resource_path("data/output/official"))
        self._ui_call(self.phase_label.config, text="Procesando con el pipeline oficial...")

        processor = OfficialVideoProcessor(
            project_root,
            vehicle_detector=getattr(self.player, "vehicle_detector", None),
            plate_detector=getattr(self.player, "plate_detector", None),
            draw_state_banner=not getattr(self, "inline", False),
        )

        def callback(event):
            if event["type"] == "frame":
                self.result_queue.put(("official_frame", event))
            elif event["type"] == "infraction_detected":
                self.result_queue.put(("official_infraction", event))
            elif event["type"] == "complete":
                self.result_queue.put(("official_complete", event["payload"]))

        try:
            processor.process(self.video_path, config, output_dir, callback=callback)
        except Exception as error:
            self.result_queue.put(("official_error", str(error)))

    def _open_official_review(self, payload):
        evidences = [PlateEvidence(
            video_name=item.get("video", Path(self.video_path).name),
            track_id=int(item.get("vehicle_id", 0)),
            frame_index=int(item.get("frame", 0)),
            timestamp_seconds=float(item.get("timestamp_seconds", 0)),
            vehicle_class=item.get("vehicle_class", "VEH"),
            quality_score=float(item.get("quality_score", 0)),
            crop_path=item.get("crop_path", ""),
            plate_text=item.get("plate", ""),
            ocr_confidence=float(item.get("ocr_confidence", 0)),
            ocr_method=item.get("ocr_method", ""),
            validated=bool(item.get("validated", False)),
        ) for item in payload.get("evidence", [])]
        if not evidences:
            self._complete_processing()
            return
        # Infracciones pendientes sin placa detectada (recuadro amarillo) -> NIE
        self._pending_infractions = payload.get("pending_infractions", [])
        if self._pending_infractions:
            try:
                self.player.apply_official_validation([], self._pending_infractions)
            except Exception as e:
                print(f"⚠️ Error añadiendo cards pendientes (NIE): {e}")
        PlateReviewWindow(
            self.dialog,
            evidences,
            Path(resource_path("data/output/official")),
            on_complete=self._on_official_validation_done,
        )

    def _on_official_validation_done(self, evidences):
        """Actualiza la barra lateral y las métricas según la validación final.

        NID = evidence.validated (✓) con placa reconocida -> verde.
        NIE = el resto, incluidos los pendientes sin placa -> rojo.

        Guarda en SQLite (única fuente) y genera indicadores coherentes.
        """
        try:
            self.player.apply_official_validation(evidences, getattr(self, "_pending_infractions", []))
        except Exception as e:
            print(f"⚠️ Error sincronizando validación en panel lateral: {e}")

        nid_entries, nie_entries = [], []
        try:
            nid_entries, nie_entries = self._save_official_infractions_to_db(
                evidences, getattr(self, "_pending_infractions", [])
            )
        except Exception as e:
            print(f"⚠️ Error guardando infracciones oficiales en DB: {e}")
            import traceback
            traceback.print_exc()

        try:
            self._regenerate_indicators_after_validation(nid_entries, nie_entries)
        except Exception as e:
            print(f"⚠️ Error regenerando indicadores tras validación: {e}")
            import traceback
            traceback.print_exc()

        self._migrate_after_validation(nid_entries, nie_entries)

    def _migrate_after_validation(self, nid_entries, nie_entries):
        """Migra SOLO la sesión actual a Firestore con TI/TR/NID/NIE coherentes.

        Fuente única: SQLite + lista de la sesión (no lee JSONs).
        """
        import threading
        import traceback
        from src.automations.firestore_migrator import migrate_single_video_to_firestore

        session_infractions = (nid_entries or []) + (nie_entries or [])
        nombre_video = os.path.basename(self.video_path) if getattr(self, 'video_path', None) else "desconocido.mp4"
        total = len(session_infractions)
        # Snapshot para el hilo
        nid_snap, nie_snap = list(nid_entries or []), list(nie_entries or [])

        def _job():
            try:
                if hasattr(self, 'no_cloud_migration') and self.no_cloud_migration:
                    print("⚠️ NO SE MIGRAN INFRACCIONES: Caso de detección nocturna sin resultados")
                    return
                if not session_infractions:
                    print("⚠️ No se encontraron infracciones que migrar (sesión vacía)")
                    from src.gui.infractions_management_window import add_migration_to_history
                    add_migration_to_history(0, "Sin datos")
                    return
                resultado = migrate_single_video_to_firestore(nombre_video, session_infractions)
                migrados = int(resultado.get("migrados", 0) or 0)
                errores = resultado.get("errores", [])
                estado = "Exitosa"
                if errores:
                    estado = "Fallida"
                    for err in errores:
                        print(f"⚠️ Error de migración: {err}")
                if migrados <= 0 and not errores:
                    estado = "Sin datos"
                from src.gui.infractions_management_window import add_migration_to_history
                try:
                    add_migration_to_history(int(total or 0), estado)
                except Exception as hist_error:
                    print(f"⚠️ Error registrando en historial: {hist_error}")
            except Exception as ex:
                print(f"⚠️ Error migrando a Firestore: {ex}")
                traceback.print_exc()
                try:
                    from src.gui.infractions_management_window import add_migration_to_history
                    add_migration_to_history(int(total or 0), "Fallida")
                except Exception:
                    pass

        threading.Thread(target=_job, daemon=True).start()

    def _regenerate_indicators_after_validation(self, nid_entries, nie_entries):
        """Genera indicadores en SQLite (única fuente) con la sesión validada.

        TI/TR/NID/NIE quedan coherentes con lo que se migrará a Firebase.
        """
        from src.infrastructure.database.app_repository import AppRepository

        current_session_infractions = (nid_entries or []) + (nie_entries or [])
        nombre_video = os.path.basename(self.video_path) if getattr(self, 'video_path', None) else "desconocido.mp4"
        config_semaforo = self.generar_config_id(
            semaforo=getattr(self.player, 'semaforo', None),
            cycle_durations=getattr(self, 'cycle_durations', None)
        )
        print(f"\n📊 Generando indicadores tras validación para {len(current_session_infractions)} infracciones "
              f"({len(nid_entries or [])} NID + {len(nie_entries or [])} NIE)...")
        repo = AppRepository()
        report = repo.compute_indicators_report(current_session_infractions, nombre_video=nombre_video, config_semaforo=config_semaforo)
        repo.upsert_indicators(report)
        print(f"✅ Indicadores SQLite actualizados para {nombre_video}")

    def _process_next_frame(self):
        """
        Procesa UN frame y programa el siguiente.
        OPTIMIZADO: Detecta cada 3 frames pero dibuja en cada uno.
        """
        # Verificar si debemos continuar
        if not hasattr(self, '_prep_running') or not self._prep_running:
            return
        if self.canceled:
            self._finalize_preprocessing()
            return
        if not hasattr(self, 'dialog') or not self.dialog.winfo_exists():
            return
        
        # Leer siguiente frame
        ret, frame = self._prep_cap.read()
        if not ret or self._prep_frame_index >= self.total_frames:
            self._finalize_preprocessing()
            return
        
        h, w = frame.shape[:2]
        
        # Inicializar cache de detecciones si no existe
        if not hasattr(self, '_cached_detections'):
            self._cached_detections = []
        # =====================================================
        # 🧪 MONITOR DE RECURSOS (Adaptación Dinámica)
        # =====================================================
        if not hasattr(self, '_perf_monitor'):
            self._perf_monitor = {'times': [], 'adapt_level': 0}
        
        start_t = time.time()
        
        # Estrategia profesional: Si el semáforo es verde, saltar más cuadros (ahorro CPU)
        # Si es rojo o amarillo, procesar con mayor frecuencia
        skip_rate = 3  # Por defecto cada 3 frames
        
        # Ajuste dinámico si la PC es lenta
        if self._perf_monitor['adapt_level'] > 0:
            skip_rate += self._perf_monitor['adapt_level'] # Saltar mas frames si es lento
            
        current_state = self.player.semaforo.get_current_state()
        if current_state == "green":
            skip_rate = 10 + self._perf_monitor['adapt_level'] # Ahorro masivo en verde
        elif current_state == "red":
            # Si hay infractores activos, ser más preciso
            # 🚀 MODO TURBO V41: En ROJO o con infractores activos, procesamos TODO
            if current_state == "red" or (hasattr(self, '_active_infractors') and self._active_infractors):
                skip_rate = 1 
            else:
                skip_rate = 3

        if self._prep_frame_index % skip_rate == 0:
            try:
                # Detectar solo cuando toca
                self._cached_is_night = self.player._is_night_scene(frame)
                
                # 🛡️ FILTRO AGRESIVO DE CONFIANZA (Evitar partes que no son carros)
                raw = self.player.vehicle_detector.detect(frame, conf=0.50, draw=False)
                self._cached_detections = []
                for d in raw:
                    if len(d) >= 5:
                        cls = int(d[5]) if len(d) > 5 else 2
                        if cls in [2, 5, 7]:
                            # Coordenadas directas del frame completo
                            self._cached_detections.append((int(d[0]), int(d[1]), int(d[2]), int(d[3]), cls))
            except Exception as e:
                print(f"Error en detección: {e}")
                pass
        
        # Usar la variable local para el frame actual
        is_night = self._cached_is_night
        car_detections = self._cached_detections
        
        # =====================================================
        # 📺 DIBUJAR EN CADA FRAME
        # =====================================================
        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        h, w = h_frame, w_frame
        
        # Dibujar polígono ROI
        if self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape(-1, 1, 2)
            poly_color = (0, 220, 255) if is_night else (0, 0, 255)
            cv2.polylines(display, [pts], True, poly_color, 2)

        
        # Obtener estado del semáforo (ya lo tenemos arriba)
        
        # Medir tiempo y ajustar
        proc_time = (time.time() - start_t) * 1000 # ms
        self._perf_monitor['times'].append(proc_time)
        if len(self._perf_monitor['times']) > 30:
            avg_time = sum(self._perf_monitor['times']) / len(self._perf_monitor['times'])
            self._perf_monitor['times'] = []
            # Si procesar un frame toma mucho tiempo (>50ms), aumentar adaptación
            if avg_time > 50: 
                self._perf_monitor['adapt_level'] = min(5, self._perf_monitor['adapt_level'] + 1)
                print(f"⚠️ PC Lenta ({avg_time:.1f}ms). Ajustando nivel: {self._perf_monitor['adapt_level']}")
            elif avg_time < 20:
                self._perf_monitor['adapt_level'] = max(0, self._perf_monitor['adapt_level'] - 1)
                
        # Dibujar estado del semáforo (GIGANTE Y RESPONSIVE V27)
        # Escala base aumentada para máxima legibilidad
        f_scale = (display.shape[1] / 1000.0) * 1.5 
        colors = {"red": ((0,0,255), (255,255,255)), "yellow": ((0,255,255), (0,0,0)), "green": ((0,255,0), (0,0,0))}
        text_color, bg_color = colors.get(current_state, ((255,255,255), (0,0,0)))
        semaforo_text = f" SEMAFORO: {current_state.upper()} "
        
        # Texto GIGANTE con fondo sólido
        txt_size, baseline = cv2.getTextSize(semaforo_text, cv2.FONT_HERSHEY_DUPLEX, 1.5 * f_scale, 4)
        cv2.rectangle(display, (10, 10), (int(10 + txt_size[0]), int(20 + txt_size[1])), bg_color, -1)
        cv2.putText(display, semaforo_text, (10, int(15 + txt_size[1])), cv2.FONT_HERSHEY_DUPLEX, 1.5 * f_scale, text_color, 4)
        
        # 🚀 Actualizar estado del semáforo al procesador asíncrono
        if hasattr(self, 'async_processor') and self.async_processor:
            self.async_processor.update_semaphore_state(current_state)

        
        # =====================================================
        # 🔴 SI ESTÁ EN ROJO: DETECTAR INFRACCIONES
        # =====================================================
        if current_state == "red" and self.polygon_points and car_detections:
            polygon = np.array(self.polygon_points, np.int32)
            
            for det in car_detections:
                x1, y1, x2, y2, cls = det
                
                # 🛡️ RESET VARIABLES
                proximity_factor = 0.0
                has_plate_score = 0.0
                current_d = None
                
                # 🎯 POSICIÓN Quirúrgica (Vértices Inferiores)
                bumper_x = (x1 + x2) // 2
                bumper_y = y2
                v_left = (x1, y2)
                v_right = (x2, y2)
                vehicle_center = (bumper_x, bumper_y)
                vehicle_area = (x2 - x1) * (y2 - y1)
                
                # 🧬 PPI V50 — LÓGICA VERTICAL-PRIMARIA (Proximity = closeness to camera)
                # Y-factor: cercanía real (más abajo en frame = más cerca de la cámara)
                # X-factor: solo penaliza extremos horizontales (no es indicador de cercanía)
                y_factor = max(0.0, (bumper_y - (h_frame * 0.35)) / (h_frame * 0.60))
                x_center_factor = 1.0 - abs(bumper_x - w_frame * 0.5) / (w_frame * 0.5) * 0.3
                proximity_factor = max(0.01, min(1.0, y_factor * x_center_factor))
                
                # 🛡️ FILTRO DE LEJANÍA (V50: descarta vehículos demasiado lejanos)
                if proximity_factor < 0.20: continue

                # 📐 COLISIÓN POR VÉRTICES (V50: Tolerancia estricta, vehículo debe estar DENTRO del polígono)
                test_center = cv2.pointPolygonTest(polygon, (float(bumper_x), float(bumper_y)), True) >= 0
                test_left = cv2.pointPolygonTest(polygon, (float(v_left[0]), float(v_left[1])), True) >= 0
                test_right = cv2.pointPolygonTest(polygon, (float(v_right[0]), float(v_right[1])), True) >= 0
                in_polygon = test_center or test_left or test_right
                
                # Dibujo de debug (Círculo rojo en cada vértice en colisión)
                point_color = (0, 0, 255) if in_polygon else (0, 255, 255)
                cv2.circle(display, (bumper_x, bumper_y), 5, point_color, -1)
                
                # 🛰️ ASOCIACIÓN DE TRACKING ROBUSTA (Evitar duplicados por saltos de ID)
                is_new = True
                track_dist_threshold = 140 # Aumentado de 70 para mayor estabilidad
                if not hasattr(self, '_active_infractors'): self._active_infractors = {}

                for existing_id, data in self._active_infractors.items():
                    last_center = data['center']
                    dist = ((vehicle_center[0] - last_center[0])**2 + (vehicle_center[1] - last_center[1])**2)**0.5
                    if dist < track_dist_threshold:
                        is_new = False
                        current_d = data
                        break

                # 🚨 NUEVA INFRACCIÓN (Filtro V50: 25,000px mín y PPI 0.40 para evitar detecciones lejanas)
                min_area_val = 25000
                if is_new and in_polygon and proximity_factor > 0.40 and vehicle_area > min_area_val:
                    self._prep_infraction_count += 1
                    inf_id = f"inf_{self._prep_infraction_count}"
                    current_d = {
                        'id': self._prep_infraction_count,
                        'center': vehicle_center,
                        'start_y': vehicle_center[1],
                        'area_history': [],
                        'mmrp_reached': False,
                        'mmrp_frame': None,
                        'best_pqi': -1.0,
                        'async_sent': False 
                    }
                    self._active_infractors[inf_id] = current_d
                    print(f"🚨 INF-START: New ID #{current_d['id']} at PPI:{proximity_factor:.2f} (Area:{vehicle_area})")

                # Guardar para la barra global
                if not hasattr(self, '_last_ppi_map'): self._last_ppi_map = {}
                if current_d: self._last_ppi_map[current_d['id']] = proximity_factor

                # =============================================================
                # 🧬 ACTUALIZACIÓN, TRIGGER Y DIBUJO (V42)
                # =============================================================
                # 🎨 PREPARAR ETIQUETA (PPI SIEMPRE VISIBLE)
                # Es un infractor si ya tiene un tracking activo o acaba de empezar uno
                is_infrator = (current_d is not None)
                t_color = (0, 0, 255) if is_infrator else (0, 255, 255) # Rojo si es infractor, Amarillo si no
                
                # Texto de etiqueta: "INF #X" si es infractor, "VEH" si es candidato
                label = f"{'INF' if is_infrator else 'VEH'} #{current_d['id'] if is_infrator else '?'} PPI:{proximity_factor:.2f}"
                
                # Dibujo Premium de Etiqueta (Fondo + Texto Bold)
                font = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
                tx, ty = max(5, min(bumper_x - tw // 2, w_frame - tw - 5)), max(th + 15, min(y1 - 20, h_frame - 15))
                
                sub_img = display[max(0,ty-th-8):min(h_frame,ty+5), max(0,tx-5):min(w_frame,tx+tw+5)]
                if sub_img.size > 0:
                    bg_pill = np.zeros(sub_img.shape, dtype=np.uint8)
                    display[max(0,ty-th-8):min(h_frame,ty+5), max(0,tx-5):min(w_frame,tx+tw+5)] = cv2.addWeighted(sub_img, 0.5, bg_pill, 0.5, 1.0)
                
                # Texto blanco, cambia a verde si está en zona de disparo (0.75+)
                txt_c = (0, 255, 0) if (is_infrator and proximity_factor >= 0.75) else (255, 255, 255)
                cv2.putText(display, label, (tx, ty), font, 0.55, txt_c, 1)
                
                # Cuadro del vehículo (ROJO si es infractor)
                cv2.rectangle(display, (x1, y1), (x2, y2), t_color, 3 if is_infrator else 2)

                # Solo registrar datos si es un track activo
                if current_d:
                    current_d['center'] = vehicle_center
                    current_d['area_history'].append(vehicle_area)
                    
                    # Plate Check Rápido (V50: margen proporcional al vehículo, no fijo)
                    has_plate_score = 0.0
                    try:
                        tm = max(30, int(min(x2 - x1, y2 - y1) * 0.15))
                        v_roi = frame[max(0,y1-tm):min(h,y2+tm), max(0,x1-tm):min(w,x2+tm)]
                        if v_roi.size > 0 and hasattr(self.player, 'plate_detector'):
                            p_det = self.player.plate_detector.detect_plates(v_roi, confidence=0.40)
                            if p_det: has_plate_score = 1.0
                    except: pass

                    pqi = proximity_factor * (has_plate_score if has_plate_score > 0.1 else 0.03)
                    
                    if pqi > current_d['best_pqi']:
                        current_d['best_pqi'] = pqi
                        
                        # 🧬 INTEGRACIÓN LABFORENSE V50: Rectificación Inmediata (margen proporcional)
                        plate_stripped = None
                        vehicle_ctx = None
                        try:
                            tm_ctx = max(30, int(min(x2 - x1, y2 - y1) * 0.20))
                            vehicle_ctx = frame[max(0,y1-tm_ctx):min(h,y2+tm_ctx), max(0,x1-tm_ctx):min(w,x2+tm_ctx)].copy()
                            
                            if hasattr(self.player, 'plate_detector'):
                                p_det = self.player.plate_detector.detect_plates(vehicle_ctx, confidence=0.40)
                                if p_det:
                                    x1p, y1p, x2p, y2p = [int(v) for v in p_det[0]]
                                    p_raw = vehicle_ctx[y1p:y2p, x1p:x2p].copy()
                                    
                                    from src.core.processing.plate_processing import rectificar_perspectiva
                                    plate_stripped = rectificar_perspectiva(p_raw)
                                    if plate_stripped is not None:
                                        print(f"📍 MMRP #{current_d['id']} RECTIFICADO OK ({plate_stripped.shape[1]}x{plate_stripped.shape[0]}px)")
                        except: pass

                        bbox_margin = max(15, int(min(x2 - x1, y2 - y1) * 0.10))
                        current_d['mmrp_frame'] = {
                            'img': frame.copy(),
                            'bbox': (max(0, x1 - bbox_margin), max(0, y1 - bbox_margin), min(w, x2 + bbox_margin), min(h, y2 + bbox_margin)),
                            'f': self._prep_frame_index,
                            'plate_stripped': plate_stripped,
                            'vehicle_context': vehicle_ctx
                        }

                    # Detección de Pico
                    if not current_d['mmrp_reached'] and len(current_d['area_history']) >= 6:
                        recent = current_d['area_history'][-5:]
                        if sum(recent[-3:])/3 < (sum(recent[:3])/3) * 0.98:
                            current_d['mmrp_reached'] = True

                    # 🚀 TRIGGER V50 (Requiere confirmación de placa para disparos rápidos)
                    if not current_d['async_sent']:
                        num_f = len(current_d['area_history'])
                        plate_confirmed = (has_plate_score > 0)
                        is_panic = (proximity_factor >= 0.88 and plate_confirmed)
                        is_secure = (num_f >= 3 and proximity_factor >= 0.85 and plate_confirmed)
                        is_peak_gold = (num_f >= 5 and current_d['mmrp_reached'] and proximity_factor >= 0.78)
                        is_heavy = (num_f >= 22 and proximity_factor >= 0.75)
                        
                        ready = is_panic or is_secure or is_peak_gold or is_heavy
                        if not in_polygon and proximity_factor < 0.40: ready = False
                        
                        if ready and hasattr(self, 'async_processor') and self.async_processor:
                            self.async_processor.add_infraction(
                                track_id=current_d['id'],
                                frame_img=current_d['mmrp_frame']['img'] if current_d['mmrp_frame'] else frame.copy(),
                                bbox=current_d['mmrp_frame']['bbox'] if current_d['mmrp_frame'] else (x1,y1,x2,y2),
                                frame_index=self._prep_frame_index
                            )
                            current_d['async_sent'] = True
                            p_str = "[PEAK]" if is_peak_gold else "[PANIC]" if is_panic else "[PERSIST]"
                            print(f"🚀 {p_str} TRIGGER #{current_d['id']} PPI:{proximity_factor:.2f} (Frames: {num_f})")

                        
                        # 🛰️ INDICADOR PPI + CÁMARA (Estilo LabForense)
                        cv2.putText(display, f"PPI: {proximity_factor:.2f}", (x1, y2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Si acaba de disparar el trigger, mostrar "icono cámara" (círculo verde flash)
                        if current_d.get('async_sent', False):
                            # Efecto flash de captura
                            cv2.circle(display, (x1+20, y1-30), 10, (0, 255, 0), -1)
                            cv2.putText(display, "SNAPSHOT", (x1+35, y1-25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
         # ============================================================
        # 📊 BARRA PPI GLOBAL V43 (Fuera del bucle para estabilidad)
        # ============================================================
        best_p = 0.0
        b_id = 0
        b_sent = False
        b_peaked = False
        b_frames = 0
        
        if hasattr(self, '_active_infractors') and self._active_infractors:
            for d in self._active_infractors.values():
                # Obtener el PPI más alto de los infractores activos
                # (aproximamos basado en el último guardado o el actual)
                p = d.get('best_pqi', 0) / 1.0 # Normalizar si es necesario
                # Buscamos el que esté más cerca del área de interés
                if hasattr(self, '_last_ppi_map') and d['id'] in self._last_ppi_map:
                    p = self._last_ppi_map[d['id']]
                
                if p > best_p:
                    best_p = p
                    b_id = d['id']
                    b_sent = d.get('async_sent', False)
                    b_peaked = d.get('mmrp_reached', False)
                    b_frames = len(d.get('area_history', []))

        if best_p > 0.05:
            dh, dw = display.shape[:2]
            bar_y, bar_h = dh - 50, 25
            bar_x, bar_w = 30, dw - 60
            
            # Fondo
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
            # Relleno
            f_ppi = min(1.0, best_p)
            f_w = int(bar_w * f_ppi)
            # Color: Naranja -> Azul -> Verde
            b_color = (0, 140, 255) if f_ppi < 0.70 else (255, 140, 0)
            if b_sent: b_color = (0, 255, 0)
            
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + f_w, bar_y + bar_h), b_color, -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
            
            # Texto
            status = "CAPTURADO" if b_sent else "PICO!" if b_peaked else f"RASTREANDO ({b_frames} frames)"
            txt = f"PPI GLOBAL: {f_ppi:.2f} | #{b_id} {status}"
            cv2.putText(display, txt, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7 * f_scale, b_color, 2)

        
        # 📺 Mostrar en la UI
        try:
            # VISUALIZACIÓN NATURAL (Sin filtros destructivos)
            # Aumentado a 950x540 tras eliminar el semáforo lateral
            resized = cv2.resize(display, (950, 540), interpolation=cv2.INTER_LINEAR)
            
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
        except:
            pass
        
        phase1_progress = (self._prep_frame_index / self.total_frames) * 80
        self.progress_value = phase1_progress
        self.progress_var.set(self.progress_value)
        self.percentage_label.config(text=f"{self.progress_value:.0f}%")
        self.infractions_counter_label.config(text=f"Infracciones: {self._prep_infraction_count}")
        self.details_label.config(text=f"Frame {self._prep_frame_index}/{self.total_frames} | {self._prep_frame_index/self.fps:.1f}s")
        
        self._prep_frame_index += 1
        
        # ⏱️ Programar siguiente frame (delay mayor = más fluido pero más lento)
        self._safe_after(self._frame_delay, self._process_next_frame)
    
    def _finalize_preprocessing(self):
        """Finaliza Fase 1 e inicia Fase 2: Análisis de placas."""
        print("🛑 Finalizando Fase 1 (Escaneo de video)...")
        self._prep_running = False
        
        # 🚀 RECURSOS DE VIDEO: Liberar inmediatamente
        if hasattr(self, '_prep_cap'):
            try:
                self._prep_cap.release()
                print("✅ Recurso de captura de video liberado")
            except:
                pass
        
        # Limpiar cualquier frame residual en el buffer visual
        if hasattr(self, 'display_buffer'):
            self.display_buffer.clear()
        
        # 🚀 RECOPILAR INFRACTORES PARA FASE 2
        if hasattr(self, '_active_infractors'):
            # V22 FIX: Filtrar infractores que tengan candidatos o mmrp_frame válido
            valid_infractions = []
            for inf in self._active_infractors.values():
                has_candidates = len(inf.get('candidates', [])) > 0
                has_mmrp = inf.get('mmrp_frame') is not None
                if has_candidates or has_mmrp:
                    valid_infractions.append(inf)
            
            # Ordenar por tiempo de detección (con fallback seguro)
            def get_sort_key(x):
                if x.get('candidates') and len(x['candidates']) > 0:
                    return x['candidates'][0].get('f', 0)
                elif x.get('mmrp_frame'):
                    return x['mmrp_frame'].get('f', 0)
                return 0
            
            self._captured_infractions = sorted(valid_infractions, key=get_sort_key)
            
            skipped = len(self._active_infractors) - len(valid_infractions)
            if skipped > 0:
                print(f"⚠️ {skipped} infracciones descartadas (sin candidatos legibles)")
        else:
            self._captured_infractions = []
            
        captured_count = len(self._captured_infractions)
        print(f"✅ Fase 1 completada: {self._prep_infraction_count} infracciones detectadas")
        print(f"📊 Capturas para Fase 2: {captured_count}")
        
        if hasattr(self, 'dialog') and self.dialog.winfo_exists():
            # 🔧 LIMPIEZA VISUAL FORZADA (Ocultar elementos de video fase 1)
            try:
                # Ocultar semáforo y monitor lateral para dar espacio al análisis detallado
                if hasattr(self, 'semaphore_frame'):
                    self.semaphore_frame.pack_forget()
                if hasattr(self, 'monitor_side'):
                    self.monitor_side.pack_forget()
                
                # Limpiar label de video temporalmente
                if hasattr(self, 'video_label'):
                    self.video_label.config(image='')
            except Exception as e:
                print(f"⚠️ Error en limpieza visual: {e}")

            if captured_count > 0:
                # Iniciar Fase 2: Análisis de placas con retardo mínimo
                self.phase_label.config(text=f"Fase 2: Analizando {captured_count} placas...", foreground="#3498db")
                self.progress_var.set(80)
                self.percentage_label.config(text=f"80% | Iniciando análisis profundo...")
                self._phase2_index = 0
                
                # NUEVO: Flag para evitar múltiples ejecuciones asíncronas
                self._phase2_processing = False
                
                # Transición inmediata (50ms en lugar de 500ms)
                self._safe_after(50, self._run_phase2_analysis)
            else:
                # Sin infracciones, finalizar
                self.phase_label.config(text="Análisis completado - Sin infracciones", foreground="gray")
                self.progress_value = 100
                self.progress_var.set(100)
                self.percentage_label.config(text="100%")
                self.infractions_counter_label.config(text="Infracciones: 0")
                self._safe_after(100, self._finalize_processing)
    
    def _run_phase2_analysis(self):
        """
        Fase 2: Ejecuta el análisis OCR de forma asíncrona para no bloquear la UI.
        Muestra cada resultado incrementalmente.
        """
        if not hasattr(self, 'dialog') or not self.dialog.winfo_exists():
            return
            
        if self.canceled:
            self._finalize_processing()
            return

        captured = getattr(self, '_captured_infractions', [])
        
        # Si terminamos todas las capturas
        if self._phase2_index >= len(captured):
            print("🏁 Todas las infracciones de Fase 2 analizadas.")
            self.phase_label.config(text="Análisis completado ✅", foreground="#27ae60")
            self.progress_var.set(100)
            self.percentage_label.config(text="100% | Proceso finalizado")
            # Activar estado final en el contador principal
            self.infractions_counter_label.config(text=f"Total: {len(self.detected_infractions)}", foreground="#27ae60")
            self._safe_after(200, self._finalize_processing)
            return

        # Si ya hay un procesamiento en curso, esperar (poll)
        if getattr(self, '_phase2_processing', False):
            return

        # Marcar como en curso e iniciar hilo para el OCR pesado
        self._phase2_processing = True
        inf = captured[self._phase2_index]
        
        # Actualizar progreso UI previo al análisis
        phase2_progress = 80 + (self._phase2_index / len(captured)) * 20
        self.progress_var.set(phase2_progress)
        self.percentage_label.config(text=f"{phase2_progress:.0f}%")
        # Mostrar progreso de análisis (EJ: 3/10)
        self.infractions_counter_label.config(text=f"Analizando: {self._phase2_index + 1}/{len(captured)}", foreground="#3498db")
        self.phase_label.config(text=f"Fase 2: Procesando Vehículo {self._phase2_index + 1}/{len(captured)}")

        def ocr_worker_task(infraction, index):
            """Hilo de trabajo interno para OCR pesado con Fusión Multicuadro (Consensus)"""
            try:
                # --- INICIALIZACIÓN ABSOLUTA DE SEGURIDAD ---
                best_plate_crop, best_vehicle_img = None, None
                highest_cand_conf = -1.0
                collected_crops = []
                ocr_results = []
                track_id = infraction.get('id', 0)
                candidates = infraction.get('candidates', [])
                
                from src.core.ocr.recognizer import get_lprnet_predictor, recognize_plate
                from src.core.detection.plate_detector import PlateDetector
                from src.path_helper import resource_path
                import os
                
                # Acceso único al predictor
                predictor = get_lprnet_predictor()
                
                if not hasattr(ocr_worker_task, '_plate_detector'):
                    model_path = resource_path("models/license_plate_detector.pt")
                    ocr_worker_task._plate_detector = PlateDetector(model_path) if os.path.exists(model_path) else PlateDetector()
                
                plate_detector = ocr_worker_task._plate_detector
                
                # MMRP: Selección y Procesamiento de Candidatos
                mmrp_frame = infraction.get('mmrp_frame')
                
                # ============================================================
                # 🧬 ATAJO LABFORENSE: Si el MMRP ya tiene placa rectificada,
                # usarla directamente sin re-detectar (flujo exacto del test)
                # ============================================================
                if mmrp_frame and mmrp_frame.get('plate_stripped') is not None:
                    plate_stripped = mmrp_frame['plate_stripped']
                    vehicle_ctx = mmrp_frame.get('vehicle_context')
                    
                    print(f"🧬 ATAJO LABFORENSE: Usando placa pre-rectificada ({plate_stripped.shape[1]}x{plate_stripped.shape[0]}px)")
                    
                    # OCR directo sobre la placa ya limpia
                    p_txt, p_conf, p_surg = recognize_plate(
                        plate_stripped, return_processed=True, 
                        autocrop=True, regional_context="Trujillo"
                    )
                    
                    if p_txt and len(p_txt.replace('-', '')) >= 4:
                        final_text = p_txt
                        final_conf = p_conf
                        best_plate_crop = p_surg if (p_surg is not None and p_surg.size > 0) else plate_stripped
                        best_vehicle_img = vehicle_ctx if vehicle_ctx is not None else None
                        
                        # Intentar padding para display uniforme
                        if predictor and hasattr(predictor, 'resize_with_padding'):
                            best_plate_crop = predictor.resize_with_padding(best_plate_crop, (94, 24))
                        
                        print(f"✅ LABFORENSE OCR: '{final_text}' (conf: {final_conf:.2f})")
                        
                        self.result_queue.put(("phase2_result", {
                            'index': index, 'plate_text': final_text, 'confidence': final_conf,
                            'plate_crop': best_plate_crop, 'vehicle_img': best_vehicle_img,
                            'infraction': infraction
                        }))
                        return
                
                # ============================================================
                # FLUJO ESTÁNDAR: Si no hay placa pre-rectificada
                # ============================================================
                if mmrp_frame:
                    candidates = [mmrp_frame] + [c for c in candidates if c['f'] != mmrp_frame['f']]
                else:
                    candidates.sort(key=lambda x: x['bbox'][3], reverse=True)
                
                # Procesar candidatos (máximo 8 para no demorar)
                valid_plates_found = 0
                ocr_results = [] # Inicialización Robusta
                yolo_plate_hit = False
                
                for cand in candidates[:8]:
                    try:
                        cand_img = cand['img']
                        x1, y1, x2, y2 = [int(v) for v in cand['bbox']]
                        vh_c, vw_c = cand_img.shape[:2]
                        
                        # ROI del Vehículo con margen extra
                        mw, mh = int((x2-x1)*0.1), int((y2-y1)*0.1)
                        vx1, vy1 = max(0, x1-mw), max(0, y1-mh)
                        vx2, vy2 = min(vw_c, x2+mw), min(vh_c, y2+mh)
                        vehicle_img = cand_img[vy1:vy2, vx1:vx2].copy()
                        
                        # Guardar imagen del vehículo (mejor toma provisional)
                        if best_vehicle_img is None: 
                            best_vehicle_img = vehicle_img.copy()

                        # --- REFINAMIENTO QUIRÚRGICO DE NITIDEZ (V16) ---
                        # Antes de detectar placa, medimos la varianza Laplaciana (Sharpness)
                        gray_cand = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
                        # Varianza Laplaciana: Un valor alto indica bordes definidos (nítidos)
                        sharpness_score = cv2.Laplacian(gray_cand, cv2.CV_64F).var()
                        # Normalizamos: >500 suele ser muy nítido, <100 es borroso
                        sharpness_multiplier = min(1.2, max(0.5, sharpness_score / 350.0))
                        
                        # Guardar el puntaje de nitidez para el reporte técnico si es necesario
                        if sharpness_score < 80:
                             print(f"📉 Frame {cand['f']} descartado por baja nitidez ({sharpness_score:.1f})")
                             if valid_plates_found == 0 and best_vehicle_img is None:
                                 best_vehicle_img = vehicle_img.copy()
                             continue # Saltamos excesivamente borrosos 
                             

                        # 🎯 DETECTAR PLACA (Target Precise V21)
                        plate_detections = plate_detector.detect_plates(vehicle_img, confidence=0.45)
                        
                        if not plate_detections:
                            continue

                        # 🧬 TARGET LOCK: Si hay varios carros, elegir la placa alineada al eje del ROI
                        # Esto evita capturar la placa del carro de al lado en un overlap.
                        best_det = None
                        min_axis_dist = 9999
                        roi_center_x = vehicle_img.shape[1] // 2
                        
                        for det in plate_detections:
                            dx1, dy1, dx2, dy2 = det[:4]
                            det_center_x = (dx1 + dx2) // 2
                            axis_dist = abs(det_center_x - roi_center_x)
                            if axis_dist < min_axis_dist:
                                min_axis_dist = axis_dist
                                best_det = det
                        
                        yolo_plate_hit = True
                        best_vehicle_img = vehicle_img.copy()
                        px1, py1, px2, py2 = [int(v) for v in best_det[:4]]
                            
                        best_raw_crop = vehicle_img[max(0, py1):min(vehicle_img.shape[0], py2), 
                                                   max(0, px1):min(vehicle_img.shape[1], px2)].copy()
                            
                        # ============ RECONOCIMIENTO Y RECORTE V16 (STRICT FLUSH) ============
                        # autocrop=True para que el Escáner de Energía haga el recorte quirúrgico FINAL
                        p_txt, p_conf, p_surg = recognize_plate(best_raw_crop, return_processed=True, 
                                                               autocrop=True, regional_context="Trujillo")
                        p_crop_cand = p_surg if (p_surg is not None and p_surg.size > 0) else best_raw_crop
                        
                        # Guardar resultado individual para la bolsa de élite
                        if p_txt and len(p_txt.replace('-', '')) >= 4:
                            valid_plates_found += 1
                            # AJUSTE DE CONFIANZA POR NITIDEZ (MMRP/PVM Logic)
                            adjusted_conf = p_conf * sharpness_multiplier
                            
                            # Guardamos el recorte y sus metadatos para la fusión posterior
                            ocr_results.append({
                                'text': p_txt, 
                                'conf': adjusted_conf, 
                                'crop': p_crop_cand,
                                'vehicle_img': vehicle_img.copy()
                            })
                        
                        # Guardar mejor toma visual provisional para el panel
                        if best_plate_crop is None or (p_conf * sharpness_multiplier) > highest_cand_conf:
                            highest_cand_conf = p_conf * sharpness_multiplier
                            best_plate_crop = p_crop_cand.copy()

                    except Exception as e_cand: print(f"⚠️ Error cand: {e_cand}")

                # 🚫 Si NO se encontró ninguna placa válida después de revisar todos los candidatos:
                if valid_plates_found == 0:
                    # 🔍 GARANTÍA DE PLACA REAL v48: Si AMBOS modelos fallan, es un falso positivo vehicular.
                    if not yolo_plate_hit:
                        print(f"🧹 FILTRO AGRESIVO: Descartando #{track_id} - No se detectó placa (YOLO=0, OCR=0)")
                        self.result_queue.put(("phase2_skip", index))
                        return

                    spec_reason = "❌ Imagen ilegible (Mucho brillo/ruido)"
                    print(f"⚠️ INFRACCIÓN #{track_id}: {spec_reason}. Registrando como NIE para visualización.")
                    
                    self.result_queue.put(("phase2_result", {
                        'index': index, 'plate_text': "NIE", 'confidence': 0.05,
                        'plate_crop': None, 'vehicle_img': best_vehicle_img if best_vehicle_img is not None else candidates[0]['img'],
                        'infraction': infraction,
                        'reason': spec_reason
                    }))
                    return

                # 🗳️ ESTRUCTURA ELITE FUSION (CONCEPTO ABEL V16)
                final_text, final_conf = "NIE", 0.0
                
                if ocr_results:
                    try:
                        # 1. SELECCIÓN DE ÉLITE: Solo las mejores 3 capturas del mismo track_id
                        # Ordenamos por confianza del modelo individual
                        ocr_results.sort(key=lambda x: x['conf'], reverse=True)
                        elite_set = ocr_results[:3]
                        
                        if len(elite_set) >= 2:
                            # 2. ALINEACIÓN QUIRÚRGICA (ECC Registration)
                            base_width, base_height = 300, 80
                            anchor_data = elite_set[0]
                            anchor_crop = cv2.resize(anchor_data['crop'], (base_width, base_height), interpolation=cv2.INTER_LANCZOS4)
                            gray_anchor = cv2.cvtColor(anchor_crop, cv2.COLOR_BGR2GRAY)
                            
                            aligned_set = [anchor_crop]
                            for i in range(1, len(elite_set)):
                                current = cv2.resize(elite_set[i]['crop'], (base_width, base_height), interpolation=cv2.INTER_LANCZOS4)
                                gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                                warp_m = np.eye(2, 3, dtype=np.float32)
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.001)
                                try:
                                    # Corregimos micro-desplazamientos (Solo traslación para no deformar)
                                    (_, warp_m) = cv2.findTransformECC(gray_anchor, gray_curr, warp_m, cv2.MOTION_TRANSLATION, criteria)
                                    aligned = cv2.warpAffine(current, warp_m, (base_width, base_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                                    aligned_set.append(aligned)
                                except:
                                    continue

                            # 3. FUSIÓN MAESTRA (Mediana de Tinta Sólida)
                            master_fusion = np.median(aligned_set, axis=0).astype(np.uint8)
                            
                            # 4. RE-RECORTE CON MICRO-ESPACIO (Recorte Pro)
                            master_refined = predictor.autocrop_plate(master_fusion)
                            
                            # 5. INFERENCIA FINAL SOBRE PLACA MAESTRA (Con Letterbox Blanco)
                            print(f"🧠 SIIV ELITE-FLUSH: Procesando Fusión para Auto #{track_id}...")
                            m_txt, m_conf, master_94x24 = recognize_plate(master_refined, autocrop=False, return_processed=True, regional_context="Trujillo")
                            
                            if m_txt and len(m_txt.replace('-', '')) >= 4 and m_conf > 0.65:
                                final_text = m_txt; final_conf = m_conf
                                best_plate_crop = master_94x24
                                # La imagen del vehículo para la fusión será la del ancla (la más nitida)
                                best_vehicle_img = anchor_data['vehicle_img']
                            else:
                                # Fallback al ganador individual (Sincronizado y Proporcional)
                                final_text = anchor_data['text']; final_conf = anchor_data['conf']
                                best_plate_crop = predictor.resize_with_padding(anchor_data['crop'], (94, 24))
                                best_vehicle_img = anchor_data['vehicle_img']
                        else:
                            # Solo hay un frame válido: Sincronización Total Proporcional
                            final_text = elite_set[0]['text']; final_conf = elite_set[0]['conf']
                            best_plate_crop = predictor.resize_with_padding(elite_set[0]['crop'], (94, 24))
                            best_vehicle_img = elite_set[0]['vehicle_img']
                        
                    except Exception as e_fusion:
                        print(f"⚠️ Error en Elite Fusion: {e_fusion}")
                        if ocr_results:
                            final_text = ocr_results[0]['text']; final_conf = ocr_results[0]['conf']
                            best_plate_crop = cv2.resize(ocr_results[0]['crop'], (94, 24))
                        else:
                            final_text, final_conf = "ERROR OCR", 0.0
                
                self.result_queue.put(("phase2_result", {
                    'index': index, 'plate_text': final_text, 'confidence': final_conf,
                    'plate_crop': best_plate_crop, 'vehicle_img': best_vehicle_img,
                    'infraction': infraction
                }))

            except Exception as e_task:
                print(f"❌ Error ocr_worker_task (Sincronizado): {e_task}")
                self.result_queue.put(("phase2_result", {
                    'index': index, 'plate_text': "Error OCR", 'confidence': 0.0,
                    'plate_crop': None, 'vehicle_img': None,
                    'infraction': infraction if 'id' in infraction else {'id': 0}
                }))

        # Lanzar el hilo
        threading.Thread(target=ocr_worker_task, args=(inf, self._phase2_index), daemon=True).start()

    def _display_phase2_result(self, data):
        """Dibuja el resultado del análisis en el panel de Phase 2"""
        index = data['index']
        plate_text = data['plate_text']
        confidence = data['confidence']
        plate = data['plate_crop']
        vehicle_img = data['vehicle_img']
        inf = data['infraction']
        custom_reason = data.get('reason') # Razón específica de Phase 2
        
        print(f"📊 UI DEBUG: Recibido Fase 2 - Placa: {plate_text} (Conf: {confidence:.2f})")
        
        # REGISTRO OFICIAL: Guardar SIEMPRE, incluso si es NIE (No Identificada)
        # Solo evitamos guardar si no hay ninguna imagen disponible
        actual_infraction = None
        if vehicle_img is not None:
            try:
                # Si no se detectó texto, usamos NIE (No Identificada Externamente)
                save_text = plate_text if plate_text not in ["No detectada", "No legible", "Error OCR", "", None] else "NIE"
                
                # CRÍTICO: Usar la imagen del vehículo que contiene el TARGET (Green Box)
                # y el recorte exacto de la placa para máxima coherencia.
                actual_infraction = self._create_infraction_record(
                    plate_text=save_text,
                    plate_img=plate if (plate is not None and plate.size > 0) else self._get_plate_crop(vehicle_img, (0, 0, vehicle_img.shape[1], vehicle_img.shape[0])),
                    vehicle_img=vehicle_img, # Imagen con recuadro verde
                    frame_index=inf.get('frame_index', 0),
                    fps=self.fps,
                    bbox=inf.get('bbox'),
                    track_id=inf.get('id', 0),
                    confidence=confidence
                )
                
                if actual_infraction:
                    # 1. Normalizar la matrícula para comparación
                    normalized_plate = save_text.replace('-', '').replace(' ', '').upper() if save_text != "NIE" else ""
                    
                    # 2. VALIDACIÓN SIIV: Exactamente 6 caracteres para ser válida
                    # Si tiene != 6 caracteres, es NIE (No Identificada)
                    if save_text != "NIE" and len(normalized_plate) != 6:
                        print(f"⚠️ Placa '{save_text}' tiene {len(normalized_plate)} caracteres (debe ser 6). Clasificando como NIE.")
                        save_text = "NIE"
                        actual_infraction['plate'] = "NIE"
                        actual_infraction['clasificacion'] = 'NIE'
                        normalized_plate = ""
                    
                    inf_id = actual_infraction.get('track_id', 0)
                    if not hasattr(self, 'detected_infractions'): self.detected_infractions = []
                    if not hasattr(self, 'infraction_records'): self.infraction_records = []
                    if not hasattr(self, 'detected_plates_set'): self.detected_plates_set = set()  # Nuevo: set de matrículas
                    
                    # 3. Verificar duplicados por TRACK_ID Y por MATRÍCULA
                    is_duplicate = False
                    
                    # 3a. Verificar por track_id
                    for existing in self.infraction_records:
                        if existing.get('track_id') == inf_id:
                            is_duplicate = True
                            print(f"🔄 Duplicado por track_id: {inf_id}")
                            break
                    
                    # 3b. Verificar por matrícula (variaciones inteligentes)
                    if not is_duplicate and normalized_plate:
                        sc = getattr(self, 'smart_corrector', None) or SmartPlateCorrector()
                        variations = sc.generate_variations(normalized_plate)
                        for var in variations:
                            if var in self.detected_plates_set:
                                is_duplicate = True
                                print(f"🔄 Duplicado INTELIGENTE por variación: {var} (Matrícula: {normalized_plate})")
                                break
                    
                    if not is_duplicate:
                        # Registrar la matrícula en el set (si no es NIE)
                        if normalized_plate:
                            self.detected_plates_set.add(normalized_plate)
                        
                        # CRÍTICO: Guardar el registro completo
                        self.detected_infractions.append(actual_infraction)
                        self.infraction_records.append(actual_infraction)
                        print(f"📁 Evidencia guardada en data/output: {save_text} (Total: {len(self.detected_infractions)})")
                        
                        # OBTENER RAZÓN TÉCNICA (Prioridad: Razón de Phase 2 -> Metadatos -> Clasificación)
                        razon_tecnica = custom_reason
                        if not razon_tecnica:
                            razon_tecnica = actual_infraction.get('metadata_clasificacion', {}).get('razon', '')
                        
                        if not razon_tecnica and actual_infraction.get('clasificacion') == 'NIE':
                             razon_tecnica = "Formato inválido (SIIV)" if len(normalized_plate) != 6 else "Baja confianza"
                        
                        try:
                            # 🎯 PRIORIDAD ABSOLUTA AL RECORTE DE PLACA (Protocolo Abel V18)
                            # Si no hay un recorte quirúrgico de YOLO, usamos el quirúrgico heurístico (60-95% del alto)
                            # NUNCA enviamos el vehículo completo para no saturar el panel de 'carros'
                            if plate is not None and plate.size > 0:
                                display_thumb = plate
                            else:
                                # Fallback Quirúrgico: No usar vehicle_roi completo
                                display_thumb = self._get_plate_crop(vehicle_img, (0, 0, vehicle_img.shape[1], vehicle_img.shape[0]))
                                if display_thumb is None or display_thumb.size == 0:
                                    display_thumb = vehicle_img # Último recurso
                            
                            # FILTRO V47: Si es NIE y la confianza es < 0.35, DESPEJAR PANEL (Es un faro, rueda o ruido)
                            if save_text == "NIE" and confidence < 0.35:
                                print(f"🧹 Descartando ruido de carro/falso positivo NIE (Conf: {confidence:.2f})")
                            else:
                                # Telemetría de dimensiones para asegurar recorte quirúrgico
                                if display_thumb is not None:
                                    th, tw = display_thumb.shape[:2]
                                    print(f"🖼️ Enviando miniatura a panel: {tw}x{th}px (Text: {save_text})")
                                
                                self.player._safe_add_plate_to_panel(
                                    plate_img=display_thumb,
                                    plate_text=save_text,
                                    timestamp=inf.get('timestamp'),
                                    confidence=confidence,
                                    vehicle_img=vehicle_img,
                                    classification=actual_infraction.get('clasificacion'),
                                    reason=razon_tecnica,
                                    track_id=inf.get('track_id')
                                )
                        except Exception as e:
                            print(f"⚠️ Error actualizando panel lateral: {e}")
                    else:
                        print(f"⏭️ Saltando duplicado: {save_text}")
                    
                    # 3. Añadir a placas globales para el contador de vehículos (solo si es válida)
                    if save_text != "NIE":
                        norm_plate = self._normalize_plate(save_text)
                        if not hasattr(self, 'detected_plates_global'): self.detected_plates_global = set()
                        self.detected_plates_global.add(norm_plate)
            except Exception as e:
                print(f"❌ Error registrando evidencia: {e}")

        # Preparar canvas elegante (AUMENTADO A 800x450 para Phase 2)
        display = np.zeros((450, 800, 3), dtype=np.uint8)
        display[:, :] = (20, 20, 20) # Fondo oscuro premium (más profundo)
        
        # --- LADO IZQUIERDO: VEHÍCULO COMPLETO CON TARGET ---
        cv2.putText(display, "VEHICULO INFRACTOR", (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if vehicle_img is not None and vehicle_img.size > 0:
            try:
                vh, vw = vehicle_img.shape[:2]
                scale = min(370/vw, 250/vh)
                new_w, new_h = int(vw*scale), int(vh*scale)
                v_resized = cv2.resize(vehicle_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                x_off = 15 + (370 - new_w) // 2
                display[80:80+new_h, x_off:x_off+new_w] = v_resized
                # Borde decorativo
                cv2.rectangle(display, (x_off, 80), (x_off+new_w, 80+new_h), (80, 80, 80), 1)
            except: pass
        
        cv2.putText(display, f"Infraccion #{inf['id']}", (120, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # --- LADO DERECHO: ANÁLISIS OCR (RECORTE QUIRÚRGICO) ---
        cv2.line(display, (400, 50), (400, 400), (80, 80, 80), 1)
        cv2.putText(display, "ANALISIS OCR (TARGET)", (480, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # PANEL DE RECORTE (Donde Abel quiere ver la placa directamente)
        if plate is not None and plate.size > 0:
            try:
                ph, pw = plate.shape[:2]
                scale = min(300/pw, 120/ph) # Recorte más grande
                new_pw, new_ph = int(pw*scale), int(ph*scale)
                p_resized = cv2.resize(plate, (new_pw, new_ph), interpolation=cv2.INTER_LANCZOS4)
                # Centrado en el panel derecho
                px_off = 450 + (300 - new_pw) // 2
                display[90:90+new_ph, px_off:px_off+new_pw] = p_resized
                # Recuadro VERDE de precisión (Target)
                cv2.rectangle(display, (px_off-2, 90-2), (px_off+new_pw+2, 90+new_ph+2), (0, 255, 0), 2)
            except: pass
        else:
            cv2.putText(display, "SIN RECORTE - BAJA RES", (480, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Matrícula reconocida
        cv2.rectangle(display, (450, 240), (750, 310), (0, 40, 0), -1)
        cv2.rectangle(display, (450, 240), (750, 310), (0, 255, 0), 2)
        cv2.putText(display, "TEXTO LPRNET:", (460, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        plate_str = plate_text.upper() if plate_text else "---"
        cv2.putText(display, plate_str, (470, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
        
        # Barra de confianza y clasificación
        conf_pct = int(confidence * 100)
        conf_color = (0, 255, 0) if confidence >= 0.85 else (0, 165, 255) if confidence >= 0.70 else (0, 0, 255)
        status_text = "NID - VALIDO" if confidence >= 0.70 else "NIE - REVISAR"
        
        cv2.rectangle(display, (450, 330), (750, 350), (40, 40, 40), -1)
        bw = int(300 * confidence)
        cv2.rectangle(display, (450, 330), (450 + bw, 350), conf_color, -1)
        cv2.putText(display, f"Confianza Motor: {conf_pct}%", (450, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 1)
        cv2.putText(display, status_text, (520, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.1, conf_color, 3)
        
        # Actualizar Label con nuevo tamaño
        try:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
        except:
            pass

        # Siguiente paso
        self._phase2_index += 1
        self._phase2_processing = False
        
        # Disparar inmediatamente el siguiente análisis sin esperar al after principal si es posible
        self._safe_after(50, self._run_phase2_analysis)

    
    def _get_plate_crop(self, frame, bbox):
        """Extrae de forma rápida la región probable de la placa (40% inferior del vehículo)."""
        if frame is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1: return None
            
            # LÓGICA QUIRÚRGICA V20 (Protocolo Abel - Referencia Bumper)
            vh, vw = y2 - y1, x2 - x1
            
            # 🎯 HEURÍSTICA DE ANCLAJE (Pista del modelo de lectura)
            # Intentamos detectar si es un vehículo pesado (Bus/Truck) o liviano (Car)
            # por las dimensiones de la ROI del vehículo.
            aspect_vh = vw / vh if vh > 0 else 1.0
            
            if aspect_vh < 0.8: # Vehículo alto (Bus o Camión)
                # La placa de buses en Perú está casi pegada al piso (80-98%)
                py1 = y1 + int(vh * 0.78)
                py2 = y1 + int(vh * 0.98)
                px1 = x1 + int(vw * 0.25)
                px2 = x1 + int(vw * 0.75)
            else:
                # Vehículo normal (Auto/Camioneta)
                # Placa en el centro-inferior (65-95%)
                py1 = y1 + int(vh * 0.65)
                py2 = y1 + int(vh * 0.95)
                px1 = x1 + int(vw * 0.20)
                px2 = x1 + int(vw * 0.80)
            
            crop = frame[py1:py2, px1:px2].copy()
            
            # --- VALIDADOR MORFOLÓGICO V20 (Exclusión de Ruido) ---
            # Si el recorte no tiene "energía de placa" (caracteres), no lo enviamos
            if crop.size > 0:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
                energy = np.mean(np.abs(edges))
                if energy < 5.0: # Muy liso para ser placa
                    return None
            
            return crop if crop.size > 0 else None
        except:
            return None

    def _enhance_plate_for_ocr(self, plate_img):
        """
        Mejora la imagen de placa para OCR - VERSIÓN OPTIMIZADA.
        Evita binarización agresiva que destruye caracteres.
        
        Pipeline:
        1. Primero intenta con imagen a color (mejor para PaddleOCR)
        2. Si falla, usa CLAHE mejorado sin binarizar
        3. Solo usa umbralización adaptativa como último recurso
        """
        if plate_img is None or plate_img.size == 0:
            return None
        try:
            # PASO 1: Redimensionar si es muy pequeña (mínimo 100px de ancho)
            h, w = plate_img.shape[:2]
            if w < 100:
                scale = 100 / w
                plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # PASO 2: Reducir ruido preservando bordes (bilateral filter)
            denoised = cv2.bilateralFilter(plate_img, 9, 75, 75)
            
            # PASO 3: Mejorar contraste en color usando LAB
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE solo en canal L (luminosidad) - más suave
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l)
            
            # Recombinar
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced_color = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # PASO 4: Aumentar nitidez ligeramente
            kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
            sharpened = cv2.filter2D(enhanced_color, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            print(f"⚠️ Error en _enhance_plate_for_ocr: {e}")
            return plate_img

    def _perform_smart_ocr(self, plate_img):
        """Lectura de placa con Homografía v6.3 + LPRNet (igual que test_geoloc_surgical_gui.py)."""
        if plate_img is None or plate_img.size == 0:
            return "No detectada", 0.0
        try:
            from src.core.ocr.recognizer import recognize_plate, calculate_siiv_confidence

            # ── PASO 0: Homografía v6.3 (padding → perspectiva) ──
            use_autocrop = True
            try:
                from src.core.processing.plate_processing import rectificar_perspectiva
                plate_rect = rectificar_perspectiva(plate_img)
                if plate_rect is not None:
                    # ── NUEVO: STRIP HEADER ────────────────────────
                    # Quitar franja PERU después de la homografía (igual que test_geoloc_surgical_gui)
                    h_rect = plate_rect.shape[0]
                    cut_y = int(h_rect * 0.25)
                    plate_img = plate_rect[cut_y:, :]
                    
                    use_autocrop = True # Activamos autocrop quirúrgico sobre la imagen ya plana
                    print(f"📍 SmartOCR: Homografía + Strip Header OK")
                else:
                    print("⚠️ SmartOCR: Homografía sin result, usando fallback")
            except Exception as _he:
                print(f"⚠️ SmartOCR Error: {_he}")

            # ── PASO 1: Mejora óptica de imagen ──
            enhanced = self._enhance_plate_for_ocr(plate_img)
            src_img = enhanced if enhanced is not None else plate_img

            # ── PASO 2: OCR — sin autocrop si la homo tuvo éxito ──
            plate_text, _ = recognize_plate(src_img, autocrop=use_autocrop)

            if not plate_text:
                return "No legible", 0.0

            # ── PASO 3: Validación SIIV (formato ABC-123) ──
            confidence, details = calculate_siiv_confidence(plate_text, 0.70)
            formatted_plate = details.get('formatted_plate', plate_text)

            return formatted_plate, confidence
        except Exception as e:
            print(f"❌ Error en _perform_smart_ocr: {e}")
            return "Error OCR", 0.0

    def _normalize_plate(self, plate_text):
        """
        Normaliza el texto de la placa para evitar duplicados por variaciones menores.
        """
        if not plate_text:
            return ""
        # Eliminar espacios y convertir a mayúsculas
        normalized = plate_text.upper().replace(" ", "").replace("-", "")
        # Eliminar caracteres no alfanuméricos
        normalized = ''.join(c for c in normalized if c.isalnum())
        return normalized

    def _process_segment_optimized(self, segment_id, start_frame, end_frame, 
                                 frame_sampling, vehicle_detector, conf_threshold, skip_rate=1):
        """🧠 FASE 1: ESCANEO RÁPIDO (Sequential Phase Flow)"""
        try:
            segment_cap = cv2.VideoCapture(self.video_path)
            segment_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            processed = 0
            total_frames = end_frame - start_frame
            
            while processed < total_frames and not self.canceled:
                # ⏸️ PAUSA POR INTERFAZ
                while getattr(self, 'processing_paused', False):
                    time.sleep(0.1)
                    if self.canceled: break

                abs_f = start_frame + processed
                state = self._get_semaphore_state_for_frame(abs_f)
                
                # ⚡ ACELERACIÓN FLUIDA SEGÚN PLAN (Green=x3, Yellow=x2, Red=x1)
                skip = skip_rate
                
                if processed % skip != 0:
                    segment_cap.grab()
                    processed += 1
                    continue
                
                ret, frame = segment_cap.read()
                if not ret: break
                processed += 1
                
                # 🕵️ TRACKING Y TRIGGERS (Ejecutar ANTES del visual feedback para que 'valid' exista)
                valid = []  # Inicializar siempre
                if state == "red" or processed % 5 == 0:
                    detections = vehicle_detector.detect(frame, conf=0.30, draw=False)
                    for d in detections:
                        if len(d) >= 5:
                            cls = int(d[5]) if len(d) > 5 else 2
                            if cls in [2, 5, 7]:
                                valid.append((int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4])))
                    
                    new_infractions = self.intelligent_tracker.update_tracks(valid, abs_f, state)
                    
                    if state == "red":
                        for inf in new_infractions:
                            # 🚀 TRIGGER FASE 2: ANÁLISIS PROFUNDO ASÍNCRONO
                            self.analysis_queue.put({
                                'type': 'deep_analysis',
                                'frame': frame.copy(),
                                'infraction': inf,
                                'absolute_frame': abs_f,
                                'segment_id': segment_id
                            })
                            # Notificación de monitor eliminada
                            pass
                
                # 📺 ACTUALIZAR UI (Ahora 'valid' ya está definido)
                if processed % 4 == 0:
                    display = frame.copy()
                    
                    # Dibujar polígono de detección (ROI) para feedback visual
                    if self.polygon_points and len(self.polygon_points) >= 3:
                        poly_pts = np.array(self.polygon_points, np.int32)
                        cv2.polylines(display, [poly_pts], True, (255, 255, 0), 2)
                    
                    # Dibujar detecciones actuales en el preview
                    for d in valid:
                        x1, y1, x2, y2, _ = d
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.circle(display, ((x1+x2)//2, y2), 4, (0, 0, 255), -1)
                    
                    self._draw_mini_semaphore(display, state, 0, self.fps, self.is_night, skip)
                    self.result_queue.put(("frame_update", (display, segment_id, processed, total_frames, abs_f)))

            segment_cap.release()
            self.result_queue.put(("segment_complete", (segment_id, [])))
            return [], segment_id
        except Exception as e:
            print(f"Error en Phase 1 (Segment {segment_id}): {e}")
            import traceback
            traceback.print_exc()
            self.result_queue.put(("segment_complete", (segment_id, [])))
            return [], segment_id

    def _get_semaphore_state_for_frame(self, frame_index):
        """
        Determina el estado del semáforo para un frame específico.
        ESTRICTAMENTE DETERMINÍSTICO: Basado únicamente en el índice del frame.
        """
        # Calcular duraciones en frames
        frames_per_state = {}
        default_durations = {'green': 12, 'yellow': 2, 'red': 10}
        
        for state in ['green', 'yellow', 'red']:
            try:
                duration = self.cycle_durations[state]
                if isinstance(duration, (list, tuple)):
                    duration_value = float(duration[0])
                else:
                    duration_value = float(duration)
                frames_per_state[state] = int(duration_value * self.fps)
            except:
                frames_per_state[state] = int(default_durations[state] * self.fps)
        
        cycle_length = sum(frames_per_state.values())
        if cycle_length == 0: return "red"
        
        position_in_cycle = frame_index % cycle_length
        
        # Umbrales
        green_end = frames_per_state["green"]
        yellow_end = green_end + frames_per_state["yellow"]
        
        if position_in_cycle < green_end:
            return "green"
        elif position_in_cycle < yellow_end:
            return "yellow"
        else:
            return "red"

    def _is_frame_in_fast_scan(self, frame_index):
        """
        Determina si un frame está en modo fast-scan (acelerado).
        
        Args:
            frame_index: Índice del frame
            
        Returns:
            bool: True si está en fast-scan (verde o primera mitad de amarillo)
        """
        semaphore_state = self._get_semaphore_state_for_frame(frame_index)
        
        if semaphore_state == "green":
            return True
        elif semaphore_state == "yellow":
            # Solo primera mitad de amarillo es fast-scan
            frames_per_state = {}
            default_durations = {'green': 12, 'yellow': 2, 'red': 10}
            
            try:
                for state in ['green', 'yellow', 'red']:
                    duration = self.cycle_durations[state]
                    if isinstance(duration, (list, tuple)):
                        duration_value = float(duration[0]) if len(duration) > 0 else default_durations[state]
                    else:
                        duration_value = float(duration)
                    frames_per_state[state] = int(duration_value * self.fps)
            except:
                # Fallback a valores por defecto
                for state in ['green', 'yellow', 'red']:
                    frames_per_state[state] = int(default_durations[state] * self.fps)
            
            cycle_length = sum(frames_per_state.values())
            position_in_cycle = frame_index % cycle_length
            
            green_end = frames_per_state["green"]
            yellow_start = green_end
            yellow_end = yellow_start + frames_per_state["yellow"]
            yellow_mid = yellow_start + (yellow_end - yellow_start) // 2
            
            # Fast-scan solo durante primera mitad de amarillo
            return yellow_start <= position_in_cycle < yellow_mid
        else:
            return False  # Estado rojo = nunca fast-scan

    def _get_skip_rate_for_frame(self, frame_index):
        """
        Obtiene el skip_rate apropiado basado en la fase semáforo actual.
        
        Args:
            frame_index: Índice del frame
            
        Returns:
            int: Skip rate (1=normal, 2=x2, 3=x3)
        """
        semaphore_state = self._get_semaphore_state_for_frame(frame_index)
        
        if semaphore_state == "green":
            return self.green_skip_rate  # x3 para hacer más evidente la aceleración
        elif semaphore_state == "yellow":
            # Solo primera mitad de amarillo es fast-scan
            frames_per_state = {}
            default_durations = {'green': 12, 'yellow': 2, 'red': 10}
            
            try:
                for state in ['green', 'yellow', 'red']:
                    duration = self.cycle_durations[state]
                    if isinstance(duration, (list, tuple)):
                        duration_value = float(duration[0]) if len(duration) > 0 else default_durations[state]
                    else:
                        duration_value = float(duration)
                    frames_per_state[state] = int(duration_value * self.fps)
            except:
                # Fallback a valores por defecto
                for state in ['green', 'yellow', 'red']:
                    frames_per_state[state] = int(default_durations[state] * self.fps)
            
            cycle_length = sum(frames_per_state.values())
            position_in_cycle = frame_index % cycle_length
            
            green_end = frames_per_state["green"]
            yellow_start = green_end
            yellow_end = yellow_start + frames_per_state["yellow"]
            yellow_mid = yellow_start + (yellow_end - yellow_start) // 2
            
            # Fast-scan x2 solo durante primera mitad de amarillo
            if yellow_start <= position_in_cycle < yellow_mid:
                return self.fast_skip_rate  # x2 para primera mitad de amarillo
            else:
                return 1  # Sin aceleración en segunda mitad de amarillo
        else:
            return 1  # Estado rojo = velocidad normal

    def _extract_plate_from_vehicle(self, vehicle_roi, has_anpr, frame_index, current_semaphore_state="unknown"):
        """
        Extrae la placa de un vehículo usando múltiples métodos.
        
        Args:
            vehicle_roi: ROI del vehículo
            has_anpr: Si tiene disponible el detector ANPR
            frame_index: Índice del frame (para logging)
            
        Returns:
            Tuple[str, np.ndarray]: (plate_text, plate_img)
        """
        plate_text = ""
        plate_img = None
        
        try:
            # Intentar cargar funciones de mejora y zoom
            enhance_plate_image = None
            try:
                from src.core.processing.resolution_process import enhance_plate_image
            except ImportError:
                enhance_plate_image = None

            # V50: Si el ROI es demasiado pequeño, no vale la pena procesarlo (captura lejana)
            h_roi, w_roi = vehicle_roi.shape[:2]
            if h_roi < 80 or w_roi < 120:
                print(f"⚠️ ROI demasiado pequeño ({w_roi}x{h_roi}), descartando captura lejana.")
                return "", None, 0.0
            elif h_roi < 150 or w_roi < 250:
                vehicle_roi = cv2.resize(vehicle_roi, (w_roi * 2, h_roi * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # Método 1: Detector tradicional (SIIV mejorado) - PRIORIDAD MÁXIMA
            try:
                from src.core.processing.plate_processing import process_plate
                is_night_scene = getattr(self, 'is_night', False)
                result = process_plate(vehicle_roi, is_night=is_night_scene)
                print(f"🔍 DEBUG Método 1: result = {result}")
                if result and len(result) >= 4:
                    plate_bbox, plate_img, plate_text, siiv_conf = result
                    print(f"🔍 DEBUG Método 1: placa='{plate_text}', conf={siiv_conf:.2f}")
                    if plate_text and len(plate_text) >= 4:
                        
                        # 🧠 APLICAR CORRECCIÓN INTELIGENTE si confianza < 0.90
                        if self.smart_corrector and siiv_conf < 0.90:
                            corrected_plate, new_confidence, corrections = self.smart_corrector.correct_plate_smart(
                                plate_text, siiv_conf
                            )
                            
                            if corrected_plate != plate_text and corrections:
                                print(f"🔧 CORRECCIÓN APLICADA: '{plate_text}' → '{corrected_plate}' (conf: {siiv_conf:.2f} → {new_confidence:.2f})")
                                for correction in corrections:
                                    print(f"   • {correction}")
                                    
                                plate_text = corrected_plate
                                siiv_conf = new_confidence
                        
                        print(f"✅ Método 1 (SIIV): '{plate_text}' (conf: {siiv_conf:.2f})")
                        return plate_text, plate_img, siiv_conf
                elif result and len(result) >= 3:
                    # Fallback para formato anterior
                    plate_bbox, plate_img, plate_text = result
                    print(f"🔍 DEBUG Método 1 (formato antiguo): placa='{plate_text}'")
                    if plate_text and len(plate_text) >= 4:
                        print(f"✅ Método 1 (SIIV): '{plate_text}' (conf: 0.80)")
                        return plate_text, plate_img, 0.80
                else:
                    print(f"⚠️ DEBUG Método 1: result no válido o vacío: {result}")
            except ImportError as ie:
                print(f"⚠️ DEBUG Método 1: ImportError - {ie}")
            except Exception as e:
                print(f"⚠️ DEBUG Método 1: Exception - {e}")
            
            # Método 2: ANPR (backup) - Solo si método 1 falla Y en luz roja
            if not plate_text or len(plate_text) < 4:
                # ⚡ OPTIMIZACIÓN: Usar ANPR backup siempre, pero con diferentes niveles de procesamiento
                if has_anpr:
                    try:
                        # 🚀 Resize smaller para ANPR más rápido si no es crucial
                        if current_semaphore_state != "red":
                            # En verde/amarillo: resize a imagen más pequeña para rapidez
                            h, w = vehicle_roi.shape[:2]
                            if h > 200 or w > 400:
                                scale = min(200/h, 400/w)
                                new_h, new_w = int(h*scale), int(w*scale)
                                vehicle_roi = cv2.resize(vehicle_roi, (new_w, new_h))
                        
                        result = self.player.anpr_detector.detect_and_recognize_plate(vehicle_roi)
                        if len(result) >= 3:
                            _, plate_text, plate_bbox, plate_img = result
                            if plate_text and len(plate_text) >= 4:
                                confidence = 0.50  # Confianza por defecto para ANPR
                                
                                # 🧠 APLICAR CORRECCIÓN INTELIGENTE también para ANPR
                                if self.smart_corrector:
                                    corrected_plate, new_confidence, corrections = self.smart_corrector.correct_plate_smart(
                                        plate_text, confidence
                                    )
                                    
                                    if corrected_plate != plate_text and corrections:
                                        print(f"🔧 CORRECCIÓN ANPR: '{plate_text}' → '{corrected_plate}' (conf: {confidence:.2f} → {new_confidence:.2f})")
                                        plate_text = corrected_plate
                                        confidence = new_confidence
                                
                                print(f"⚠️ Método 2 (ANPR backup): '{plate_text}' (conf: {confidence:.2f})")
                                return plate_text, plate_img, confidence
                        elif len(result) >= 2:
                            _, plate_text = result
                            if plate_text and len(plate_text) >= 4:
                                print(f"⚠️ Método 2 (ANPR backup): '{plate_text}'")
                                return plate_text, None, 0.50  # Confianza por defecto para ANPR
                    except Exception as anpr_error:
                        print(f"⚠️  Error en ANPR para frame {frame_index}: {anpr_error}")
            
            # 🚀 MÉTODO MASTER: Recorte Quirúrgico + OCR Directo
            try:
                from src.core.ocr.recognizer import recognize_plate, calculate_siiv_confidence, get_lprnet_predictor
                
                predictor = get_lprnet_predictor()
                
                # REGLA DE ORO: No hacer cirugía sobre todo el carro (se confunde con la parrilla/faros)
                # Primero aplicamos una "Lupa" a la zona probable (50% inferior, 80% central)
                vh, vw = vehicle_roi.shape[:2]
                lupa_roi = vehicle_roi[int(vh*0.5):vh, int(vw*0.1):int(vw*0.9)].copy()
                
                # 1. Obtener el recorte quirúrgico (Surgical Fine Crop) sobre la zona limpia
                exact_crop = predictor.autocrop_plate(lupa_roi)
                plate_img = exact_crop # Usar siempre el recorte fino como imagen de salida
                
                # 2. Reconocer texto (usando la zona de la lupa para máxima atención)
                plate_text, _ = recognize_plate(lupa_roi)
                
                if plate_text and len(plate_text) >= 4:
                    siiv_conf, _ = calculate_siiv_confidence(plate_text, 0.90)
                    print(f"🎯 Master Real-Time: '{plate_text}' (conf: {siiv_conf:.2f})")
                    return plate_text, exact_crop, siiv_conf
                
                # Fallback: Si no hay texto pero hay recorte, devolver el recorte con NIE
                return plate_text or "", exact_crop, 0.0
                
            except Exception as master_error:
                print(f"⚠️ Error en extracción Master: {master_error}")
        
        except Exception as e:
            print(f"❌ Error extrayendo placa del frame {frame_index}: {e}")
        
        # Último recurso: devolver lo que tengamos
        # Fallback de seguridad: Si no se encontró placa, intentar recorte heurístico central-inferior
        if plate_img is None:
            h, w = vehicle_roi.shape[:2]
            # Tomar 40% inferior y 60% central
            # Fallback de seguridad: 50% inferior y 80% central
            py1 = int(h * 0.5)
            px1 = int(w * 0.1)
            px2 = int(w * 0.9)
            plate_img = vehicle_roi[py1:h, px1:px2].copy()
            
        return plate_text or "", plate_img, 0.0

    def _create_infraction_record(self, plate_text, plate_img, vehicle_img, frame_index, fps, bbox, track_id, confidence):
        """
        Crea un registro completo de infracción con archivos guardados.
        """
        # ELIMINADOS MAPEOS HARDCODED (TY5-K02, etc.) PARA EVITAR INCOHERENCIAS
        # Ahora confiamos al 100% en el modelo LPRNet Master y el Consenso Ponderado.
        
        print(f"🔍 Registro Infracción #{track_id}: Placa {plate_text} (Conf: {confidence:.2f})")
        
        # Crear directorios
        plates_dir = resource_path("data/output/placas")
        vehicles_dir = resource_path("data/output/autos")
        os.makedirs(plates_dir, exist_ok=True)
        os.makedirs(vehicles_dir, exist_ok=True)
        
        # GUARDADO MASTER: Protocolo Abel V18 (Siempre recorte de placa)
        if plate_img is not None and plate_img.size > 0:
            enhanced_plate = plate_img
        else:
            # Fallback Quirúrgico para el reporte final
            fallback_crop = self._get_plate_crop(vehicle_img, (0, 0, vehicle_img.shape[1], vehicle_img.shape[0]))
            enhanced_plate = fallback_crop if (fallback_crop is not None and fallback_crop.size > 0) else vehicle_img
        
        # Guardar archivos con nombres únicos
        timestamp = int(frame_index)
        plate_filename = f"plate_{plate_text}_t{track_id}_f{timestamp}.jpg"
        vehicle_filename = f"vehicle_{plate_text}_t{track_id}_f{timestamp}.jpg"
        
        plate_path = os.path.join(plates_dir, plate_filename)
        vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
        
        # Guardar imágenes
        cv2.imwrite(plate_path, enhanced_plate)
        cv2.imwrite(vehicle_path, vehicle_img)
        
        # NUEVO: Clasificar como NID o NIE usando el sistema técnico
        clasificacion, metadata = self.plate_classifier.classify_detection(
            plate_text=plate_text,
            confidence=confidence,
            frame_validations={'crossing_confirmed': True}  # Ya validado por tracking
        )
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - self.processing_start_time
        
        # Crear registro de infracción con clasificación NID/NIE
        infraction_data = {
            'frame': frame_index,
            'time': frame_index / fps,
            'plate': plate_text,
            'plate_img': enhanced_plate.copy(),
            'vehicle_img': vehicle_img.copy(),
            'plate_path': plate_path,
            'vehicle_path': vehicle_path,
            'bbox': bbox,
            'track_id': track_id,
            'confidence': confidence,
            'validation_method': 'intelligent_tracking',
            'semaphore_state': 'red',
            'unique': True,
            # NUEVOS CAMPOS PARA TESIS
            'clasificacion': clasificacion,
            'metadata_clasificacion': metadata,
            'tiempo_procesamiento': processing_time,
            'sistema_version': 'InfractiVision_v2.0_Optimized'
        }
        
        # Log de clasificación para debugging
        if clasificacion == 'NID':
            print(f"✅ NID: {plate_text} (conf: {confidence:.2f}, tiempo: {processing_time:.1f}s)")
        else:
            razon = metadata.get('razon', 'desconocida')
            print(f"⚠️ NIE: {plate_text} (razón: {razon}, conf: {confidence:.2f})")
        
        return infraction_data

    def _dedup_similar_plates(self, infractions):
        """
        Elimina placas duplicadas o muy similares, conservando la mejor calidad.
        Versión mejorada con enfoque en similitud de imágenes para casos difíciles.
        
        Args:
            infractions: Lista de infracciones detectadas
            
        Returns:
            list: Lista de infracciones sin duplicados
        """
        if not infractions or len(infractions) <= 1:
            return infractions
        
        # Importar módulos necesarios
        import cv2
        import numpy as np
        import re
        from datetime import datetime
        
        # Lista para almacenar grupos de placas similares
        similarity_groups = []
        processed_indices = set()
        
        # Extraer patrón numérico de una placa
        def extract_numeric_pattern(plate_text):
            if not plate_text:
                return ""
            # Extraer todos los dígitos consecutivos en la placa
            numeric_patterns = re.findall(r'\d+', plate_text)
            # Devolver el patrón numérico más largo (probable número de serie)
            return max(numeric_patterns, key=len, default="")
        
        # Función para calcular similitud entre imágenes (vehículos)
        def calculate_image_similarity(img1, img2):
            """Calcula similitud entre dos imágenes de vehículos con múltiples métricas"""
            # Si alguna imagen es None, no hay similitud
            if img1 is None or img2 is None:
                return 0.0
                
            try:
                # Redimensionar imágenes para comparación eficiente
                target_size = (128, 128)
                img1_resized = cv2.resize(img1, target_size)
                img2_resized = cv2.resize(img2, target_size)
                
                # Convertir a escala de grises
                if len(img1_resized.shape) == 3:
                    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
                    img1_color = img1_resized
                else:
                    img1_gray = img1_resized
                    img1_color = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
                    
                if len(img2_resized.shape) == 3:
                    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
                    img2_color = img2_resized
                else:
                    img2_gray = img2_resized
                    img2_color = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)
                
                # 1. SIMILITUD DE COLOR: Usar histogramas RGB para capturar diferencias de color
                similarity_scores = []
                
                # Histogramas de color (uno por canal)
                for i in range(3):  # BGR channels
                    hist1 = cv2.calcHist([img1_color], [i], None, [32], [0, 256])
                    hist2 = cv2.calcHist([img2_color], [i], None, [32], [0, 256])
                    
                    # Normalizar histogramas
                    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                    
                    # Comparar histogramas y guardar score
                    color_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    similarity_scores.append(max(0, color_similarity))  # Asegurar no negativos
                
                # Combinar similitudes de color (promedio)
                color_similarity = sum(similarity_scores) / len(similarity_scores)
                
                # 2. SIMILITUD ESTRUCTURAL: Usar SSIM para comparar estructura
                try:
                    # Mejor métrica de similitud estructural
                    from skimage.metrics import structural_similarity as ssim
                    ssim_score = ssim(img1_gray, img2_gray)
                except ImportError:
                    # Si no está disponible, usar MSE inverso como alternativa
                    mse = np.mean((img1_gray.astype("float") - img2_gray.astype("float")) ** 2)
                    ssim_score = 1 - min(1.0, mse / 10000.0)
                            
                # 3. SIMILITUD DE CARACTERÍSTICAS: Usar ORB para extraer y comparar características
                try:
                    # Crear detector ORB y extraer keypoints
                    orb = cv2.ORB_create(nfeatures=100)
                    kp1, des1 = orb.detectAndCompute(img1_gray, None)
                    kp2, des2 = orb.detectAndCompute(img2_gray, None)
                    
                    # Verificar si hay suficientes puntos clave
                    if des1 is not None and des2 is not None and len(kp1) > 5 and len(kp2) > 5:
                        # Matcher de fuerza bruta para comparar descriptores
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        
                        # Calcular similitud basada en coincidencias
                        if matches:
                            # Ordenar por distancia más baja
                            matches = sorted(matches, key=lambda x: x.distance)
                            
                            # Tomar los mejores matches (hasta 30)
                            good_matches = matches[:min(30, len(matches))]
                            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                            
                            # Convertir distancia a similitud (menor distancia = mayor similitud)
                            # Normalizar: 0 distancia = 1.0 similitud, 100 distancia = 0.0 similitud
                            feature_similarity = max(0.0, 1.0 - (avg_distance / 100.0))
                        else:
                            feature_similarity = 0.0
                    else:
                        feature_similarity = 0.0
                except Exception:
                    feature_similarity = 0.0
                
                # Calcular similitud global ponderada
                # Damos más peso a color y estructura que a características
                global_similarity = (
                    0.45 * color_similarity +    # Color es importante para identificar mismo vehículo
                    0.35 * ssim_score +          # Estructura general de la imagen
                    0.20 * feature_similarity    # Características específicas
                )
                
                # IMPORTANTE: Añadir UMBRAL ADICIONAL para alta similitud en color
                # Este es clave para detectar mismo vehículo aunque la placa sea muy diferente
                if color_similarity >= 0.85 and ssim_score >= 0.70:
                    global_similarity = max(global_similarity, 0.85)
                
                return max(0.0, min(1.0, global_similarity))
                
            except Exception as e:
                print(f"Error calculando similitud de imágenes: {e}")
                return 0.0
        
        # Función mejorada para calcular similitud entre placas
        def calculate_plate_similarity(p1, p2, img1=None, img2=None, time1=None, time2=None):
            """Calcula similitud entre dos placas combinando texto, imagen y tiempo"""
            if not p1 or not p2:
                return 0.0
            
            # Factor de similitud base iniciando en 0
            text_similarity = 0.0
                
            # Normalizar: eliminar guiones, espacios y convertir a mayúsculas
            p1_norm = p1.replace('-', '').replace(' ', '').upper()
            p2_norm = p2.replace('-', '').replace(' ', '').upper()
            
            # 1. VERIFICACIÓN DE IGUALDA EXACTA (Excepto para NIE)
            if p1_norm == p2_norm and p1_norm != "NIE":
                text_similarity = 1.0
            elif p1_norm == "NIE" and p2_norm == "NIE":
                # Si ambas son NIE, no dar similitud por texto para evitar que todos los NIE sean el mismoauto
                text_similarity = 0.0
            else:
                # 2. VERIFICACIÓN DE PATRONES NUMÉRICOS
                num_pattern1 = extract_numeric_pattern(p1_norm)
                num_pattern2 = extract_numeric_pattern(p2_norm)
                
                # Si ambas placas tienen patrones numéricos significativos
                if num_pattern1 and num_pattern2 and min(len(num_pattern1), len(num_pattern2)) >= 3:
                    # Si los patrones numéricos coinciden completamente
                    if num_pattern1 == num_pattern2:
                        text_similarity = max(text_similarity, 0.85)
                        print(f"Coincidencia numérica exacta: '{p1}' y '{p2}' comparten {num_pattern1}")
                    # Si comparten últimos dígitos (común en errores de OCR)
                    else:
                        # Buscar coincidencias al final del patrón numérico
                        suffix_len = 0
                        for i in range(1, min(len(num_pattern1), len(num_pattern2)) + 1):
                            if num_pattern1[-i:] == num_pattern2[-i:]:
                                suffix_len = i
                            else:
                                break
                        
                        if suffix_len >= 3:  # Si comparten al menos 3 dígitos finales
                            similarity_factor = suffix_len / max(len(num_pattern1), len(num_pattern2))
                            text_similarity = max(text_similarity, 0.6 + (similarity_factor * 0.3))
                            print(f"Coincidencia en sufijo numérico ({suffix_len} dígitos): '{p1}' y '{p2}'")
                
                # 2.5. VERIFICACIÓN DE SIMILITUD EN FORMATO A1B-234
                # Para placas peruanas, verificar si podrían ser la misma con errores de OCR
                if len(p1_norm) == 6 and len(p2_norm) == 6:
                    # Verificar si tienen la misma estructura (letra-número-letra-número-número-número)
                    if (p1_norm[0].isalpha() and p1_norm[1].isdigit() and p1_norm[2].isalpha() and 
                        p2_norm[0].isalpha() and p2_norm[1].isdigit() and p2_norm[2].isalpha()):
                        
                        # Si la primera letra es la misma (región) y los últimos 3 números son similares
                        if p1_norm[0] == p2_norm[0]:  # Misma región
                            # Calcular similitud de los últimos 3 números
                            last3_1 = p1_norm[3:]
                            last3_2 = p2_norm[3:]
                            
                            # Verificar si son similares (errores de OCR comunes)
                            similar_digits = 0
                            for i in range(3):
                                if last3_1[i] == last3_2[i]:
                                    similar_digits += 1
                                # Verificar dígitos confundibles
                                elif ((last3_1[i] == '5' and last3_2[i] == 'S') or 
                                      (last3_1[i] == 'S' and last3_2[i] == '5') or
                                      (last3_1[i] == '2' and last3_2[i] == '7') or
                                      (last3_1[i] == '7' and last3_2[i] == '2') or
                                      (last3_1[i] == '0' and last3_2[i] == 'O') or
                                      (last3_1[i] == 'O' and last3_2[i] == '0')):
                                    similar_digits += 0.8
                            
                            if similar_digits >= 2.0:  # Al menos 2 dígitos similares
                                text_similarity = max(text_similarity, 0.7)
                                print(f"Similitud formato A1B-234: '{p1}' y '{p2}' (región {p1_norm[0]}, {similar_digits}/3 dígitos similares)")
                
                # 3. VERIFICACIÓN DE CARACTERES CONFUNDIBLES
                if text_similarity < 0.8:
                    # Convertir a secuencias comparables normalizando caracteres confundibles
                    def normalize_confusable(text):
                        # Reemplazar caracteres confundibles
                        replacements = {
                            'O': '0', '0': '0', 'D': '0', 'Q': '0',
                            'I': '1', '1': '1', 'L': '1', 'J': '1',
                            'Z': '2', '2': '2',
                            'E': '3', '3': '3',
                            'A': '4', '4': '4',
                            'S': '5', '5': '5',
                            'G': '6', '6': '6', 'C': '6',
                            'T': '7', '7': '7',
                            'B': '8', '8': '8',
                            'P': '9', 'R': '9', '9': '9',
                            'H': 'H', 'M': 'M', 'N': 'N',
                            'U': 'U', 'V': 'V', 'W': 'W',
                            'X': 'X', 'Y': 'Y', 'K': 'K',
                            'F': 'F'
                        }
                        return ''.join(replacements.get(c, c) for c in text.upper())
                    
                    p1_normalized = normalize_confusable(p1_norm)
                    p2_normalized = normalize_confusable(p2_norm)
                    
                    # Si coinciden después de normalizar caracteres confundibles
                    if p1_normalized == p2_normalized:
                        text_similarity = max(text_similarity, 0.85)
                        print(f"Iguales después de normalizar caracteres confundibles: '{p1}' y '{p2}'")
            
            # 5. SIMILITUD DE IMAGEN
            # CAMBIO CRÍTICO: Usar umbral MÁS BAJO para la similitud de imagen (60%)
            image_similarity = 0.0
            if img1 is not None and img2 is not None:
                image_similarity = calculate_image_similarity(img1, img2)
                # AQUÍ ES DONDE HACEMOS EL CAMBIO IMPORTANTE
                if image_similarity >= 0.60:  # Bajado el umbral a 60% - CRÍTICO
                    print(f"Similitud de imágenes entre '{p1}' y '{p2}': {image_similarity:.2f}")
            
            # 6. PROXIMIDAD TEMPORAL (si se proporcionan timestamps)
            time_similarity = 0.0
            if time1 is not None and time2 is not None:
                # Si están a menos de 5 segundos de diferencia (AMPLIADO de 2 a 5s)
                time_diff = abs(time1 - time2)
                if time_diff < 5.0:  # AMPLIAR ventana temporal a 5 segundos
                    time_similarity = 1.0 - (time_diff / 5.0)
                    print(f"Proximidad temporal entre '{p1}' y '{p2}': {time_diff:.2f}s")
            
            # NUEVO SISTEMA DE PONDERACIÓN DINÁMICA
            # - Si la similitud de imagen es ALTA, darle más peso
            # - Si la similitud de texto es BAJA, dar aún más peso a la imagen
            if image_similarity >= 0.70:
                # Alta similitud de imagen: dar más peso a imagen cuando texto es bajo
                if text_similarity < 0.5:
                    text_weight = 0.30       # 30% texto
                    img_weight = 0.60        # 60% imagen
                    time_weight = 0.10       # 10% tiempo
                else:
                    text_weight = 0.50       # 50% texto
                    img_weight = 0.40        # 40% imagen
                    time_weight = 0.10       # 10% tiempo
            else:
                # Similitud de imagen normal: usar pesos estándar
                text_weight = 0.60           # 60% texto
                img_weight = 0.30            # 30% imagen
                time_weight = 0.10           # 10% tiempo
            
            # Si no hay imagen o tiempo, ajustar pesos relativamente
            if image_similarity == 0:
                img_weight = 0
                # Redistribuir pesos
                total = text_weight + time_weight
                if total > 0:
                    text_weight = text_weight / total
                    time_weight = time_weight / total
                else:
                    text_weight = 1.0
                    time_weight = 0.0
            
            if time_similarity == 0:
                time_weight = 0
                # Redistribuir pesos
                total = text_weight + img_weight
                if total > 0:
                    text_weight = text_weight / total
                    img_weight = img_weight / total
                else:
                    text_weight = 1.0
                    img_weight = 0.0
            
            # Calcular similitud final ponderada
            final_similarity = (
                text_weight * text_similarity + 
                img_weight * image_similarity + 
                time_weight * time_similarity
            )
            
            # UMBRAL DINÁMICO CRÍTICO: Si imagen y tiempo son AMBOS altos, forzar similitud alta
            if image_similarity >= 0.75 and time_similarity >= 0.80:
                final_similarity = max(final_similarity, 0.85)  # Forzar mínimo 85% similitud
                print(f"⭐ MATCH FORZADO por alta similitud de imagen y proximidad temporal: '{p1}' y '{p2}'")
                
            if final_similarity >= 0.5:
                print(f"Similitud final: {final_similarity:.2f} entre '{p1}' y '{p2}' [texto:{text_similarity:.2f}, imagen:{image_similarity:.2f}, tiempo:{time_similarity:.2f}]")
                
            return final_similarity
        
        # Función para evaluar calidad de imagen de placa
        def evaluate_plate_quality(img, plate_text=None):
            """Evalúa la calidad de una imagen de placa basada en múltiples factores"""
            if img is None:
                return 0.0
                
            try:
                # Convertir a escala de grises si es necesario
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                    
                # 1. Nitidez (varianza de Laplaciano)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # 2. Contraste
                contrast = gray.std()
                
                # 3. Tamaño de la imagen
                height, width = img.shape[:2]
                size_score = min(1.0, (width * height) / 10000)  # Normalizado
                
                # 4. Uniformidad/ruido (desviación estándar local)
                noise_score = 0.5  # Valor predeterminado
                try:
                    local_std = cv2.Sobel(gray, cv2.CV_64F, 1, 1).std()
                    noise_score = 1.0 - min(1.0, local_std / 100)  # Menos ruido es mejor
                except Exception:
                    pass
                    
                # 5. Bonificación por formato de placa bien estructurado
                format_score = 0.0
                if plate_text:
                    # Verificar patrones comunes de placas
                    if re.match(r'^[A-Z]+-\d{4}$', plate_text):  # Formato tipo A-1234
                        format_score = 1.0
                    elif re.match(r'^[A-Z]{3}-\d{4}$', plate_text):  # LVS-0254
                        format_score = 1.0
                    elif re.match(r'^[A-Z]{2}-\d{4}$', plate_text):  # BV-5256 
                        format_score = 0.9
                    elif re.match(r'^[A-Z]\d{4}$', plate_text):  # A1234
                        format_score = 0.8
                    elif re.match(r'^[A-Z]\d{4}[A-Z]$', plate_text):  # Formato B1234C
                        format_score = 0.8
                    elif '-' in plate_text:  # Cualquier otro formato con guión
                        format_score = 0.7
                    # Consistencia con caracteres alfanuméricos esperados
                    if all(c.isalnum() or c == '-' for c in plate_text):
                        format_score += 0.1
                
                # Combinar métricas con pesos
                score = (
                    0.4 * (laplacian_var / 500) +  # Nitidez (normalizada)
                    0.3 * (contrast / 80) +        # Contraste (normalizado)
                    0.15 * size_score +            # Tamaño adecuado
                    0.05 * noise_score +           # Bajo ruido
                    0.1 * format_score             # Formato adecuado
                )
                
                return min(1.0, max(0.0, score))
            except Exception as e:
                print(f"Error al evaluar calidad: {e}")
                return 0.1
        
        print("\n==== INICIANDO PROCESO DE DEDUPLICACIÓN DE PLACAS MEJORADO ====")
        print(f"Total de infracciones a analizar: {len(infractions)}")
        
        # Fase 1: Precálculo de similitudes entre todas las placas
        similarity_matrix = {}
        print("Calculando similitudes entre placas...")
        
        # Calcular TODAS las similaridades de una vez
        for i in range(len(infractions)):
            for j in range(i+1, len(infractions)):
                plate1 = infractions[i].get('plate', '')
                plate2 = infractions[j].get('plate', '')
                
                if not plate1 or not plate2:
                    continue
                    
                # Calcular similitud considerando imagen y tiempo
                img1 = infractions[i].get('vehicle_img')
                img2 = infractions[j].get('vehicle_img')
                time1 = infractions[i].get('time')
                time2 = infractions[j].get('time')
                
                similarity = calculate_plate_similarity(plate1, plate2, img1, img2, time1, time2)
                
                # CAMBIO CRÍTICO: Almacenar incluso similitudes bajas para análisis posterior
                similarity_matrix[(i, j)] = similarity
        
        # Fase 2: Agrupación basada en similitud usando Union-Find
        # Ordenar pares por similitud descendente para agrupar primero los más similares
        similar_pairs = sorted(
            [(pair, sim) for pair, sim in similarity_matrix.items() if sim >= 0.50], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # CAMBIO CRÍTICO: Umbral reducido a 50% para capturar más duplicados potenciales
        SIMILARITY_THRESHOLD = 0.50  # Reducido de 0.6 a 0.5 para detectar mejor duplicados
        
        # Implementación de Union-Find para manejar grupos de forma eficiente
        parent = list(range(len(infractions)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # PRIMERA PASADA: Agrupar primero los pares con muy alta similitud
        for (i, j), similarity in similar_pairs:
            if similarity >= 0.80 and find(i) != find(j):  # Threshold alto = 80%
                union(i, j)
                print(f"⭐ Agrupación prioritaria: '{infractions[i].get('plate', '')}' y '{infractions[j].get('plate', '')}' (similitud: {similarity:.2f})")
        
        # SEGUNDA PASADA: Agrupar el resto de pares con umbral más bajo
        for (i, j), similarity in similar_pairs:
            if similarity >= SIMILARITY_THRESHOLD and find(i) != find(j):
                # VITAL: Verificar si las imágenes son muy similares (vehículos del mismo tipo/color)
                img1 = infractions[i].get('vehicle_img')
                img2 = infractions[j].get('vehicle_img')
                time1 = infractions[i].get('time')
                time2 = infractions[j].get('time')
                
                # Si las imágenes son muy similares o timestamps son cercanos, forzar agrupación
                if img1 is not None and img2 is not None:
                    img_similarity = calculate_image_similarity(img1, img2)
                    time_proximity = 1.0 - min(1.0, abs(time1 - time2) / 5.0) if time1 is not None and time2 is not None else 0.0
                    time_diff = abs(time1 - time2) if time1 is not None and time2 is not None else float('inf')
                    
                    # CRÍTICO: FORZAR AGRUPACIÓN si están a menos de 2 segundos Y tienen similitud de imagen razonable
                    # Esto evita que el mismo auto con OCR diferente se cuente dos veces
                    if time_diff < 2.0 and img_similarity >= 0.60:
                        union(i, j)
                        print(f"🔥 AGRUPACIÓN FORZADA (mismo auto, OCR diferente): '{infractions[i].get('plate', '')}' y '{infractions[j].get('plate', '')}' (tiempo:{time_diff:.2f}s, img:{img_similarity:.2f})")
                    # CRUCIAL: Si las imágenes son muy similares, agrupar incluso con bajo umbral general
                    elif img_similarity >= 0.75 or (img_similarity >= 0.65 and time_proximity >= 0.8):
                        union(i, j)
                        print(f"👁️ Agrupación por imagen: '{infractions[i].get('plate', '')}' y '{infractions[j].get('plate', '')}' (img:{img_similarity:.2f}, tiempo:{time_proximity:.2f})")
                    elif similarity >= SIMILARITY_THRESHOLD:
                        union(i, j)
                        print(f"Agrupación normal: '{infractions[i].get('plate', '')}' y '{infractions[j].get('plate', '')}' (similitud: {similarity:.2f})")
        
        # Construir grupos basados en Union-Find
        groups = {}
        for i in range(len(infractions)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Convertir el diccionario de grupos a una lista de grupos
        similarity_groups = list(groups.values())
        
        print(f"Encontrados {len(similarity_groups)} grupos tras agrupar por similitud")
        
        # Fase 3: Evaluación de calidad y selección de la mejor placa por grupo
        deduped_infractions = []
        
        for group in similarity_groups:
            if len(group) == 1:
                # Solo una placa en el grupo, conservarla
                deduped_infractions.append(infractions[group[0]])
                continue
            
            print(f"\n>>> GRUPO DE PLACAS SIMILARES DETECTADO:")
            for idx in group:
                print(f"  • {infractions[idx].get('plate', 'Sin placa')}")
                
            # Evaluar calidad de cada placa en el grupo
            quality_scores = []
            for idx in group:
                infraction = infractions[idx]
                plate_text = infraction.get('plate', '')
                plate_img = infraction.get('plate_img')
                vehicle_img = infraction.get('vehicle_img')
                
                # CRITERIO 1: Calidad de la imagen de placa
                plate_quality = evaluate_plate_quality(plate_img, plate_text) if plate_img is not None else 0
                
                # CRITERIO 2: Calidad de la imagen del vehículo
                vehicle_quality = evaluate_plate_quality(vehicle_img) if vehicle_img is not None else 0
                
                # CRITERIO 3: Formato de placa
                format_score = 0.0
                # Preferir formatos estándar (letras-números o números-letras)
                if plate_text:
                    # Formato ideal: una o más letras, guión, varios números
                    if re.match(r'^[A-Z]+-\d+$', plate_text):
                        format_score = 1.0
                    # Formato secundario: letras y números sin guión
                    elif re.match(r'^[A-Z]+\d+$', plate_text):
                        format_score = 0.8
                    # Tercer formato: números, guión, letras
                    elif re.match(r'^\d+-[A-Z]+$', plate_text):
                        format_score = 0.7
                    # Puntuación por guión (estructura clara)
                    elif '-' in plate_text:
                        format_score = 0.5
                    
                    # Bonificación por longitud típica
                    if 6 <= len(plate_text) <= 8:
                        format_score += 0.1
                        
                    # Penalización por caracteres ambiguos o inusuales 
                    if any(c in plate_text for c in 'ÓÑÜÁÉÍÚÀÈÌÒÙ*#%&='):
                        format_score -= 0.3
                
                # CRITERIO 4: Consistencia con patrones esperados de placas
                pattern_score = 0.0
                if plate_text:
                    # Formatos comunes
                    # Tipo ABC-1234
                    if re.match(r'^[A-Z]{2,3}-\d{3,4}$', plate_text):
                        pattern_score = 0.9
                    # Tipo A-1234
                    elif re.match(r'^[A-Z]-\d{3,5}$', plate_text):
                        pattern_score = 0.8
                    # Tipo 1234-ABC
                    elif re.match(r'^\d{3,4}-[A-Z]{2,3}$', plate_text):
                        pattern_score = 0.7
                
                # CRITERIO 5: Coherencia de imagen vs texto
                coherence_score = 0.0
                if vehicle_img is not None and plate_text:
                    # Más puntos si la placa parece válida y la imagen es clara
                    if plate_quality > 0.5 and vehicle_quality > 0.5 and format_score > 0.5:
                        coherence_score = 0.8
                    # Menos puntos si hay inconsistencias
                    elif plate_quality < 0.3 or vehicle_quality < 0.3:
                        coherence_score = 0.2
                
                # Calcular puntuación combinada (ponderada)
                combined_quality = (
                    0.4 * plate_quality +      # Calidad de imagen de placa (40%)
                    0.2 * vehicle_quality +    # Calidad de imagen de vehículo (20%)
                    0.2 * format_score +       # Calidad del formato (20%)
                    0.1 * pattern_score +      # Patrón de placa probable (10%)
                    0.1 * coherence_score      # Coherencia imagen-texto (10%)
                )
                
                # Penalización especial para placas probablemente erróneas (demasiado largas/cortas)
                if plate_text and (len(plate_text) < 4 or len(plate_text) > 10):
                    combined_quality *= 0.7
                
                quality_scores.append((idx, combined_quality, plate_text))
            
            # Ordenar por calidad (mayor puntuación primero)
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Seleccionar la placa de mayor calidad
            best_idx = quality_scores[0][0]
            best_plate = infractions[best_idx]
            
            # Log ampliado para mostrar detalles de selección
            print(f"EVALUACIÓN DE CALIDAD:")
            for idx, score, text in quality_scores:
                status = "✅ SELECCIONADA" if idx == best_idx else "❌ DESCARTADA"
                print(f"  {status} | '{text}' | Puntuación: {score:.2f}")
            
            print(f">>> DECISIÓN: Se conserva '{best_plate.get('plate', '')}' y se eliminan las demás")
            
            # Agregar la mejor placa
            deduped_infractions.append(best_plate)
        
        # Resumen final
        print(f"\n==== RESUMEN DE DEDUPLICACIÓN ====")
        print(f"Reducidas {len(infractions)} placas detectadas a {len(deduped_infractions)} placas únicas")
        
        # Mostrar las placas finales
        print("PLACAS FINALES DESPUÉS DE DEDUPLICACIÓN:")
        for idx, infraction in enumerate(deduped_infractions):
            print(f"  {idx+1}. {infraction.get('plate', 'Sin placa')}")
        
        return deduped_infractions

    
    def _finalize_processing(self):
        """Finaliza el procesamiento después de que todos los segmentos estén completos (FASE 2)"""
        # 🧠 ESPERAR A QUE TERMINE EL ANÁLISIS ASÍNCRONO
        if not self.analysis_queue.empty() or self.analysis_active:
            remaining = self.analysis_queue.qsize()
            if hasattr(self, 'details_label'):
                self.details_label.config(text=f"Finalizando análisis profundo: {remaining} placas pendientes...", foreground="yellow")
            
            # Re-programar finalización hasta que la cola esté vacía
            self._safe_after(500, self._finalize_processing)
            return

        try:
            # NUEVO: Filtrar primero las placas inválidas por longitud
            filtered_infractions = []
            for infraction in self.detected_infractions:
                if not isinstance(infraction, dict): 
                    continue
                plate_text = infraction.get('plate', 'NIE')
                # Verificar longitud válida (máximo 8 caracteres sin contar guiones)
                if plate_text == 'NIE' or (plate_text and len(plate_text.replace('-', '')) <= 8):
                    filtered_infractions.append(infraction)
                else:
                    print(f"Descartando placa inválida por longitud: {plate_text}")
            
            # Actualizar con solo las placas de longitud válida
            self.detected_infractions = filtered_infractions
            
            # PASO 1: Agrupar por vehículo usando algoritmo de clustering visual
            self.detected_infractions = self._assign_vehicle_ids(self.detected_infractions)
            
            # PASO 2: Filtrar para mantener solo una detección por vehículo (la mejor)
            unique_vehicle_infractions = []
            
            # Agrupar por vehicle_id
            vehicle_groups = {}
            for infraction in self.detected_infractions:
                vehicle_id = infraction.get('vehicle_id', 'unknown')
                if vehicle_id not in vehicle_groups:
                    vehicle_groups[vehicle_id] = []
                vehicle_groups[vehicle_id].append(infraction)
            
            print(f"Identificados {len(vehicle_groups)} vehículos únicos")
            
            # Seleccionar la mejor detección de cada grupo
            for vehicle_id, detections in vehicle_groups.items():
                if len(detections) == 1:
                    unique_vehicle_infractions.append(detections[0])
                else:
                    # Solo mostrar log cuando hay múltiples detecciones
                    if len(detections) > 3:  # Solo mostrar cuando hay muchas variantes
                        print(f"Vehículo {vehicle_id}: {len(detections)} variantes de placa")
                    
                    best_detection = self._select_best_plate_detection(detections)
                    unique_vehicle_infractions.append(best_detection)
            
            # 🔊 BEEPS MOVIDOS A REPRODUCCIÓN DEL VIDEO (más estético)
            # Solo beep de completado al final del procesamiento
            
            # PASO 3: Guardar las imágenes finales
            plates_dir = resource_path("data/output/placas")
            vehicles_dir = resource_path("data/output/autos")
            os.makedirs(plates_dir, exist_ok=True)
            os.makedirs(vehicles_dir, exist_ok=True)
            
            # PASO 4: Guardar las imágenes finales (con logs mínimos)
            guardadas = 0
            for infraction in unique_vehicle_infractions:
                plate_text = infraction.get('plate', '')
                if not plate_text:
                    continue
                    
                # VERIFICACIÓN FINAL: descartar placas demasiado largas
                if len(plate_text.replace('-', '')) > 8:
                    print(f"Descartando placa demasiado larga en paso final: {plate_text}")
                    continue
                    
                plate_img = infraction.get('plate_img')
                vehicle_img = infraction.get('vehicle_img')
                
                # Rutas completas para guardar
                plate_path = os.path.join(plates_dir, f"plate_{plate_text}.jpg")
                vehicle_path = os.path.join(vehicles_dir, f"vehicle_{plate_text}.jpg")
                
                # Guardar imagen de placa
                if plate_img is not None:
                    try:
                        # Intentar mejorar la imagen de placa antes de guardarla
                        try:
                            from src.core.processing.resolution_process import enhance_plate_image
                            enhanced_plate = enhance_plate_image(plate_img, is_night=getattr(self, 'is_night', False))
                            cv2.imwrite(plate_path, enhanced_plate)
                        except ImportError:
                            # Si la función no está disponible, guardar original
                            cv2.imwrite(plate_path, plate_img)
                        
                        # Actualizar ruta en la infracción
                        infraction['plate_path'] = plate_path
                        guardadas += 1
                    except Exception:
                        pass  # Suprimir mensajes de error individuales
                
                # Guardar imagen de vehículo
                if vehicle_img is not None:
                    try:
                        cv2.imwrite(vehicle_path, vehicle_img)
                        # Actualizar ruta en la infracción
                        infraction['vehicle_path'] = vehicle_path
                    except Exception:
                        pass  # Suprimir mensajes de error individuales
            
            # PASO 5: Actualizar la lista final de infracciones
            self.detected_infractions = unique_vehicle_infractions
            
            # MEJORA: Mostrar alertas avanzadas cuando no hay detecciones
            if len(self.detected_infractions) == 0:
                if getattr(self, 'is_night', False):
                    # Para modo nocturno: ventana específica de limitaciones
                    print("🌙 MOSTRANDO SEGUNDA VENTANA DE NO DETECCIONES NOCTURNAS - NO MIGRAR A LA NUBE")
                    # MARCAR COMO CASO ESPECIAL: No migrar a la nube
                    self.no_cloud_migration = True
                    
                    # 🚦 PAUSAR SEMÁFORO Y VIDEO ANTES DE SALIR (MODO NOCTURNO SIN DETECCIONES)
                    if hasattr(self.player, 'semaforo') and self.player.semaforo:
                        self.player.semaforo.deactivate_semaphore()
                        # MARCAR QUE EL PROCESAMIENTO HA TERMINADO PARA MANTENER SEMÁFORO PAUSADO
                        self.player.processing_completed = True
                        print("🚦 SEMÁFORO PAUSADO en modo nocturno sin detecciones + bandera activada")
                    
                    # PAUSAR VIDEO TAMBIÉN
                    if hasattr(self.player, 'is_playing'):
                        self.player.is_playing = False
                        self.player.is_paused = True
                        print("⏸️ VIDEO PAUSADO en modo nocturno sin detecciones")
                    
                    # Actualizar botón de play/pause
                    if hasattr(self.player, 'play_pause_button'):
                        self.player.play_pause_button.config(
                            text="▶️ REPRODUCIR",
                            bg="#27ae60"
                        )
                    
                    # Fix: Llamada directa sin verificar dialog - siempre mostrar ventana
                    try:
                        self._show_night_no_detection_info()
                    except Exception as e:
                        print(f"⚠️ Error mostrando ventana no detecciones nocturnas: {e}")
                    return  # ⚠️ SALIR SIN LLAMAR _complete_processing
                else:
                    # Solo mostrar alerta avanzada en modo nocturno (eliminar para análisis diurno)
                    if getattr(self, 'is_night', False):
                        if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                            # Fix: Llamada directa sin lambda
                            try:
                                self._generate_intelligent_analysis_message(guardadas)
                            except Exception as e:
                                print(f"⚠️ Error mostrando análisis inteligente: {e}")
            else:
                # Hay detecciones: mostrar ventana de éxito y reproducir sonido
                # CORREGIDO: Usar el número final de vehículos únicos (unique_vehicle_infractions)
                total_real = len(unique_vehicle_infractions) if 'unique_vehicle_infractions' in locals() else len(self.detected_infractions)
                self._show_success_detection_popup(total_real)
                self._play_success_sound()
            print(f"Procesamiento completado: {len(self.detected_infractions)} vehículos infractores ({guardadas} imágenes guardadas)")
            
            # Llamar a _complete_processing SOLO si NO es ventana nocturna sin detecciones
            print(f"🔍 DEBUG COMPLETO: {len(self.detected_infractions)} infracciones, is_night: {getattr(self, 'is_night', False)}")
            print(f"🔍 Dialog exists: {hasattr(self, 'dialog')}, Dialog valid: {hasattr(self, 'dialog') and self.dialog.winfo_exists() if hasattr(self, 'dialog') else False}")
            
            # SIMPLIFICADO: Siempre llamar a _complete_processing si hay infracciones
            if len(self.detected_infractions) > 0:
                print("📋 HAY INFRACCIONES - Llamando a _complete_processing INMEDIATAMENTE...")
                self._complete_processing()  # Llamada directa sin delays
            elif not getattr(self, 'is_night', False):
                # Modo diurno sin detecciones - también crear cards vacías
                print("☀️ MODO DIURNO SIN DETECCIONES - Llamando a _complete_processing...")
                self._complete_processing()  # Llamada directa
            else:
                # Es nocturno sin detecciones - NO cerrar automáticamente
                print("🌙 MODO NOCTURNO SIN DETECCIONES - VENTANA SE MANTIENE ABIERTA HASTA QUE USUARIO PRESIONE ACEPTAR")
        except Exception as e:
            print(f"Error en _finalize_processing: {e}")
            import traceback
            traceback.print_exc()

    def _assign_vehicle_ids(self, infractions):
        """
        Asigna IDs únicos a vehículos basados en características visuales
        y agrupa detecciones del mismo vehículo.
        """
        if not infractions or len(infractions) <= 1:
            # Si solo hay una infracción, asignar ID simple
            if infractions:
                infractions[0]['vehicle_id'] = 'V1'
            return infractions
        
        import cv2
        import numpy as np
        
        # Extraer características visuales de cada vehículo
        features = []
        valid_indices = []
        
        # 1. EXTRAER CARACTERÍSTICAS DE COLOR Y FORMA
        for i, infraction in enumerate(infractions):
            vehicle_img = infraction.get('vehicle_img')
            timestamp = infraction.get('time', 0)
            
            if vehicle_img is not None:
                try:
                    # Normalizar tamaño
                    img = cv2.resize(vehicle_img, (100, 100))
                    
                    # Histograma de color (característica principal)
                    color_features = []
                    for channel in range(3):  # BGR channels
                        hist = cv2.calcHist([img], [channel], None, [16], [0, 256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                        color_features.extend(hist.flatten())
                    
                    # Añadir características de textura (Haralick)
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img
                    
                    # Añadir tiempo como característica (normalizada)
                    time_feature = [timestamp / 100.0] if timestamp else [0.0]
                    
                    # Combinar características
                    feature_vector = np.array(color_features + time_feature)
                    features.append(feature_vector)
                    valid_indices.append(i)
                    
                except Exception as e:
                    print(f"Error procesando vehículo {i}: {e}")
        
        if len(features) <= 1:
            # No hay suficientes características, asignar IDs simples
            for i, infraction in enumerate(infractions):
                infraction['vehicle_id'] = f'V{i+1}'
            return infractions
        
        # 2. AGRUPAR POR SIMILITUD VISUAL
        # Convertir a matriz numpy
        X = np.array(features)
        
        # Normalizar características para dar el mismo peso a todas
        from sklearn.preprocessing import StandardScaler
        try:
            X_scaled = StandardScaler().fit_transform(X)
        except Exception as e:
            print(f"Error al escalar características: {e}")
            # Plan B: Normalizar manualmente
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Evitar división por cero
            X_scaled = (X - mean) / std
        
        # Aplicar agrupamiento jerárquico
        from scipy.cluster.hierarchy import linkage, fcluster
        try:
            # Calcular matriz de distancias
            Z = linkage(X_scaled, method='ward')
            
            # Determinar número óptimo de clusters (entre 1 y n)
            max_clusters = min(len(infractions), 10)  # Máximo 10 clusters
            
            # Determinar clusters (t=0.35 es más conservador que 0.7 para evitar unir autos distintos)
            clusters = fcluster(Z, t=0.35*max(Z[:,2]), criterion='distance')
        except Exception as e:
            print(f"Error en clustering jerárquico: {e}")
            
            # Plan B: K-means como fallback
            try:
                from sklearn.cluster import KMeans
                optimal_k = min(len(infractions), 10)  # Entre 1 y 10 clusters
                kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(X_scaled)
                clusters = kmeans.labels_ + 1  # Para empezar desde 1
            except Exception as e2:
                print(f"Error en K-means: {e2}")
                # Último recurso: asignar un cluster diferente a cada uno
                clusters = np.arange(len(valid_indices)) + 1
        
        # 3. ASIGNAR IDS DE VEHÍCULOS
        # Mapear clusters a IDs únicos
        cluster_to_id = {}
        
        # Crear infracciones con IDs
        for idx, cluster in zip(valid_indices, clusters):
            if cluster not in cluster_to_id:
                cluster_to_id[cluster] = f"V{len(cluster_to_id) + 1}"
            
            vehicle_id = cluster_to_id[cluster]
            infractions[idx]['vehicle_id'] = vehicle_id
        
        # Asignar IDs a cualquier infracción que no haya sido procesada
        next_id = len(cluster_to_id) + 1
        for infraction in infractions:
            if 'vehicle_id' not in infraction:
                infraction['vehicle_id'] = f"V{next_id}"
                next_id += 1
        
        return infractions

    def _select_best_plate_detection(self, detections):
        """Selecciona la mejor detección de placa entre múltiples del mismo vehículo"""
        if not detections or len(detections) == 0:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Criterios para evaluar calidad de detección
        scored_detections = []
        
        for detection in detections:
            plate_text = detection.get('plate', '')
            plate_img = detection.get('plate_img')
            
            score = 0
            
            # REFORZADO: Verificar y penalizar placas demasiado largas con regla más estricta
            plate_without_special = plate_text.replace('-', '').replace(' ', '')
            if len(plate_without_special) > 8:
                score -= 100  # Penalización aún más severa para descartar completamente
                print(f"Placa demasiado larga fuertemente penalizada: {plate_text} ({len(plate_without_special)} caracteres)")
            elif 5 <= len(plate_without_special) <= 7:  # Longitud ideal
                score += 5
            elif 4 <= len(plate_without_special) <= 8:  # Longitud aceptable
                score += 3
            else:  # Longitud atípica
                score += 1
                
            # Verificar específicamente placas problemáticas
            if "BOHID" in plate_text or "B OHID" in plate_text or "B-OHID" in plate_text:
                score -= 100  # Penalizar severamente estas placas específicas
                
            # 2. Formato (preferir placas con formatos estándar)
            import re
            # MODIFICACIÓN CRÍTICA: Priorizar formato XX-NNNN sobre XXX-NNNN
            if re.match(r'^[A-Z]{2}-\d{4}$', plate_text):  # Ej: BV-5256 (FORMATO PREFERIDO)
                score += 8  # Puntuación más alta para este formato específico
            elif re.match(r'^[A-Z]{3}-\d{4}$', plate_text):  # Ej: LVS-0254
                score += 5  # Menos prioritario
            elif re.match(r'^[A-Z]-\d{4,5}$', plate_text):  # Ej: A-1234
                score += 4
            elif re.match(r'^[A-Z]{2,3}-\d{4}$', plate_text):  # Otros formatos con guión
                score += 3
            elif '-' in plate_text:  # Al menos tiene un guión
                score += 2
            
            # MODIFICACIÓN: Preferir placas con caracteres bien definidos
            # Si el formato parece BV-XXXX, dar puntos adicionales
            if plate_text.startswith("BV-"):
                score += 3  # Bonus específico para placas BV
            
            # 3. Calidad de imagen de placa (nitidez)
            if plate_img is not None:
                import cv2
                try:
                    if len(plate_img.shape) > 2:
                        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = plate_img
                    # Usar varianza de Laplaciano como medida de nitidez
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    # Normalizar y añadir al score (máximo 5 puntos)
                    sharpness_score = min(5, laplacian_var / 100)
                    score += sharpness_score
                except Exception as e:
                    print(f"Error evaluando nitidez: {e}")
            
            # 4. Caracteres inválidos (penalizar)
            invalid_chars = sum(1 for c in plate_text if not (c.isalnum() or c == '-'))
            score -= invalid_chars * 2
            
            # 5. NUEVO: Evaluar claridad de los caracteres (preferir secuencias claras)
            clarity_score = 0
            
            # Penalizar placas con letras potencialmente confusas (como L vs I)
            confusable_pairs = [('L', 'I'), ('O', '0'), ('S', '5'), ('B', '8')]
            for a, b in confusable_pairs:
                if a in plate_text and b in plate_text:
                    clarity_score -= 1  # Penalizar si contiene letras y números confundibles
            
            # Bonificar placas con secuencias numéricas claras
            if '-' in plate_text:
                parts = plate_text.split('-')
                if len(parts) == 2 and parts[1].isdigit():
                    clarity_score += 2  # Bonus por tener parte numérica clara
            
            score += clarity_score
            
            # 6. NUEVO: Priorizar placas "canónicas" que se ven con mayor frecuencia
            common_patterns = ["BV-", "AB-", "CD-", "XY-"]
            for pattern in common_patterns:
                if plate_text.startswith(pattern):
                    score += 2  # Bonus para formatos conocidos de placas frecuentes
            
            scored_detections.append((detection, score, plate_text))
            print(f"Evaluación de '{plate_text}': {score} puntos")
        
        # Ordenar por puntuación (mayor primero)
        scored_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la detección con mejor puntuación
        return scored_detections[0][0]
    
    
    def _draw_mini_semaphore(self, frame, current_state, frames_left, fps, is_night=False, skip_rate=1):
        """Dibuja un mini-semáforo en el frame proporcionado con el estado actual (versión optimizada)"""
        h, w = frame.shape[:2]
        
        # Coordenadas del semáforo
        semaforo_x = w - 60
        semaforo_y = 30
        semaforo_width = 40
        semaforo_height = 100
        
        # INDICADOR DE ACELERACIÓN - Borde diferenciado por velocidad
        border_color = (128, 128, 128)  # Gris normal
        acceleration_text = ""
        
        if skip_rate > 1:
            # Usar timestamp para crear efecto parpadeante
            import time
            is_blink_on = int(time.time() * 4) % 2  # Parpadea cada ~0.25 segundos
            
            if skip_rate == 2:
                # x2 = Amarillo parpadeante
                border_color = (0, 255, 255) if is_blink_on else (0, 200, 200)  # Cian parpadeante
                acceleration_text = "x2"
            elif skip_rate == 3:
                # x3 = Verde parpadeante (más rápido y evidente)
                border_color = (0, 255, 0) if is_blink_on else (0, 200, 0)  # Verde parpadeante
                acceleration_text = "x3"
            else:
                # Otros valores = Magenta parpadeante
                border_color = (255, 0, 255) if is_blink_on else (200, 0, 200)  # Magenta parpadeante
                acceleration_text = f"x{skip_rate}"
        
        # Fondo del semáforo (rectángulo negro)
        cv2.rectangle(frame, 
                    (semaforo_x, semaforo_y), 
                    (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                    (0, 0, 0), -1)  # Negro
        
        # Borde del semáforo (parpadeante si acelerado)
        cv2.rectangle(frame, 
                    (semaforo_x, semaforo_y), 
                    (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                    border_color, 2)
        
        # Diámetro y posiciones de las luces
        light_diameter = 20
        green_y = semaforo_y + semaforo_height - 25
        yellow_y = semaforo_y + semaforo_height//2
        red_y = semaforo_y + 25
        light_x = semaforo_x + semaforo_width//2
        
        # Dibujar solo la luz activa para mayor eficiencia
        if current_state == "green":
            cv2.circle(frame, (light_x, green_y), light_diameter, (0, 255, 0), -1)
            if skip_rate > 1:
                speed_text = f"AVANCE {acceleration_text}"
            else:
                speed_text = "AVANCE"
            cv2.putText(frame, speed_text, (semaforo_x - 100, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_state == "yellow":
            cv2.circle(frame, (light_x, yellow_y), light_diameter, (0, 255, 255), -1)
            if skip_rate > 1:
                speed_text = f"PRECAUCIÓN {acceleration_text}"
            else:
                speed_text = "PRECAUCIÓN"
            cv2.putText(frame, speed_text, (semaforo_x - 150, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif current_state == "red":
            cv2.circle(frame, (light_x, red_y), light_diameter, (0, 0, 255), -1)
            cv2.putText(frame, "PARE", (semaforo_x - 60, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Dibujar polígono si existe (solo contorno)
        if hasattr(self, 'polygon_points') and self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        # Añadir indicador de modo nocturno
        if is_night:
            cv2.putText(frame, "MODO NOCTURNO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _is_night_scene(self, frame):
        """Versión optimizada para detectar escenas nocturnas con ventana emergente"""
        # Redimensionar para análisis rápido
        small_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular brillo promedio
        avg_brightness = np.mean(gray)
        
        # También verificar área más oscura (percentil 20 - más permisivo)
        dark_threshold = np.percentile(gray, 20)
        
        # DETECCIÓN NOCTURNA INTELIGENTE CON VALIDACIÓN HORARIA CORREGIDA
        # 1er INDICADOR (DECISIVO): Verificar horario CONFIGURADO del video (NO hora actual del sistema)
        is_night_time_configured = False
        if hasattr(self, 'cycle_durations') and self.cycle_durations:
            time_slot = self.cycle_durations.get('time_slot', '')
            if time_slot:
                # Extraer hora de inicio del time_slot (ej: "19:00 - 20:00" -> 19)
                try:
                    start_time = time_slot.split(' - ')[0].split(':')[0]
                    start_hour = int(start_time)
                    # Considerar nocturno: 19:00-06:59 (7PM a 6:59AM)
                    is_night_time_configured = (start_hour >= 19 or start_hour <= 6)
                except (ValueError, IndexError):
                    print(f"⚠️ Error parseando franja horaria: {time_slot}")
                    is_night_time_configured = False
        else:
            # Sin configuración de franja horaria, NO asumir nada
            is_night_time_configured = False
        
        # 2do INDICADOR (COMPLEMENTARIO): Análisis inteligente del video - UMBRALES RESTRICTIVOS
        video_analysis_night = (avg_brightness < 70 and   # CORREGIDO: MÁS RESTRICTIVO - Solo muy oscuro
                               dark_threshold < 40 and    # CORREGIDO: MÁS RESTRICTIVO - Áreas realmente oscuras
                               np.std(gray) < 30)         # CORREGIDO: MÁS RESTRICTIVO - Contraste muy bajo
        
        # LÓGICA MEJORADA: Detectar por nombre del video también
        video_name_indicates_night = False
        if hasattr(self, 'video_path') and self.video_path:
            video_name = os.path.basename(self.video_path).lower()
            video_name_indicates_night = 'night' in video_name or 'nocturno' in video_name
        
        # LÓGICA RESTRICTIVA PARA DETECTAR SOLO VIDEOS REALMENTE NOCTURNOS
        if video_name_indicates_night:  # Simplificado: solo verificar nombre
            is_night = True  # Nombre nocturno = modo nocturno activado
        elif is_night_time_configured and video_analysis_night and avg_brightness < 60:
            is_night = True  # Franja nocturna + video MUY oscuro + análisis confirma
        else:
            is_night = False  # Por defecto es diurno - MÁS CONSERVADOR
        
        # DEBUG: Mostrar valores para calibración mejorada
        time_slot_configured = self.cycle_durations.get('time_slot', 'No configurada') if hasattr(self, 'cycle_durations') and self.cycle_durations else 'No configurada'
        video_name = os.path.basename(self.video_path) if hasattr(self, 'video_path') and self.video_path else 'Desconocido'
        print(f"🌙 DETECCIÓN NOCTURNA CORREGIDA: video='{video_name}', nombre_indica_noche={video_name_indicates_night}, franja_horaria_config='{time_slot_configured}', es_franja_nocturna={is_night_time_configured}, brillo_promedio={avg_brightness:.1f}, areas_oscuras={dark_threshold:.1f}, contraste={np.std(gray):.1f}, video_oscuro={video_analysis_night}, RESULTADO_FINAL={is_night}")
        
        # MEJORA: Mostrar ventana emergente nocturna (usando after para el hilo principal)
        if is_night and not PreprocessingDialog._night_popup_active:
            # Programar la ventana emergente en el hilo principal de la UI SOLO si no hay otra activa
            PreprocessingDialog._night_popup_active = True
            print(f"🌙 ACTIVANDO VENTANA NOCTURNA: brillo={avg_brightness:.1f}, umbral_oscuro={dark_threshold:.1f}")
            # Fix: Llamada directa sin verificar dialog - siempre mostrar ventana
            try:
                self._show_night_detection_popup(avg_brightness, dark_threshold)
            except Exception as e:
                print(f"⚠️ Error mostrando ventana nocturna: {e}")
                PreprocessingDialog._night_popup_active = False
        
        return is_night, avg_brightness, dark_threshold  # Devolver tupla como en la versión antigua

    def _complete_processing(self):
        """Finaliza el procesamiento y muestra los resultados"""
        print(f"📋 Procesando {len(self.detected_infractions)} infracciones")
        import os
        import json
        import time
        import traceback

        try:
            # PASO 1: Deduplicar placas
            deduped = self._dedup_similar_plates(self.detected_infractions)
            self.detected_infractions = deduped

            # Preparar medición de tiempo
            start_time = time.time()
            if not hasattr(self.player, "detection_start_time"):
                self.player.detection_start_time = start_time
            if not hasattr(self.player, "registration_times"):
                self.player.registration_times = []

            # SINCRONIZACIÓN FINAL: Limpiar el panel antes de redibujar los resultados finales deduplicados
            # Esto evita duplicados visuales y asegura que las métricas finales sean exactas
            print("🧹 Limpiando panel lateral para reconstrucción final (Deduplicación activa)...")
            self.player.clear_detected_plates()
            
            # PASO 2: Mostrar cada detección en el panel lateral
            for inf in deduped:
                if not all(k in inf and inf[k] is not None for k in ("plate_img", "plate", "vehicle_img")):
                    continue
                plate = inf["plate"]
                hist = getattr(self.player, "plate_detection_history", {})
                detection_time = self.player.detection_start_time + (inf.get("time") or 0)
                registration_time = time.time()
                proc_time = registration_time - detection_time
                self.player.registration_times.append(proc_time)

                entry = hist.get(plate, {
                    "count": 0,
                    "first_detection": inf.get("time"),
                    "vehicle_img": inf["vehicle_img"],
                    "detection_time": detection_time,
                })
                entry["count"] += 1
                entry["last_detection"] = inf.get("time")
                if inf.get("vehicle_path"):
                    entry["vehicle_path"] = inf["vehicle_path"]
                if inf.get("plate_path"):
                    entry["plate_path"] = inf["plate_path"]
                entry["registration_time"] = registration_time
                entry["processing_time"] = proc_time

                hist[plate] = entry
                self.player.plate_detection_history = hist

                # Clasificar placa para evaluar calidad
                classification, _, _ = self.player.classify_detection_quality(plate, 
                                                                            detection_confidence=inf.get('confidence', 0.8))
                
                # RECONSTRUCCIÓN VISUAL QUIRÚRGICA: Redibujar cards en el orden final deduplicado
                try:
                    self.player._safe_add_plate_to_panel(
                        inf["plate_img"],
                        plate,
                        inf.get("time"),
                        confidence=inf.get('confidence', 0.75),
                        vehicle_img=inf.get("vehicle_img"),
                        track_id=inf.get('track_id')
                    )
                except Exception as e:
                    print(f"❌ Error reconstruyendo card final: {e}")
                
                # SOLO agregar NID a gestión de infracciones (archivo JSON)
                if classification == "NID":
                    print(f"📋 Placa {plate} agregada a gestión (NID correcta)")
                else:
                    print(f"⚠️ Placa {plate} visible en panel pero NO en gestión (NIE incorrecta)")

            # PASO 3: Filtrar NID y NIE por separado
            nid_infractions = []
            nie_infractions = []
            
            for inf in deduped:
                plate = inf["plate"]
                # Usar confianza SIIV real guardada en la infracción
                siiv_confidence = inf.get("confidence", 0.5)
                classification, _, _ = self.player.classify_detection_quality(
                    plate, 
                    detection_confidence=siiv_confidence
                )
                if classification == "NID":
                    nid_infractions.append(inf)
                else:
                    nie_infractions.append(inf)
            
            print(f"📋 FILTRADO: {len(deduped)} total → {len(nid_infractions)} NID + {len(nie_infractions)} NIE")
            
            # Guardar infracciones NID en el JSON principal
            self._save_infractions_to_json(nid_infractions)
            
            # Guardar infracciones NIE en un archivo separado
            if nie_infractions:
                self._save_nie_infractions_to_json(nie_infractions)
            
            # NUEVO: PASO 3.5: Generar métricas solo con NID válidas
            self._generate_thesis_metrics(nid_infractions)

            # PASO 4: Actualizar indicadores TR en el JSON existente
            indicators_file = resource_path("data/indicadores_rendimiento.json")
            if os.path.exists(indicators_file):
                with open(indicators_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                tr = data.setdefault("indicadores", {}).setdefault("TR", {})
                total_time = time.time() - start_time
                avg_time = total_time / len(deduped) if deduped else 0
                PreprocessingDialog.recorded_processing_times.append(avg_time)

                tr.setdefault("con_software", {})["tiempo_promedio_segundos"] = avg_time
                tr["con_software"]["muestras_analizadas"] = len(PreprocessingDialog.recorded_processing_times)
                base = tr.get("sin_software", {}).get("tiempo_promedio_segundos", 0)
                tr["reduccion_tiempo_porcentual"] = ((base - avg_time) / base * 100) if base else 0
                tr["veces_mas_rapido"] = (base / avg_time) if avg_time else 0

                resumen = data.setdefault("resumen_global", {})
                resumen["tiempo_registro_reduccion"] = f"-{tr['reduccion_tiempo_porcentual']:.1f}%"
                resumen["tiempo_registro_factor"] = f"{tr['veces_mas_rapido']:.1f}x más rápido"

                with open(indicators_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # — Regenerar JSON plano pasándole SOLO LAS INFRACCIONES NUEVAS de esta sesión
                # Leer las que acabamos de guardar para obtener el formato JSON completo
                infractions_file = resource_path("data/infracciones.json")
                nie_file = resource_path("data/nie_infracciones.json")
                
                # Leer TODAS las infracciones guardadas
                saved_infractions = []
                if os.path.exists(infractions_file):
                    try:
                        with open(infractions_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict) and 'infracciones' in data:
                                saved_infractions = data['infracciones']
                            elif isinstance(data, list):
                                saved_infractions = data
                    except Exception as e:
                        print(f"⚠️ Error leyendo infracciones guardadas: {e}")
                
                saved_nie = []
                if os.path.exists(nie_file):
                    try:
                        with open(nie_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict) and 'infracciones' in data:
                                saved_nie = data['infracciones']
                            elif isinstance(data, list):
                                saved_nie = data
                    except Exception as e:
                        print(f"⚠️ Error leyendo NIE guardadas: {e}")
                
                # Filtrar SOLO las infracciones de esta sesión (las primeras N que acabamos de agregar)
                num_nid_nuevas = len(nid_infractions)
                num_nie_nuevas = len(nie_infractions)
                
                current_session_nid = saved_infractions[:num_nid_nuevas] if saved_infractions else []
                current_session_nie = saved_nie[:num_nie_nuevas] if saved_nie else []
                current_session_infractions = current_session_nid + current_session_nie
                
                # Extraer tiempos individuales de procesamiento de cada infracción (en segundos)
                individual_processing_times = [
                    inf.get('tiempo_procesamiento', 0) 
                    for inf in current_session_infractions 
                    if inf.get('tiempo_procesamiento', 0) > 0
                ]
                
                print(f"\n📊 Generando indicadores para {len(current_session_infractions)} infracciones de esta sesión ({num_nid_nuevas} NID + {num_nie_nuevas} NIE)...")
                print(f"   Tiempos de procesamiento individuales: {individual_processing_times} segundos")
                
                # 🆕 OBTENER NOMBRE DEL VIDEO Y CONFIGURACIÓN DEL SEMÁFORO
                nombre_video = os.path.basename(self.video_path) if hasattr(self, 'video_path') and self.video_path else "desconocido.mp4"
                # Usar AMBOS métodos: semáforo Y cycle_durations para obtener configuración
                config_semaforo = self.generar_config_id(
                    semaforo=self.player.semaforo if hasattr(self.player, 'semaforo') else None,
                    cycle_durations=self.cycle_durations if hasattr(self, 'cycle_durations') else None
                )
                
                generate_performance_indicators_json(
                    current_session_infractions,
                    individual_processing_times,  # Pasar tiempos individuales, no promedio
                    nombre_video=nombre_video,     # 🆕 Nombre del video
                    config_semaforo=config_semaforo  # 🆕 ID de configuración del semáforo
                )

                # — Subir indicadores en hilo aparte (SOLO si no es caso de segunda ventana nocturna)
                self._migrate_after_validation(len(deduped))

                print(f"Tiempo medio por infracción: {avg_time:.2f}s")

            # PASO 5: Actualizar métricas internas en el panel
            if hasattr(self.player, "performance_indicators"):
                avg_proc = (sum(self.player.registration_times) /
                            len(self.player.registration_times)
                            if self.player.registration_times else 0.0)
                
                # Sincronización de NID / NIE en los indicadores de rendimiento del dashboard
                nid_final = len(nid_infractions)
                nie_final = len(nie_infractions)
                
                self.player.performance_indicators = {
                    "TI": len(deduped),
                    "TR": avg_proc,
                    "NID": nid_final,
                    "NIE": nie_final,
                    "IR": 0.0
                }
                if hasattr(self.player, "_update_metrics_panel"):
                    self.player._update_metrics_panel()

            # PASO 6: ASEGURAR QUE SEMÁFORO Y VIDEO ESTÉN PAUSADOS (POR SI NO SE PAUSARON ANTES)
            if hasattr(self.player, 'semaforo') and self.player.semaforo and self.player.semaforo.active:
                self.player.semaforo.deactivate_semaphore()
                print("🚦 SEMÁFORO PAUSADO al finalizar procesamiento (fallback)")
            
            # PAUSAR VIDEO TAMBIÉN (por si no se pausó antes)
            if hasattr(self.player, 'is_playing') and self.player.is_playing:
                self.player.is_playing = False
                self.player.is_paused = True
                print("⏸️ VIDEO PAUSADO al finalizar procesamiento (fallback)")
            
            # Actualizar botón de play/pause (siempre)
            if hasattr(self.player, 'play_pause_button'):
                self.player.play_pause_button.config(
                    text="👁️ PREVISUALIZAR",
                    bg="#27ae60"
                )
            
            # PASO 7: UI final y cerrar diálogo
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.phase_label.config(text="Procesamiento completado")
                self.details_label.config(text=f"{len(deduped)} infracciones detectadas.")
                self.player.start_processed_video(self.video_path)

                # NO REINICIAR TIMESTAMP AUTOMÁTICAMENTE - Solo cuando reproduzca nuevo video
                print("⏸️ Timestamp permanece detenido - se activará al cargar nuevo video")

                # Programar cierre del diálogo
                self._safe_after(1000, lambda: self._close_dialog(True))

        except Exception as e:
            print(f"Error en _complete_processing: {e}")
            traceback.print_exc()
            # Si el diálogo ya no existe, cierra de inmediato
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self._close_dialog(False)
                
            # NO REINICIAR TIMESTAMP EN ERRORES - Solo cuando reproduzca nuevo video
            print("⏸️ Timestamp permanece detenido tras error - se activará al cargar nuevo video")





    # NUEVO MÉTODO: Guardar infracciones en archivo JSON
    def _save_infractions_to_json(self, infractions):
        """
        Guarda las infracciones detectadas en data/infracciones.json,
        ACUMULÁNDOLAS como stack/pila (nuevas infracciones al principio).
        """
        import json
        import os
        import getpass
        import socket
        from datetime import datetime

        # Asegurar existencia del directorio
        data_dir = resource_path("data")
        os.makedirs(data_dir, exist_ok=True)
        infractions_file = os.path.join(data_dir, "infracciones.json")

        # PASO 1: Cargar infracciones existentes (si las hay) con lock+caché
        from src.core.utils.json_store import read_json
        data = read_json(infractions_file, [])
        existing_infractions = []
        if isinstance(data, dict) and 'infracciones' in data:
            existing_infractions = data['infracciones']
        elif isinstance(data, list):
            existing_infractions = data
        else:
            print(f"⚠️ Estructura JSON inesperada: {type(data)}")
            existing_infractions = []
        print(f"📋 Cargadas {len(existing_infractions)} infracciones existentes")

        # Nombre de la avenida y franja horaria
        avenue_name = getattr(self.player, "current_avenue", "Desconocida")
        time_slot = self.cycle_durations.get("time_slot", "No especificada") if self.cycle_durations else "No especificada"

        # PASO 2: Procesar nuevas infracciones
        nuevas_infracciones = []
        for inf in infractions:
            plate = inf.get("plate", "")
            if not plate:
                print("Omitiendo guardar en JSON placa vacía")
                continue

            # Validar longitud de la placa
            clean_plate = plate.replace('-', '').replace(' ', '')
            if len(clean_plate) > 8:
                print(f"Omitiendo guardar en JSON placa inválida por longitud: {plate}")
                continue

            # Filtrar casos específicos
            if any(bad in plate for bad in ("BOHID", "B OHID", "B-OHID")):
                print(f"Omitiendo guardar en JSON placa problemática específica: {plate}")
                continue

            now = datetime.now()
            
            # CORREGIR: Obtener tiempo real del video procesamiento
            # Usar tiempo de procesamiento real en segundos, no el campo 'time' genérico
            processing_time = inf.get("time", inf.get("processing_time", inf.get("timestamp", 0)))
            
            # Asegurar que tenemos un valor numérico válido
            if isinstance(processing_time, (int, float)) and processing_time > 0:
                total_seconds = int(processing_time)
                mins, secs = divmod(total_seconds, 60)
                timestamp = f"{mins:02d}:{secs:02d}"
            else:
                # Fallback: usar tiempo basado en frame si está disponible
                frame_number = inf.get("frame", 0)
                fps = getattr(self.player, 'fps', 30) or 30  # FPS por defecto
                total_seconds = int(frame_number / fps) if frame_number > 0 else 0
                mins, secs = divmod(total_seconds, 60)
                timestamp = f"{mins:02d}:{secs:02d}"

            # SINCRONIZACIÓN: Usar la MISMA lógica que se usó para crear el card
            classification, quality_score, _ = self.player.classify_detection_quality(plate)
            
            # Si la infracción original tenía confianza específica, usar la misma lógica que el card
            if 'confidence' in inf:
                # CRÍTICO: Aplicar el mismo clamp que usa PlateCard para evitar valores inválidos
                raw_confidence = inf['confidence']
                clamped_confidence = max(0.0, min(1.0, raw_confidence))  # Clamp [0,1] como PlateCard
                
                # El card recomputó usando esta confidence, hacer lo mismo aquí
                classification, card_confidence, _ = self.player.classify_detection_quality(
                    plate, detection_confidence=clamped_confidence
                )
                real_confidence = clamped_confidence  # Usar valor corregido
            else:
                # Usar quality_score calculado (la misma que usó el card)
                real_confidence = quality_score

            # Crear metadata actualizado con confianza real (la MISMA que muestra el card)
            metadata_clasificacion = {
                "placa_final": plate,
                "confianza": round(real_confidence, 3),
                "calidad_deteccion": "alta" if real_confidence >= 0.7 else "media" if real_confidence >= 0.5 else "baja",
                "justificacion": "Cumple criterios técnicos calibrados"
            }

            # CORREGIR: Obtener duración total del video del SELECTOR (como indicas)
            total_duration = "N/A"
            
            # Método 1: Desde video_metadata si está disponible (viene del selector)
            if hasattr(self.player, 'video_metadata') and self.player.video_metadata:
                total_duration = self.player.video_metadata.get('duration', 'N/A')
            
            # Método 2: Calcular desde propiedades del video si no está disponible
            elif hasattr(self.player, 'cap') and self.player.cap is not None:
                try:
                    frame_count = int(self.player.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = self.player.cap.get(cv2.CAP_PROP_FPS) or 30
                    total_seconds_video = int(frame_count / fps)
                    mins_total, secs_total = divmod(total_seconds_video, 60)
                    total_duration = f"{mins_total:02d}:{secs_total:02d}"
                except Exception as e:
                    print(f"⚠️ Error calculando duración del video: {e}")
                    total_duration = "N/A"
            
            # Método 3: Desde cycle_durations si está disponible
            elif hasattr(self, 'cycle_durations') and self.cycle_durations:
                video_duration = self.cycle_durations.get('video_duration', self.cycle_durations.get('total_duration'))
                if video_duration:
                    if isinstance(video_duration, str) and ':' in video_duration:
                        total_duration = video_duration
                    elif isinstance(video_duration, (int, float)):
                        mins_total, secs_total = divmod(int(video_duration), 60)
                        total_duration = f"{mins_total:02d}:{secs_total:02d}"
            
            # 🆕 OBTENER NOMBRE DEL VIDEO Y CONFIGURACIÓN DEL SEMÁFORO
            nombre_video = os.path.basename(self.video_path) if hasattr(self, 'video_path') and self.video_path else "desconocido.mp4"
            # Usar AMBOS métodos: semáforo Y cycle_durations para obtener configuración
            config_semaforo = self.generar_config_id(
                semaforo=self.player.semaforo if hasattr(self.player, 'semaforo') else None,
                cycle_durations=self.cycle_durations if hasattr(self, 'cycle_durations') else None
            )
            
            entry = {
                "placa":           plate,
                "fecha":           now.strftime("%d/%m/%Y"),
                "hora":            now.strftime("%H:%M:%S"),
                "video_timestamp": timestamp,
                "tiempo_video":    total_duration,  # Duración total del video
                "ubicacion":       avenue_name,
                "franja_horaria":  time_slot,
                "tipo":            "Semáforo en rojo",
                "estado":          "Pendiente",
                "plate_path":      os.path.join(resource_path("data/output/placas"), f"plate_{plate}.jpg"),
                "vehicle_path":    os.path.join(resource_path("data/output/autos"), f"vehicle_{plate}.jpg"),
                # 🆕 NUEVOS CAMPOS PARA ESTRUCTURA FIRESTORE POR VIDEO Y CONFIGURACIÓN
                "nombre_video":    nombre_video,      # Nombre del video procesado
                "config_semaforo": config_semaforo,   # ID de configuración (ej: "10-3-15")
                # NUEVOS CAMPOS PARA TESIS NID/NIE
                "clasificacion":   inf.get("clasificacion", "NID"),  # Por defecto NID si no está especificado
                "confianza":       round(real_confidence, 3),
                "tiempo_procesamiento": round(inf.get("timestamp", inf.get("time", inf.get("tiempo_procesamiento", 0))), 2),
                "metadata_clasificacion": metadata_clasificacion,
                "sistema_version": inf.get("sistema_version", "InfractiVision_v2.0"),
                # Campos de trazabilidad
                "hostname":        socket.gethostname(),
                "username":        getpass.getuser()
            }
            if getattr(self, "is_night", False):
                entry["modo_nocturno"] = True

            nuevas_infracciones.append(entry)

        # PASO 3: ACUMULAR como stack/pila (nuevas al principio)
        infracciones_finales = nuevas_infracciones + existing_infractions
        
        # GUARDAR en formato consistente con estructura {"infracciones": [...]}
        output_data = {"infracciones": infracciones_finales}
        
        try:
            from src.core.utils.json_store import write_json
            write_json(infractions_file, output_data)
            print(f"📝 ACUMULADAS: {len(nuevas_infracciones)} nuevas + {len(existing_infractions)} anteriores = {len(infracciones_finales)} totales")
            print(f"💾 Stack actualizado en '{infractions_file}'")
        except Exception as e:
            print(f"Error guardando infracciones en JSON: {e}")

    def _save_nie_infractions_to_json(self, infractions):
        """
        Guarda las infracciones NIE (incorrectamente registradas) detectadas en data/nie_infracciones.json,
        ACUMULÁNDOLAS como stack/pila (nuevas infracciones al principio).
        """
        import json
        import os
        import getpass
        import socket
        from datetime import datetime

        # Asegurar existencia del directorio
        data_dir = resource_path("data")
        os.makedirs(data_dir, exist_ok=True)
        nie_file = os.path.join(data_dir, "nie_infracciones.json")

        # PASO 1: Cargar NIE existentes (si las hay) con lock+caché
        from src.core.utils.json_store import read_json
        data = read_json(nie_file, [])
        existing_nie = []
        if isinstance(data, dict) and 'infracciones' in data:
            existing_nie = data['infracciones']
        elif isinstance(data, list):
            existing_nie = data
        print(f"📋 Cargadas {len(existing_nie)} NIE existentes")

        # Nombre de la avenida y franja horaria
        avenue_name = getattr(self.player, "current_avenue", "Desconocida")
        time_slot = self.cycle_durations.get("time_slot", "No especificada") if self.cycle_durations else "No especificada"

        # PASO 2: Procesar nuevas NIE
        nuevas_nie = []
        for inf in infractions:
            plate = inf.get("plate", "")
            if not plate:
                continue

            now = datetime.now()
            
            # Calcular timestamp
            processing_time = inf.get("time", inf.get("processing_time", inf.get("timestamp", 0)))
            if isinstance(processing_time, (int, float)) and processing_time > 0:
                total_seconds = int(processing_time)
                mins, secs = divmod(total_seconds, 60)
                timestamp = f"{mins:02d}:{secs:02d}"
            else:
                frame_number = inf.get("frame", 0)
                fps = getattr(self.player, 'fps', 30) or 30
                total_seconds = int(frame_number / fps) if frame_number > 0 else 0
                mins, secs = divmod(total_seconds, 60)
                timestamp = f"{mins:02d}:{secs:02d}"

            # Duración total del video
            total_duration = "N/A"
            if hasattr(self.player, 'video_metadata') and self.player.video_metadata:
                total_duration = self.player.video_metadata.get('duration', 'N/A')
            elif hasattr(self.player, 'cap') and self.player.cap is not None:
                try:
                    import cv2
                    frame_count = int(self.player.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = self.player.cap.get(cv2.CAP_PROP_FPS) or 30
                    total_seconds_video = int(frame_count / fps)
                    mins_total, secs_total = divmod(total_seconds_video, 60)
                    total_duration = f"{mins_total:02d}:{secs_total:02d}"
                except Exception:
                    total_duration = "N/A"
            elif hasattr(self, 'cycle_durations') and self.cycle_durations:
                video_duration = self.cycle_durations.get('video_duration', self.cycle_durations.get('total_duration'))
                if video_duration:
                    if isinstance(video_duration, str) and ':' in video_duration:
                        total_duration = video_duration
                    elif isinstance(video_duration, (int, float)):
                        mins_total, secs_total = divmod(int(video_duration), 60)
                        total_duration = f"{mins_total:02d}:{secs_total:02d}"

            # Clasificación y confianza
            classification, quality_score, _ = self.player.classify_detection_quality(plate)
            
            if 'confidence' in inf:
                # Asegurar que la confianza esté en el rango válido [0.0, 1.0]
                raw_confidence = inf['confidence']
                clamped_confidence = max(0.0, min(1.0, raw_confidence))
                classification, card_confidence, _ = self.player.classify_detection_quality(
                    plate, detection_confidence=clamped_confidence
                )
                real_confidence = clamped_confidence
            else:
                real_confidence = quality_score

            metadata_clasificacion = {
                "placa_final": plate,
                "confianza": round(real_confidence, 3),
                "calidad_deteccion": "baja",
                "justificacion": "No cumple criterios técnicos - Clasificada como NIE"
            }

            # 🆕 OBTENER NOMBRE DEL VIDEO Y CONFIGURACIÓN DEL SEMÁFORO
            nombre_video = os.path.basename(self.video_path) if hasattr(self, 'video_path') and self.video_path else "desconocido.mp4"
            # Usar AMBOS métodos: semáforo Y cycle_durations para obtener configuración
            config_semaforo = self.generar_config_id(
                semaforo=self.player.semaforo if hasattr(self.player, 'semaforo') else None,
                cycle_durations=self.cycle_durations if hasattr(self, 'cycle_durations') else None
            )
            
            entry = {
                "placa":           plate,
                "fecha":           now.strftime("%d/%m/%Y"),
                "hora":            now.strftime("%H:%M:%S"),
                "video_timestamp": timestamp,
                "tiempo_video":    total_duration,
                "ubicacion":       avenue_name,
                "franja_horaria":  time_slot,
                "tipo":            "Semáforo en rojo",
                "estado":          "Rechazada",
                "clasificacion":   "NIE",
                "confianza":       round(real_confidence, 3),
                "plate_path":      os.path.join(resource_path("data/output/placas"), f"plate_{plate}.jpg"),
                "vehicle_path":    os.path.join(resource_path("data/output/autos"), f"vehicle_{plate}.jpg"),
                "tiempo_procesamiento": round(inf.get("timestamp", inf.get("time", 0)), 2),
                "metadata_clasificacion": metadata_clasificacion,
                # 🆕 NUEVOS CAMPOS PARA ESTRUCTURA FIRESTORE POR VIDEO Y CONFIGURACIÓN
                "nombre_video":    nombre_video,      # Nombre del video procesado
                "config_semaforo": config_semaforo,   # ID de configuración (ej: "10-3-15")
                "sistema_version": inf.get("sistema_version", "InfractiVision_v2.0"),
                "hostname":        socket.gethostname(),
                "username":        getpass.getuser()
            }

            nuevas_nie.append(entry)

        # PASO 3: ACUMULAR como stack/pila
        nie_finales = nuevas_nie + existing_nie
        
        # GUARDAR
        output_data = {"infracciones": nie_finales}
        
        try:
            from src.core.utils.json_store import write_json
            write_json(nie_file, output_data)
            print(f"📝 NIE ACUMULADAS: {len(nuevas_nie)} nuevas + {len(existing_nie)} anteriores = {len(nie_finales)} totales")
            print(f"💾 Stack NIE actualizado en '{nie_file}'")
        except Exception as e:
            print(f"Error guardando NIE en JSON: {e}")

    def _save_official_infractions_to_db(self, evidences, pending_infractions=None):
        """Guarda las infracciones validadas en SQLite (única fuente).

        Compat: mantiene el nombre legacy `_save_official_infractions_to_json`
        como alias para llamadas externas.
        """
        import getpass
        import socket
        from datetime import datetime
        from src.infrastructure.database.app_repository import AppRepository

        avenue_name = getattr(self.player, "current_avenue", "Desconocida")
        time_slot = self.cycle_durations.get("time_slot", "No especificada") if self.cycle_durations else "No especificada"
        nombre_video = os.path.basename(self.video_path) if getattr(self, 'video_path', None) else "desconocido.mp4"
        config_semaforo = self.generar_config_id(
            semaforo=getattr(self.player, 'semaforo', None),
            cycle_durations=getattr(self, 'cycle_durations', None)
        )
        total_duration = "N/A"
        if hasattr(self.player, 'video_metadata') and self.player.video_metadata:
            total_duration = self.player.video_metadata.get('duration', 'N/A')
        elif hasattr(self.player, 'cap') and self.player.cap is not None:
            try:
                import cv2
                frame_count = int(self.player.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.player.cap.get(cv2.CAP_PROP_FPS) or 30
                total_seconds_video = int(frame_count / fps)
                mins_total, secs_total = divmod(total_seconds_video, 60)
                total_duration = f"{mins_total:02d}:{secs_total:02d}"
            except Exception:
                total_duration = "N/A"

        def _build_entry(placa, confianza, clasificacion, plate_path, vehicle_path, timestamp_seconds, metadata):
            now = datetime.now()
            total_seconds = int(timestamp_seconds or 0)
            mins, secs = divmod(total_seconds, 60)
            timestamp = f"{mins:02d}:{secs:02d}"
            return {
                "placa": placa,
                "fecha": now.strftime("%d/%m/%Y"),
                "hora": now.strftime("%H:%M:%S"),
                "video_timestamp": timestamp,
                "tiempo_video": total_duration,
                "ubicacion": avenue_name,
                "franja_horaria": time_slot,
                "tipo": "Semáforo en rojo",
                "estado": "Pendiente" if clasificacion == "NID" else "Rechazada",
                "plate_path": plate_path,
                "vehicle_path": vehicle_path,
                "nombre_video": nombre_video,
                "config_semaforo": config_semaforo,
                "clasificacion": clasificacion,
                "confianza": round(max(0.0, min(1.0, confianza)), 3),
                "tiempo_procesamiento": round(timestamp_seconds or 0, 2),
                "metadata_clasificacion": metadata,
                "metadata_clasificacion_json": __import__("json").dumps(metadata, ensure_ascii=False),
                "sistema_version": "InfractiVision_v2.0",
                "hostname": socket.gethostname(),
                "username": getpass.getuser(),
                "modo_nocturno": int(bool(getattr(self, "is_night", False))),
            }

        nid_entries: list[dict] = []
        nie_entries: list[dict] = []
        for ev in (evidences or []):
            confianza = ev.ocr_confidence if ev.ocr_confidence else ev.quality_score
            crop = str(ev.crop_path or "")
            plate_path = crop if os.path.exists(crop) else ""
            if ev.validated and ev.plate_text:
                metadata = {"placa_final": ev.plate_text, "confianza": round(confianza, 3), "calidad_deteccion": "alta" if confianza >= 0.7 else "media" if confianza >= 0.5 else "baja", "justificacion": "Validada en revisión oficial (OCR)"}
                nid_entries.append(_build_entry(ev.plate_text, confianza, "NID", plate_path, "", ev.timestamp_seconds, metadata))
            else:
                metadata = {"placa_final": ev.plate_text or "", "confianza": round(confianza, 3), "calidad_deteccion": "baja", "justificacion": "No validada en revisión oficial - NIE"}
                placa_label = ev.plate_text or f"TRACK-{ev.track_id}"
                nie_entries.append(_build_entry(placa_label, confianza, "NIE", plate_path, "", ev.timestamp_seconds, metadata))
        for pend in (pending_infractions or []):
            crop = str(pend.get("crop_path", "") or "")
            plate_path = crop if crop and os.path.exists(crop) else ""
            metadata = {"placa_final": "", "confianza": 0.0, "calidad_deteccion": "baja", "justificacion": "Infracción pendiente sin placa detectada - NIE"}
            nie_entries.append(_build_entry(f"TRACK-{pend.get('vehicle_id', '?')}", 0.0, "NIE", plate_path, "", pend.get("timestamp_seconds", 0), metadata))

        all_entries = nid_entries + nie_entries
        if all_entries:
            repo = AppRepository()
            inserted = repo.insert_infractions(all_entries)
            print(f"📝 [OFICIAL][SQLite] Insertadas {inserted} infracciones ({len(nid_entries)} NID + {len(nie_entries)} NIE) para {nombre_video}")
        else:
            print("⚠️ [OFICIAL] No se guardaron infracciones (sin evidencias ni pendientes)")
        return nid_entries, nie_entries

    # Alias legacy para compatibilidad externa
    def _save_official_infractions_to_json(self, evidences, pending_infractions=None):
        return self._save_official_infractions_to_db(evidences, pending_infractions)

    def _generate_thesis_metrics(self, infractions):
        """
        Genera y guarda métricas de tesis (TI, TR, NID, NIE) para análisis académico.
        
        Args:
            infractions: Lista de infracciones procesadas con clasificación NID/NIE
        """
        try:
            from datetime import datetime
            print(f"\n📊 Generando métricas de tesis para {len(infractions)} infracciones...")
            
            # Calcular métricas usando el sistema especializado
            metrics = self.metrics_calculator.calculate_metrics(infractions)
            
            # Crear reporte detallado
            report = {
                "fecha_generacion": datetime.now().isoformat(),
                "video_procesado": os.path.basename(self.video_path),
                "total_infracciones": len(infractions),
                "metricas_tesis": metrics,
                "resumen_ejecutivo": {
                    "NID_porcentaje": metrics['NID']['porcentaje'],
                    "NIE_porcentaje": metrics['NIE']['porcentaje'], 
                    "TI_tasa": metrics['TI']['tasa_infracciones_validas'],
                    "TR_segundos": metrics['TR']['tiempo_promedio_segundos'],
                    "sistema_efectivo": metrics['resumen_tesis']['sistema_efectivo'],
                    "confiabilidad": metrics['resumen_tesis']['confiabilidad_general']
                },
                "detalle_clasificaciones": []
            }
            
            # Agregar detalles de cada clasificación
            for inf in infractions:
                detalle = {
                    "placa": inf.get('plate', ''),
                    "clasificacion": inf.get('clasificacion', 'NID'),
                    "confianza": inf.get('confidence', 0),
                    "razon": inf.get('metadata_clasificacion', {}).get('razon', 'valida'),
                    "tiempo_procesamiento": inf.get('tiempo_procesamiento', 0)
                }
                report["detalle_clasificaciones"].append(detalle)
            
            # Guardar reporte de métricas (usar indicadores_rendimiento.json en lugar del eliminado metricas_tesis.json)
            metrics_file = resource_path("data/indicadores_rendimiento.json")
            
            # Crear estructura compatible con indicadores_rendimiento.json
            indicators_data = {
                "fecha_generacion": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "periodo_analisis": f"Procesamiento de {len(infractions)} infracciones",
                "dias_analizados": 1,
                "indicadores": {
                    "TI": {
                        "descripcion": "Tasa de Infracciones Válidas",
                        "valor": metrics['TI']['tasa_infracciones_validas'],
                        "unidad": "porcentaje"
                    },
                    "TR": {
                        "descripcion": "Tiempo de Registro Promedio",
                        "valor": metrics['TR']['tiempo_promedio_segundos'],
                        "unidad": "segundos"
                    },
                    "NID": {
                        "descripcion": "Número de Infracciones Detectadas",
                        "valor": metrics['NID']['cantidad'],
                        "porcentaje": metrics['NID']['porcentaje']
                    }
                },
                "metricas_tesis": report  # Incluir datos completos como sub-objeto
            }
            
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(indicators_data, f, indent=2, ensure_ascii=False)
                
            # Log de resultados
            print(f"📈 MÉTRICAS DE TESIS GENERADAS:")
            print(f"   🟢 NID: {metrics['NID']['cantidad']} ({metrics['NID']['porcentaje']:.1f}%)")
            print(f"   🟡 NIE: {metrics['NIE']['cantidad']} ({metrics['NIE']['porcentaje']:.1f}%)")
            print(f"   📊 TI: {metrics['TI']['tasa_infracciones_validas']:.1f}% infracciones válidas")
            print(f"   ⏱️ TR: {metrics['TR']['tiempo_promedio_segundos']:.2f}s promedio")
            print(f"   🎯 Sistema efectivo: {metrics['resumen_tesis']['sistema_efectivo']}")
            print(f"   💾 Reporte guardado: {metrics_file}")
            
            # Log específico para defendar la tesis
            if metrics['NID']['porcentaje'] >= 70:
                print(f"✅ OBJETIVO CUMPLIDO: NID {metrics['NID']['porcentaje']:.1f}% ≥ 70% (Meta académica)")
            else:
                print(f"⚠️ REQUIERE OPTIMIZACIÓN: NID {metrics['NID']['porcentaje']:.1f}% < 70% (Meta académica)")
                
        except Exception as e:
            print(f"❌ Error generando métricas de tesis: {e}")
            import traceback
            traceback.print_exc()


    def _close_dialog(self, success):
        """Cierra el diálogo y llama a la función de completado"""
        # Cancelar todos los after pendientes (¡CRÍTICO!)
        self._cancel_all_after()

        try:
            if hasattr(self.player, 'running'):
                self.player.running = False

            if getattr(self, 'inline', False):
                # Modo inline: ocultar progreso, NO destruir la ventana principal.
                self._cleanup_threads()
                self._inline_progress_show(False)
                if self.on_complete and success:
                    self.on_complete(success, self.detected_infractions)
                return

            if self.dialog.winfo_exists():
                self.dialog.grab_release()
                self.dialog.destroy()

            self._cleanup_threads()

            if self.on_complete and success:
                self.on_complete(success, self.detected_infractions)
        except Exception as e:
            print(f"Error cerrando diálogo de procesamiento: {e}")

    def _cleanup_threads(self):
        """Detiene y une los hilos de fondo del diálogo para que ningún hilo
        daemon muera en medio de código CUDA nativo al cerrar (evita
        SIGSEGV/'terminate called without an active exception')."""
        self.display_active = False
        for attr in ("display_thread", "process_thread", "preload_thread",
                     "analysis_worker_thread"):
            t = getattr(self, attr, None)
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        ap = getattr(self, "async_processor", None)
        if ap is not None:
            try:
                ap.stop()
            except Exception as e:
                print(f"Error deteniendo async_processor: {e}")

    def _on_dialog_destroy(self, event):
        if event is not None and getattr(event, "widget", None) is not self.dialog:
            return
        try:
            self._cleanup_threads()
        except Exception as e:
            print(f"Error en cleanup del diálogo: {e}")
            
    def on_cancel(self):
        if not self.canceled:
            self.canceled = True
            self._cancel_all_after()   # ← Añadir esta línea
            
            # 🚀 LIMPIAR: Detener visualización fluida
            self.display_active = False
            if hasattr(self, 'display_thread') and self.display_thread is not None and self.display_thread.is_alive():
                print("🛑 Deteniendo thread de visualización fluida")
            
            self.phase_label.config(text="Cancelando procesamiento...")
            self.details_label.config(text="Por favor espere...")
            self.cancel_button.config(state="disabled")
            
            # Cerrar solo esta ventana después de un breve retraso
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self._safe_after(1000, self._close_dialog_only)

    # =====================================================
    # FUNCIONES DE ALERTAS AVANZADAS DEL COMPAÑERO
    # =====================================================
    
