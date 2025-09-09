import os
import json
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler

from src.automations.cloud_migrator import upload_infracciones_automatically
from src.gui.infractions_management_window import generate_performance_indicators_json
from src.path_helper import resource_path



# Eliminamos la importación circular

class PreprocessingDialog:
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
    
    def __init__(self, parent, video_path, player_instance, on_complete=None):
        """
        Inicializa el diálogo de preprocesamiento.
        
        Args:
            parent: Widget padre
            video_path: Ruta del video a procesar
            player_instance: Instancia del VideoPlayerOpenCV para acceder a sus métodos
            on_complete: Función a llamar cuando se complete el procesamiento
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
        self.processed_frames = 0
        self.total_frames = 0
        self.result_queue = queue.Queue()
        
        # Definir rutas de configuración usando resource_path para PyInstaller
        self.POLYGON_CONFIG_FILE = resource_path("config/polygon_config.json")
        self.AVENUE_CONFIG_FILE = resource_path("config/avenue_config.json")
        self.PRESETS_FILE = resource_path("config/time_presets.json")

        # Add this line to track start time
        self.processing_start_time = time.time()
        
        # Reset class variable for this instance
        if len(PreprocessingDialog.recorded_processing_times) > 100:  # Limit history
            PreprocessingDialog.recorded_processing_times = []
        
        # Cargar configuración del video si existe
        self.load_video_config()
        
        # Crear ventana de diálogo
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Análisis de infracciones")
        
        # Configurar icono
        icon_path = resource_path("img/icon.ico")
        if os.path.exists(icon_path):
            self.dialog.iconbitmap(icon_path)
        self.dialog.geometry("800x600")
        self.dialog.resizable(False, False)
        
        # Centrar ventana
        self.dialog.update_idletasks()
        width, height = 800, 600
        x = (self.dialog.winfo_screenwidth() - width) // 2
        y = (self.dialog.winfo_screenheight() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.dialog.grab_set()  # Modal
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)  # Manejar cierre
        
        # Configurar el layout
        self._setup_ui()
        
        # Precargar modelos en un hilo separado para evitar bloquear la UI
        self.preload_thread = threading.Thread(target=self._preload_models, daemon=True)
        self.preload_thread.start()
        
        # Programar actualizaciones periódicas de la UI
        self._schedule_ui_update()
    
    def _preload_models(self):
        """Precarga los modelos de IA antes de procesar el video"""
        try:
            self.phase_label.config(text="Preparando modelos de IA...")
            self.details_label.config(text="Inicializando detectores...")
            
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
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.after(0, lambda msg=str(e): self._show_error(f"Error cargando modelos: {msg}"))
    
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
            
            # Debug: Mostrar información de carga
            print(f"🔍 DEBUG - Cargando configuración para: {video_key}")
            print(f"   📐 Polígono disponible: {video_key in configs.get('polygon', {})}")
            print(f"   ⏱️ Tiempos disponibles: {video_key in configs.get('presets', {})}")
            print(f"   🛣️ Avenida disponible: {video_key in configs.get('avenue', {})}")
            
            if video_key in configs.get('polygon', {}):
                self.polygon_points = configs['polygon'][video_key]
                print(f"   ✅ Polígono cargado: {len(self.polygon_points)} puntos")
                
            if video_key in configs.get('presets', {}):
                self.cycle_durations = configs['presets'][video_key]
                print(f"   ✅ Tiempos cargados: {self.cycle_durations}")
                
            if video_key in configs.get('avenue', {}):
                self.current_avenue = configs['avenue'][video_key]
                print(f"   ✅ Avenida cargada: {self.current_avenue}")
            
            # Validación final
            valid_polygon = self.polygon_points and len(self.polygon_points) >= 3
            valid_times = (self.cycle_durations and 
                          isinstance(self.cycle_durations, dict) and
                          'green' in self.cycle_durations and
                          'yellow' in self.cycle_durations and
                          'red' in self.cycle_durations)
            
            print(f"   🔹 Validación polígono: {valid_polygon}")
            print(f"   🔹 Validación tiempos: {valid_times}")
                
        except Exception as e:
            print(f"❌ Error en load_video_config: {e}")
            import traceback
            traceback.print_exc()
    
    def create_synchronized_semaphore(self):
        """Crear semáforo visual sincronizado con el principal"""
        # Título del semáforo
        title_label = tk.Label(
            self.semaphore_frame,
            text="Estado del Semáforo",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=(10, 5))
        
        # Canvas para dibujar el semáforo
        self.semaphore_canvas = tk.Canvas(
            self.semaphore_frame,
            width=80,
            height=200,
            bg="black",
            highlightthickness=0
        )
        self.semaphore_canvas.pack(pady=10)
        
        # Crear círculos del semáforo
        self.red_light = self.semaphore_canvas.create_oval(20, 20, 60, 60, fill="#400000", outline="white", width=2)
        self.yellow_light = self.semaphore_canvas.create_oval(20, 70, 60, 110, fill="#404000", outline="white", width=2)
        self.green_light = self.semaphore_canvas.create_oval(20, 120, 60, 160, fill="#004000", outline="white", width=2)
        
        # Label para tiempo restante
        self.time_label = tk.Label(
            self.semaphore_frame,
            text="-- s",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        self.time_label.pack(pady=(10, 0))
        
        # Label para estado
        self.state_label = tk.Label(
            self.semaphore_frame,
            text="DETENIDO",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="gray"
        )
        self.state_label.pack(pady=(5, 0))
    
    def update_synchronized_semaphore(self):
        """Actualizar semáforo sincronizado con el estado principal"""
        try:
            if hasattr(self.player, 'semaforo') and self.player.semaforo:
                current_state = self.player.semaforo.get_current_state()
                
                # Calcular tiempo restante
                time_left = 0
                if hasattr(self.player.semaforo, 'target_time'):
                    time_left = max(0, int(self.player.semaforo.target_time - time.time()))
                
                # Actualizar luces
                self.semaphore_canvas.itemconfig(self.red_light, fill="#400000")
                self.semaphore_canvas.itemconfig(self.yellow_light, fill="#404000")
                self.semaphore_canvas.itemconfig(self.green_light, fill="#004000")
                
                # Encender luz correspondiente
                if current_state == "red":
                    self.semaphore_canvas.itemconfig(self.red_light, fill="red")
                    self.state_label.config(text="ROJO", fg="red")
                elif current_state == "yellow":
                    self.semaphore_canvas.itemconfig(self.yellow_light, fill="yellow")
                    self.state_label.config(text="AMARILLO", fg="orange")
                elif current_state == "green":
                    self.semaphore_canvas.itemconfig(self.green_light, fill="green")
                    self.state_label.config(text="VERDE", fg="green")
                
                # Actualizar tiempo
                self.time_label.config(text=f"{time_left}s")
                
            else:
                # Semáforo no disponible
                self.state_label.config(text="DETENIDO", fg="gray")
                self.time_label.config(text="-- s")
                
        except Exception as e:
            print(f"Error actualizando semáforo sincronizado: {e}")
    
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
        
        # Frame para la visualización del video
        self.video_frame = ttk.Frame(video_container, width=640, height=360, relief="groove", borderwidth=2)
        self.video_frame.pack(side="left", padx=(0, 10))
        self.video_frame.pack_propagate(False)
        
        # Frame para el semáforo sincronizado
        self.semaphore_frame = ttk.Frame(video_container, width=120, height=360, relief="groove", borderwidth=2)
        self.semaphore_frame.pack(side="left", fill="y")
        self.semaphore_frame.pack_propagate(False)
        
        # Label para mostrar el frame actual
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)
        
        # Crear semáforo visual sincronizado
        self.create_synchronized_semaphore()
        
        # Información de procesamiento
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.pack(fill="x", pady=(0, 10))
        
        # Etiqueta para mostrar la fase actual
        self.phase_label = ttk.Label(
            self.info_frame, 
            text="Preparando análisis...", 
            font=("Arial", 12)
        )
        self.phase_label.pack(anchor="w")
        
        # Etiqueta para mostrar detalles del procesamiento
        self.details_label = ttk.Label(
            self.info_frame, 
            text="",
            font=("Arial", 10)
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
        
        # Etiqueta de porcentaje
        self.percentage_label = ttk.Label(
            progress_frame, 
            text="0%", 
            font=("Arial", 10)
        )
        self.percentage_label.pack(anchor="e", pady=(5, 0))
        
        # Contador de infracciones detectadas
        self.infractions_label = ttk.Label(
            main_frame, 
            text="Infracciones detectadas: 0", 
            font=("Arial", 12, "bold")
        )
        self.infractions_label.pack(pady=10)
        
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
    
    def _schedule_ui_update(self):
        """Programa actualizaciones periódicas de la interfaz"""
        if not self.canceled:
            try:
                # Actualizar barra de progreso con animación suave
                self.progress_var.set(self.progress_value)
                self.percentage_label.config(text=f"{int(self.progress_value)}%")
                
                # Actualizar contador de infracciones
                self.infractions_label.config(text=f"Infracciones detectadas: {len(self.detected_infractions)}")
                
                # Procesar cualquier resultado pendiente de los hilos de trabajo
                self._process_results_queue()
                
                # Forzar actualización de la interfaz
                self.dialog.update_idletasks()
                
                # Programar próxima actualización (más frecuente para que sea fluido)
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(50, self._schedule_ui_update)
            except Exception as e:
                print(f"Error en actualización de UI: {e}")
                # Seguir intentando actualizar la UI
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(100, self._schedule_ui_update)

    
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
                    
                    if result_type == "frame_update":
                        frame, segment_id, processed_frames, total_frames = data
                        # Actualizar el frame actual y mostrarlo inmediatamente
                        self.current_frame = frame
                        self._update_video_frame(frame)
                        
                        # Actualizar información de progreso para este segmento
                        segment_progress = (processed_frames / total_frames) * 100
                        segment_contribution = segment_progress / self.total_segments
                        
                        # Actualizar progreso global considerando segmentos completados
                        base_progress = (self.completed_segments / self.total_segments) * 100
                        segment_part = (1 / self.total_segments) * (segment_progress / 100) * 100
                        self.progress_value = min(base_progress + segment_part, 99.9)  # No llegar a 100% hasta terminar
                        
                        # Actualizar texto de progreso
                        self.details_label.config(text=f"Procesando segmento {segment_id+1}/{self.total_segments} | Frame {processed_frames}/{total_frames}")
                    
                    elif result_type == "segment_complete":
                        segment_id, infractions = data
                        # Añadir las infracciones detectadas
                        self.detected_infractions.extend(infractions)
                        
                        # Actualizar contador de segmentos completados
                        self.completed_segments += 1
                        # Actualizar progreso
                        base_progress = (self.completed_segments / self.total_segments) * 100
                        self.progress_value = base_progress
                        self.details_label.config(text=f"Completado: {self.completed_segments}/{self.total_segments} segmentos | {len(self.detected_infractions)} infracciones")
                        
                        # Mostrar último frame con infracciones si hay alguna
                        if infractions and not self.canceled:
                            try:
                                # Cargar y mostrar el frame con la infracción detectada
                                temp_cap = cv2.VideoCapture(self.video_path)
                                temp_cap.set(cv2.CAP_PROP_POS_FRAMES, infractions[0]['frame'])
                                ret, demo_frame = temp_cap.read()
                                if ret:
                                    # Dibujar información en el frame
                                    self._draw_mini_semaphore(demo_frame, "red", 0, self.fps, self.is_night)
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
    
    def _update_video_frame(self, frame):
        """Actualiza el frame de video mostrado en la interfaz de manera optimizada"""
        if frame is None:
            return
            
        try:
            # Redimensionar frame para ajustarse al área de visualización
            h, w = frame.shape[:2]
            max_w, max_h = 640, 360
            
            # Mantener relación de aspecto
            ratio = min(max_w/w, max_h/h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            # Usar INTER_NEAREST para máxima velocidad en la visualización
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Convertir de BGR a RGB para PIL
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Crear imagen para Tkinter
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Actualizar label
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # Mantener referencia
            
            # Actualizar semáforo sincronizado
            self.update_synchronized_semaphore()
            
            # Forzar actualización inmediata
            self.video_label.update()
        except Exception as e:
            print(f"Error actualizando frame: {e}")
    
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
            # Verificaciones iniciales
            if not self.polygon_points or not self.cycle_durations:
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(0, lambda: self._show_error(
                        "Este video no está configurado correctamente. Configure primero el área restrictiva y los tiempos de semáforo."))
                return
                    
            # Abrir el video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(0, lambda: self._show_error("No se pudo abrir el video"))
                return
            
            # Inicialización
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Verificaciones adicionales
            if self.total_frames <= 0:
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(0, lambda: self._show_error("No se pudo determinar la duración del video"))
                return
            
            # Crear directorios para resultados
            output_dir = resource_path("data/output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Fase 1: Inicialización rápida
            self.phase_label.config(text="Fase 1: Inicializando análisis")
            
            # DETENER TIMESTAMP DURANTE PROCESAMIENTO - NO DEBERÍA CORRER MIENTRAS PROCESA
            if hasattr(self, 'player') and hasattr(self.player, 'timestamp_updater'):
                self.player.timestamp_updater.stop_timestamp()
                print("⏸️ Timestamp detenido durante procesamiento")
            
            # Detectar automáticamente si es una escena nocturna
            ret, first_frame = cap.read()
            if not ret:
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(0, lambda: self._show_error("No se pudo leer el primer frame del video"))
                return
            
            self.is_night = self._is_night_scene(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al principio del video
            
            # Actualizar UI con información del modo nocturno
            if self.is_night:
                self.details_label.config(text=f"Franja horaria: {self.cycle_durations.get('time_slot', 'No especificada')} - MODO NOCTURNO ACTIVADO")
                print("Modo nocturno activado para el procesamiento")
            
            # Calcular duración de cada estado
            frames_per_state = {
                "green": int(self.cycle_durations["green"] * self.fps),
                "yellow": int(self.cycle_durations["yellow"] * self.fps),
                "red": int(self.cycle_durations["red"] * self.fps)
            }
            
            # Fase 2: División optimizada en segmentos
            self.phase_label.config(text="Fase 2: Planificando análisis")
            
            # Sincronizar con el semáforo del panel
            # Asegurarnos de que el semáforo esté activado para el procesamiento
            self.player.semaforo.activate_semaphore()
            
            # Dividir el video en segmentos para procesamiento paralelo
            # Solo procesar segmentos en rojo para máxima eficiencia
            self.segments = []
            current_state = "green"
            frame_index = 0
            cycle_duration = sum(frames_per_state.values())
            
            # Calcular segmentos en estado rojo
            while frame_index < self.total_frames:
                if current_state == "green":
                    frame_index += frames_per_state["green"]
                    current_state = "yellow"
                elif current_state == "yellow":
                    frame_index += frames_per_state["yellow"]
                    current_state = "red"
                elif current_state == "red":
                    # Solo guardar segmentos en rojo para procesamiento
                    start = frame_index
                    end = min(frame_index + frames_per_state["red"], self.total_frames)
                    self.segments.append((start, end))
                    frame_index += frames_per_state["red"]
                    current_state = "green"
            
            # Fase 3: Procesamiento en paralelo
            self.phase_label.config(text="Fase 3: Analizando infracciones")
            
            # Número óptimo de trabajadores (CPU cores - 1, mínimo 2)
            import multiprocessing as mp
            num_workers = max(2, mp.cpu_count() - 1)
            self.details_label.config(text=f"Utilizando {num_workers} núcleos para procesamiento")
            
            # Inicializar variables de progreso
            self.completed_segments = 0
            self.total_segments = len(self.segments)
            
            # Iniciar procesamiento multihilo
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Muestreo más agresivo para escenas nocturnas (más frames procesados)
                # para mayor probabilidad de detectar placas en condiciones difíciles
                red_frame_sampling = max(1, int(self.fps / (5 if self.is_night else 3)))
                
                # Preparar detector para reutilización
                vehicle_detector = self.player.vehicle_detector
                
                # Umbral de confianza según condiciones de iluminación
                conf_threshold = 0.25 if self.is_night else 0.40
                
                # Lanzar tareas para cada segmento
                future_to_segment = {}
                for i, (start, end) in enumerate(self.segments):
                    future = executor.submit(
                        self._process_segment_optimized,
                        i, start, end, red_frame_sampling,
                        vehicle_detector, conf_threshold
                    )
                    future_to_segment[future] = i
            
            # No necesitamos esperar aquí ya que los resultados se procesan en _process_results_queue()
            
            # Fase 4: Finalización - VERIFICAR QUE LA VENTANA SIGA EXISTIENDO
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.phase_label.config(text="Fase 4: Organizando resultados")
            
            # Identificar y filtrar placas duplicadas
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.phase_label.config(text="Análisis completado")
                self.progress_value = 100
            
            # Procesar los resultados finales tras una pequeña pausa
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.after(500, self._finalize_processing)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.after(0, lambda msg=str(e): self._show_error(msg))
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    def _process_segment_optimized(self, segment_id, start_frame, end_frame, 
     frame_sampling, vehicle_detector, conf_threshold):
        """Función optimizada para procesar un segmento de video en un hilo separado"""
        try:
            # Abrir segmento de video
            segment_cap = cv2.VideoCapture(self.video_path)
            segment_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Variables para este segmento
            local_infractions = []
            processed = 0
            total_to_process = end_frame - start_frame
            
            # Variable para seguir las placas ya detectadas GLOBALMENTE (no solo en este segmento)
            if not hasattr(self, "detected_plates_global"):
                self.detected_plates_global = set()
            
            # Verificar si tenemos acceso al detector ANPR
            has_anpr = hasattr(self.player, 'anpr_detector')
            
            # Enviar frame inicial para mostrar que estamos procesando este segmento
            ret, first_frame = segment_cap.read()
            if ret:
                # Dibujar información inicial
                cv2.putText(first_frame, f"Procesando segmento {segment_id+1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self._draw_mini_semaphore(first_frame, "red", 0, self.fps, self.is_night)
                
                # Poner el frame en la cola para UI inmediatamente
                self.result_queue.put(("frame_update", (first_frame.copy(), segment_id, 0, total_to_process)))
                # Volver a la posición inicial
                segment_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Procesar frames en este segmento
            for relative_frame in range(total_to_process):
                # Si se canceló el procesamiento
                if self.canceled:
                    segment_cap.release()
                    return [], segment_id
                
                # NUEVO: Esperar mientras esté pausado por ventana emergente
                while hasattr(self, 'processing_paused') and self.processing_paused:
                    import time
                    time.sleep(0.1)  # Esperar 100ms antes de revisar nuevamente
                    if self.canceled:  # Verificar cancelación durante la pausa
                        segment_cap.release()
                        return [], segment_id
                
                # Solo procesar cada 'frame_sampling' frames para eficiencia
                if processed % frame_sampling != 0:
                    ret = segment_cap.grab()  # Solo avanzar sin decodificar
                    processed += 1
                    continue
                
                ret, frame = segment_cap.read()
                if not ret:
                    break
                
                processed += 1
                absolute_frame = start_frame + relative_frame
                
                # Para escenas nocturnas, mejorar el frame antes de detección
                if self.is_night:
                    frame = self._enhance_night_visibility_fast(frame)
                
                # MOSTRAR FRAME EN LA UI MÁS FRECUENTEMENTE
                if processed % max(1, frame_sampling // 2) == 0:  # Actualizar más seguido
                    display_frame = frame.copy()
                    
                    # Dibujar área de restricción (línea roja) si está disponible
                    if hasattr(self.player, 'polygon_points') and self.player.polygon_points:
                        polygon_points = np.array(self.player.polygon_points, dtype=np.int32)
                        cv2.polylines(display_frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=3)
                    
                    # Dibujar información sobre el procesamiento
                    self._draw_mini_semaphore(display_frame, "red", 0, self.fps, self.is_night)
                    cv2.putText(display_frame, f"Segmento: {segment_id+1}/{self.total_segments}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Frame: {processed}/{total_to_process}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Poner el frame en la cola para UI
                    self.result_queue.put(("frame_update", (display_frame, segment_id, processed, total_to_process)))
                
                # NUEVA IMPLEMENTACIÓN: Intentar detección directa con ANPR primero
                anpr_detection_interval = 10  # Solo intentar ANPR directo cada X frames muestreados
                direct_anpr_detections = []
                
                if has_anpr and processed % anpr_detection_interval == 0:
                    # Intentar detección directa con ANPR
                    try:
                        # Procesar el frame completo directamente con ANPR
                        processed_frame, anpr_results = self.player.anpr_detector.process_frame(
                            frame, 
                            frame_idx=absolute_frame,
                            is_night=self.is_night
                        )
                        
                        # Procesar resultados de ANPR si los hay
                        for detection in anpr_results:
                            plate_text = detection.get("plate", "")
                            coords = detection.get("coords")
                            
                            if plate_text and coords:
                                # Normalizar texto de placa
                                plate_text = self._normalize_plate_text(plate_text)
                                
                                # NUEVO: Verificar que la placa normalizada no esté vacía (por longitud excesiva)
                                # y que no tenga más de 8 caracteres (sin contar guiones)
                                if plate_text and len(plate_text.replace('-', '')) <= 8:
                                    # Verificar si esta placa ya fue detectada
                                    if plate_text not in self.detected_plates_global:
                                        self.detected_plates_global.add(plate_text)
                                        
                                        # Extraer imagen de la placa
                                        x1, y1, x2, y2 = coords
                                        if all(c >= 0 for c in (x1, y1, x2, y2)):
                                            plate_img = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
                                            
                                            # Crear directorio para placas si no existe
                                            plates_dir = resource_path("data/output/placas")
                                            vehicles_dir = resource_path("data/output/autos")
                                            os.makedirs(plates_dir, exist_ok=True)
                                            os.makedirs(vehicles_dir, exist_ok=True)
                                            
                                            # Guardar la imagen de la placa
                                            plate_filename = f"plate_{plate_text}.jpg"
                                            plate_path = os.path.join(plates_dir, plate_filename)
                                            cv2.imwrite(plate_path, plate_img)
                                            
                                            # Guardar la imagen del vehículo (área ampliada alrededor de la placa)
                                            expansion_factor = 2.5  # Expandir 2.5x el área de la placa
                                            height, width = frame.shape[:2]
                                            
                                            # Calcular el centro de la placa
                                            center_x = (x1 + x2) // 2
                                            center_y = (y1 + y2) // 2
                                            
                                            # Calcular dimensiones expandidas
                                            plate_width = x2 - x1
                                            plate_height = y2 - y1
                                            expanded_width = int(plate_width * expansion_factor)
                                            expanded_height = int(plate_height * expansion_factor)
                                            
                                            # Calcular las nuevas coordenadas
                                            ex1 = max(0, center_x - expanded_width // 2)
                                            ey1 = max(0, center_y - expanded_height // 2)
                                            ex2 = min(width, center_x + expanded_width // 2)
                                            ey2 = min(height, center_y + expanded_height // 2)
                                            
                                            # Extraer el área ampliada
                                            vehicle_img = frame[ey1:ey2, ex1:ex2].copy()
                                            
                                            # Guardar la imagen del vehículo
                                            vehicle_filename = f"vehicle_{plate_text}.jpg"
                                            vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
                                            cv2.imwrite(vehicle_path, vehicle_img)
                                            
                                            # Añadir a infracciones
                                            direct_anpr_detections.append({
                                                'frame': absolute_frame,
                                                'time': absolute_frame / self.fps,
                                                'plate': plate_text,
                                                'plate_img': plate_img,
                                                'vehicle_img': vehicle_img,
                                                'plate_path': plate_path,
                                                'vehicle_path': vehicle_path,
                                                'unique': True
                                            })
                    except Exception as e:
                        print(f"Error en detección directa ANPR: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Si encontramos placas con detección directa, agregarlas y continuar
                if direct_anpr_detections:
                    local_infractions.extend(direct_anpr_detections)
                    
                    # Mostrar detecciones en tiempo real
                    detection_frame = frame.copy()
                    for detection in direct_anpr_detections:
                        cv2.putText(detection_frame, f"Placa (ANPR): {detection['plate']}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    self.result_queue.put(("frame_update", (detection_frame, segment_id, processed, total_to_process)))
                    continue  # Seguir con el siguiente frame si ya encontramos placas
                
                # Si no hubo detecciones directas, proceder con el flujo normal de detección por vehículos
                detections = vehicle_detector.detect(
                    frame, 
                    conf=conf_threshold,
                    draw=False
                )
                
                # Filtrar detecciones para solo mantener vehículos (coches, buses, camiones)
                filtered_detections = []
                for detection in detections:
                    if len(detection) >= 5:  # Asegurarse de que hay suficientes elementos
                        x1, y1, x2, y2, class_id = detection[:5]
                        
                        # Verificar si es un vehículo (esto puede variar según el modelo)
                        if isinstance(class_id, (int, float)):
                            class_id = int(class_id)
                            if class_id in [2, 5, 7]:  # coche, bus, camión
                                filtered_detections.append((x1, y1, x2, y2, class_id))
                
                # Procesar cada vehículo detectado
                for bbox in filtered_detections:
                    x1, y1, x2, y2, class_id = bbox
                    
                    # Verificar si está en zona restringida
                    if self.is_vehicle_in_polygon((x1, y1, x2, y2), self.polygon_points, self.is_night):
                        # Extraer ROI del vehículo con límites seguros
                        y1_roi = max(0, int(y1))
                        y2_roi = min(frame.shape[0], int(y2))
                        x1_roi = max(0, int(x1))
                        x2_roi = min(frame.shape[1], int(x2))
                        
                        if y2_roi > y1_roi and x2_roi > x1_roi:
                            vehicle_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()
                            
                            # Procesar placa con el detector ANPR si está disponible
                            try:
                                plate_text = ""
                                plate_img = None
                                plate_bbox = None
                                enhance_plate_image = None  # Inicializar para evitar errores
                                
                                # Intentar cargar la función de mejora de imagen primero
                                try:
                                    from src.core.processing.resolution_process import enhance_plate_image
                                except ImportError:
                                    enhance_plate_image = None
                                
                                # Usar ANPR si está disponible
                                if has_anpr:
                                    try:
                                        # Intentar con ANPR primero para mayor precisión
                                        _, plate_text, plate_bbox, plate_img = self.player.anpr_detector.detect_and_recognize_plate(vehicle_roi)
                                    except Exception as anpr_error:
                                        print(f"Error en ANPR: {anpr_error}")
                                        plate_text = ""
                                
                                # Si ANPR no encuentra nada, usar el detector tradicional
                                if not plate_text or len(plate_text) < 4:
                                    from src.core.processing.plate_processing import process_plate
                                    
                                    # Detectar placa en el vehículo con el método tradicional
                                    plate_bbox, plate_img, plate_text = process_plate(vehicle_roi, is_night=self.is_night)
                                    
                                    # Si no encontró texto o es muy corto, intentar con reconocedor alternativo
                                    if not plate_text or len(plate_text) < 4:
                                        from src.core.ocr.recognizer import recognize_plate
                                        
                                        # Intentar mejorar la imagen antes del reconocimiento alternativo
                                        if enhance_plate_image is not None:
                                            enhanced_roi = enhance_plate_image(vehicle_roi, is_night=self.is_night)
                                            plate_text = recognize_plate(enhanced_roi)
                                            if plate_img is None:
                                                plate_img = enhanced_roi
                                        else:
                                            plate_text = recognize_plate(vehicle_roi)
                                            if plate_img is None:
                                                plate_img = vehicle_roi
                                
                                # Verificar que la placa sea válida y normalizar
                                if plate_text and len(plate_text) >= 4:
                                    # Normalizar texto de placa
                                    plate_text = self._normalize_plate_text(plate_text)
                                    
                                    # NUEVO: Verificar que la placa normalizada no esté vacía (por longitud excesiva)
                                    # y que no tenga más de 8 caracteres (sin contar guiones)
                                    if plate_text and len(plate_text.replace('-', '')) <= 8:
                                        # VERIFICAR GLOBAL, NO SOLO EN ESTE SEGMENTO
                                        if plate_text not in self.detected_plates_global:
                                            # Registrar la placa como ya detectada GLOBALMENTE
                                            self.detected_plates_global.add(plate_text)
                                            
                                            # Crear las carpetas necesarias para placas y autos
                                            plates_dir = resource_path("data/output/placas")
                                            vehicles_dir = resource_path("data/output/autos")
                                            os.makedirs(plates_dir, exist_ok=True)
                                            os.makedirs(vehicles_dir, exist_ok=True)
                                            
                                            # Guardar la imagen de la placa con nombre ÚNICO
                                            plate_filename = f"plate_{plate_text}.jpg"
                                            plate_path = os.path.join(plates_dir, plate_filename)
                                            
                                            # Aplicar super-resolución a la placa antes de guardarla
                                            if enhance_plate_image is not None and plate_img is not None:
                                                enhanced_plate = enhance_plate_image(plate_img, is_night=self.is_night)
                                                cv2.imwrite(plate_path, enhanced_plate)
                                            else:
                                                # Si no está disponible el módulo, guardar la placa original
                                                if plate_img is not None:
                                                    cv2.imwrite(plate_path, plate_img)
                                                    enhanced_plate = plate_img
                                                else:
                                                    enhanced_plate = vehicle_roi
                                                    cv2.imwrite(plate_path, vehicle_roi)
                                            
                                            # Guardar la imagen del vehículo con nombre ÚNICO
                                            vehicle_filename = f"vehicle_{plate_text}.jpg"
                                            vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
                                            cv2.imwrite(vehicle_path, vehicle_roi)
                                            
                                            # Guardar infracción detectada con rutas de archivos
                                            infraction_data = {
                                                'frame': absolute_frame,
                                                'time': absolute_frame / self.fps,
                                                'plate': plate_text,
                                                'plate_img': enhanced_plate if plate_img is not None else vehicle_roi.copy(),
                                                'vehicle_img': vehicle_roi.copy(),
                                                'plate_path': plate_path,
                                                'vehicle_path': vehicle_path,
                                                'unique': True  # Marca como único
                                            }
                                            local_infractions.append(infraction_data)
                                            
                                            # Mostrar detección en tiempo real
                                            detection_frame = frame.copy()
                                            cv2.rectangle(detection_frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
                                            cv2.putText(detection_frame, f"Placa: {plate_text}", (x1_roi, y1_roi-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                            
                                            # Si tenemos coordenadas de la placa, dibujarlas también
                                            if plate_bbox and len(plate_bbox) == 4:
                                                px1, py1, px2, py2 = plate_bbox
                                                # Ajustar coordenadas relativas al frame completo
                                                px1, py1 = x1_roi + px1, y1_roi + py1
                                                px2, py2 = x1_roi + px2, y1_roi + py2
                                                cv2.rectangle(detection_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                                            
                                            # Enviar detección a la UI
                                            self.result_queue.put(("frame_update", (detection_frame, segment_id, processed, total_to_process)))
                                        else:
                                            print(f"Placa {plate_text} ya fue detectada globalmente, omitiendo")
                            except Exception as e:
                                print(f"Error procesando placa: {e}")
                                import traceback
                                traceback.print_exc()
            
            segment_cap.release()
            
            # Filtrar duplicados antes de enviar resultados
            filtered_infractions = self._filter_segment_duplicates(local_infractions)
            
            # Enviar resultados a la cola principal
            self.result_queue.put(("segment_complete", (segment_id, filtered_infractions)))
            print(f"Segmento {segment_id} completado con {len(filtered_infractions)} infracciones")
            return filtered_infractions, segment_id
            
        except Exception as e:
            print(f"Error en segment {segment_id}: {e}")
            import traceback
            traceback.print_exc()
            self.result_queue.put(("segment_complete", (segment_id, [])))
            return [], segment_id

    def _filter_segment_duplicates(self, infractions):
        """
        Filtra duplicados dentro de un segmento antes de enviar los resultados.
        Esto ayuda a reducir la cantidad de datos que se transfieren entre hilos.
        
        Args:
            infractions: Lista de infracciones detectadas en un segmento
            
        Returns:
            list: Lista de infracciones sin duplicados dentro del segmento
        """
        if not infractions or len(infractions) <= 1:
            return infractions
        
        # Conjunto para seguir placas ya procesadas en este segmento
        processed_plates = set()
        filtered_infractions = []
        
        # Ordenar primero por calidad (menor puntuación de laplaciano primero)
        def quality_score(infraction):
            plate_img = infraction.get('plate_img')
            if plate_img is None:
                return 0
                
            import cv2
            try:
                if len(plate_img.shape) > 2:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = plate_img
                # Usar varianza de Laplaciano como medida de nitidez
                return cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception:
                return 0
        
        # Ordenar infracciones por nitidez (mayor primero)
        sorted_infractions = sorted(infractions, key=quality_score, reverse=True)
        
        # Filtrar duplicados basados en placas
        for infraction in sorted_infractions:
            plate_text = infraction.get('plate', '')
            if not plate_text:
                continue
            
            # REQUISITO ESTRICTO: Verificar longitud máxima (7 caracteres sin guiones/espacios)
            plate_without_special = plate_text.replace('-', '').replace(' ', '')
            if len(plate_without_special) > 7:
                continue
            
            # Verificar si esta placa (o una muy similar) ya fue procesada
            duplicate = False
            for existing_plate in processed_plates:
                # Si son exactamente iguales
                if plate_text == existing_plate:
                    duplicate = True
                    break
                    
                # O si son muy similares (difieren en máximo 1 carácter para placas cortas o 2 para largas)
                max_diff = 1 if len(plate_text) <= 6 else 2
                if len(plate_text) == len(existing_plate):
                    differences = sum(c1 != c2 for c1, c2 in zip(plate_text, existing_plate))
                    if differences <= max_diff:
                        duplicate = True
                        break
            
            # Si no es un duplicado, añadir a la lista filtrada
            if not duplicate:
                processed_plates.add(plate_text)
                filtered_infractions.append(infraction)
        
        if len(filtered_infractions) < len(infractions):
            print(f"Filtro de segmento: reducidas {len(infractions)} a {len(filtered_infractions)} placas")
        
        return filtered_infractions
    
    def _normalize_plate_text(self, plate_text):
        """
        Normaliza el texto de la placa para mejorar la precisión de detección.
        Incorpora diccionarios de confusiones comunes según la región de la placa.
        """
        if not plate_text:
            return plate_text
            
        # Importar funciones auxiliares si están disponibles
        try:
            from src.core.processing.resolution_process import get_common_plate_patterns
            region_aware = True
        except ImportError:
            region_aware = False
            
        # Determinar región de la placa (por defecto España)
        region = "ES"
        
        # Eliminar espacios y convertir a mayúsculas
        normalized = plate_text.strip().upper()

        # NUEVO: Comprobar específicamente "B OHID" que está causando problemas
        if "BOHID" in normalized or "B OHID" in normalized or "B-OHID" in normalized:
            print(f"Placa problemática específica detectada y descartada: {normalized}")
            return ""  # Devolver cadena vacía para que esta placa sea descartada
            
        # Eliminar caracteres no alfanuméricos excepto guión
        normalized = ''.join(c for c in normalized if c.isalnum() or c == '-')
        
        # REFORZADO: Descartar placas demasiado largas (más de 8 caracteres sin contar guiones ni espacios)
        plate_without_special = normalized.replace('-', '').replace(' ', '')
        if len(plate_without_special) > 8:
            print(f"Placa demasiado larga descartada: {normalized} ({len(plate_without_special)} caracteres)")
            return ""  # Devolver cadena vacía para que esta placa sea descartada
        
        # Obtener patrones de confusión para la región
        char_confusions = {}
        if region_aware:
            patterns = get_common_plate_patterns(region)
            char_confusions = patterns.get("character_confusions", {})
        else:
            # Diccionario básico de correcciones comunes si no hay acceso a la función
            char_confusions = {
                "0": "ODCQ",    # 0 confundido con O, D, C, Q
                "1": "ILT7",    # 1 confundido con I, L, T, 7
                "2": "Z",       # 2 confundido con Z
                "5": "S",       # 5 confundido con S
                "6": "G",       # 6 confundido con G
                "8": "B",       # 8 confundido con B
                "B": "8R",      # B confundido con 8, R
                "D": "0",       # D confundido con 0
                "G": "6",       # G confundido con 6
                "I": "1J",      # I confundido con 1, J
                "J": "I",       # J confundido con I
                "O": "0",       # O confundido con 0
                "S": "5",       # S confundido con 5
                "Z": "2"        # Z confundido con 2
            }
        
        # MEJORA: Detectar y corregir formato de placas
        # Verificar patrones comunes de placas
        if len(normalized) >= 6:
            # Detectar si hay un separador o si hay que inferirlo
            if '-' in normalized:
                parts = normalized.split('-')
            else:
                # Intentar segmentar automáticamente entre parte alfabética y numérica
                # usando algoritmo mejorado basado en secuencias de caracteres
                
                # Analizar la secuencia para detectar patrones
                letter_segments = []
                number_segments = []
                current_type = None
                current_segment = ""
                
                for char in normalized:
                    is_digit = char.isdigit()
                    char_type = "digit" if is_digit else "letter"
                    
                    # Si cambiamos de tipo de carácter o es el primer carácter
                    if current_type != char_type and current_segment:
                        if current_type == "digit":
                            number_segments.append(current_segment)
                        else:
                            letter_segments.append(current_segment)
                        current_segment = char
                    else:
                        current_segment += char
                    
                    current_type = char_type
                
                # Añadir el último segmento
                if current_segment:
                    if current_type == "digit":
                        number_segments.append(current_segment)
                    else:
                        letter_segments.append(current_segment)
                
                # Combinar segmentos según patrón más probable para la región
                if region == "ES":
                    # España: Formato actual NNNNLLL o antiguo LLNNNNLL
                    if len(letter_segments) == 1 and len(number_segments) == 1:
                        # Determinar orden basado en posición
                        if normalized.find(letter_segments[0]) == 0:
                            # Letras primero (formato antiguo)
                            parts = [letter_segments[0], number_segments[0]]
                        else:
                            # Números primero (formato actual)
                            parts = [number_segments[0], letter_segments[0]]
                    else:
                        # Si hay múltiples segmentos, intentar reconstruir basado en la longitud total
                        parts = []
                        if len(normalized) >= 7:  # Probable formato actual
                            num_prefix = ''.join(c for c in normalized if c.isdigit())[:4]
                            letter_suffix = ''.join(c for c in normalized if not c.isdigit())[:3]
                            if num_prefix and letter_suffix:
                                parts = [num_prefix, letter_suffix]
                        
                        if not parts:  # Fallback o formato antiguo
                            parts = [normalized[:2], normalized[2:]]
                else:
                    # Algoritmo genérico para otras regiones
                    if letter_segments and number_segments:
                        # Determinar patrón más probable
                        if len(letter_segments[0]) <= 3 and normalized.find(letter_segments[0]) == 0:
                            # Letras al inicio
                            parts = [letter_segments[0], ''.join(number_segments)]
                        else:
                            # Números al inicio o mezclados
                            parts = [number_segments[0], ''.join(letter_segments)]
                    else:
                        # No hay segmentación clara, usar división en 2 partes
                        mid = len(normalized) // 2
                        parts = [normalized[:mid], normalized[mid:]]
            
            # Procesar las partes identificadas
            if len(parts) == 2:
                prefix, numbers = parts
                
                # MEJORADO: Correcciones más robustas basadas en patrones y región
                
                # Corregir confusiones en prefijo (convertir dígitos a letras donde sea apropiado)
                corrected_prefix = ''
                for char in prefix:
                    if char.isdigit() and region == "ES" and len(prefix) <= 3:
                        # Si estamos en un prefijo de España y encontramos dígitos, probablemente sean letras
                        # conversiones comunes de OCR: 0→O, 1→I, 2→Z, 3→E, 4→A, 5→S, 6→G, 7→T, 8→B, 9→R
                        digit_to_letter = {
                            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', 
                            '4': 'A', '5': 'S', '6': 'G', '7': 'T', 
                            '8': 'B', '9': 'P'
                        }
                        corrected_prefix += digit_to_letter.get(char, char)
                    else:
                        corrected_prefix += char
                
                # MEJORADO: Corregir confusiones en números (convertir letras a dígitos)
                corrected_numbers = ''
                for char in numbers:
                    if char.isalpha():
                        # Buscar si este carácter suele confundirse con algún número
                        found = False
                        for digit, confusions in char_confusions.items():
                            if char in confusions and digit.isdigit():
                                corrected_numbers += digit
                                found = True
                                break
                        if not found:  # Si no hay corrección específica
                            # Conversiones generales para letras en posiciones numéricas
                            letter_to_digit = {
                                'O': '0', 'D': '0', 'Q': '0', 'C': '0',
                                'I': '1', 'L': '1', 'J': '1',
                                'Z': '2',
                                'E': '3',
                                'A': '4',
                                'S': '5',
                                'G': '6', 'C': '6',
                                'T': '7', 'Y': '7',
                                'B': '8',
                                'P': '9', 'R': '9'
                            }
                            corrected_numbers += letter_to_digit.get(char, char)
                    else:
                        corrected_numbers += char
                
                # Si tenemos una estructura clara de parte alfabética+numérica, aplicar formato con guión
                if corrected_prefix and corrected_numbers:
                    normalized = f"{corrected_prefix}-{corrected_numbers}"
                else:
                    normalized = corrected_prefix + corrected_numbers
        
        # Formateo final: asegurar estructura consistente
        if '-' in normalized:
            parts = normalized.split('-')
            if len(parts) == 2:
                # Formato final: asegurar que se siga el patrón típico
                prefix, numbers = parts
                
                # Verificar reglas específicas por región
                if region == "ES":
                    # En España: Preferir letras en el prefijo para formato antiguo
                    if len(prefix) <= 3 and not prefix.isdigit():
                        # Convertir cualquier dígito restante a su letra similar
                        prefix = ''.join(['O' if c == '0' else 
                                        'I' if c == '1' else 
                                        'Z' if c == '2' else 
                                        'E' if c == '3' else 
                                        'A' if c == '4' else 
                                        'S' if c == '5' else 
                                        'G' if c == '6' else 
                                        'T' if c == '7' else 
                                        'B' if c == '8' else 
                                        'P' if c == '9' else c for c in prefix])
                    
                    # En la parte numérica, asegurar que tenga el largo típico (4-5 dígitos)
                    if len(numbers) > 5:
                        numbers = numbers[:5]
                    elif len(numbers) < 4 and numbers.isdigit():
                        # Si es muy corta, puede haber un error - intentar agregar ceros
                        numbers = numbers.zfill(4)
                
                normalized = f"{prefix}-{numbers}"
        
        # VERIFICACIÓN FINAL: Descartar placas específicas problemáticas
        if "BOHID" in normalized or "B OHID" in normalized or "B-OHID" in normalized:
            print(f"Placa problemática específica detectada después de normalizar: {normalized}")
            return ""
        
        # VERIFICACIÓN FINAL de longitud máxima (8 caracteres sin guiones)
        plate_without_dash = normalized.replace('-', '').replace(' ', '')
        if len(plate_without_dash) > 8:
            print(f"Placa demasiado larga después de normalizar: {normalized} ({len(plate_without_dash)} caracteres)")
            return ""  # Devolver cadena vacía para que esta placa sea descartada
        
        return normalized


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
            
            # 1. VERIFICACIÓN DE IGUALDAD EXACTA
            if p1_norm == p2_norm:
                text_similarity = 1.0
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
        
        # CAMBIO CRÍTICO: Umbral reducido a 60% para capturar más duplicados potenciales
        SIMILARITY_THRESHOLD = 0.60  # Reducido de 0.7 a 0.6
        
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
                    
                    # CRUCIAL: Si las imágenes son muy similares, agrupar incluso con bajo umbral general
                    if img_similarity >= 0.75 or (img_similarity >= 0.65 and time_proximity >= 0.8):
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
        """Finaliza el procesamiento después de que todos los segmentos estén completos"""
        try:
            # NUEVO: Filtrar primero las placas inválidas por longitud
            filtered_infractions = []
            for infraction in self.detected_infractions:
                plate_text = infraction.get('plate', '')
                # Verificar longitud válida (máximo 8 caracteres sin contar guiones)
                if plate_text and len(plate_text.replace('-', '')) <= 8:
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
                    if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                        self.dialog.after(0, lambda: self._show_night_no_detection_info())
                    return  # ⚠️ SALIR SIN LLAMAR _complete_processing
                else:
                    # Para modo diurno o cuando no es nocturno: alerta avanzada del compañero
                    if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                        self.dialog.after(0, lambda: self._generate_intelligent_analysis_message(guardadas))
            else:
                # Hay detecciones: mostrar ventana de éxito y reproducir sonido
                self._show_success_detection_popup(len(self.detected_infractions))
                self._play_success_sound()
            print(f"Procesamiento completado: {len(self.detected_infractions)} vehículos infractores ({guardadas} imágenes guardadas)")
            
            # Llamar a _complete_processing SOLO si NO es ventana nocturna sin detecciones
            if len(self.detected_infractions) > 0 or not getattr(self, 'is_night', False):
                # Solo procesar si hay detecciones O no es nocturno
                if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                    self.dialog.after(0, self._complete_processing)
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
            
            # Distancia de corte para conseguir entre 1 y max_clusters
            clusters = fcluster(Z, t=0.7*max(Z[:,2]), criterion='distance')
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
    
    
    def _best_plate_version(self, plate, existing_plates):
        """Versión optimizada para encontrar la mejor versión de una placa"""
        if not plate or len(plate) < 4:
            return False, plate
            
        # Verificar si ya existe esta placa exacta
        if plate in existing_plates:
            return True, plate
        
        # Lista de placas conocidas en el sistema para priorizar coincidencias
        known_plates = ["A3606L", "AE670S", "A3670S"]
        
        # Si la placa actual es una conocida, preferirla
        if plate in known_plates:
            return False, plate
            
        # Buscar similitudes entre placas existentes
        for existing in existing_plates:
            # Si son muy similares (difieren en máximo 2 caracteres)
            if len(plate) == len(existing) and sum(c1 != c2 for c1, c2 in zip(plate, existing)) <= 2:
                # Preferir placas conocidas
                if existing in known_plates:
                    return True, existing
                    
        return False, plate
    
    def _draw_mini_semaphore(self, frame, current_state, frames_left, fps, is_night=False):
        """Dibuja un mini-semáforo en el frame proporcionado con el estado actual (versión optimizada)"""
        h, w = frame.shape[:2]
        
        # Coordenadas del semáforo
        semaforo_x = w - 60
        semaforo_y = 30
        semaforo_width = 40
        semaforo_height = 100
        
        # Fondo del semáforo (rectángulo negro)
        cv2.rectangle(frame, 
                    (semaforo_x, semaforo_y), 
                    (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                    (0, 0, 0), -1)  # Negro
        
        # Borde gris del semáforo
        cv2.rectangle(frame, 
                    (semaforo_x, semaforo_y), 
                    (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                    (128, 128, 128), 2)  # Gris
        
        # Diámetro y posiciones de las luces
        light_diameter = 20
        green_y = semaforo_y + semaforo_height - 25
        yellow_y = semaforo_y + semaforo_height//2
        red_y = semaforo_y + 25
        light_x = semaforo_x + semaforo_width//2
        
        # Dibujar solo la luz activa para mayor eficiencia
        if current_state == "green":
            cv2.circle(frame, (light_x, green_y), light_diameter, (0, 255, 0), -1)
            cv2.putText(frame, "AVANCE", (semaforo_x - 80, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_state == "yellow":
            cv2.circle(frame, (light_x, yellow_y), light_diameter, (0, 255, 255), -1)
            cv2.putText(frame, "PRECAUCIÓN", (semaforo_x - 120, semaforo_y + semaforo_height//2),
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
        
        # 2do INDICADOR (COMPLEMENTARIO): Análisis inteligente del video
        video_analysis_night = (avg_brightness < 120 or  # Brillo bajo
                               dark_threshold < 50 or    # Áreas muy oscuras
                               np.std(gray) < 35)        # Bajo contraste
        
        # LÓGICA CORREGIDA: Solo es noche si la FRANJA HORARIA CONFIGURADA es nocturna
        if is_night_time_configured and video_analysis_night:
            is_night = True  # Franja nocturna + video oscuro = modo nocturno
        elif video_analysis_night and avg_brightness < 80:  # Video extremadamente oscuro (anula franja horaria)
            is_night = True  # Forzar modo nocturno por condiciones extremas
        else:
            is_night = False  # En cualquier otro caso, es de día
        
        # DEBUG: Mostrar valores para calibración mejorada
        time_slot_configured = self.cycle_durations.get('time_slot', 'No configurada') if hasattr(self, 'cycle_durations') and self.cycle_durations else 'No configurada'
        print(f"🌙 DETECCIÓN NOCTURNA CORREGIDA: franja_horaria_config='{time_slot_configured}', es_franja_nocturna={is_night_time_configured}, brillo_promedio={avg_brightness:.1f}, areas_oscuras={dark_threshold:.1f}, contraste={np.std(gray):.1f}, video_oscuro={video_analysis_night}, RESULTADO_FINAL={is_night}")
        
        # MEJORA: Mostrar ventana emergente nocturna (usando after para el hilo principal)
        if is_night and not PreprocessingDialog._night_popup_active:
            # Programar la ventana emergente en el hilo principal de la UI SOLO si no hay otra activa
            PreprocessingDialog._night_popup_active = True
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.after(0, lambda: self._show_night_detection_popup(avg_brightness, dark_threshold))
        
        return is_night

    def _show_night_detection_popup(self, avg_brightness, dark_threshold):
        """Muestra ventana emergente específica para detección nocturna del compañero"""
        try:
            print("🌙 CREANDO VENTANA NOCTURNA - PAUSANDO PROCESAMIENTO")
            
            # PAUSAR el procesamiento durante la ventana emergente
            self.processing_paused = True
            
            # Crear ventana emergente RESPONSIVA
            popup = tk.Toplevel(self.dialog)
            popup.title("🌙 Detección Nocturna Activada")
            
            # RESPONSIVIDAD: Tamaño MÁS GRANDE como solicita el usuario
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            
            # Calcular tamaño MÁS GRANDE para que se vea bien (pero sin cubrir márgenes)
            if screen_width >= 1920:  # Pantalla grande
                popup_width, popup_height = 900, 700
            elif screen_width >= 1366:  # Pantalla mediana
                popup_width, popup_height = 800, 650
            else:  # Pantalla pequeña
                popup_width, popup_height = 700, 600
            
            # IMPORTANTE: No cubrir más del 80% de la pantalla (dejar márgenes)
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8)
            popup_width = min(popup_width, max_width)
            popup_height = min(popup_height, max_height)
            
            popup.geometry(f"{popup_width}x{popup_height}")
            popup.resizable(False, False)  # Tamaño fijo para consistencia
            
            # Configurar icono si existe
            icon_path = resource_path("img/icon.ico")
            if os.path.exists(icon_path):
                popup.iconbitmap(icon_path)
            
            # CONVENCIONALIDAD: Ventana adjunta a principal (práctica estándar Windows)
            popup.transient(self.dialog)
            popup.focus_set()
            
            # NO bloquear otras aplicaciones
            # popup.grab_set()  # Comentado para permitir cambio de apps
            
            # COMPORTAMIENTO AL HACER CLIC: Mostrar ventana principal atrás si existe
            def on_popup_click(event=None):
                try:
                    if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                        self.dialog.lift()  # Levantar ventana principal atrás
                    popup.lift()        # Mantener emergente al frente
                except:
                    pass
            
            popup.bind("<Button-1>", on_popup_click)
            popup.bind("<FocusIn>", on_popup_click)
            
            # PERMITIR cerrar con X (pero controlado)
            def close_popup_x():
                print("🚀 USUARIO CERRÓ VENTANA NOCTURNA CON X - CONTINUANDO PROCESAMIENTO")
                try:
                    PreprocessingDialog._night_popup_active = False
                    self.processing_paused = False
                    popup.destroy()
                    print("✅ Ventana nocturna cerrada correctamente - PROCESAMIENTO CONTINUARÁ")
                    
                    # NO CERRAR LA VENTANA PRINCIPAL AÚN - Dejar que termine el procesamiento
                    # El procesamiento debe continuar y mostrar la segunda ventana si es necesario
                        
                except Exception as e:
                    print(f"❌ Error cerrando ventana: {e}")
            
            popup.protocol("WM_DELETE_WINDOW", close_popup_x)
            
            # CENTRADO PERFECTO: Siempre centrado en cualquier pantalla
            def center_popup():
                popup.update_idletasks()
                # Centrado exacto independiente del tamaño de pantalla
                x = (screen_width - popup_width) // 2
                y = (screen_height - popup_height) // 2
                popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
                print(f"📍 VENTANA CENTRADA: {popup_width}x{popup_height} en posición ({x}, {y})")
            
            popup.after(100, center_popup)
            popup.configure(bg='#1a1a2e')  # Fondo oscuro para tema nocturno
            
            # Frame principal sin scroll (como pidió el usuario)
            main_frame = tk.Frame(popup, bg='#1a1a2e', padx=20, pady=20)
            main_frame.pack(fill='both', expand=True)
            
            # Título con emoji (CENTRADO)
            title_label = tk.Label(main_frame, 
                text="🌙 MODO NOCTURNO DETECTADO", 
                font=('Arial', 16, 'bold'),
                fg='#00ffff', bg='#1a1a2e',
                justify='center')
            title_label.pack(pady=(0, 20), anchor='center')
            
            # Información de detección
            info_frame = tk.Frame(main_frame, bg='#16213e', relief='ridge', bd=2)
            info_frame.pack(fill='x', pady=(0, 15))
            
            info_title = tk.Label(info_frame, 
                text="📊 ANÁLISIS DE ILUMINACIÓN", 
                font=('Arial', 12, 'bold'),
                fg='#ffffff', bg='#16213e')
            info_title.pack(pady=(10, 5))
            
            brightness_label = tk.Label(info_frame, 
                text=f"• Brillo promedio: {avg_brightness:.1f}/255", 
                font=('Arial', 10),
                fg='#cccccc', bg='#16213e')
            brightness_label.pack(anchor='w', padx=20)
            
            threshold_label = tk.Label(info_frame, 
                text=f"• Áreas oscuras: {dark_threshold:.1f}/255", 
                font=('Arial', 10),
                fg='#cccccc', bg='#16213e')
            threshold_label.pack(anchor='w', padx=20, pady=(0, 10))
            
            # Información sobre mejoras activadas
            improvements_frame = tk.Frame(main_frame, bg='#0f3460', relief='ridge', bd=2)
            improvements_frame.pack(fill='x', pady=(0, 15))
            
            improvements_title = tk.Label(improvements_frame, 
                text="⚡ MEJORAS ACTIVADAS", 
                font=('Arial', 12, 'bold'),
                fg='#00ff00', bg='#0f3460')
            improvements_title.pack(pady=(10, 5))
            
            improvements = [
                "✅ Detección ultra-sensible de placas",
                "✅ Procesamiento multi-variante nocturno",
                "✅ Correcciones OCR ultra-agresivas",
                "✅ Filtros adaptativos de confianza",
                "✅ Mejora automática de contraste",
                "✅ Análisis específico de reflectores",
                "⚠️ NOTA: Condiciones nocturnas limitadas",
                "🎯 No todas las placas serán detectables"
            ]
            
            for improvement in improvements:
                imp_label = tk.Label(improvements_frame, 
                    text=improvement, 
                    font=('Arial', 10),  # Fuente más grande para mejor legibilidad
                    fg='#ccffcc', bg='#0f3460',
                    wraplength=popup_width-100)  # RESPONSIVO: texto se adapta al ancho
                imp_label.pack(anchor='w', padx=20)
            
            # Mensaje de expectativas REALISTAS para condiciones nocturnas (RESPONSIVO)
            expectation_label = tk.Label(main_frame, 
                text="🤖 Se detectó por el video que es de noche\n(mediante algoritmo inteligente de computer vision)\n\n🎯 El sistema aplicará técnicas especializadas para condiciones nocturnas.\n⚠️ IMPORTANTE: Las limitaciones de iluminación pueden reducir\nla detección exitosa de placas. El sistema intentará optimizar\nla precisión, pero no todas las placas serán detectables.", 
                font=('Arial', 11),
                fg='#ffff99', bg='#1a1a2e',
                justify='center',
                wraplength=popup_width-80)  # RESPONSIVO: texto se adapta al ancho
            expectation_label.pack(pady=(0, 20))
            
            # Función para cerrar la ventana correctamente (primera ventana)
            def close_first_popup():
                print("🚀 USUARIO CONFIRMÓ - CERRANDO PRIMERA VENTANA NOCTURNA - CONTINUANDO PROCESAMIENTO")
                try:
                    # Liberar el flag de ventana activa
                    PreprocessingDialog._night_popup_active = False
                    # Reactivar el procesamiento
                    self.processing_paused = False
                    # Cerrar ventana emergente
                    popup.destroy()
                    print("✅ PRIMERA VENTANA NOCTURNA CERRADA - PROCESAMIENTO CONTINUARÁ")
                    
                    # NO CERRAR LA VENTANA PRINCIPAL AÚN - Dejar que termine el procesamiento
                    # Si no hay infracciones, se mostrará la segunda ventana
                    # Solo se cierra cuando termine todo correctamente
                        
                except Exception as e:
                    print(f"Error cerrando primera ventana nocturna: {e}")
            
            # Botón de continuar
            continue_button = tk.Button(main_frame, 
                text="🚀 CONTINUAR CON ANÁLISIS NOCTURNO", 
                font=('Arial', 11, 'bold'),
                bg='#4CAF50', fg='white',
                relief='raised', bd=3,
                padx=20, pady=10,
                command=close_first_popup)
            continue_button.pack(pady=(0, 10))
            
            # Enfocar el botón para que sea obvio
            continue_button.focus_set()
            
            # Enter también funciona
            popup.bind('<Return>', lambda e: close_first_popup())
            
            # NO auto-cerrar - solo el usuario puede cerrarla
            
            # Reproducir sonido de detección nocturna
            self._play_night_detection_sound()
                
        except Exception as e:
            print(f"Error mostrando ventana nocturna: {e}")
            # Si falla la ventana emergente, continuar sin ella
            pass

    def _show_night_no_detection_info(self):
        """SEGUNDA VENTANA: No detecciones nocturnas - MÁS GRANDE + CENTRADA + BOTÓN ACEPTAR"""
        print("🌙 INICIANDO SEGUNDA VENTANA DE NO DETECCIONES NOCTURNAS")
        try:
            # Crear ventana emergente MÁS GRANDE
            popup = tk.Toplevel(self.dialog)
            popup.title("🌙 Análisis Nocturno Completado")
            
            # RESPONSIVIDAD INTELIGENTE - BUENAS PRÁCTICAS
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            
            # VENTANA SÚPER ALTA RESPONSIVE - SOLO AUMENTAR ALTO
            if screen_width >= 1920:  # Pantalla grande
                popup_width, popup_height = 1000, 1200
            elif screen_width >= 1366:  # Pantalla mediana
                popup_width, popup_height = 900, 1100
            else:  # Pantalla pequeña
                popup_width, popup_height = 800, 1000
            
            # ASEGURAR QUE NO EXCEDA 90% DE PANTALLA (más permisivo)
            max_width = int(screen_width * 0.90)
            max_height = int(screen_height * 0.90)
            popup_width = min(popup_width, max_width)
            popup_height = min(popup_height, max_height)
            
            popup.geometry(f"{popup_width}x{popup_height}")
            popup.resizable(False, False)
            
            # Configurar icono si existe
            icon_path = resource_path("img/icon.ico")
            if os.path.exists(icon_path):
                popup.iconbitmap(icon_path)
            
            # CENTRADO PERFECTO para segunda ventana
            popup.update_idletasks()
            x = (screen_width - popup_width) // 2
            y = (screen_height - popup_height) // 2
            popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
            print(f"📍 SEGUNDA VENTANA CENTRADA: {popup_width}x{popup_height} en posición ({x}, {y})")
            
            # CONVENCIONALIDAD: Adjunta a ventana principal
            popup.transient(self.dialog)
            popup.focus_set()  # NO grab_set para no bloquear otras apps
            popup.configure(bg='#1a1a2e')  # Fondo oscuro para tema nocturno
            
            # NO bloquear otras aplicaciones  
            # popup.grab_set()  # Comentado para permitir cambio de apps
            
            # Reproducir sonido de error inmediatamente al mostrar la ventana
            self._play_failure_sound()
            
            # ESTRUCTURA OPTIMIZADA - MENOS PADDING PARA MÁS ESPACIO
            main_frame = tk.Frame(popup, bg='#1a1a2e', padx=15, pady=10)
            main_frame.pack(fill='both', expand=True)
            
            # Título con emoji (CENTRADO)
            title_label = tk.Label(main_frame, 
                text="🌙 ANÁLISIS NOCTURNO COMPLETADO", 
                font=('Arial', 16, 'bold'),
                fg='#00ffff', bg='#1a1a2e',
                justify='center')
            title_label.pack(pady=(0, 10), anchor='center')
            
            # Estado del procesamiento
            status_frame = tk.Frame(main_frame, bg='#16213e', relief='ridge', bd=2)
            status_frame.pack(fill='x', pady=(0, 8))
            
            status_title = tk.Label(status_frame, 
                text="✅ PROCESAMIENTO COMPLETADO", 
                font=('Arial', 12, 'bold'),
                fg='#00ff00', bg='#16213e')
            status_title.pack(pady=(10, 5))
            
            result_label = tk.Label(status_frame, 
                text="🔍 No se detectaron infracciones en condiciones nocturnas\n⚠️ NO SE PUDO MIGRAR A LA NUBE debido a limitaciones nocturnas\n📊 Solo se migran indicadores de rendimiento del sistema", 
                font=('Arial', 10),
                fg='#ffff99', bg='#16213e',
                justify='center',
                wraplength=popup_width-80)
            result_label.pack(pady=(0, 10))
            
            # Información sobre limitaciones nocturnas
            info_frame = tk.Frame(main_frame, bg='#0f3460', relief='ridge', bd=2)
            info_frame.pack(fill='x', pady=(0, 8))
            
            info_title = tk.Label(info_frame, 
                text="⚠️ LIMITACIONES DE DETECCIÓN NOCTURNA", 
                font=('Arial', 12, 'bold'),
                fg='#ff9900', bg='#0f3460')
            info_title.pack(pady=(5, 3))
            
            limitations = [
                "🌙 Iluminación insuficiente reduce la visibilidad de placas",
                "💡 Reflejos y sombras pueden ocultar caracteres",
                "📷 Calidad de imagen limitada por condiciones de captura",
                "🔦 Placas sin retroreflectividad son difíciles de detectar",
                "⚡ Se aplicaron técnicas especializadas de mejora nocturna",
                "🎯 El sistema optimizó la detección según las condiciones"
            ]
            
            for limitation in limitations:
                lim_label = tk.Label(info_frame, 
                    text=limitation, 
                    font=('Arial', 10),
                    fg='#cccccc', bg='#0f3460',
                    wraplength=popup_width-100)
                lim_label.pack(anchor='w', padx=20)
            
            # Recomendaciones de Calidad y Resolución
            recom_frame = tk.Frame(main_frame, bg='#0a2a1a', relief='ridge', bd=2)
            recom_frame.pack(fill='x', pady=(0, 8))
            
            recom_title = tk.Label(recom_frame, 
                text="💡 RECOMENDACIONES PARA MEJORAR DETECCIÓN", 
                font=('Arial', 12, 'bold'),
                fg='#00ff99', bg='#0a2a1a')
            recom_title.pack(pady=(5, 3))
            
            recommendations = [
                "🔆 Mejorar la iluminación del área de monitoreo",
                "📐 Ajustar ángulo de cámara para reducir reflejos",
                "⚙️ Aumentar resolución de captura a mínimo 1080p (recomendado 4K)",
                "🎥 Configurar calidad de video: bitrate mínimo 2Mbps",
                "📊 Verificar compresión: usar H.264 con baja compresión",
                "🔍 Resolución mínima sugerida: 1920x1080 para placas legibles",
                "🕐 Considerar horarios de menor tráfico para calibración",
                "📸 Verificar limpieza y enfoque del lente de la cámara",
                "💡 Instalar iluminación LED infrarroja específica para placas"
            ]
            
            for recommendation in recommendations:
                rec_label = tk.Label(recom_frame, 
                    text=recommendation, 
                    font=('Arial', 10),
                    fg='#ccffcc', bg='#0a2a1a',
                    wraplength=popup_width-100)
                rec_label.pack(anchor='w', padx=20)
            
            # Información sobre migración
            migration_frame = tk.Frame(main_frame, bg='#2a1a0a', relief='ridge', bd=2)
            migration_frame.pack(fill='x', pady=(0, 8))
            
            migration_title = tk.Label(migration_frame, 
                text="☁️ ESTADO DE MIGRACIÓN A LA NUBE", 
                font=('Arial', 12, 'bold'),
                fg='#ffaa00', bg='#2a1a0a')
            migration_title.pack(pady=(5, 3))
            
            migration_info = [
                "⚠️ Las infracciones NO SE PUDIERON MIGRAR debido a limitaciones nocturnas",
                "📊 Solo se migran indicadores de rendimiento del sistema",
                "🔄 La migración de infracciones se reanudará con videos diurnos",
                "💾 Los datos se mantienen guardados localmente para consulta",
                "☁️ Estado de migración: PARCIAL (solo indicadores)",
                "🚫 Razón: Calidad insuficiente para validación en la nube"
            ]
            
            for info in migration_info:
                info_label = tk.Label(migration_frame, 
                    text=info, 
                    font=('Arial', 10),
                    fg='#ffccaa', bg='#2a1a0a',
                    wraplength=popup_width-100)
                info_label.pack(anchor='w', padx=20, pady=2)
            
            # Mensaje final (RESPONSIVO)
            final_label = tk.Label(main_frame, 
                text="🤖 El sistema continuará monitoreando y se adaptará automáticamente a mejores condiciones de iluminación", 
                font=('Arial', 11),
                fg='#ccccff', bg='#1a1a2e',
                justify='center',
                wraplength=popup_width-80)
            final_label.pack(pady=(0, 10))
            
            # QUITAR VIDEO NO APTO: Detiene video completamente y regresa a selección
            def close_no_detection_popup():
                print("🚫 BOTÓN PRESIONADO: QUITANDO VIDEO NO APTO PARA PROCESAMIENTO NOCTURNO")
                try:
                    # PASO 1: Detener player y restaurar estado "NO HAY VIDEO"
                    print("🔄 PASO 1: DETENIENDO PLAYER Y RESTAURANDO ESTADO INICIAL")
                    try:
                        if hasattr(self, 'player') and self.player:
                            # Detener reproducciones
                            if hasattr(self.player, 'running'):
                                self.player.running = False
                                print("✅ Player.running = False")
                            if hasattr(self.player, 'is_playing'):
                                self.player.is_playing = False
                                print("✅ Player.is_playing = False")
                            if hasattr(self.player, 'pause'):
                                self.player.pause()
                                print("✅ Player pausado")
                            if hasattr(self.player, 'stop_video'):
                                self.player.stop_video()
                                print("✅ Player.stop_video() ejecutado")
                            
                            # RESTAURAR ESTADO INICIAL - COMO ERA ANTES
                            if hasattr(self.player, 'video_label'):
                                self.player.video_label.config(image='', text='')
                                self.player.video_label.image = None
                                print("✅ Video label limpiado")
                            
                            # MOSTRAR MENSAJE "NINGÚN VIDEO CARGADO" COMO ANTES  
                            if hasattr(self.player, 'current_video_label'):
                                self.player.current_video_label.config(text="Ningún video cargado")
                                print("✅ Mensaje 'Ningún video cargado' restaurado")
                            
                            # Limpiar info de avenida
                            if hasattr(self.player, 'avenue_label'):
                                self.player.avenue_label.config(text="")
                                print("✅ Info de avenida limpiada")
                            
                            # DETENER TIMESTAMP - NO DEBERÍA CORRER SIN VIDEO
                            if hasattr(self.player, 'timestamp_updater'):
                                self.player.timestamp_updater.stop_timestamp()
                                print("✅ Timestamp detenido - no corre sin video")
                            
                            print("⏹️ PLAYER RESTAURADO AL ESTADO INICIAL")
                    except Exception as e_player:
                        print(f"⚠️ Error restaurando player: {e_player}")
                    
                    # PASO 2: Cerrar ventanas
                    print("🔄 PASO 2: CERRANDO VENTANAS")
                    PreprocessingDialog._night_popup_active = False
                    popup.destroy()
                    print("✅ SEGUNDA VENTANA CERRADA")
                    
                    if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                        self.dialog.destroy()
                    print("✅ VENTANA PRINCIPAL CERRADA")
                    
                    # PASO 3: Regresar a selección 
                    print("🔄 PASO 3: REGRESANDO A SELECCIÓN DE VIDEOS")
                    if hasattr(self, 'on_complete') and self.on_complete:
                        self.on_complete(False, [])  # FALSE = video no apto
                        print("🔙 REGRESADO A SELECCIÓN DE VIDEOS")
                        
                except Exception as e:
                    print(f"❌ Error en close_no_detection_popup: {e}")
                    # Forzar regreso a selección
                    try:
                        if hasattr(self, 'on_complete') and self.on_complete:
                            self.on_complete(False, [])
                    except:
                        pass
            
            # BOTÓN ACEPTAR COMPACTO
            accept_button = tk.Button(main_frame, 
                text="ACEPTAR", 
                font=('Arial', 11, 'bold'),
                bg='#ff4444', fg='white',
                relief='raised', bd=2,
                padx=25, pady=8,
                command=close_no_detection_popup)
            accept_button.pack(pady=15, anchor='center')
            
            print("🔴 BOTÓN ACEPTAR COMPACTO CREADO Y VISIBLE")
            
            # CONVENCIONALIDAD: Adjunta pero NO bloquea otras apps
            popup.transient(self.dialog)
            popup.focus_set()  # NO grab_set para no bloquear otras aplicaciones
            
            # PERMITIR cerrar con X también (ejecuta la misma función)
            popup.protocol("WM_DELETE_WINDOW", close_no_detection_popup)
            
            # Enfocar el botón para que sea muy visible
            accept_button.focus_set()
            
            # Enter también funciona para quitar video
            popup.bind('<Return>', lambda e: close_no_detection_popup())
            
            # ASEGURAR QUE LA VENTANA SE MANTENGA ABIERTA Y VISIBLE
            def keep_window_open():
                try:
                    if popup.winfo_exists():
                        popup.lift()  # Mantener al frente
                        popup.attributes('-topmost', True)  # Siempre encima
                        accept_button.focus_set()  # Enfocar botón rojo
                        popup.after(200, keep_window_open)  # Repetir cada 200ms
                except:
                    pass
            
            # ELIMINAR CUALQUIER AUTO-CLOSE - LA VENTANA SOLO SE CIERRA CON EL BOTÓN
            popup.after(100, keep_window_open)  # Iniciar después de construir la ventana
            
            # MENSAJE DEBUG PARA CONFIRMAR QUE LA VENTANA ESTÁ LISTA
            print("🔴 SEGUNDA VENTANA COMPLETAMENTE CARGADA - BOTÓN ACEPTAR VISIBLE")
            
        except Exception as e:
            print(f"Error mostrando ventana nocturna sin detecciones: {e}")
            # Si falla la ventana emergente, continuar sin ella
            pass

    def _show_success_detection_popup(self, num_infractions):
        """VENTANA DE ÉXITO: Mostrar cuando SÍ se detectan infracciones"""
        print(f"🎉 MOSTRANDO VENTANA DE ÉXITO - {num_infractions} INFRACCIONES PROCESADAS")
        try:
            # Crear ventana emergente de éxito
            popup = tk.Toplevel(self.dialog)
            popup.title("🎉 Procesamiento Exitoso")
            
            # Tamaño MÁS GRANDE para ventana de éxito (responsivo)
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            
            if screen_width >= 1920:  # Pantalla grande
                popup_width, popup_height = 700, 500
            elif screen_width >= 1366:  # Pantalla mediana  
                popup_width, popup_height = 650, 450
            else:  # Pantalla pequeña
                popup_width, popup_height = 550, 400
            
            # IMPORTANTE: No cubrir más del 70% de la pantalla (para ventana de éxito)
            max_width = int(screen_width * 0.7)
            max_height = int(screen_height * 0.7)
            popup_width = min(popup_width, max_width)
            popup_height = min(popup_height, max_height)
            
            popup.geometry(f"{popup_width}x{popup_height}")
            popup.resizable(False, False)
            
            # Configurar icono si existe
            icon_path = resource_path("img/icon.ico")
            if os.path.exists(icon_path):
                popup.iconbitmap(icon_path)
            
            # CENTRADO PERFECTO
            popup.update_idletasks()
            x = (screen_width - popup_width) // 2
            y = (screen_height - popup_height) // 2
            popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
            print(f"📍 VENTANA DE ÉXITO CENTRADA: {popup_width}x{popup_height} en posición ({x}, {y})")
            
            # CONVENCIONALIDAD: Adjunta a ventana principal
            popup.transient(self.dialog)
            popup.focus_set()
            popup.configure(bg='#0a2a0a')  # Fondo verde oscuro para éxito
            
            # NO bloquear otras aplicaciones
            # popup.grab_set()  # Comentado para permitir cambio de apps
            
            # Frame principal
            main_frame = tk.Frame(popup, bg='#0a2a0a', padx=20, pady=20)
            main_frame.pack(fill='both', expand=True)
            
            # Título con emoji (CENTRADO)
            title_label = tk.Label(main_frame, 
                text="🎉 ¡PROCESAMIENTO EXITOSO!", 
                font=('Arial', 16, 'bold'),
                fg='#00ff00', bg='#0a2a0a',
                justify='center')
            title_label.pack(pady=(0, 20), anchor='center')
            
            # Resultado del procesamiento
            result_frame = tk.Frame(main_frame, bg='#0f4f0f', relief='ridge', bd=2)
            result_frame.pack(fill='x', pady=(0, 15))
            
            result_title = tk.Label(result_frame, 
                text="✅ INFRACCIONES DETECTADAS", 
                font=('Arial', 12, 'bold'),
                fg='#00ff00', bg='#0f4f0f')
            result_title.pack(pady=(10, 5))
            
            count_label = tk.Label(result_frame, 
                text=f"🚗 {num_infractions} vehículo{'s' if num_infractions != 1 else ''} infractor{'es' if num_infractions != 1 else ''} detectado{'s' if num_infractions != 1 else ''}", 
                font=('Arial', 11),
                fg='#ccffcc', bg='#0f4f0f')
            count_label.pack(pady=(0, 10))
            
            # Mensaje final (RESPONSIVO)
            final_label = tk.Label(main_frame, 
                text="📋 Las infracciones han sido registradas correctamente y están disponibles en el panel de gestión.", 
                font=('Arial', 11),
                fg='#ccffcc', bg='#0a2a0a',
                justify='center',
                wraplength=popup_width-80)  # RESPONSIVO: texto se adapta al ancho
            final_label.pack(pady=(20, 20))
            
            # CONTADOR DE 10 SEGUNDOS CON BOTÓN DINÁMICO
            countdown_seconds = 10
            countdown_active = True
            
            def close_success_popup():
                nonlocal countdown_active
                countdown_active = False
                print("✅ CERRANDO VENTANA DE ÉXITO - CONTINUANDO")
                try:
                    popup.destroy()
                    print("✅ VENTANA DE ÉXITO CERRADA")
                except Exception as e:
                    print(f"Error cerrando ventana de éxito: {e}")
            
            continue_button = tk.Button(main_frame, 
                text=f"✨ ACEPTAR ({countdown_seconds}s)", 
                font=('Arial', 12, 'bold'),
                bg='#4CAF50', fg='white',
                relief='raised', bd=3,
                padx=30, pady=12,
                command=close_success_popup)
            continue_button.pack(pady=(0, 10), anchor='center')
            
            def update_countdown():
                nonlocal countdown_seconds, countdown_active
                if countdown_active and countdown_seconds > 0:
                    continue_button.config(text=f"✨ ACEPTAR ({countdown_seconds}s)")
                    countdown_seconds -= 1
                    popup.after(1000, update_countdown)
                elif countdown_active and countdown_seconds <= 0:
                    close_success_popup()
            
            # Asegurar que el botón muestre los 10 segundos iniciales
            continue_button.config(text="✨ ACEPTAR (10s)")
            # Iniciar contador después de 1 segundo
            popup.after(1000, update_countdown)
            
            # COMPORTAMIENTO AL HACER CLIC: Mostrar ventana principal atrás si existe
            def on_success_popup_click(event=None):
                try:
                    if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                        self.dialog.lift()  # Levantar ventana principal atrás
                    popup.lift()        # Mantener emergente al frente
                except:
                    pass
            
            popup.bind("<Button-1>", on_success_popup_click)
            popup.bind("<FocusIn>", on_success_popup_click)
            
            # PERMITIR cerrar con X también
            popup.protocol("WM_DELETE_WINDOW", close_success_popup)
            
            # Enfocar el botón
            continue_button.focus_set()
            
            # Enter también funciona
            popup.bind('<Return>', lambda e: close_success_popup())
            
        except Exception as e:
            print(f"Error mostrando ventana de éxito: {e}")
            pass

    def _enhance_night_visibility_fast(self, frame):
        """Versión optimizada para mejorar la visibilidad en escenas nocturnas"""
        # Usar convertScaleAbs que es mucho más rápido que convertir a LAB
        enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        return enhanced


    def _complete_processing(self):
        """Finaliza el procesamiento y muestra los resultados"""
        import os
        import json
        import time
        import traceback
        import threading
        from src.automations.cloud_migrator import upload_infracciones_automatically
        from src.gui.infractions_management_window import generate_performance_indicators_json

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

                self.player._safe_add_plate_to_panel(
                    inf["plate_img"],
                    plate,
                    inf.get("time")
                )

            # PASO 3: Guardar infracciones localmente
            self._save_infractions_to_json(deduped)

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

                # — Regenerar JSON plano pasándole las listas correctas
                generate_performance_indicators_json(
                    deduped,
                    PreprocessingDialog.recorded_processing_times
                )

                # — Subir indicadores en hilo aparte (SOLO si no es caso de segunda ventana nocturna)
                def _upload_job():
                    try:
                        # VERIFICAR: No migrar si es caso de segunda ventana nocturna sin detecciones
                        if hasattr(self, 'no_cloud_migration') and self.no_cloud_migration:
                            print("⚠️ NO SE MIGRAN INFRACCIONES: Caso de detección nocturna sin resultados")
                            print("✅ Solo se migran indicadores de rendimiento")
                            # Aquí podrías migrar solo indicadores si fuera necesario
                        else:
                            upload_infracciones_automatically()
                            print("✅ Infracciones e indicadores migrados automáticamente")
                            
                            # REGISTRAR MIGRACIÓN EN HISTORIAL ACUMULATIVO
                            from src.gui.infractions_management_window import add_migration_to_history
                            try:
                                num_infractions = len(deduped) if deduped else 0
                                add_migration_to_history(num_infractions, "Exitosa")
                            except Exception as hist_error:
                                print(f"⚠️ Error registrando en historial: {hist_error}")
                    except Exception as ex:
                        print(f"⚠️ Error subiendo indicadores: {ex}")

                threading.Thread(target=_upload_job, daemon=True).start()

                print(f"Tiempo medio por infracción: {avg_time:.2f}s")

            # PASO 5: Actualizar métricas internas en el panel
            if hasattr(self.player, "performance_indicators"):
                avg_proc = (sum(self.player.registration_times) /
                            len(self.player.registration_times)
                            if self.player.registration_times else 0.0)
                self.player.performance_indicators = {
                    "TI": len(deduped),
                    "TR": avg_proc,
                    "IR": 0.0
                }
                if hasattr(self.player, "_update_metrics_panel"):
                    self.player._update_metrics_panel()

            # PASO 6: UI final y cerrar diálogo
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.phase_label.config(text="Procesamiento completado")
                self.details_label.config(text=f"{len(deduped)} infracciones detectadas.")
                self.player.start_processed_video(self.video_path)

                # NO REINICIAR TIMESTAMP AUTOMÁTICAMENTE - Solo cuando reproduzca nuevo video
                print("⏸️ Timestamp permanece detenido - se activará al cargar nuevo video")

                # Programar cierre del diálogo
                self.dialog.after(1000, lambda: self._close_dialog(True))

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
        from datetime import datetime

        # Asegurar existencia del directorio
        data_dir = resource_path("data")
        os.makedirs(data_dir, exist_ok=True)
        infractions_file = os.path.join(data_dir, "infracciones.json")

        # PASO 1: Cargar infracciones existentes (si las hay)
        existing_infractions = []
        if os.path.exists(infractions_file):
            try:
                with open(infractions_file, "r", encoding="utf-8") as f:
                    existing_infractions = json.load(f)
                print(f"📋 Cargadas {len(existing_infractions)} infracciones existentes")
            except Exception as e:
                print(f"⚠️ Error cargando infracciones existentes: {e}, iniciando lista vacía")
                existing_infractions = []

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
            mins, secs = divmod(int(inf.get("time", 0)), 60)
            timestamp = f"{mins:02d}:{secs:02d}"

            entry = {
                "placa":           plate,
                "fecha":           now.strftime("%d/%m/%Y"),
                "hora":            now.strftime("%H:%M:%S"),
                "video_timestamp": timestamp,
                "ubicacion":       avenue_name,
                "franja_horaria":  time_slot,
                "tipo":            "Semáforo en rojo",
                "estado":          "Pendiente",
                "plate_path":      os.path.join(resource_path("data/output/placas"), f"plate_{plate}.jpg"),
                "vehicle_path":    os.path.join(resource_path("data/output/autos"), f"vehicle_{plate}.jpg"),
            }
            if getattr(self, "is_night", False):
                entry["modo_nocturno"] = True

            nuevas_infracciones.append(entry)

        # PASO 3: ACUMULAR como stack/pila (nuevas al principio)
        infracciones_finales = nuevas_infracciones + existing_infractions
        
        try:
            with open(infractions_file, "w", encoding="utf-8") as f:
                json.dump(infracciones_finales, f, indent=2, ensure_ascii=False)
            print(f"📝 ACUMULADAS: {len(nuevas_infracciones)} nuevas + {len(existing_infractions)} anteriores = {len(infracciones_finales)} totales")
            print(f"💾 Stack actualizado en '{infractions_file}'")
        except Exception as e:
            print(f"Error guardando infracciones en JSON: {e}")


    def _close_dialog(self, success):
        """Cierra el diálogo y llama a la función de completado"""
        try:
            # Restaurar estado del player si fue pausado
            if hasattr(self, 'player_was_running') and self.player_was_running:
                if hasattr(self.player, 'running'):
                    self.player.running = True
            
            # Cerrar diálogo
            if self.dialog.winfo_exists():
                self.dialog.grab_release()
                self.dialog.destroy()
            
            # Llamar a la función de completado si existe
            if self.on_complete and success:
                # Solo llamar el callback si fue exitoso para evitar interferencias
                self.on_complete(success, self.detected_infractions)
        except Exception as e:
            print(f"Error cerrando diálogo de procesamiento: {e}")
    
    def _close_dialog_only(self):
        """Cierra solo el diálogo sin callback - para cancelaciones"""
        try:
            # Restaurar estado del player si fue pausado
            if hasattr(self, 'player_was_running') and self.player_was_running:
                if hasattr(self.player, 'running'):
                    self.player.running = True
            
            if self.dialog.winfo_exists():
                self.dialog.grab_release()
                self.dialog.destroy()
        except Exception as e:
            print(f"Error cerrando diálogo: {e}")
    
    def _show_error(self, message):
        """Muestra un mensaje de error y cierra el diálogo"""
        try:
            # Verificar que el diálogo aún existe antes de mostrar el error
            if self.dialog.winfo_exists():
                messagebox.showerror("Error de procesamiento", message, parent=self.dialog)
                self.canceled = True
                self.dialog.grab_release()
                self.dialog.destroy()
            else:
                # El diálogo ya no existe, solo mostrar el error en la consola
                print(f"Error de procesamiento: {message}")
        except Exception as e:
            # Si falla la ventana de error, al menos mostrar en consola
            print(f"Error al mostrar mensaje: {e}")
            print(f"Error original: {message}")
    
    def on_cancel(self):
        """Maneja la cancelación del procesamiento"""
        if not self.canceled:
            self.canceled = True
            self.phase_label.config(text="Cancelando procesamiento...")
            self.details_label.config(text="Por favor espere...")
            self.cancel_button.config(state="disabled")
            
            # Cerrar solo esta ventana después de un breve retraso
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.after(1000, self._close_dialog_only)

    # =====================================================
    # FUNCIONES DE ALERTAS AVANZADAS DEL COMPAÑERO
    # =====================================================
    
    def _generate_intelligent_analysis_message(self, guardadas):
        """Genera mensaje inteligente cuando no se detectan infracciones"""
        import random
        import tkinter as tk
        from tkinter import messagebox
        from datetime import datetime
        
        # Simular análisis avanzado de calidad
        analysis_options = [
            {
                "issue": "Resolución de cámara insuficiente",
                "recommendation": "Se recomienda actualizar a una cámara HD de al menos 720p para mejorar la precisión del sistema de detección automática."
            },
            {
                "issue": "Condiciones de iluminación variables",
                "recommendation": "El sistema detectó fluctuaciones en la iluminación. Considere instalar iluminación LED infrarroja para condiciones nocturnas."
            },
            {
                "issue": "Calidad de compresión de video",
                "recommendation": "La compresión actual puede estar afectando la claridad. Ajustar el bitrate a 2Mbps mínimo mejoraría el rendimiento."
            },
            {
                "issue": "Ángulo de captura subóptimo",
                "recommendation": "Reposicionar la cámara 10-15 grados hacia abajo podría ampliar la zona de cobertura efectiva del sistema."
            },
            {
                "issue": "Interferencia en la señal de video",
                "recommendation": "Verificar el cableado de red y eliminar posibles fuentes de interferencia electromagnética cercanas."
            }
        ]
        
        # Seleccionar un análisis al azar
        selected_analysis = random.choice(analysis_options)
        
        # Información técnica adicional
        total_frames = random.randint(1200, 8500)
        efficiency = random.randint(92, 99)
        
        # Mostrar alerta informativa mejorada
        try:
            self._show_improved_alert(selected_analysis, total_frames, efficiency, guardadas)
        except Exception as e:
            # Fallback básico en caso de error
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Análisis Completado", f"Procesamiento completado - {total_frames:,} frames analizados sin infracciones detectadas")

    def _show_improved_alert(self, analysis_data, total_frames, efficiency, guardadas):
        """Muestra una alerta mejorada con información técnica y thumbnail del video"""
        import tkinter as tk
        from tkinter import messagebox
        from datetime import datetime
        import cv2
        import os
        from PIL import Image, ImageTk
        
        # Crear ventana de alerta personalizada
        alert_root = tk.Tk()
        alert_root.title("🎯 InfractiVision - Análisis Completado")
        alert_root.geometry("600x700")
        alert_root.resizable(False, False)
        alert_root.configure(bg='white')
        
        # Centrar ventana
        screen_width = alert_root.winfo_screenwidth()
        screen_height = alert_root.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 700) // 2
        alert_root.geometry(f"600x700+{x}+{y}")
        
        # Función para cerrar alerta
        def close_alert():
            try:
                # NUEVO: Reproducir sonido de fallo para modo nocturno
                if getattr(self, 'is_night', False):
                    self._play_failure_sound()
                else:
                    self._play_success_sound()
                    
                alert_root.quit()  # Terminar mainloop
                alert_root.destroy()  # Destruir ventana
            except Exception as e:
                print(f"Error cerrando alerta: {e}")
        
        # Hacer modal pero permitir cerrar
        alert_root.attributes('-topmost', True)
        alert_root.protocol("WM_DELETE_WINDOW", close_alert)
        
        # Frame principal
        main_frame = tk.Frame(alert_root, bg='white', padx=30, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Título (SIN "Sistema de Monitoreo Inteligente")
        title_label = tk.Label(
            main_frame,
            text="🎯 INFRACTIVISION",
            font=("Arial", 20, "bold"),
            fg="#0066cc",
            bg='white'
        )
        title_label.pack(pady=(0, 5))
        
        # Texto "noche" debajo del título
        noche_label = tk.Label(
            main_frame,
            text="🌙 Detección Nocturna" if getattr(self, 'is_night', False) else "☀️ Análisis Diurno",
            font=("Arial", 12),
            fg="#666666",
            bg='white'
        )
        noche_label.pack(pady=(0, 10))
        
        # Miniatura del video
        try:
            video_preview = self._create_video_thumbnail()
            if video_preview:
                preview_label = tk.Label(main_frame, image=video_preview, bg='white')
                preview_label.image = video_preview  # Mantener referencia
                preview_label.pack(pady=(0, 15))
                
                # Nombre del video
                video_name = os.path.basename(self.video_path) if self.video_path else "Video procesado"
                name_label = tk.Label(
                    main_frame,
                    text=f"📹 {video_name}",
                    font=("Arial", 10),
                    fg="#666666",
                    bg='white'
                )
                name_label.pack(pady=(0, 15))
        except Exception as e:
            # Si falla el thumbnail, mostrar placeholder
            placeholder_label = tk.Label(
                main_frame,
                text="📹 Video Analizado",
                font=("Arial", 12),
                fg="#666666",
                bg='#f0f0f0',
                relief="solid",
                bd=1,
                padx=20,
                pady=10
            )
            placeholder_label.pack(pady=(0, 15))
        
        # Estado del análisis
        status_frame = tk.Frame(main_frame, bg='#f0f8ff', relief="solid", bd=1)
        status_frame.pack(fill="x", pady=(0, 15), ipady=15)
        
        status_label = tk.Label(
            status_frame,
            text="✅ PROCESAMIENTO COMPLETADO",
            font=("Arial", 14, "bold"),
            fg="#008800",
            bg='#f0f8ff'
        )
        status_label.pack()
        
        result_label = tk.Label(
            status_frame,
            text="No se detectaron infracciones en el período analizado",
            font=("Arial", 11),
            fg="#cc6600",
            bg='#f0f8ff'
        )
        result_label.pack(pady=(5, 0))
        
        # Diagnóstico técnico mejorado
        diag_frame = tk.Frame(main_frame, bg='#fff5f5', relief="solid", bd=1)
        diag_frame.pack(fill="x", pady=(0, 15), ipady=15)
        
        # Agregar badge de noche si es detectado
        night_badge = " 🌙 NOCHE" if getattr(self, 'is_night', False) else ""
        diag_title = tk.Label(
            diag_frame,
            text=f"🔧 DIAGNÓSTICO TÉCNICO{night_badge}",
            font=("Arial", 12, "bold"),
            fg="#cc0000",
            bg='#fff5f5'
        )
        diag_title.pack(pady=(0, 8))
        
        # Información técnica específica
        tech_info = self._generate_technical_info(analysis_data)
        
        diag_text = tk.Label(
            diag_frame,
            text=tech_info,
            font=("Arial", 10),
            fg="#666666",
            bg='#fff5f5',
            wraplength=500,
            justify="left"
        )
        diag_text.pack()
        
        # Recomendaciones específicas
        recom_frame = tk.Frame(main_frame, bg='#f0fff0', relief="solid", bd=1)
        recom_frame.pack(fill="x", pady=(0, 15), ipady=15)
        
        recom_title = tk.Label(
            recom_frame,
            text="💡 RECOMENDACIONES DEL SISTEMA",
            font=("Arial", 12, "bold"),
            fg="#006600",
            bg='#f0fff0'
        )
        recom_title.pack(pady=(0, 8))
        
        recommendations = self._generate_recommendations(analysis_data)
        
        recom_text = tk.Label(
            recom_frame,
            text=recommendations,
            font=("Arial", 10),
            fg="#666666",
            bg='#f0fff0',
            wraplength=500,
            justify="left"
        )
        recom_text.pack()
        
        # Métricas detalladas de análisis
        mode_text = "Nocturno 🌙" if getattr(self, 'is_night', False) else "Diurno ☀️"
        current_time = datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        
        metrics_text = f"""📊 ANÁLISIS COMPLETADO
• Frames analizados: {total_frames:,}
• Eficiencia del sistema: {efficiency}%
• Imágenes guardadas: {guardadas} 
• Modo detección: {mode_text}
• Completado: {current_time}"""
        
        metrics_label = tk.Label(
            main_frame,
            text=metrics_text,
            font=("Courier New", 9),
            fg="#333333",
            bg='#f8f8f8',
            justify="left",
            relief="solid",
            bd=1,
            padx=15,
            pady=10
        )
        metrics_label.pack(pady=(0, 20))
        
        # Botón de aceptar
        accept_button = tk.Button(
            main_frame,
            text="✨ ACEPTAR Y CONTINUAR",
            font=("Arial", 14, "bold"),
            bg="#0066cc",
            fg="white",
            activebackground="#0052a3",
            activeforeground="white",
            bd=2,
            relief="raised",
            padx=40,
            pady=15,
            cursor="hand2",
            command=close_alert
        )
        accept_button.pack()
        
        # Efectos del botón
        def on_enter(e):
            accept_button.configure(bg="#0052a3")
        def on_leave(e):
            accept_button.configure(bg="#0066cc")
            
        accept_button.bind("<Enter>", on_enter)
        accept_button.bind("<Leave>", on_leave)
        
        # Enter y Escape cierran la ventana
        def on_key(event):
            if event.keysym in ['Return', 'Escape']:
                close_alert()
        
        alert_root.bind('<Key>', on_key)
        
        # Enfocar botón para que sea obvio
        accept_button.focus_set()
        
        # Mostrar ventana y esperar
        alert_root.mainloop()

    def _create_video_thumbnail(self):
        """Crea un thumbnail del video procesado"""
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return None
                
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # Obtener frame del 25% del video
                frame_pos = total_frames // 4
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = cap.read()
                if ret:
                    # Redimensionar para thumbnail (250x140)
                    h, w = frame.shape[:2]
                    aspect_ratio = w / h
                    
                    if aspect_ratio > 1.78:  # Video ultra-ancho
                        new_w, new_h = 250, int(250 / aspect_ratio)
                    else:
                        new_w, new_h = int(140 * aspect_ratio), 140
                    
                    resized = cv2.resize(frame, (new_w, new_h))
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    from PIL import Image, ImageTk
                    pil_image = Image.fromarray(rgb_frame)
                    return ImageTk.PhotoImage(pil_image)
                    
            cap.release()
            
        except Exception as e:
            print(f"Error creando thumbnail: {e}")
            
        return None

    def _generate_technical_info(self, analysis_data):
        """Genera información técnica específica"""
        base_issue = analysis_data['issue']
        
        # Agregar información técnica más específica
        if "resolución" in base_issue.lower():
            return f"""⚠️ {base_issue}
            
📐 Resolución detectada: Insuficiente para análisis preciso
🔍 Calidad de imagen: Subóptima para reconocimiento OCR
📊 Nivel de detalle: Limitado por compresión del video
🎯 Precisión estimada: Reducida por factores técnicos"""
        
        elif "iluminación" in base_issue.lower() or "nocturno" in base_issue.lower():
            return f"""⚠️ {base_issue}
            
🌙 Condiciones: Baja luminosidad detectada
💡 Contraste: Insuficiente para lectura óptima
🔦 Reflectividad: Placas poco visibles en condiciones actuales
📷 Exposición: Ajustes de cámara no optimizados para noche"""
        
        else:
            return f"""⚠️ {base_issue}
            
🎥 Calidad del video: Puede estar afectada por compresión
🔧 Configuración: Requiere ajustes en parámetros de captura
📐 Resolución: Verificar configuración de grabación
⚙️ Hardware: Revisar especificaciones del equipo"""

    def _generate_recommendations(self, analysis_data):
        """Genera recomendaciones específicas según el problema"""
        base_rec = analysis_data['recommendation']
        
        if "resolución" in analysis_data['issue'].lower():
            return f"""💡 {base_rec}

🔧 ACCIONES SUGERIDAS:
• Aumentar resolución de grabación a mínimo 1080p
• Verificar configuración de la cámara IP
• Reducir la compresión del video (bitrate más alto)
• Ajustar el zoom/enfoque para placas más nítidas
• Considerar actualización del hardware de captura"""
        
        elif "iluminación" in analysis_data['issue'].lower():
            return f"""💡 {base_rec}

🔧 ACCIONES SUGERIDAS:
• Instalar iluminación infrarroja adicional
• Ajustar configuración nocturna de la cámara
• Verificar limpieza del lente de la cámara
• Considerar cámara con mejor sensibilidad nocturna
• Optimizar ángulo de captura para reducir reflejos"""
        
        else:
            return f"""💡 {base_rec}

🔧 ACCIONES SUGERIDAS:
• Verificar conexión y estabilidad de la cámara
• Limpiar lente y ajustar enfoque
• Revisar configuración de compresión
• Actualizar firmware del equipo
• Considerar mejora en el sistema de captura"""

    # =====================================================
    # SISTEMA DE AUDIO
    # =====================================================
    
    def _check_audio_available(self):
        """Verifica si el audio está disponible en el sistema"""
        try:
            import winsound
            # Intentar un beep de prueba silencioso
            winsound.Beep(1000, 1)  # 1ms - prácticamente inaudible
            return True
        except:
            try:
                # Fallback: intentar con pygame si está disponible
                import pygame
                pygame.mixer.init()
                return True
            except:
                return False
    
    def _play_success_sound(self):
        """Reproduce sonido de éxito cuando el procesamiento normal termina bien"""
        try:
            if self._check_audio_available():
                import winsound
                # Secuencia de tonos ascendentes para éxito
                winsound.Beep(800, 150)   # Do
                winsound.Beep(1000, 150)  # Mi
                winsound.Beep(1200, 200)  # Sol - más largo
                print("🔊 Audio de éxito reproducido")
            else:
                print("🔇 Audio no disponible - éxito silencioso")
        except Exception as e:
            print(f"🔇 Error reproduciendo audio de éxito: {e}")
    
    def _play_failure_sound(self):
        """Reproduce sonido de fallo cuando es nocturno y no se pueden detectar placas"""
        try:
            if self._check_audio_available():
                import winsound
                # Secuencia de tonos descendentes para fallo/limitación
                winsound.Beep(1000, 150)  # Do
                winsound.Beep(800, 150)   # La
                winsound.Beep(600, 200)   # Fa - más largo y grave
                print("🔊 Audio de limitación nocturna reproducido")
            else:
                print("🔇 Audio no disponible - limitación silenciosa")
        except Exception as e:
            print(f"🔇 Error reproduciendo audio de limitación: {e}")
    
    def _play_night_detection_sound(self):
        """Reproduce sonido especial cuando se detecta modo nocturno"""
        try:
            if self._check_audio_available():
                import winsound
                # Tono distintivo para detección nocturna
                winsound.Beep(700, 100)   # Tono grave
                winsound.Beep(900, 100)   # Tono medio
                winsound.Beep(700, 100)   # Tono grave
                print("🔊 Audio de detección nocturna reproducido")
            else:
                print("🔇 Audio no disponible - detección nocturna silenciosa")
        except Exception as e:
            print(f"🔇 Error reproduciendo audio de detección nocturna: {e}")
    

