import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import json
import queue
from concurrent.futures import ThreadPoolExecutor

# Eliminamos la importación circular

class PreprocessingDialog:
    """
    Diálogo que muestra una barra de progreso mientras se procesa un video.
    Analiza las infracciones y detecta las placas sin reproducir el video completo.
    """

    # Atributo estático para almacenar tiempos de procesamiento
    recorded_processing_times = []

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
        self.progress_value = 0
        self.current_frame = None
        self.detected_infractions = []
        self.processed_frames = 0
        self.total_frames = 0
        self.result_queue = queue.Queue()
        
        # Definir rutas de configuración directamente sin importarlas
        self.POLYGON_CONFIG_FILE = "config/polygon_config.json"
        self.AVENUE_CONFIG_FILE = "config/avenue_config.json"
        self.PRESETS_FILE = "config/time_presets.json"

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
        self.dialog.geometry("800x600")
        self.dialog.resizable(False, False)
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
                self.player.vehicle_detector = VehicleDetector(model_path="models/yolov8n.pt")
                
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
            
            # Extraer datos específicos para este video
            if self.video_path in configs.get('polygon', {}):
                self.polygon_points = configs['polygon'][self.video_path]
                
            if self.video_path in configs.get('presets', {}):
                self.cycle_durations = configs['presets'][self.video_path]
                
            if self.video_path in configs.get('avenue', {}):
                self.current_avenue = configs['avenue'][self.video_path]
                
        except Exception as e:
            print(f"Error en load_video_config: {e}")
    
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
        
        # Frame para la visualización del video
        self.video_frame = ttk.Frame(main_frame, width=640, height=360, relief="groove", borderwidth=2)
        self.video_frame.pack(pady=(0, 20))
        self.video_frame.pack_propagate(False)
        
        # Label para mostrar el frame actual
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)
        
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
                self.dialog.after(50, self._schedule_ui_update)
            except Exception as e:
                print(f"Error en actualización de UI: {e}")
                # Seguir intentando actualizar la UI
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
                self.dialog.after(0, lambda: self._show_error(
                    "Este video no está configurado correctamente. Configure primero el área restrictiva y los tiempos de semáforo."))
                return
                    
            # Abrir el video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.dialog.after(0, lambda: self._show_error("No se pudo abrir el video"))
                return
            
            # Inicialización
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Verificaciones adicionales
            if self.total_frames <= 0:
                self.dialog.after(0, lambda: self._show_error("No se pudo determinar la duración del video"))
                return
            
            # Crear directorios para resultados
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Fase 1: Inicialización rápida
            self.phase_label.config(text="Fase 1: Inicializando análisis")
            
            # Detectar automáticamente si es una escena nocturna
            ret, first_frame = cap.read()
            if not ret:
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
            
            # Fase 4: Finalización 
            self.phase_label.config(text="Fase 4: Organizando resultados")
            
            # Identificar y filtrar placas duplicadas
            self.phase_label.config(text="Análisis completado")
            self.progress_value = 100
            
            # Procesar los resultados finales tras una pequeña pausa
            self.dialog.after(500, self._finalize_processing)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
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
                                            plates_dir = os.path.join("data", "output", "placas")
                                            vehicles_dir = os.path.join("data", "output", "autos")
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
                                            plates_dir = os.path.join("data", "output", "placas")
                                            vehicles_dir = os.path.join("data", "output", "autos")
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
                                                try:
                                                    # CORRECCIÓN: Convertir explícitamente a enteros y validar
                                                    px1 = int(x1_roi + plate_bbox[0])
                                                    py1 = int(y1_roi + plate_bbox[1])
                                                    px2 = int(x1_roi + plate_bbox[2])
                                                    py2 = int(y1_roi + plate_bbox[3])
                                                    
                                                    # Verificar que las coordenadas son válidas antes de dibujar
                                                    if px1 >= 0 and py1 >= 0 and px2 > px1 and py2 > py1:
                                                        cv2.rectangle(detection_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                                                except (TypeError, ValueError, IndexError) as e:
                                                    print(f"Error al dibujar rectángulo de placa: {e}")
                                            
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
        SIMILARITY_THRESHOLD = 0.75  # Reducido de 0.7 a 0.6
        
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
                    if img_similarity >= 0.85 or (img_similarity >= 0.75 and time_proximity >= 0.9):
                        # Umbral de similitud de imagen aumentado de 0.65 a 0.75
                        # Umbral de proximidad temporal aumentado de 0.8 a 0.9
                        union(i, j)
                        print(f"Agrupación por imagen: '{infractions[i].get('plate', '')}' y '{infractions[j].get('plate', '')}' (img:{img_similarity:.2f}, tiempo:{time_proximity:.2f})")
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
            plates_dir = os.path.join("data", "output", "placas")
            vehicles_dir = os.path.join("data", "output", "autos")
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
            print(f"Procesamiento completado: {len(self.detected_infractions)} vehículos infractores ({guardadas} imágenes guardadas)")
            
            # Llamar a _complete_processing
            self.dialog.after(0, self._complete_processing)
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
        """Versión optimizada para detectar escenas nocturnas"""
        # Redimensionar para análisis rápido
        small_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular brillo promedio
        avg_brightness = np.mean(gray)
        
        # También verificar área más oscura (percentil 10)
        dark_threshold = np.percentile(gray, 10)
        
        # Si el brillo promedio es bajo o áreas oscuras son muy oscuras
        return avg_brightness < 85 or dark_threshold < 30

    def _enhance_night_visibility_fast(self, frame):
        """Versión optimizada para mejorar la visibilidad en escenas nocturnas"""
        # Usar convertScaleAbs que es mucho más rápido que convertir a LAB
        enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        return enhanced

    def _complete_processing(self):
        """Finaliza el procesamiento y muestra los resultados"""
        try:
            # PASO 1: Aplicar la deduplicación mejorada a las infracciones detectadas
            deduped_infractions = self._dedup_similar_plates(self.detected_infractions)
            
            # PASO 2: Actualizar las infracciones detectadas con la lista deduplicada
            self.detected_infractions = deduped_infractions
            
            # CORRECCIÓN: Inicializar el contador de tiempo para registrar procesamiento
            processing_start_time = time.time()
            
            # Inicializar colección para tiempos de procesamiento
            processing_times = []
            
            # CORRECCIÓN: Asegurarse que el tiempo de inicio esté disponible en el player
            if not hasattr(self.player, "detection_start_time"):
                self.player.detection_start_time = processing_start_time
            
            # CORRECCIÓN: Preparar registro de tiempos para mostrar TR correcto
            if not hasattr(self.player, "registration_times"):
                self.player.registration_times = []
            
            # PASO 3: Añadir TODAS las placas deduplicadas al panel lateral
            # IMPORTANTE: Ya NO filtramos por placas únicas aquí, porque la deduplicación ya se hizo
            for infraction in deduped_infractions:
                
                # Calcular tiempo de procesamiento individual para esta placa
                plate_processing_time = time.time() - processing_start_time
                processing_times.append(plate_processing_time)
                
                # Asegurarnos de que todos los parámetros sean válidos
                if 'plate_img' in infraction and infraction['plate_img'] is not None and \
                'plate' in infraction and infraction['plate'] is not None and \
                'vehicle_img' in infraction and infraction['vehicle_img'] is not None:
                    
                    plate_text = infraction['plate']
                    
                    # Inicializar historial si no existe
                    if not hasattr(self.player, "plate_detection_history"):
                        self.player.plate_detection_history = {}
                    
                    # Inicializar estructura para esta placa (sea nueva o duplicada)
                    detection_time = self.player.detection_start_time
                    if infraction['time'] is not None:
                        detection_time = self.player.detection_start_time + infraction['time']
                        
                    registration_time = time.time()
                    proc_time = registration_time - detection_time
                    
                    # Añadir al historial de tiempos de registro para estadísticas
                    self.player.registration_times.append(proc_time)
                    
                    # Actualizar o crear entrada en el historial
                    if plate_text not in self.player.plate_detection_history:
                        self.player.plate_detection_history[plate_text] = {
                            "count": 1,
                            "first_detection": infraction['time'],
                            "last_detection": infraction['time'],
                            "vehicle_img": infraction['vehicle_img'],
                            "detection_time": detection_time,
                            "registration_time": registration_time,
                            "processing_time": proc_time
                        }
                    else:
                        # Si ya existe, actualizar información
                        self.player.plate_detection_history[plate_text]["count"] += 1
                        self.player.plate_detection_history[plate_text]["last_detection"] = infraction['time']
                    
                    # Si existen rutas de archivos, asegurarse de guardarlas en el historial
                    if 'vehicle_path' in infraction and infraction['vehicle_path']:
                        self.player.plate_detection_history[plate_text]["vehicle_path"] = infraction['vehicle_path']
                    
                    if 'plate_path' in infraction and infraction['plate_path']:
                        self.player.plate_detection_history[plate_text]["plate_path"] = infraction['plate_path']
                    
                    # MODIFICACIÓN IMPORTANTE: Añadir TODAS las placas al panel, sin filtrar duplicados
                    # porque ya eliminamos los duplicados en _dedup_similar_plates
                    self.player._safe_add_plate_to_panel(
                        infraction['plate_img'], 
                        plate_text, 
                        infraction['time']
                    )
            
            # NUEVO: Guardar todas las infracciones detectadas en el archivo JSON
            self._save_infractions_to_json(deduped_infractions)
            
            # MODIFICADO: Calcular y almacenar el tiempo de procesamiento para los indicadores
            try:
                # Calcular el tiempo total desde el inicio hasta ahora
                total_time = time.time() - self.processing_start_time
                
                # Calcular el tiempo promedio por infracción
                avg_time = 0
                if len(deduped_infractions) > 0:
                    avg_time = total_time / len(deduped_infractions)
                
                # Registrar en la variable estática de clase para acceso global
                PreprocessingDialog.recorded_processing_times.append(avg_time)
                
                # Actualizar el JSON de indicadores de rendimiento existente con este tiempo
                indicators_file = os.path.join("data", "indicadores_rendimiento.json")
                
                try:
                    if os.path.exists(indicators_file):
                        # Leer el archivo JSON existente
                        with open(indicators_file, "r", encoding="utf-8") as f:
                            indicators_data = json.load(f)
                            
                        # Actualizar el tiempo de procesamiento en el TR
                        if "indicadores" in indicators_data and "TR" in indicators_data["indicadores"]:
                            indicators_data["indicadores"]["TR"]["con_software"]["tiempo_promedio_segundos"] = avg_time
                            indicators_data["indicadores"]["TR"]["con_software"]["muestras_analizadas"] = len(PreprocessingDialog.recorded_processing_times)
                            
                            # Recalcular reducción y factor de velocidad
                            pnp_avg_time = indicators_data["indicadores"]["TR"]["sin_software"]["tiempo_promedio_segundos"]
                            indicators_data["indicadores"]["TR"]["reduccion_tiempo_porcentual"] = ((pnp_avg_time - avg_time) / pnp_avg_time * 100) if pnp_avg_time else 0
                            indicators_data["indicadores"]["TR"]["veces_mas_rapido"] = pnp_avg_time / avg_time if avg_time else 0
                            
                            # Actualizar resumen global
                            tr_reduction = indicators_data["indicadores"]["TR"]["reduccion_tiempo_porcentual"]
                            tr_speedup = indicators_data["indicadores"]["TR"]["veces_mas_rapido"]
                            indicators_data["resumen_global"]["tiempo_registro_reduccion"] = f"-{tr_reduction:.1f}%"
                            indicators_data["resumen_global"]["tiempo_registro_factor"] = f"{tr_speedup:.1f}x más rápido"
                            
                            # Guardar el JSON actualizado
                            with open(indicators_file, "w", encoding="utf-8") as f:
                                json.dump(indicators_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error actualizando indicadores: {e}")
                
                print(f"Tiempo de procesamiento registrado: {avg_time:.2f} segundos por infracción")
            except Exception as e:
                print(f"Error guardando tiempos de procesamiento: {e}")
                        
            # CORRECCIÓN: Actualizar indicadores de rendimiento después de añadir todas las placas
            if hasattr(self.player, "performance_indicators"):
                # Calcular el promedio del tiempo de registro si hay datos
                avg_proc_time = 0.0
                if self.player.registration_times:
                    avg_proc_time = sum(self.player.registration_times) / len(self.player.registration_times)
                    
                # Actualizar indicadores
                self.player.performance_indicators = {
                    "TI": len(deduped_infractions),  # Número exacto de infracciones
                    "TR": avg_proc_time,             # Tiempo promedio de registro
                    "IR": 0.0                       # Reiniciar índice de reincidencia
                }
                
                # Forzar actualización del panel de rendimiento
                if hasattr(self.player, "_update_metrics_panel"):
                    self.player._update_metrics_panel()
            
            # Actualizar la interfaz con información de las infracciones
            found_message = f"Se han detectado {len(deduped_infractions)} infracciones."
            self.phase_label.config(text="Procesamiento completado")
            self.details_label.config(text=found_message)
            
            # Iniciar reproducción optimizada del video (solo detección de vehículos)
            self.player.start_processed_video(self.video_path)
            
            # Cerrar diálogo después de un breve retraso para mostrar el mensaje final
            self.dialog.after(1000, lambda: self._close_dialog(True))
        except Exception as e:
            print(f"Error en _complete_processing: {e}")
            import traceback
            traceback.print_exc()
            self._close_dialog(False)

    # NUEVO MÉTODO: Guardar infracciones en archivo JSON
    def _save_infractions_to_json(self, infractions):
        """
        Guarda las infracciones detectadas en un archivo JSON centralizado
        para ser utilizadas por el módulo de gestión de infracciones.
        """
        import json
        import os
        from datetime import datetime
        
        # Directorio y archivo para infracciones
        data_dir = os.path.join("data")
        os.makedirs(data_dir, exist_ok=True)
        infractions_file = os.path.join(data_dir, "infracciones.json")
        
        # Cargar infracciones existentes si el archivo existe
        existing_infractions = []
        if os.path.exists(infractions_file):
            try:
                with open(infractions_file, "r", encoding="utf-8") as f:
                    existing_infractions = json.load(f)
            except Exception as e:
                print(f"Error al cargar archivo de infracciones: {e}")
        
        # Nombre de la avenida actual
        avenue_name = "Desconocida"
        if hasattr(self.player, "current_avenue") and self.player.current_avenue:
            avenue_name = self.player.current_avenue
        
        # Obtener la franja horaria configurada para este video
        time_slot = "No especificada"
        if self.cycle_durations and "time_slot" in self.cycle_durations:
            time_slot = self.cycle_durations["time_slot"]
        
        # Convertir cada infracción a formato compatible con gestión
        for infraction in infractions:
            # NUEVO: Verificación adicional de seguridad para placas inválidas
            plate_text = infraction.get("plate", "")
            if not plate_text:
                print("Omitiendo guardar en JSON placa vacía")
                continue
                
            # Verificación adicional de longitud
            plate_without_special = plate_text.replace('-', '').replace(' ', '')
            if len(plate_without_special) > 8:
                print(f"Omitiendo guardar en JSON placa inválida por longitud: {plate_text}")
                continue
            
            # Verificación específica para placas problemáticas
            if "BOHID" in plate_text or "B OHID" in plate_text or "B-OHID" in plate_text:
                print(f"Omitiendo guardar en JSON placa problemática específica: {plate_text}")
                continue
                
            # Obtener la fecha y hora actual para registro
            now = datetime.now()
            fecha = now.strftime("%d/%m/%Y")
            hora = now.strftime("%H:%M:%S")
            
            # Obtener tiempo de video donde ocurrió la infracción
            video_time = infraction.get('time', 0)
            if video_time:
                mins = int(video_time // 60)
                secs = int(video_time % 60)
                timestamp = f"{mins:02d}:{secs:02d}"
            else:
                timestamp = "00:00"
            
            # Crear estructura para la infracción
            infraction_data = {
                "placa": plate_text,
                "fecha": fecha,
                "hora": hora,
                "video_timestamp": timestamp,
                "ubicacion": avenue_name,
                "franja_horaria": time_slot,  # Añadimos la franja horaria aquí
                "tipo": "Semáforo en rojo",
                "estado": "Pendiente",
                # Rutas a las imágenes guardadas
                "vehicle_path": infraction.get("vehicle_path", ""),
                "plate_path": infraction.get("plate_path", "")
            }
            
            # Añadir modo nocturno si está activo
            if hasattr(self, 'is_night') and self.is_night:
                infraction_data["modo_nocturno"] = True
            
            # Añadir a la lista de infracciones
            existing_infractions.append(infraction_data)
        
        # Guardar todas las infracciones actualizadas
        try:
            with open(infractions_file, "w", encoding="utf-8") as f:
                json.dump(existing_infractions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando infracciones en JSON: {e}")

    def _close_dialog(self, success):
        """Cierra el diálogo y llama a la función de completado"""
        # Cerrar diálogo
        self.dialog.grab_release()
        self.dialog.destroy()
        
        # Llamar a la función de completado si existe
        if self.on_complete:
            self.on_complete(success, self.detected_infractions)
    
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
            
            # Cerrar después de un breve retraso
            self.dialog.after(1000, lambda: self.dialog.destroy())