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
                # Enviar cada frame procesado para visualización en tiempo real
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
                
                # Detectar vehículos (sin el parámetro classes que causaba el error)
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
                        # Clases típicas de YOLO: 2=coche, 5=bus, 7=camión
                        if isinstance(class_id, (int, float)):
                            class_id = int(class_id)
                            if class_id in [2, 5, 7]:
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
                            
                            # Procesar placa
                            try:
                                from src.core.processing.plate_processing import process_plate
                                
                                # Verificar si está disponible el módulo de super-resolución
                                try:
                                    from src.core.processing.resolution_process import enhance_plate_image
                                except ImportError:
                                    print("Módulo de super-resolución no disponible, usando procesamiento estándar")
                                    enhance_plate_image = None
                                
                                # Detectar placa en el vehículo
                                bbox_plate, plate_img, plate_text = process_plate(vehicle_roi, is_night=self.is_night)
                                
                                # Si no encontró texto o es muy corto, intentar con reconocedor alternativo
                                if not plate_text or len(plate_text) < 4:
                                    from src.core.ocr.recognizer import recognize_plate
                                    
                                    # Intentar mejorar la imagen antes del reconocimiento alternativo
                                    if enhance_plate_image is not None:
                                        enhanced_roi = enhance_plate_image(vehicle_roi, is_night=self.is_night)
                                        plate_text = recognize_plate(enhanced_roi)
                                        plate_img = enhanced_roi
                                    else:
                                        plate_text = recognize_plate(vehicle_roi)
                                
                                # Verificar que la placa sea válida
                                if plate_text and len(plate_text) >= 4:
                                    # Normalizar texto de placa
                                    plate_text = self._normalize_plate_text(plate_text)
                                    
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
                                        if enhance_plate_image is not None:
                                            enhanced_plate = enhance_plate_image(plate_img, is_night=self.is_night, output_path=plate_path)
                                        else:
                                            # Si no está disponible el módulo, guardar la placa original
                                            cv2.imwrite(plate_path, plate_img)
                                            enhanced_plate = plate_img
                                        
                                        # Guardar la imagen del vehículo con nombre ÚNICO
                                        vehicle_filename = f"vehicle_{plate_text}.jpg"
                                        vehicle_path = os.path.join(vehicles_dir, vehicle_filename)
                                        cv2.imwrite(vehicle_path, vehicle_roi)
                                        
                                        # Guardar infracción detectada con rutas de archivos
                                        infraction_data = {
                                            'frame': absolute_frame,
                                            'time': absolute_frame / self.fps,
                                            'plate': plate_text,
                                            'plate_img': enhanced_plate.copy(),  # Usar la versión mejorada
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
                                        
                                        # Enviar detección a la UI
                                        self.result_queue.put(("frame_update", (detection_frame, segment_id, processed, total_to_process)))
                                    else:
                                        print(f"Placa {plate_text} ya fue detectada globalmente, omitiendo")
                            except Exception as e:
                                print(f"Error procesando placa: {e}")
                                import traceback
                                traceback.print_exc()
            
            segment_cap.release()
            
            # Enviar resultados a la cola principal
            self.result_queue.put(("segment_complete", (segment_id, local_infractions)))
            print(f"Segmento {segment_id} completado con {len(local_infractions)} infracciones")
            return local_infractions, segment_id
            
        except Exception as e:
            print(f"Error en segment {segment_id}: {e}")
            import traceback
            traceback.print_exc()
            self.result_queue.put(("segment_complete", (segment_id, [])))
            return [], segment_id
    
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
        
        # Eliminar caracteres no alfanuméricos excepto guión
        normalized = ''.join(c for c in normalized if c.isalnum() or c == '-')
        
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
        
        return normalized
    
    def _consolidate_plate_detections(self):
        """Consolida múltiples detecciones de la misma placa para mejorar la precisión"""
        if not self.detected_infractions:
            return []
        
        # Agrupar por placas similares
        plate_groups = {}
        
        for infraction in self.detected_infractions:
            plate = infraction['plate']
            best_match = None
            best_similarity = 0
            
            # Buscar el grupo más similar
            for group_key in plate_groups.keys():
                import difflib
                similarity = difflib.SequenceMatcher(None, plate, group_key).ratio()
                if similarity > 0.7 and similarity > best_similarity:  # 70% similar
                    best_similarity = similarity
                    best_match = group_key
            
            # Añadir al grupo existente o crear uno nuevo
            if best_match:
                plate_groups[best_match].append(infraction)
            else:
                plate_groups[plate] = [infraction]
        
        # Para cada grupo, elegir la mejor detección o consolidar información
        consolidated_infractions = []
        
        for group_key, detections in plate_groups.items():
            if len(detections) == 1:
                # Solo una detección, añadirla directamente
                consolidated_infractions.append(detections[0])
            else:
                # Múltiples detecciones, elegir la mejor calidad o consolidar
                best_detection = max(detections, key=lambda x: 
                                    cv2.Laplacian(x['plate_img'], cv2.CV_64F).var())
                
                # Conservar la detección de mejor calidad
                consolidated_infractions.append(best_detection)
                
                # Añadir este método al procesamiento final en _finalize_processing
        
        return consolidated_infractions
    
    def _filter_segment_duplicates(self, infractions):
        """Filtra duplicados dentro de un mismo segmento"""
        if not infractions:
            return []
            
        filtered = []
        seen_plates = set()
        
        for infraction in infractions:
            plate = infraction['plate']
            
            # Verificar si ya existe esta placa exacta en este segmento
            if plate not in seen_plates:
                seen_plates.add(plate)
                filtered.append(infraction)
                
        return filtered
    
    def _finalize_processing(self):
        """Finaliza el procesamiento después de que todos los segmentos estén completos"""
        try:
            # NUEVO: Usar el método de consolidación para mejorar precisión
            final_infractions = self._consolidate_plate_detections()

            # Identificar y filtrar placas duplicadas entre todos los segmentos
            final_infractions = []
            global_detected_plates = set()
            
            for infraction in self.detected_infractions:
                plate = infraction['plate']
                
                # Verificar si ya hemos añadido esta placa
                if plate not in global_detected_plates:
                    global_detected_plates.add(plate)
                    final_infractions.append(infraction)
            
            # Actualizar la lista de infracciones detectadas
            self.detected_infractions = final_infractions
            
            # Llamar a _complete_processing
            self.dialog.after(0, self._complete_processing)
        except Exception as e:
            print(f"Error en _finalize_processing: {e}")
            import traceback
            traceback.print_exc()
    
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
            # Debug para verificar que tenemos placas para añadir
            print(f"Procesamiento completo: {len(self.detected_infractions)} infracciones detectadas")
            
            # Conjunto para seguir placas únicas procesadas
            unique_plates = set()
            unique_infractions = []
            
            # Filtrar solo infracciones únicas
            for infraction in self.detected_infractions:
                plate_text = infraction['plate']
                if plate_text not in unique_plates:
                    unique_plates.add(plate_text)
                    unique_infractions.append(infraction)
                    print(f"Añadiendo placa única: {plate_text}")
                else:
                    print(f"Omitiendo placa duplicada: {plate_text}")
            
            print(f"Infracciones únicas a añadir: {len(unique_infractions)} de {len(self.detected_infractions)}")
            
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
            
            # Añadir las placas detectadas al panel lateral (solo las únicas)
            for infraction in unique_infractions:
                print(f"Procesando placa: {infraction['plate']} al tiempo {infraction['time']}")
                
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
                    
                    # Inicializar estructura para esta placa si no existe
                    if plate_text not in self.player.plate_detection_history:
                        # CORRECCIÓN: Añadir datos precisos de tiempo
                        detection_time = self.player.detection_start_time
                        if infraction['time'] is not None:
                            detection_time = self.player.detection_start_time + infraction['time']
                            
                        registration_time = time.time()
                        proc_time = registration_time - detection_time
                        
                        # Añadir al historial de tiempos de registro para estadísticas
                        self.player.registration_times.append(proc_time)
                        
                        self.player.plate_detection_history[plate_text] = {
                            "count": 1,  # Siempre 1, evitamos duplicados
                            "first_detection": infraction['time'],
                            "last_detection": infraction['time'],
                            "vehicle_img": infraction['vehicle_img'],
                            "detection_time": detection_time,
                            "registration_time": registration_time,
                            "processing_time": proc_time
                        }
                        
                        # Si existen rutas de archivos, asegurarse de guardarlas en el historial
                        if 'vehicle_path' in infraction and infraction['vehicle_path']:
                            self.player.plate_detection_history[plate_text]["vehicle_path"] = infraction['vehicle_path']
                        
                        if 'plate_path' in infraction and infraction['plate_path']:
                            self.player.plate_detection_history[plate_text]["plate_path"] = infraction['plate_path']
                        
                        # Llamar directamente al método para añadir la placa al panel
                        self.player._safe_add_plate_to_panel(
                            infraction['plate_img'], 
                            plate_text, 
                            infraction['time']
                        )
                    else:
                        print(f"Placa {plate_text} ya está en el historial, no se añade duplicado")
            
            # NUEVO: Guardar todas las infracciones detectadas en el archivo JSON
            self._save_infractions_to_json(unique_infractions)
                        
            # CORRECCIÓN: Actualizar indicadores de rendimiento después de añadir todas las placas
            if hasattr(self.player, "performance_indicators"):
                # Calcular el promedio del tiempo de registro si hay datos
                avg_proc_time = 0.0
                if self.player.registration_times:
                    avg_proc_time = sum(self.player.registration_times) / len(self.player.registration_times)
                    
                # Actualizar indicadores
                self.player.performance_indicators = {
                    "TI": len(unique_plates),  # Número exacto de infracciones
                    "TR": avg_proc_time,      # Tiempo promedio de registro
                    "IR": 0.0                  # Reiniciar índice de reincidencia
                }
                
                # Forzar actualización del panel de rendimiento
                if hasattr(self.player, "_update_metrics_panel"):
                    self.player._update_metrics_panel()
            
            # Actualizar la interfaz con información de las infracciones
            found_message = f"Se han detectado {len(unique_plates)} infracciones únicas."
            self.phase_label.config(text="Procesamiento completado")
            self.details_label.config(text=found_message)
            
            # Registrar el número correcto en los logs
            print(f"Análisis completado: {len(unique_plates)} infracciones detectadas")
            
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
        
        # Convertir cada infracción a formato compatible con gestión
        for infraction in infractions:
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
                "placa": infraction["plate"],
                "fecha": fecha,
                "hora": hora,
                "video_timestamp": timestamp,
                "ubicacion": avenue_name,
                "tipo": "Semáforo en rojo",
                "estado": "Pendiente",
                # Rutas a las imágenes guardadas
                "vehicle_path": infraction.get("vehicle_path", ""),
                "plate_path": infraction.get("plate_path", "")
            }
            
            # Añadir a la lista de infracciones
            existing_infractions.append(infraction_data)
        
        # Guardar todas las infracciones actualizadas
        try:
            with open(infractions_file, "w", encoding="utf-8") as f:
                json.dump(existing_infractions, f, indent=2, ensure_ascii=False)
            
            print(f"Se guardaron {len(infractions)} infracciones en {infractions_file}")
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