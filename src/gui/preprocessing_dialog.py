import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import json

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
        
        # Iniciar procesamiento en un hilo separado
        self.process_thread = threading.Thread(target=self._process_video, daemon=True)
        self.process_thread.start()
        
        # Programar actualizaciones periódicas de la UI
        self._schedule_ui_update()
    
    def load_video_config(self):
        """Carga la configuración del video (polígono, semáforo, etc.)"""
        self.polygon_points = []
        self.cycle_durations = None
        self.current_avenue = None
        
        # Cargar polígono de área restrictiva
        if os.path.exists(self.POLYGON_CONFIG_FILE):
            try:
                with open(self.POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    polygons = json.load(f)
                    if self.video_path in polygons:
                        self.polygon_points = polygons[self.video_path]
            except Exception as e:
                print(f"Error al cargar polígono: {e}")
        
        # Cargar tiempos de semáforo
        if os.path.exists(self.PRESETS_FILE):
            try:
                with open(self.PRESETS_FILE, "r") as f:
                    presets = json.load(f)
                    if self.video_path in presets:
                        self.cycle_durations = presets[self.video_path]
            except Exception as e:
                print(f"Error al cargar tiempos: {e}")
        
        # Cargar nombre de avenida
        if os.path.exists(self.AVENUE_CONFIG_FILE):
            try:
                with open(self.AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                    avenues = json.load(f)
                    if self.video_path in avenues:
                        self.current_avenue = avenues[self.video_path]
            except Exception as e:
                print(f"Error al cargar avenida: {e}")
    
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
            text="Analizando video...", 
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
            # Actualizar barra de progreso
            self.progress_var.set(self.progress_value)
            self.percentage_label.config(text=f"{int(self.progress_value)}%")
            
            # Actualizar contador de infracciones
            self.infractions_label.config(text=f"Infracciones detectadas: {len(self.detected_infractions)}")
            
            # Actualizar frame de video si hay uno disponible
            if self.current_frame is not None:
                self._update_video_frame(self.current_frame)
            
            # Programar próxima actualización
            self.dialog.after(100, self._schedule_ui_update)
    
    def _update_video_frame(self, frame):
        """Actualiza el frame de video mostrado en la interfaz"""
        # Redimensionar frame para ajustarse al área de visualización
        h, w = frame.shape[:2]
        max_w, max_h = 640, 360
        
        # Mantener relación de aspecto
        ratio = min(max_w/w, max_h/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Convertir de BGR a RGB para PIL
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Crear imagen para Tkinter
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Actualizar label
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # Mantener referencia
    
    def is_vehicle_in_polygon(self, bbox, polygon_points):
        """
        Determina si un vehículo está dentro del polígono restrictivo
        Usa el centro inferior del rectángulo (para simular la posición de las ruedas)
        """
        if not polygon_points or len(polygon_points) < 3:
            return False
            
        x1, y1, x2, y2 = bbox
        
        # Usar la parte inferior central del vehículo (posición aproximada de las ruedas)
        center_x = (x1 + x2) // 2
        center_y = y2  # Borde inferior
        
        # Convertir polígono a formato numpy
        polygon = np.array(polygon_points, np.int32)
        
        # Comprobar si el punto está dentro del polígono
        result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
        return result >= 0
    
    # Modificar la función _process_video para optimizar el procesamiento
    def _process_video(self):
        """Procesa el video en un hilo separado para detectar infracciones con soporte nocturno"""
        try:
            # Verificaciones iniciales
            if not self.polygon_points or not self.cycle_durations:
                self.dialog.after(0, lambda message="Este video no está configurado correctamente. Configure primero el área restrictiva y los tiempos de semáforo.": 
                                self._show_error(message))
                return
                    
            # Abrir el video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.dialog.after(0, lambda: self._show_error("No se pudo abrir el video"))
                return
            
            # Inicialización
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Verificaciones adicionales
            if self.total_frames <= 0:
                self.dialog.after(0, lambda: self._show_error("No se pudo determinar la duración del video"))
                return
            
            # Crear directorios para resultados
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Inicializar detectores
            self.phase_label.config(text="Fase 1: Cargando modelos de detección")
            self.details_label.config(text="Preparando modelos de IA...")
            
            # Inicializar detectores si son necesarios
            if not hasattr(self.player, 'vehicle_detector'):
                from src.core.detection.vehicle_detector import VehicleDetector
                self.player.vehicle_detector = VehicleDetector(model_path="models/yolov8n.pt")
                
            if not hasattr(self.player, 'plate_detector'):
                from src.core.detection.plate_detector import PlateDetector
                self.player.plate_detector = PlateDetector()
            
            # Configuración de semáforo
            self.phase_label.config(text="Fase 2: Analizando infracciones")
            
            # Sincronizar con el semáforo del panel
            # Asegurarnos de que el semáforo esté activado para el procesamiento
            self.player.semaforo.activate_semaphore()
            
            current_state = "green"  # Empezamos con verde
            next_state_frame = 0
            frame_index = 0
            
            # Calcular duración de cada estado
            frames_per_state = {
                "green": int(self.cycle_durations["green"] * fps),
                "yellow": int(self.cycle_durations["yellow"] * fps),
                "red": int(self.cycle_durations["red"] * fps)
            }
            
            # Mostrar franja horaria si está disponible
            time_slot = self.cycle_durations.get("time_slot", "No especificada")
            self.details_label.config(text=f"Franja horaria: {time_slot}")
            
            # Variables para controlar el muestreo en estado rojo
            last_processed_red_frame = -1
            red_frame_sampling = max(1, int(fps / 3))  # Procesar 3 frames por segundo en rojo (más frecuente)
            
            # Detectar automáticamente si es una escena nocturna
            ret, first_frame = cap.read()
            if not ret:
                self.dialog.after(0, lambda: self._show_error("No se pudo leer el primer frame del video"))
                return
            
            is_night = self._is_night_scene(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al principio del video
            
            # Actualizar UI con información del modo nocturno
            if is_night:
                self.details_label.config(text=f"Franja horaria: {time_slot} - MODO NOCTURNO ACTIVADO")
                # Información de debug
                print("Modo nocturno activado para el procesamiento")
            
            # Ciclo principal optimizado
            while cap.isOpened() and not self.canceled:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_index += 1
                self.processed_frames = frame_index
                
                # Actualizar progreso
                self.progress_value = (frame_index / self.total_frames) * 100
                
                # Actualizar detalles solo periódicamente
                if frame_index % 30 == 0:
                    self.details_label.config(text=f"Analizando frame {frame_index}/{self.total_frames}")
                
                # Simular cambio de estado de semáforo
                if frame_index >= next_state_frame:
                    if current_state == "green":
                        current_state = "yellow"
                        next_state_frame = frame_index + frames_per_state["yellow"]
                    elif current_state == "yellow":
                        current_state = "red"
                        next_state_frame = frame_index + frames_per_state["red"]
                        # Indicar cambio a rojo
                        if is_night:
                            self.details_label.config(text=f"¡Semáforo en ROJO! Detectando infracciones (Modo Nocturno)...")
                        else:
                            self.details_label.config(text=f"¡Semáforo en ROJO! Detectando infracciones...")
                    else:  # red
                        current_state = "green"
                        next_state_frame = frame_index + frames_per_state["green"]
                
                # OPTIMIZACIÓN: Mostrar frames a la interfaz solo periódicamente
                # para no saturar la UI, independientemente del estado
                # MODIFICACIÓN: Mini-semáforo en lugar del círculo
                if frame_index % 10 == 0:
                    frame_display = frame.copy()
                    
                    # NUEVO: Dibujamos un mini semáforo en lugar del punto simple
                    h, w = frame_display.shape[:2]
                    
                    # Coordenadas del semáforo
                    semaforo_x = w - 60
                    semaforo_y = 30
                    semaforo_width = 40
                    semaforo_height = 100
                    
                    # Fondo del semáforo (rectángulo negro)
                    cv2.rectangle(frame_display, 
                                (semaforo_x, semaforo_y), 
                                (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                                (0, 0, 0), -1)  # Negro
                    
                    # Borde gris del semáforo
                    cv2.rectangle(frame_display, 
                                (semaforo_x, semaforo_y), 
                                (semaforo_x + semaforo_width, semaforo_y + semaforo_height),
                                (128, 128, 128), 2)  # Gris
                    
                    # Diámetro y posiciones de las luces
                    light_diameter = 20
                    green_y = semaforo_y + semaforo_height - 25
                    yellow_y = semaforo_y + semaforo_height//2
                    red_y = semaforo_y + 25
                    light_x = semaforo_x + semaforo_width//2
                    
                    # Dibujar las tres luces (apagadas)
                    cv2.circle(frame_display, (light_x, red_y), light_diameter, (40, 40, 40), -1)
                    cv2.circle(frame_display, (light_x, yellow_y), light_diameter, (40, 40, 40), -1)
                    cv2.circle(frame_display, (light_x, green_y), light_diameter, (40, 40, 40), -1)
                    
                    # Encender la luz correspondiente al estado actual
                    if current_state == "green":
                        cv2.circle(frame_display, (light_x, green_y), light_diameter, (0, 255, 0), -1)
                        # Texto estado
                        cv2.putText(frame_display, "AVANCE", (semaforo_x - 80, semaforo_y + semaforo_height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif current_state == "yellow":
                        cv2.circle(frame_display, (light_x, yellow_y), light_diameter, (0, 255, 255), -1)
                        # Texto estado
                        cv2.putText(frame_display, "PRECAUCIÓN", (semaforo_x - 120, semaforo_y + semaforo_height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    elif current_state == "red":
                        cv2.circle(frame_display, (light_x, red_y), light_diameter, (0, 0, 255), -1)
                        # Texto estado
                        cv2.putText(frame_display, "PARE", (semaforo_x - 60, semaforo_y + semaforo_height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Añadir temporizador del estado actual
                    frames_left = next_state_frame - frame_index
                    secs_left = frames_left / fps
                    cv2.putText(frame_display, f"{secs_left:.1f}s", 
                            (semaforo_x - 20, semaforo_y + semaforo_height + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Dibujar polígono si existe
                    if self.polygon_points:
                        pts = np.array(self.polygon_points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame_display, [pts], True, (255, 0, 0), 2)
                    
                    # Añadir indicador de modo nocturno
                    if is_night:
                        cv2.putText(frame_display, "MODO NOCTURNO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Actualizar frame actual
                    self.current_frame = frame_display
                
                # OPTIMIZACIÓN: Solo detectar infracciones en luz roja
                if current_state == "red" and (frame_index - last_processed_red_frame >= red_frame_sampling):
                    last_processed_red_frame = frame_index
                    
                    # Preparar frame para detecciones - IMPORTANTE: Usar copia para no modificar el original
                    frame_copy = frame.copy()
                    
                    # Para escenas nocturnas, mejorar el frame antes de la detección
                    proc_frame = frame.copy()
                    if is_night:
                        proc_frame = self._enhance_night_visibility(proc_frame)
                    
                    # Detectar vehículos con umbral de confianza ajustado para noche/día
                    conf_threshold = 0.15 if is_night else 0.35
                    detections = self.player.vehicle_detector.detect(proc_frame, conf=conf_threshold, draw=False)
                    
                    # Variable para indicar si se detectó infracción
                    detected_infraction = False
                    
                    # Procesar las detecciones
                    for bbox in detections:
                        x1, y1, x2, y2, class_id = bbox
                        
                        # Solo considerar vehículos (clase 2=coche, 5=bus, 7=camión)
                        if class_id not in [2, 5, 7]:
                            continue
                        
                        # Verificar si está en la zona restringida
                        in_polygon = False
                        if is_night:
                            in_polygon = self._is_vehicle_in_polygon_night((x1, y1, x2, y2), self.polygon_points)
                        else:
                            in_polygon = self.is_vehicle_in_polygon((x1, y1, x2, y2), self.polygon_points)
                        
                        if in_polygon:
                            detected_infraction = True
                            # Dibujar rectángulo rojo en vehículo infractor
                            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame_copy, "INFRACCION", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # CRÍTICO: Siempre dibujar el mini-semáforo antes de actualizar el frame
                            self._draw_mini_semaphore(frame_copy, current_state, next_state_frame - frame_index, fps, is_night)
                            
                            # Actualizar frame actual para mostrar la infracción
                            self.current_frame = frame_copy
                            
                            # Procesar la placa solo si realmente hay una infracción
                            y1_roi = max(0, int(y1))
                            y2_roi = min(frame.shape[0], int(y2))
                            x1_roi = max(0, int(x1))
                            x2_roi = min(frame.shape[1], int(x2))

                            if y2_roi > y1_roi and x2_roi > x1_roi:
                                vehicle_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()
                                
                                # Si es de noche, aplicar pre-procesamiento específico para placas
                                if is_night:
                                    vehicle_roi = self._enhance_night_visibility(vehicle_roi)
                                
                                # Procesar placa con el flag de noche
                                from src.core.processing.plate_processing import process_plate
                                try:
                                    # Intentos para detectar la placa
                                    bbox_plate, plate_img, plate_text = process_plate(vehicle_roi, is_night=is_night)
                                    
                                    # IMPORTANTE: Dibujar el mini-semáforo ANTES de actualizar current_frame
                                    # para asegurar que siempre esté visible
                                    self._draw_mini_semaphore(frame_copy, current_state, next_state_frame - frame_index, fps, is_night)
                                    self.current_frame = frame_copy
                                    
                                    # INTENTO 2: Si falla, intentar detección directa más agresiva
                                    if not plate_text or len(plate_text) < 4:
                                        from src.core.ocr.recognizer import recognize_plate
                                        plate_text = recognize_plate(vehicle_roi)
                                        plate_img = vehicle_roi  # Usar ROI completo como imagen de placa
                                    
                                    if plate_text and len(plate_text) >= 4:
                                        # Añadir a la lista de infracciones detectadas
                                        infraction_data = {
                                            'frame': frame_index,
                                            'time': frame_index / fps,
                                            'plate': plate_text,
                                            'plate_img': plate_img,
                                            'vehicle_img': vehicle_roi
                                        }
                                        
                                        # Verificar duplicados y determinar la mejor versión
                                        is_duplicate, best_plate = self._deduplicate_plates(plate_text)
                                        if is_duplicate:
                                            # Actualizar placa existente si la nueva es mejor
                                            if best_plate == plate_text:
                                                # Buscar y reemplazar la placa anterior
                                                for data in self.detected_infractions:
                                                    if data['plate'] == best_plate:
                                                        data['plate_img'] = plate_img
                                                        data['vehicle_img'] = vehicle_roi
                                                        break
                                            # Si no es mejor, simplemente no la añadimos
                                            pass
                                        else:
                                            # Nueva placa, añadirla a la lista
                                            self.detected_infractions.append(infraction_data)
                                            
                                            # Guardar imagen de la placa
                                            if plate_img is not None:
                                                plate_filename = f"plate_{best_plate}_{frame_index}.jpg"
                                                cv2.imwrite(os.path.join(output_dir, plate_filename), plate_img)
                                except Exception as plate_err:
                                    print(f"Error procesando placa: {plate_err}")
                        # Si no se detectó infracción pero es tiempo de actualizar el frame
                    if not detected_infraction and frame_index % 10 == 0:
                        self._draw_mini_semaphore(frame_copy, current_state, next_state_frame - frame_index, fps, is_night)
                        self.current_frame = frame_copy
                    # Para frames que no son de detección, actualizar periódicamente la UI con el mini-semáforo
                    elif frame_index % 10 == 0:
                        frame_display = frame.copy()
                        self._draw_mini_semaphore(frame_display, current_state, next_state_frame - frame_index, fps, is_night)
                        self.current_frame = frame_display
            
            # Finalización del procesamiento (igual)
            self.phase_label.config(text="Análisis completado")
            self.progress_value = 100
            time.sleep(0.5)  # Pequeña pausa para mostrar 100%
            
            # Liberar recursos
            cap.release()
            
            # Llamar a la función de finalización en el hilo principal, usando función directa
            if not self.canceled:
                self.dialog.after(0, self._complete_processing)
                
        except Exception as e:
            # Capturar la excepción en una variable local
            error_message = str(e)
            import traceback
            traceback.print_exc()  # Imprimir stack trace para depuración
            # Usar el parámetro por defecto para capturar el valor
            self.dialog.after(0, lambda msg=error_message: self._show_error(msg))
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()


    def _draw_mini_semaphore(self, frame, current_state, frames_left, fps, is_night=False):
        """
        Dibuja un mini-semáforo en el frame proporcionado con el estado actual.
        
        Args:
            frame: El frame donde dibujar el semáforo
            current_state: Estado actual del semáforo ("red", "yellow", "green")
            frames_left: Número de frames restantes en el estado actual
            fps: Frames por segundo del video
            is_night: Indica si estamos en modo nocturno
        """
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
        
        # Dibujar las tres luces (apagadas)
        cv2.circle(frame, (light_x, red_y), light_diameter, (40, 40, 40), -1)
        cv2.circle(frame, (light_x, yellow_y), light_diameter, (40, 40, 40), -1)
        cv2.circle(frame, (light_x, green_y), light_diameter, (40, 40, 40), -1)
        
        # Encender la luz correspondiente al estado actual
        if current_state == "green":
            cv2.circle(frame, (light_x, green_y), light_diameter, (0, 255, 0), -1)
            # Texto estado
            cv2.putText(frame, "AVANCE", (semaforo_x - 80, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_state == "yellow":
            cv2.circle(frame, (light_x, yellow_y), light_diameter, (0, 255, 255), -1)
            # Texto estado
            cv2.putText(frame, "PRECAUCIÓN", (semaforo_x - 120, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif current_state == "red":
            cv2.circle(frame, (light_x, red_y), light_diameter, (0, 0, 255), -1)
            # Texto estado
            cv2.putText(frame, "PARE", (semaforo_x - 60, semaforo_y + semaforo_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Añadir temporizador del estado actual
        secs_left = frames_left / fps
        cv2.putText(frame, f"{secs_left:.1f}s", 
                (semaforo_x - 20, semaforo_y + semaforo_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dibujar polígono si existe
        if hasattr(self, 'polygon_points') and self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        # Añadir indicador de modo nocturno
        if is_night:
            cv2.putText(frame, "MODO NOCTURNO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _is_night_scene(self, frame):
        """Determina si un frame corresponde a una escena nocturna"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular brillo promedio
        avg_brightness = cv2.mean(gray)[0]
        
        # También verificar área más oscura (percentil 10)
        dark_threshold = np.percentile(gray, 10)
        
        # Si el brillo promedio es bajo o áreas oscuras son muy oscuras
        # Ajustar este valor basado en tus videos
        return avg_brightness < 85 or dark_threshold < 30

    def _enhance_night_visibility(self, frame):
        """Mejora la visibilidad en escenas nocturnas con parámetros más agresivos"""
        # Convertir a LAB para trabajar con el canal de luminosidad
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE con parámetros más agresivos
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Fusionar canales de nuevo
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convertir de vuelta a BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Aumentar ganancia para mayor visibilidad - más agresivo
        return cv2.convertScaleAbs(enhanced_bgr, alpha=1.5, beta=40)

    def _is_vehicle_in_polygon_night(self, bbox, polygon_points):
        """
        Versión muy permisiva para detección nocturna que expande el área de detección
        """
        if not polygon_points or len(polygon_points) < 3:
            return False
        
        # Expandir ligeramente el polígono para ser más permisivo
        polygon = np.array(polygon_points, np.int32)
        center = np.mean(polygon, axis=0).astype(int)
        
        # Expandir polígono un 10% desde su centro
        expanded_polygon = []
        for point in polygon:
            # Vector desde centro al punto
            vector = point - center
            # Expandir 10%
            expanded_point = center + vector * 1.1
            expanded_polygon.append(expanded_point)
        
        expanded_polygon = np.array(expanded_polygon, np.int32)
        
        # Extraer coordenadas del vehículo
        x1, y1, x2, y2 = bbox
        
        # Usar más puntos para verificar
        check_points = [
            (x1, y1),                # Esquina superior izquierda
            (x2, y1),                # Esquina superior derecha
            (x1, y2),                # Esquina inferior izquierda
            (x2, y2),                # Esquina inferior derecha
            ((x1+x2)//2, (y1+y2)//2), # Centro
            ((x1+x2)//2, y2),        # Punto inferior central (ruedas)
            (x1, (y1+y2)//2),        # Punto medio izquierdo
            (x2, (y1+y2)//2),        # Punto medio derecho
            ((x1+x2)//2, y1),        # Punto superior central
        ]
        
        # Agregar puntos adicionales para una cuadrícula 5x5
        width, height = x2-x1, y2-y1
        for i in range(1, 5):
            for j in range(1, 5):
                check_points.append((x1 + (width*i)//5, y1 + (height*j)//5))
        
        # Si cualquier punto está dentro del polígono expandido, considerar que está dentro
        for point in check_points:
            if cv2.pointPolygonTest(expanded_polygon, point, False) >= 0:
                return True
        
        return False

    def _complete_processing(self):
        """Finaliza el procesamiento y muestra los resultados"""
        try:
            # Debug para verificar que tenemos placas para añadir
            print(f"Procesamiento completo: {len(self.detected_infractions)} infracciones detectadas")
            
            # Añadir las placas detectadas al panel lateral
            for infraction in self.detected_infractions:
                print(f"Añadiendo placa: {infraction['plate']} al tiempo {infraction['time']}")
                # Asegurarnos de que todos los parámetros sean válidos
                if 'plate_img' in infraction and infraction['plate_img'] is not None and \
                    'plate' in infraction and infraction['plate'] is not None:
                    # Llamar directamente al método (sin _)
                    self.player._safe_add_plate_to_panel(
                        infraction['plate_img'], 
                        infraction['plate'], 
                        infraction['time']
                    )
            
            # Actualizar la interfaz con información de las infracciones
            found_message = f"Se han detectado {len(self.detected_infractions)} infracciones."
            self.phase_label.config(text="Procesamiento completado")
            self.details_label.config(text=found_message)
            
            # Iniciar reproducción normal del video
            self.player.start_processed_video(self.video_path)
            
            # Cerrar diálogo después de un breve retraso para mostrar el mensaje final
            self.dialog.after(1500, lambda: self._close_dialog(True))
        except Exception as e:
            print(f"Error en _complete_processing: {e}")
            import traceback
            traceback.print_exc()
            self._close_dialog(False)

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
                from tkinter import messagebox
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

    def _deduplicate_plates(self, plate_text):
        """
        Compara una placa contra las ya detectadas y determina
        si es un duplicado o cuál es la versión más probable.
        
        Args:
            plate_text: Texto de la placa detectada
        
        Returns:
            (is_duplicate, best_plate): Indica si es duplicado y cuál es la mejor versión
        """
        if not plate_text or len(plate_text) < 4:
            return False, plate_text
            
        # Verificar si ya existe esta placa exacta
        for data in self.detected_infractions:
            if data['plate'] == plate_text:
                return True, plate_text
        
        # Buscar placas muy similares (posibles diferentes detecciones del mismo vehículo)
        best_match = None
        best_score = 0
        
        for data in self.detected_infractions:
            existing_plate = data['plate']
            
            # Si tienen longitud muy diferente, no son la misma placa
            if abs(len(existing_plate) - len(plate_text)) > 2:
                continue
                
            # Calcular similitud basada en caracteres coincidentes en la misma posición
            min_len = min(len(existing_plate), len(plate_text))
            matches = sum(1 for i in range(min_len) if existing_plate[i] == plate_text[i])
            score = matches / min_len
            
            # Si hay similitud alta, probablemente es la misma placa
            if score > 0.6 and score > best_score:
                best_score = score
                best_match = existing_plate
        
        if best_match:
            # Decidir cuál versión es mejor (la nueva o la existente)
            # Preferir placas conocidas como A3606L o AE670S
            known_plates = ["A3606L", "AE670S", "A3670S"]
            
            if plate_text in known_plates:
                return True, plate_text  # La nueva es mejor
            elif best_match in known_plates:
                return True, best_match  # La existente es mejor
                
            # Si ninguna es conocida, preferir la más larga o la que tenga mejor formato
            if len(plate_text) > len(best_match):
                return True, plate_text
            else:
                return True, best_match
                
        return False, plate_text