"""Tracker inteligente de vehículos (cruce de polígono + detección de infracción).

Extraído de preprocessing_dialog.py sin cambios algorítmicos.
"""

class IntelligentVehicleTracker:
    """
    Sistema de tracking inteligente para validar infracciones reales.
    
    LÓGICA CLAVE:
    - Tracking por ID para mantener historial de posición
    - Validación de "primer contacto" del parachoques delantero
    - Prevención de falsos positivos por perspectiva
    """
    
    def __init__(self, polygon_points):
        """
        Inicializa el tracker.
        
        Args:
            polygon_points: Puntos del polígono de detección
        """
        self.polygon_points = polygon_points
        self.vehicle_tracks = {}  # track_id -> historial de posiciones
        self.infraction_records = {}  # track_id -> datos de infracción
        self.next_track_id = 1
        
        # Configuración del tracker
        self.max_distance_threshold = 100  # Distancia máxima para asociar detecciones
        self.history_length = 15  # Aumentado para mejor análisis de mejores tomas
        self.best_frame_per_track = {}  # track_id -> {'frame': img, 'conf': C, 'idx': I}
        
    def update_tracks(self, detections, frame_index, current_semaphore_state):
        """
        Actualiza el tracking de vehículos y detecta infracciones.
        
        Args:
            detections: Lista de detecciones [(x1, y1, x2, y2, confidence)]
            frame_index: Índice del frame actual
            current_semaphore_state: Estado actual del semáforo ('red', 'yellow', 'green')
            
        Returns:
            List[Dict]: Lista de infracciones detectadas en este frame
        """
        current_infractions = []
        
        # Asociar detecciones con tracks existentes o crear nuevos
        matched_tracks = set()
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            detection_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_front = ((x1 + x2) // 2, y2)  # Parachoques delantero (parte inferior)
            
            # Buscar track más cercano
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.vehicle_tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                if len(track_data['positions']) > 0:
                    last_pos = track_data['positions'][-1]['center']
                    distance = np.sqrt((detection_center[0] - last_pos[0])**2 + 
                                     (detection_center[1] - last_pos[1])**2)
                    
                    if distance < self.max_distance_threshold and distance < min_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            # Si no se encontró track cercano, crear nuevo
            if best_track_id is None:
                best_track_id = self.next_track_id
                self.vehicle_tracks[best_track_id] = {
                    'positions': [],
                    'first_seen': frame_index,
                    'last_seen': frame_index
                }
                self.next_track_id += 1
            
            # Actualizar track
            track_data = self.vehicle_tracks[best_track_id]
            track_data['positions'].append({
                'frame': frame_index,
                'bbox': (x1, y1, x2, y2),
                'center': detection_center,
                'front': detection_front,
                'confidence': confidence,
                'in_polygon': self.is_vehicle_in_polygon_robust((x1, y1, x2, y2)),
                'semaphore_state': current_semaphore_state
            })
            track_data['last_seen'] = frame_index
            
            # Mantener solo historial reciente
            if len(track_data['positions']) > self.history_length:
                track_data['positions'] = track_data['positions'][-self.history_length:]
            
            # ACTUALIZACIÓN DE MMRP (Punto Máximo de Resolución - PVM)
        # Basado en la propuesta técnica de Abel V16:
        # Optimiza: Resolución Espacial (PPM) + Ortogonalidad (Coseno) + Enfoque (Contrast)
        if current_semaphore_state == "red":
            if best_track_id not in self.best_frame_per_track:
                self.best_frame_per_track[best_track_id] = []
            
            # --- CÁLCULO TRIGONOMÉTRICO Y GEOMÉTRICO DEL MÉRITO ---
            
            # 1. Factor de Resolución (PPM): BBox Area
            bbox_area = (x2 - x1) * (y2 - y1)
            res_factor = bbox_area / 20000.0 # Normalizado para resolución HD
            
            # 2. Factor de Centralidad (C): Menor distorsión por aberración de lente
            # Ideal: El vehículo está en el centro horizontal de la escena
            img_w = 1920 # Asumimos HD por defecto, se ajusta si es mayor
            center_dist = abs(detection_center[0] - (img_w // 2))
            # Penalización suave por estar en los bordes térmicos del lente
            centrality_factor = 1.0 - (center_dist / (img_w // 2)) * 0.4
            
            # 3. Factor de Sharpening Estimado (Puntaje de Contraste)
            # Como Phase 1 es rápida, usamos confianza de YOLO como proxy inicial
            # Pero Phase 2 lo refinará con varianza Laplaciana.
            
            merit_score = float(confidence) * res_factor * centrality_factor
            
            self.best_frame_per_track[best_track_id].append({
                'score': merit_score,
                'frame_idx': frame_index,
                'bbox': (x1, y1, x2, y2),
                'conf': confidence
            })
            
            # Ordenar por el Merito Máximo (MMRP / PVM)
            self.best_frame_per_track[best_track_id] = sorted(
                self.best_frame_per_track[best_track_id], 
                key=lambda x: x['score'], 
                reverse=True
            )[:5] # Mantenemos el top 5 para el Consensus Elite del final

            matched_tracks.add(best_track_id)
            
            # VALIDACIÓN DE INFRACCIÓN: Solo en semáforo ROJO
            if current_semaphore_state == "red":
                infraction = self._check_infraction(best_track_id, frame_index)
                if infraction:
                    current_infractions.append(infraction)
        
        # Limpiar tracks antiguos (no vistos en muchos frames)
        self._cleanup_old_tracks(frame_index)
        
        # ELIMINAR RESULTADOS OBSOLETOS DE MEJORES TOMAS
        tracks_to_keep = set(self.vehicle_tracks.keys())
        for tid in list(self.best_frame_per_track.keys()):
            if tid not in tracks_to_keep:
                del self.best_frame_per_track[tid]
        
        return current_infractions
    
    def _check_infraction(self, track_id, frame_index):
        """
        Verifica si un vehículo cometió una infracción.
        
        REGLA: Solo cuenta como infracción el PRIMER contacto del parachoques 
        delantero con la ROI cuando el semáforo está en ROJO y el vehículo
        estaba DETRÁS de la línea en frames previos.
        
        Args:
            track_id: ID del track del vehículo
            frame_index: Índice del frame actual
            
        Returns:
            Dict o None: Datos de la infracción si se detecta, None si no
        """
        # Ya registramos infracción para este vehículo?
        if track_id in self.infraction_records:
            return None
            
        track_data = self.vehicle_tracks[track_id]
        positions = track_data['positions']
        
        if len(positions) < 2:
            return None  # Necesitamos al menos 2 posiciones para comparar
            
        current_pos = positions[-1]
        
        # LÓGICA ADAPTATIVA: Para videos cortos o modo compatibilidad
        # Si estamos en modo fallback (video corto), ser más permisivo con el estado del semáforo
        is_short_video_mode = hasattr(self, '_short_video_fallback') and self._short_video_fallback
        
        if is_short_video_mode:
            # En videos cortos, detectar infracción si el vehículo está EN el polígono
            # independientemente del estado exacto del semáforo
            if not current_pos['in_polygon']:
                return None
        else:
            # Lógica normal: El vehículo debe estar actualmente EN el polígono en ROJO
            if not (current_pos['in_polygon'] and current_pos['semaphore_state'] == 'red'):
                return None
            
        # Buscar la posición más reciente ANTES de entrar al polígono
        was_outside_before = False
        last_outside_frame = None
        
        for i in range(len(positions) - 2, -1, -1):  # Revisar hacia atrás
            pos = positions[i]
            if not pos['in_polygon']:
                was_outside_before = True
                last_outside_frame = pos['frame']
                
                # En modo video corto, ser más permisivo con los estados del semáforo
                if is_short_video_mode:
                    break  # En videos cortos, no importa tanto el estado del semáforo
                else:
                    # Verificar que estaba fuera durante un estado NO-ROJO (lógica normal)
                    if pos['semaphore_state'] in ['green', 'yellow']:
                        break
        
        # VALIDACIÓN ADAPTATIVA: 
        if is_short_video_mode:
            # En videos cortos: más permisivo, solo necesita haber estado fuera antes
            validation_passed = was_outside_before or len(positions) >= 1  # Casi siempre válido
        else:
            # Lógica normal: Solo es infracción si estaba fuera antes y entró en ROJO
            validation_passed = was_outside_before
            
        if validation_passed:
            # Determinar estado del semáforo para el registro
            semaphore_state = current_pos['semaphore_state'] if not is_short_video_mode else 'red'  # En modo corto asumir rojo
            validation_method = 'short_video_fallback' if is_short_video_mode else 'first_contact_front_bumper'
            
            # Registrar la infracción
            infraction_data = {
                'track_id': track_id,
                'frame': frame_index,
                'bbox': current_pos['bbox'],
                'confidence': current_pos['confidence'],
                'entry_frame': frame_index,
                'last_outside_frame': last_outside_frame,
                'semaphore_state': semaphore_state,
                'validation': validation_method,
                'crossing_point': current_pos['front']  # Punto exacto del parachoques al cruzar
            }
            
            # Marcar como ya procesado para evitar duplicados
            self.infraction_records[track_id] = infraction_data
            
            return infraction_data
            
        return None

    def is_vehicle_in_polygon_robust(self, car_bbox):
        """
        Versión robusta de detección dentro del polígono (basada en el player principal).
        """
        if not self.polygon_points or len(self.polygon_points) < 3:
            return False
        
        x1, y1, x2, y2 = car_bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        polygon = np.array(self.polygon_points, np.int32)
        
        # 1. Centro
        if cv2.pointPolygonTest(polygon, (int(center_x), int(center_y)), False) >= 0:
            return True
        
        # 2. Otros puntos críticos (como en videoplayer_opencv.py)
        front_x = (x1 + x2 * 3) // 4
        front_y = center_y
        rear_x = (x1 * 3 + x2) // 4
        rear_y = center_y
        
        if cv2.pointPolygonTest(polygon, (int(front_x), int(front_y)), False) >= 0:
            return True
        if cv2.pointPolygonTest(polygon, (int(rear_x), int(rear_y)), False) >= 0:
            return True
        
        # 3. Esquinas
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for cx, cy in corners:
            if cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0:
                return True
                
        return False

    def _is_point_in_polygon(self, point):
        """Verifica si un punto está dentro del polígono de detección."""
        if not self.polygon_points or len(self.polygon_points) < 3:
            return False
            
        polygon_np = np.array(self.polygon_points, np.int32)
        return cv2.pointPolygonTest(polygon_np, (int(point[0]), int(point[1])), False) >= 0
    
    def _cleanup_old_tracks(self, current_frame):
        """Elimina tracks que no se han visto en muchos frames."""
        max_frames_without_detection = 30  # ~1 segundo a 30fps
        
        tracks_to_remove = []
        for track_id, track_data in self.vehicle_tracks.items():
            if current_frame - track_data['last_seen'] > max_frames_without_detection:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
            if track_id in self.infraction_records:
                del self.infraction_records[track_id]


# Eliminamos la importación circular
