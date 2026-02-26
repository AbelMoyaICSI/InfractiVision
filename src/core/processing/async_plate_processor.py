"""
🚀 Pipeline Asíncrono de Procesamiento de Placas
Aprovecha los intervalos VERDE/AMARILLO para pre-procesar infracciones.

Concepto de Abel (2026):
- Durante ROJO: Captura infracciones
- Durante VERDE/AMARILLO: Procesa recortes + Super-Resolución
- Fase 2: Solo OCR (ultra rápido)
"""

import threading
import queue
import time
import cv2
from collections import deque


class AsyncPlateProcessor:
    """
    Procesador asíncrono que trabaja durante los tiempos vacíos del semáforo.
    """
    
    def __init__(self):
        # Cola de infracciones pendientes de procesar
        self.pending_queue = queue.Queue()
        
        # Resultados ya procesados (listos para OCR)
        self.processed_results = {}  # track_id -> {'plate_crop': img, 'vehicle_img': img, 'sr_applied': bool}
        
        # Control del worker
        self.worker_thread = None
        self.running = False
        self.current_semaphore_state = "unknown"
        
        # Estadísticas
        self.stats = {
            'processed_count': 0,
            'sr_applied_count': 0,
            'avg_processing_time_ms': 0
        }
        
        # Cargar FSRCNN
        self.upscaler = None
        self._load_upscaler()
        
        # Cargar detector de placas
        self.plate_detector = None
        self._load_plate_detector()
        
    def _load_upscaler(self):
        """Carga el módulo de super-resolución"""
        try:
            from src.core.ocr.super_resolution import get_upscaler
            self.upscaler = get_upscaler()
            print("✅ AsyncProcessor: FSRCNN cargado")
        except Exception as e:
            print(f"⚠️ AsyncProcessor: No se pudo cargar FSRCNN: {e}")
            
    def _load_plate_detector(self):
        """Carga el detector de placas"""
        try:
            from src.core.detection.plate_detector import PlateDetector
            from src.path_helper import resource_path
            import os
            
            model_path = resource_path("models/license_plate_detector.pt")
            if os.path.exists(model_path):
                self.plate_detector = PlateDetector(model_path)
                print("✅ AsyncProcessor: Detector de placas cargado")
        except Exception as e:
            print(f"⚠️ AsyncProcessor: No se pudo cargar detector: {e}")
    
    def start(self):
        """Inicia el worker en background"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 AsyncProcessor: Worker iniciado")
        
    def stop(self):
        """Detiene el worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        print("🛑 AsyncProcessor: Worker detenido")
        
    def update_semaphore_state(self, state):
        """Actualiza el estado del semáforo para saber cuándo procesar"""
        self.current_semaphore_state = state
        
    def add_infraction(self, track_id, frame_img, bbox, frame_index):
        """Añade una infracción a la cola de procesamiento"""
        self.pending_queue.put({
            'track_id': track_id,
            'frame_img': frame_img.copy(),
            'bbox': bbox,
            'frame_index': frame_index,
            'added_time': time.time()
        })
        
    def get_processed_result(self, track_id):
        """Obtiene el resultado procesado para un track_id"""
        return self.processed_results.get(track_id)
        
    def is_processed(self, track_id):
        """Verifica si un track_id ya fue procesado"""
        return track_id in self.processed_results
        
    def _worker_loop(self):
        """Loop principal del worker - procesa durante VERDE/AMARILLO"""
        while self.running:
            try:
                # Solo procesar si estamos en VERDE o primera mitad de AMARILLO
                # O si hay mucho en cola (emergencia)
                can_process = (
                    self.current_semaphore_state in ["green", "yellow"] or
                    self.pending_queue.qsize() > 5  # Emergencia: cola muy llena
                )
                
                if can_process and not self.pending_queue.empty():
                    # Procesar una infracción
                    item = self.pending_queue.get(timeout=0.1)
                    self._process_item(item)
                else:
                    # Esperar un poco antes de revisar de nuevo
                    time.sleep(0.05)
                    
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                print(f"⚠️ AsyncProcessor error: {e}")
                time.sleep(0.1)
                
    def _process_item(self, item):
        """Procesa una infracción: recorte + super-resolución"""
        start_time = time.time()
        
        track_id = item['track_id']
        frame_img = item['frame_img']
        bbox = item['bbox']
        
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return
                
            # Recortar vehículo con margen amplio (150px como en test_geoloc_surgical_gui)
            m = min(150, y1, x1, h-y2, w-x2)  # No exceder bordes
            vehicle_img = frame_img[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)].copy()
            
            # Detectar placa dentro del vehículo
            plate_crop = None
            sr_applied = False
            
            if self.plate_detector:
                detections = self.plate_detector.detect_plates(vehicle_img, confidence=0.4)
                if detections:
                    px1, py1, px2, py2 = [int(v) for v in detections[0]]
                    vh, vw = vehicle_img.shape[:2]
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(vw, px2), min(vh, py2)
                    
                    if px2 > px1 and py2 > py1:
                        # PASO 1: Recorte crudo de la placa (plate_raw)
                        plate_raw = vehicle_img[py1:py2, px1:px2].copy()
                        
                        # ── HOMOGRAFÍA v6.3 (Flow LabForense exacto) ──
                        # Aplicar sobre plate_raw, NO sobre vehicle_img
                        homo_ok = False
                        try:
                            from src.core.processing.plate_processing import rectificar_perspectiva
                            plate_rect = rectificar_perspectiva(plate_raw)
                            if plate_rect is not None:
                                plate_crop = plate_rect  # Ya viene sin header
                                homo_ok = True
                                print(f"📍 AsyncProc: Homografía v6.3 OK → {plate_crop.shape[1]}x{plate_crop.shape[0]}px")
                        except Exception as _he:
                            pass

                        # Fallback: usar plate_raw si homografía falló
                        if not homo_ok:
                            plate_crop = plate_raw
                            try:
                                from src.core.ocr.recognizer import get_lprnet_predictor
                                predictor = get_lprnet_predictor()
                                if predictor and hasattr(predictor, 'autocrop_plate'):
                                    plate_crop = predictor.autocrop_plate(plate_crop)
                                print(f"⚠️ AsyncProc: Fallback autocrop (homo falló)")
                            except:
                                pass
                            
                        # Dibujar recuadro verde en vehicle_img
                        cv2.rectangle(vehicle_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            
            # Si no se detectó placa, usar recorte heurístico
            if plate_crop is None or plate_crop.size == 0:
                vh, vw = vehicle_img.shape[:2]
                heuristic_y1 = int(vh * 0.55)
                plate_crop = vehicle_img[heuristic_y1:vh, :].copy()
            
            # Aplicar super-resolución si es necesario
            if plate_crop is not None and plate_crop.size > 0:
                ph, pw = plate_crop.shape[:2]
                if pw < 80 and self.upscaler:
                    plate_crop = self.upscaler.upscale(plate_crop, min_width=80)
                    sr_applied = True
                    self.stats['sr_applied_count'] += 1
            
            # Guardar resultado
            self.processed_results[track_id] = {
                'plate_crop': plate_crop,
                'vehicle_img': vehicle_img,
                'sr_applied': sr_applied,
                'frame_index': item['frame_index'],
                'bbox': bbox
            }
            
            # Actualizar estadísticas
            processing_time = (time.time() - start_time) * 1000
            self.stats['processed_count'] += 1
            self.stats['avg_processing_time_ms'] = (
                (self.stats['avg_processing_time_ms'] * (self.stats['processed_count'] - 1) + processing_time) 
                / self.stats['processed_count']
            )
            
            print(f"⚡ Async: Track {track_id} procesado en {processing_time:.1f}ms (SR: {sr_applied})")
            
        except Exception as e:
            print(f"⚠️ Error procesando track {track_id}: {e}")
            
    def get_stats(self):
        """Retorna estadísticas del procesador"""
        return {
            **self.stats,
            'pending_count': self.pending_queue.qsize(),
            'processed_total': len(self.processed_results)
        }


# Singleton
_processor_instance = None

def get_async_processor():
    """Obtiene la instancia singleton del procesador asíncrono"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AsyncPlateProcessor()
    return _processor_instance
