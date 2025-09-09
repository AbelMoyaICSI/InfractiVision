import cv2
import torch
import psutil
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Cargar modelo con configuración optimizada
        self.model = YOLO(model_path)
        
        # Dispositivo óptimo (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.using_gpu = torch.cuda.is_available()
        
        # MEJORA: Detección avanzada de hardware y configuración ultra-adaptativa
        self.hardware_info = self._detect_hardware_capabilities()
        self._configure_adaptive_settings()
        
        # Caché para evitar procesar frames similares
        self.last_frame_hash = None
        self.last_detections = []
        
        # MEJORA: Estadísticas de rendimiento
        self.detection_stats = {
            'total_detections': 0,
            'processing_times': [],
            'average_fps': 0,
            'hardware_score': self.hardware_info['score']
        }
        
        print(f"🚀 VehicleDetector: {self.hardware_info['description']}")

    def _detect_hardware_capabilities(self):
        """Detecta capacidades avanzadas del hardware para configuración óptima"""
        import time
        start_time = time.time()
        
        # Información de GPU
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory': 0,
            'compute_capability': None
        }
        
        if gpu_info['available']:
            try:
                gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_info['compute_capability'] = torch.cuda.get_device_properties(0).major
                gpu_info['name'] = torch.cuda.get_device_properties(0).name
            except:
                pass
        
        # Información de CPU
        cpu_info = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 2000,
            'memory': psutil.virtual_memory().total / (1024**3)  # GB
        }
        
        # Calcular score de hardware (0-100)
        score = 0
        description = ""
        
        if gpu_info['available'] and gpu_info['memory'] > 0:
            # GPU disponible - calcular score basado en memoria y compute capability
            gpu_score = min(50, gpu_info['memory'] * 5)  # Hasta 50 puntos por memoria
            if gpu_info['compute_capability'] and gpu_info['compute_capability'] >= 6:
                gpu_score += 20  # +20 para compute capability moderna
            if gpu_info['compute_capability'] and gpu_info['compute_capability'] >= 8:
                gpu_score += 10  # +10 para última generación
            score += gpu_score
            description = f"GPU {gpu_info.get('name', 'CUDA')} ({gpu_info['memory']:.1f}GB) + CPU {cpu_info['cores']}C"
        else:
            # Solo CPU - calcular score basado en cores y frecuencia
            cpu_score = min(40, cpu_info['cores'] * 5)  # Hasta 40 puntos por cores
            freq_score = min(20, (cpu_info['frequency'] - 1000) / 100)  # Hasta 20 por frecuencia
            memory_score = min(20, cpu_info['memory'] * 2)  # Hasta 20 por memoria
            score = cpu_score + freq_score + memory_score
            description = f"CPU {cpu_info['cores']}C/{cpu_info['threads']}T @ {cpu_info['frequency']:.0f}MHz"
        
        detection_time = time.time() - start_time
        
        return {
            'gpu': gpu_info,
            'cpu': cpu_info,
            'score': min(100, max(10, score)),  # Entre 10-100
            'description': description,
            'detection_time': detection_time
        }
    
    def _configure_adaptive_settings(self):
        """Configura parámetros adaptativos según hardware detectado"""
        score = self.hardware_info['score']
        
        if score >= 80:  # Hardware muy potente
            self.imgsz = 832
            self.conf_threshold = 0.25
            self.max_det = 150
            self.batch_size = 4
            print("🔥 Configuración ULTRA: Hardware de gama alta detectado")
        elif score >= 60:  # Hardware potente  
            self.imgsz = 640
            self.conf_threshold = 0.3
            self.max_det = 100
            self.batch_size = 2
            print("🚀 Configuración ALTA: Hardware potente detectado")
        elif score >= 40:  # Hardware medio
            self.imgsz = 480
            self.conf_threshold = 0.35
            self.max_det = 75
            self.batch_size = 1
            print("⚡ Configuración MEDIA: Hardware estándar detectado")
        else:  # Hardware básico
            self.imgsz = 320
            self.conf_threshold = 0.5
            self.max_det = 50
            self.batch_size = 1
            print("💻 Configuración BÁSICA: Hardware limitado detectado")
        
        # Configuración adicional para GPU
        if self.using_gpu and self.hardware_info['gpu']['memory'] > 0:
            # Ajustar batch_size según memoria de GPU
            if self.hardware_info['gpu']['memory'] >= 8:
                self.batch_size = min(self.batch_size * 2, 8)
            elif self.hardware_info['gpu']['memory'] < 4:
                self.batch_size = 1
    
    def get_adaptive_conf_for_conditions(self, is_night=False, image_brightness=None):
        """Retorna umbral de confianza adaptativo según condiciones"""
        base_conf = self.conf_threshold
        
        # Ajustar para condiciones nocturnas
        if is_night:
            base_conf *= 0.7  # Reducir umbral para detectar más en condiciones difíciles
        
        # Ajustar según brillo de imagen si está disponible
        if image_brightness is not None:
            if image_brightness < 80:  # Imagen muy oscura
                base_conf *= 0.6
            elif image_brightness > 200:  # Imagen muy brillante
                base_conf *= 0.9
        
        return max(0.15, min(0.8, base_conf))  # Mantener entre límites razonables

    def detect(self, image_bgr, conf=None, draw=False, is_night=False):
        """
        Detecta vehículos en una imagen con configuración adaptativa.
        
        Args:
            image_bgr: Imagen en formato BGR (OpenCV)
            conf: Umbral de confianza para detecciones (None=automático)
            draw: Si es True, dibuja las detecciones en la imagen
            is_night: Indica si es una escena nocturna
            
        Returns:
            Lista de detecciones [x1,y1,x2,y2,cls_id] o imagen con detecciones dibujadas
        """
        import time
        start_time = time.time()
        
        # MEJORA: Configuración adaptativa automática
        if conf is None:
            # Calcular brillo promedio de la imagen
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            image_brightness = np.mean(gray)
            conf = self.get_adaptive_conf_for_conditions(is_night, image_brightness)
        
        # 1. Verificar si el frame es muy similar al anterior usando hash perceptual
        if image_bgr.shape[0] > 200:  # Solo para imágenes grandes
            small = cv2.resize(image_bgr, (8, 8))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            img_hash = hash(gray.tobytes())
            
            if hasattr(self, 'last_frame_hash') and self.last_frame_hash == img_hash:
                return self.last_detections
            
            self.last_frame_hash = img_hash
        
        # 2. Procesar en tamaño adaptativo según hardware
        orig_shape = image_bgr.shape
        if orig_shape[0] > self.imgsz or orig_shape[1] > self.imgsz:
            scale = min(self.imgsz / orig_shape[0], self.imgsz / orig_shape[1])
            new_shape = (int(orig_shape[1] * scale), int(orig_shape[0] * scale))
            resized = cv2.resize(image_bgr, new_shape)
            results = self.model.predict(resized, conf=conf, verbose=False, max_det=self.max_det)
            scale_factor = (orig_shape[1] / new_shape[0], orig_shape[0] / new_shape[1])
        else:
            results = self.model.predict(image_bgr, conf=conf, verbose=False, max_det=self.max_det)
            scale_factor = (1.0, 1.0)
        
        # 3. Extraer detecciones - MANTENER FORMATO DE 5 VALORES
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Escalar coordenadas al tamaño original
                if scale_factor != (1.0, 1.0):
                    x1, x2 = int(x1 * scale_factor[0]), int(x2 * scale_factor[0])
                    y1, y2 = int(y1 * scale_factor[1]), int(y2 * scale_factor[1])
                
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                
                # IMPORTANTE: Solo guardar 5 valores (sin la confianza)
                # Usar la confianza internamente para filtrar pero no devolverla
                if conf_val >= conf:
                    detections.append((x1, y1, x2, y2, cls_id))
        
        # 4. Guardar en caché
        self.last_detections = detections
        
        # MEJORA: Actualizar estadísticas de rendimiento
        processing_time = time.time() - start_time
        self.detection_stats['total_detections'] += len(detections)
        self.detection_stats['processing_times'].append(processing_time)
        
        # Mantener solo las últimas 100 mediciones para calcular promedio
        if len(self.detection_stats['processing_times']) > 100:
            self.detection_stats['processing_times'] = self.detection_stats['processing_times'][-100:]
        
        # Calcular FPS promedio
        if self.detection_stats['processing_times']:
            avg_time = sum(self.detection_stats['processing_times']) / len(self.detection_stats['processing_times'])
            self.detection_stats['average_fps'] = 1.0 / avg_time if avg_time > 0 else 0
        
        # 5. Dibujar si es necesario
        if draw:
            for (x1, y1, x2, y2, cls_id) in detections:
                # Color verde para todos los vehículos, independientemente del tipo
                color = (0, 255, 0)  # Verde
                
                # Dibujar rectángulo
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                
                # Añadir etiqueta con clase (sin confianza)
                label = self._get_class_name(cls_id)
                cv2.putText(image_bgr, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return image_bgr
        
        return detections
    
    def get_performance_stats(self):
        """Retorna estadísticas de rendimiento del detector"""
        return {
            'total_detections': self.detection_stats['total_detections'],
            'average_fps': round(self.detection_stats['average_fps'], 2),
            'hardware_score': self.detection_stats['hardware_score'],
            'hardware_description': self.hardware_info['description'],
            'current_config': {
                'imgsz': self.imgsz,
                'conf_threshold': self.conf_threshold,
                'max_det': self.max_det,
                'batch_size': getattr(self, 'batch_size', 1)
            }
        }