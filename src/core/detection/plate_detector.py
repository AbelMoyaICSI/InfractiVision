import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class PlateDetector:
    """
    Clase optimizada para detectar placas de vehículos usando YOLO.
    Incluye mejoras de rendimiento, cache y estadísticas.
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa el detector de placas con búsqueda robusta del modelo.
        
        Args:
            model_path: Ruta al modelo YOLO para detección de placas
        """
        try:
            # FIXED: Better model path resolution
            model_paths = [
                model_path,
                # Direct paths first
                "models/license_plate_detector.pt",
                os.path.join("models", "license_plate_detector.pt"),
                # Otras rutas alternativas
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "license_plate_detector.pt"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "license_plate_detector.pt"),
                os.path.join(os.getcwd(), "models", "license_plate_detector.pt"),
            ]
            
            model_loaded = False
            for path in model_paths:
                if path is not None and os.path.exists(path):  # FIXED: Check for None
                    print(f"PlateDetector: Cargando modelo desde {path}")
                    try:
                        self.model = YOLO(path)
                        model_loaded = True
                        print("PlateDetector: Modelo cargado correctamente")
                        break
                    except Exception as e:
                        print(f"PlateDetector: Error loading {path}: {e}")
                        continue
            
            if not model_loaded:
                raise FileNotFoundError("No se encontró el modelo de detección de placas")
                
        except Exception as e:
            print(f"Error al cargar modelo de detección de placas: {e}")
            self.model = None
        
        # 🚀 CONFIGURACIÓN EDGE TURBO (GPU + FP16)
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = torch.cuda.is_available()
        
        if self.model and torch.cuda.is_available():
            self.model.to(self.device)
            if self.half:
                self.model.half()
            print(f"🚀 PlateDetector: MODO TURBO ACTIVADO ({self.device})")
        
        # Estadísticas de rendimiento
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_confidence': 0.0,
            'night_detections': 0,
            'night_failures': 0
        }
        
        # Night detection settings
        self.night_detection_enabled = True
        self.night_brightness_threshold = 85  # More sensitive to dark conditions
    
    def detect(self, image, conf=0.5, classes=[0], draw=False, is_night=None):
        """
        Detecta placas en la imagen con técnicas mejoradas.
        
        Args:
            image: Imagen donde buscar placas
            conf: Umbral de confianza para detecciones (0-1)
            classes: Lista de IDs de clases a detectar (0=placa por defecto)
            draw: Si es True, dibuja las detecciones en la imagen
            is_night: Si es True, aplica técnicas de detección nocturna
            
        Returns:
            Lista de detecciones en formato (x1, y1, x2, y2, score, class_id)
        """
        if self.model is None:
            print("PlateDetector: No hay modelo cargado")
            return []
        
        try:
            # Actualizar estadísticas
            self.detection_stats['total_detections'] += 1
            
            # Auto-detect night conditions if not specified
            if is_night is None:
                is_night = self._detect_night_conditions(image)
            
            if is_night:
                self.detection_stats['night_detections'] += 1
                print("🌙 PlateDetector: Enhanced night mode activated with multi-capture")
            
            # Optimizar imagen para mejor detección con multi-capture para noche
            if is_night:
                enhanced_image = self._select_best_night_enhancement(image)
            else:
                enhanced_image = self._enhance_image_for_detection(image, is_night)
            
            # Configuración adaptativa basada en las características de la imagen
            brightness = np.mean(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))
            
            # ULTRA LOW confidence thresholds for night detection
            if brightness < 100:  # Dark/night image
                adaptive_conf = max(0.05, conf * 0.2)  # Ultra low for night
            elif brightness > 200:  # Imagen muy brillante
                adaptive_conf = max(0.1, conf * 0.4)  # Low for bright
            else:  # Imagen normal
                adaptive_conf = max(0.15, conf * 0.5)  # Lower than before
            
            # Ejecutar inferencia con YOLO con parámetros optimizados
            results = self.model(
                enhanced_image, 
                conf=adaptive_conf, 
                classes=classes,
                iou=0.45,  # IoU threshold optimizado
                agnostic_nms=True,  # NMS mejorado
                verbose=False
            )
            
            # Extraer y filtrar detecciones
            detections = []
            
            for result in results:
                if not result.boxes:
                    continue
                    
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes_detected = result.boxes.cls.cpu().numpy()
                
                # Procesar cada detección con filtros mejorados
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    score = float(confs[i])
                    class_id = int(classes_detected[i])
                    
                    # === FILTROS GEOMÉTRICOS Y ESPACIALES AVANZADOS (Protocolo Abel V19 - Equilibrado) ===
                    width, height = x2 - x1, y2 - y1
                    if height == 0 or width == 0:
                        continue
                        
                    aspect_ratio = width / height
                    area = width * height
                    img_h, img_w = image.shape[:2]
                    image_area = img_h * img_w
                    area_ratio = area / image_area
                    
                    # 1. Filtro de Aspect Ratio (Más permisivo para ángulos)
                    # Placa peruana: ~2.85. Rango: 1.5 a 6.0
                    is_valid_shape = 1.5 <= aspect_ratio <= 6.0
                    
                    # 2. Filtro de Tamaño Relativo (Física del Objeto)
                    is_valid_size = 0.0004 <= area_ratio <= 0.25
                    
                    # 3. FILTRO DE ANCHO RELATIVO (Fundamental para Buses)
                    width_ratio = width / img_w
                    is_valid_width = width_ratio <= 0.65 

                    # 4. FILTRO DE GEOLOCALIZACIÓN INTERNA (Grounding Espacial V20)
                    y_center_rel = (y1 + y2) / (2 * img_h)
                    # BUMPER AFFINITY: Las placas están en el parachoques (Bumper)
                    # CAR/SUV: 60-95% | BUS/TRUCK: 75-98%
                    # Permitimos un rango base de 35-98% pero damos prioridad a la zona de parachoques.
                    is_in_plate_zone = 0.35 <= y_center_rel <= 0.98
                    
                    # 5. ESCÁNER DE ENERGÍA DE CARACTERES (V20)
                    has_plate_energy = True
                    if width > 35 and height > 12:
                        roi = image[y1:y2, x1:x2]
                        if roi.size > 0:
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            edges_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                            energy = np.mean(np.abs(edges_x))
                            # Si es zona de 'luces' (arriba) pedimos energía extrema
                            # Si la confianza de YOLO es alta (>0.85), somos más permisivos
                            # de lo contrario, aplicamos Escáner de Energía (V21 - Densidad de Caracteres)
                            if score < 0.85:
                                # 🔠 DENSIDAD DE CARACTERES: Las placas tienen picos de cambios negros/blancos
                                # proyectamos los bordes horizontalmente para ver "clústeres" de letras
                                projection = np.sum(np.abs(edges_x), axis=0)
                                peaks = np.sum(projection > (np.max(projection) * 0.5))
                                # Una placa real tiene al menos 5-10 zonas de alta densidad (letras)
                                plate_density = peaks / width
                                has_plate_energy = (energy > 5.5) and (0.15 <= plate_density <= 0.85)

                    if is_valid_shape and is_valid_size and is_valid_width and is_in_plate_zone and has_plate_energy and width >= 18 and height >= 6:
                        detections.append((x1, y1, x2, y2, score, class_id))
                        
                        # Dibujar si se solicita
                        if draw:
                            color = (0, 255, 0)
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image, f"PLACA OK: {score:.2f}", (x1, y1 - 10),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Actualizar estadísticas
            if detections:
                self.detection_stats['successful_detections'] += 1
                avg_conf = sum(d[4] for d in detections) / len(detections)
                current_avg = self.detection_stats['average_confidence']
                total = self.detection_stats['successful_detections']
                self.detection_stats['average_confidence'] = (current_avg * (total-1) + avg_conf) / total
            
            return detections
            
        except Exception as e:
            print(f"Error en detección de placas: {e}")
            return []

    def detect_plates(self, image, confidence=0.5):
        """
        Método de compatibilidad para detectar placas y devolver las coordenadas.
        Este método se añade para resolver el error AttributeError: 'PlateDetector' object has no attribute 'detect_plates'
        
        Args:
            image: Imagen donde buscar placas
            confidence: Umbral de confianza
            
        Returns:
            Lista de coordenadas de placas en formato [(x1, y1, x2, y2), ...]
        """
        try:
            # Usar el método detect y extraer solo las coordenadas
            detections = self.detect(image, conf=confidence, classes=[0], draw=False)
            
            # Convertir a formato requerido
            plates = []
            for detection in detections:
                if len(detection) >= 4:  # Asegurarse de que hay al menos coordenadas
                    x1, y1, x2, y2 = detection[:4]
                    plates.append((x1, y1, x2, y2))
            
            return plates
        except Exception as e:
            print(f"Error en detect_plates: {e}")
            return []
    
    def _enhance_image_for_detection(self, image, is_night=False):
        """Mejora la imagen para una mejor detección de placas con soporte nocturno"""
        try:
            enhanced = image.copy()
            
            # Corrección de gamma adaptativa
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if is_night:
                # Mejoras específicas para condiciones nocturnas
                enhanced = self._apply_night_enhancement(enhanced)
            elif mean_brightness < 100:
                # Imagen oscura - aumentar brillo
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
            elif mean_brightness > 180:
                # Imagen muy brillante - reducir exposición
                enhanced = cv2.convertScaleAbs(enhanced, alpha=0.9, beta=-10)
            
            # Mejora de contraste local
            if len(enhanced.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                if is_night:
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  # Más agresivo para noche
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Reducción de ruido adaptativa
            if is_night:
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)  # Más fuerte para noche
            else:
                enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            return enhanced
            
        except Exception:
            return image

    def _detect_night_conditions(self, image):
        """Detecta si la imagen fue tomada en condiciones nocturnas"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Check for night conditions
            is_low_brightness = mean_brightness < self.night_brightness_threshold
            is_low_contrast = std_brightness < 40
            
            # Additional check: percentage of very dark pixels
            dark_pixels_ratio = np.sum(gray < 30) / gray.size
            has_many_dark_pixels = dark_pixels_ratio > 0.3
            
            return is_low_brightness or (is_low_contrast and has_many_dark_pixels)
            
        except Exception as e:
            print(f"Error detecting night conditions: {e}")
            return False
    
    def _apply_night_enhancement(self, image):
        """Aplica mejoras ultra-agresivas para imágenes nocturnas"""
        try:
            enhanced = image.copy()
            
            # 1. Ultra-aggressive brightness and contrast enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=3.0, beta=80)
            
            # 2. Histogram equalization for better global contrast
            if len(enhanced.shape) == 3:
                # Convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # 3. Multiple gamma corrections
            gamma_versions = []
            for gamma in [0.3, 0.5, 0.7]:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                gamma_corrected = cv2.LUT(enhanced, table)
                gamma_versions.append(gamma_corrected)
            
            # Blend gamma versions based on local brightness
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            enhanced = np.where(gray[..., np.newaxis] < 50, gamma_versions[0], enhanced)
            enhanced = np.where((gray[..., np.newaxis] >= 50) & (gray[..., np.newaxis] < 100), gamma_versions[1], enhanced)
            enhanced = np.where((gray[..., np.newaxis] >= 100) & (gray[..., np.newaxis] < 150), gamma_versions[2], enhanced)
            
            # 4. Ultra-aggressive enhancement of plate-like colors
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Very permissive masks for potential plate regions
            mask_white = cv2.inRange(hsv, (0, 0, 100), (180, 60, 255))  # Much more permissive
            mask_yellow = cv2.inRange(hsv, (10, 20, 80), (50, 255, 255))  # Much more permissive
            mask_gray = cv2.inRange(hsv, (0, 0, 80), (180, 80, 200))  # Include grays
            
            # Combine all masks
            plate_mask = cv2.bitwise_or(mask_white, mask_yellow)
            plate_mask = cv2.bitwise_or(plate_mask, mask_gray)
            
            # Ultra-aggressive enhancement to potential plate regions
            enhanced[plate_mask > 0] = cv2.convertScaleAbs(
                enhanced[plate_mask > 0], alpha=2.5, beta=60
            )
            
            # 5. Edge enhancement for better character visibility
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 80)  # Lower thresholds for night
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            enhanced = cv2.addWeighted(enhanced, 0.75, edges_3ch, 0.25, 0)
            
            # 6. Final aggressive contrast boost
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in night enhancement: {e}")
            return image
    
    def _select_best_night_enhancement(self, image):
        """Selecciona la mejor mejora nocturna probando múltiples técnicas"""
        try:
            print("🔍 PlateDetector: Generating multiple night enhancement variants...")
            
            # Generar múltiples variantes
            variants = []
            
            # 1. Mejora nocturna estándar
            standard = self._apply_night_enhancement(image)
            variants.append(("Standard", standard))
            
            # 2. Mejora ultra-agresiva
            ultra = image.copy()
            ultra = cv2.convertScaleAbs(ultra, alpha=5.0, beta=120)
            gamma = 0.15
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            ultra = cv2.LUT(ultra, table)
            variants.append(("Ultra", ultra))
            
            # 3. Mejora específica para reflectores
            reflective = self._enhance_reflective_areas(image)
            variants.append(("Reflective", reflective))
            
            # 4. Mejora de contraste local extremo
            contrast = self._extreme_contrast_enhancement(image)
            variants.append(("Contrast", contrast))
            
            # 5. Mejora específica para semáforos rojos
            red_light = self._red_light_compensation(image)
            variants.append(("RedLight", red_light))
            
            # Evaluar cada variante
            best_variant = None
            best_score = 0
            
            for name, variant in variants:
                score = self._evaluate_night_variant(variant)
                print(f"🔍 Variant '{name}': score {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_variant = variant
            
            if best_variant is not None:
                print(f"🎯 Selected best night variant with score {best_score:.3f}")
                return best_variant
            else:
                print("⚠️ No good variant found, using standard enhancement")
                return self._apply_night_enhancement(image)
                
        except Exception as e:
            print(f"Error selecting best night enhancement: {e}")
            return self._apply_night_enhancement(image)
    
    def _enhance_reflective_areas(self, image):
        """Mejora específica para áreas reflectivas"""
        try:
            enhanced = image.copy()
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Detectar áreas muy brillantes (potenciales reflectores)
            bright_areas = gray > 180
            mid_areas = (gray >= 100) & (gray <= 180)
            dark_areas = gray < 100
            
            # Aplicar diferentes mejoras
            enhanced[bright_areas] = cv2.convertScaleAbs(enhanced[bright_areas], alpha=0.8, beta=-30)
            enhanced[mid_areas] = cv2.convertScaleAbs(enhanced[mid_areas], alpha=2.0, beta=40)
            enhanced[dark_areas] = cv2.convertScaleAbs(enhanced[dark_areas], alpha=4.0, beta=100)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in reflective enhancement: {e}")
            return image
    
    def _extreme_contrast_enhancement(self, image):
        """Mejora de contraste extrema para condiciones nocturnas"""
        try:
            enhanced = image.copy()
            
            # Convertir a LAB
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE extremo
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Mejora adicional
            enhanced = cv2.convertScaleAbs(enhanced, alpha=2.5, beta=50)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in extreme contrast enhancement: {e}")
            return image
    
    def _red_light_compensation(self, image):
        """Compensación específica para condiciones de semáforo rojo"""
        try:
            enhanced = image.copy()
            
            # Convertir a HSV para manipular mejor los colores
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Reducir la saturación del rojo
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Reducir saturación en áreas rojas
            hsv[red_mask > 0, 1] = hsv[red_mask > 0, 1] * 0.2
            
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Aplicar mejoras generales
            enhanced = cv2.convertScaleAbs(enhanced, alpha=3.0, beta=80)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in red light compensation: {e}")
            return image
    
    def _evaluate_night_variant(self, variant):
        """Evalúa la calidad de una variante nocturna"""
        try:
            gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
            
            # 1. Contraste general
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 60)
            
            # 2. Distribución de brillo
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            
            # Penalizar distribuciones muy concentradas
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
            entropy_score = min(1.0, entropy / 8.0)
            
            # 3. Detección de bordes (importante para placas)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 10)
            
            # 4. Brillo promedio (no muy oscuro ni muy brillante)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            
            # Puntuación combinada
            total_score = (contrast_score * 0.3 + entropy_score * 0.2 + 
                          edge_score * 0.3 + brightness_score * 0.2)
            
            return total_score
            
        except Exception as e:
            print(f"Error evaluating variant: {e}")
            return 0.0
    
    def get_detection_stats(self):
        """Obtener estadísticas de detección"""
        total = max(1, self.detection_stats['total_detections'])
        return {
            'total_detections': self.detection_stats['total_detections'],
            'successful_detections': self.detection_stats['successful_detections'],
            'success_rate': self.detection_stats['successful_detections'] / total * 100,
            'average_confidence': self.detection_stats['average_confidence'],
            'night_detections': self.detection_stats['night_detections'],
            'night_failures': self.detection_stats['night_failures']
        }
    
