import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    """
    Clase para detectar placas de vehículos usando YOLO.
    """
    
    def __init__(self, model_path="models/license_plate_detector.pt"):
        """
        Inicializa el detector de placas.
        
        Args:
            model_path: Ruta al modelo YOLO para detección de placas
        """
        try:
            self.model = YOLO(model_path)
            print("PlateDetector: Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo de detección de placas: {e}")
            # Crear un modelo alternativo fallback
            try:
                self.model = YOLO("yolov8n.pt")
                print("PlateDetector: Usando modelo genérico como fallback")
            except Exception as e:
                print(f"Error crítico, no se pudo cargar ningún modelo: {e}")
                self.model = None
    
    def detect(self, image, conf=0.5, classes=[0], draw=False):
        """
        Detecta placas en la imagen.
        
        Args:
            image: Imagen donde buscar placas
            conf: Umbral de confianza para detecciones (0-1)
            classes: Lista de IDs de clases a detectar (0=placa por defecto)
            draw: Si es True, dibuja las detecciones en la imagen
            
        Returns:
            Lista de detecciones en formato (x1, y1, x2, y2, score, class_id)
        """
        if self.model is None:
            print("PlateDetector: No hay modelo cargado")
            return []
        
        try:
            # Ejecutar inferencia con YOLO
            results = self.model(image, conf=conf, classes=classes)
            
            # Extraer detecciones
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                # Extraer coordenadas, puntuaciones y clases
                for i in range(len(boxes)):
                    # Obtener cuadro delimitador
                    box = boxes[i].xyxy[0].cpu().numpy()  # formato xyxy
                    x1, y1, x2, y2 = box
                    
                    # Obtener puntuación y clase
                    score = float(boxes[i].conf)
                    class_id = int(boxes[i].cls)
                    
                    # Añadir a la lista
                    detections.append((x1, y1, x2, y2, score, class_id))
                    
                    # Dibujar si se solicita
                    if draw:
                        color = (0, 255, 0)  # Verde para placas
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(image, f"Placa: {score:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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