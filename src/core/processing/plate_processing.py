import os
import time
import cv2
from src.core.detection.plate_detector import PlateDetector
from src.core.processing.superresolution import enhance_plate
from src.core.ocr.recognizer import recognize_plate

_detector = None

def get_plate_detector():
    global _detector
    if _detector is None:
        _detector = PlateDetector("models/yolov8n.pt")  # Usa el modelo general de YOLOv8 por ahora
    return _detector

# Añade esta función para mejorar la detección de la región de la placa
def find_plate_region(vehicle_image, is_night=False):
    """
    Intenta encontrar regiones con características de placas vehiculares
    usando técnicas de procesamiento de imagen avanzadas.
    
    Args:
        vehicle_image: Imagen del vehículo
        is_night: Bandera que indica si es una escena nocturna
        
    Returns:
        Coordenadas de la región [x1, y1, x2, y2] o None si no se encuentra
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
    
    # En la noche, primero mejorar contraste
    if is_night:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Umbralización adaptativa con parámetros ajustados según condición
    block_size = 11 if is_night else 13
    c_value = 1 if is_night else 2
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, block_size, c_value)
    
    # Operaciones morfológicas para mejorar detección
    kernel_size = 7 if is_night else 5
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    
    # Encontrar contornos
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = vehicle_image.shape[:2]
    
    # Umbral de área adaptativo según condiciones
    min_area_factor = 0.005 if is_night else 0.01  # Más permisivo en la noche
    max_area_factor = 0.25 if is_night else 0.2    # Más permisivo en la noche
    
    min_area = w * h * min_area_factor
    max_area = w * h * max_area_factor
    
    plate_regions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Relación de aspecto más permisiva para la noche
            min_ratio = 1.2 if is_night else 1.5
            max_ratio = 8.0 if is_night else 7.0
            
            aspect_ratio = width / float(height)
            if min_ratio < aspect_ratio < max_ratio:
                # Verificar que está en la mitad inferior (más permisivo en la noche)
                lower_threshold = h/3 if is_night else h/2.5
                if y > lower_threshold:
                    plate_regions.append((x, y, x+width, y+height))
    
    # Si encontramos regiones candidatas, devolver la mejor
    if plate_regions:
        # Ordenar por área (mayor primero)
        plate_regions.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
        return plate_regions[0]
    
    return None

def enhance_night_image(img):
    """Mejora visibilidad en imágenes nocturnas"""
    # Convertir a LAB para trabajar mejor con brillo
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE al canal L
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Fusionar canales de nuevo
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Aumentar brillo y contraste
    return cv2.convertScaleAbs(enhanced, alpha=1.4, beta=30)

def enhance_plate_night(plate_bgr):
    """Versión específica para optimizar placas en condiciones nocturnas"""
    try:
        # Verificar tamaño mínimo
        h, w = plate_bgr.shape[:2]
        if h < 10 or w < 20:
            return plate_bgr
            
        # Crear una copia para no modificar el original
        enhanced = plate_bgr.copy()
        
        # 1. Mayor zoom para ver mejor los detalles
        scale = 4.0
        enhanced = cv2.resize(enhanced, (int(w * scale), int(h * scale)), 
                            interpolation=cv2.INTER_CUBIC)
        
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # 3. Ecualización de histograma adaptativa con parámetros ajustados para noche
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        cl = clahe.apply(gray)
        
        # 4. Filtrado bilateral para reducir ruido preservando bordes
        filtered = cv2.bilateralFilter(cl, 11, 17, 17)
        
        # 5. Umbralización adaptativa para mejor segmentación en condiciones de baja luz
        thresh = cv2.adaptiveThreshold(
            filtered, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15, 
            4
        )
        
        # 6. Operaciones morfológicas para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 7. Convertir de vuelta a BGR para OCR (muchos OCR funcionan mejor con la umbralización invertida)
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return result
    except Exception as e:
        print(f"Error en enhance_plate_night: {e}")
        return plate_bgr

def process_plate(vehicle_bgr, is_night=False):
    """
    Procesa un ROI para extraer la placa vehicular
    
    Args:
        vehicle_bgr: Imagen (ROI) del vehículo
        is_night: Flag que indica si es escena nocturna
    
    Returns:
        (bbox, plate_sr, plate_text)
    """
    try:
        if is_night:
            vehicle_bgr = enhance_night_image(vehicle_bgr)
        
        # Obtener detector de placas
        plate_detector = get_plate_detector()
        
        # Ajustar umbral de confianza según condiciones de luz
        conf_threshold = 0.15 if is_night else 0.25  # Umbral más bajo para noche
        
        # Detectar placa en la imagen del vehículo
        plate_dets = plate_detector(vehicle_bgr, conf=conf_threshold)
        
        if not plate_dets:
            # Si YOLO no detecta placas, intentar con métodos tradicionales
            plate_region = find_plate_region(vehicle_bgr, is_night)
            if plate_region:
                x1, y1, x2, y2 = plate_region
                bbox = [x1, y1, x2, y2]
                plate_img = vehicle_bgr[y1:y2, x1:x2].copy()
            else:
                # Si no se encuentra la placa, devolver el ROI completo
                h, w = vehicle_bgr.shape[:2]
                bbox = [0, 0, w, h]
                plate_img = vehicle_bgr.copy()
        else:
            # Usar la primera detección (más confiable)
            x1, y1, x2, y2 = plate_dets[0]
            
            # Expandir un poco para asegurar que toda la placa esté incluida
            expansion = 10 if is_night else 5  # Mayor expansión para escenas nocturnas
            x1 = max(0, int(x1 - expansion))
            y1 = max(0, int(y1 - expansion))
            x2 = min(vehicle_bgr.shape[1], int(x2 + expansion))
            y2 = min(vehicle_bgr.shape[0], int(y2 + expansion))
            
            bbox = [x1, y1, x2, y2]
            plate_img = vehicle_bgr[y1:y2, x1:x2].copy()
        
        # Aplicar superresolución con parámetros específicos para día/noche
        if is_night:
            plate_sr = enhance_plate_night(plate_img)
        else:
            plate_sr = enhance_plate(plate_img)
        
        # Reconocer texto de la placa
        from src.core.ocr.recognizer import recognize_plate
        ocr_text = recognize_plate(plate_sr, is_night)
        
        return bbox, plate_sr, ocr_text
    except Exception as e:
        print(f"Error en process_plate: {e}")
        import traceback
        traceback.print_exc()
        # En caso de error, devolver valores por defecto
        h, w = vehicle_bgr.shape[:2]
        return [0, 0, w, h], vehicle_bgr, ""