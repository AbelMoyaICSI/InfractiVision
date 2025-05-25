import os
import time
import cv2
from src.core.processing.resolution_process import enhance_plate_image
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

import cv2
import numpy as np
import os

def process_plate(vehicle_roi, is_night=False):
    """
    Detect a license plate in the vehicle ROI and process it for better OCR.
    Returns (bbox, plate_img, plate_text) where bbox is relative to the vehicle_roi
    """
    from src.core.detection.plate_detector import PlateDetector
    from src.core.ocr.recognizer import recognize_plate
    
    try:
        # Initialize plate detector if not already done
        if not hasattr(process_plate, "plate_detector"):
            process_plate.plate_detector = PlateDetector()
        
        # Detect plate in the vehicle_roi using the correct method
        # Lower confidence threshold at night
        plate_confidence = 0.3 if is_night else 0.5
        
        # Check if detect_plates exists, otherwise use detect method
        if hasattr(process_plate.plate_detector, "detect_plates"):
            plates = process_plate.plate_detector.detect_plates(
                vehicle_roi, 
                confidence=plate_confidence
            )
        else:
            # Use the regular detect method and extract just the coordinates
            detections = process_plate.plate_detector.detect(
                vehicle_roi, 
                conf=plate_confidence,
                classes=[0],  # Assuming 0 is license plate class
                draw=False
            )
            
            # Convert detections to the format needed
            plates = []
            for det in detections:
                if len(det) >= 4:
                    x1, y1, x2, y2 = map(float, det[:4])  # Extract only coordinates
                    plates.append((x1, y1, x2, y2))
        
        # If no plate found, try enhancing the image
        if not plates:
            # Apply CLAHE to improve contrast, especially useful at night
            if is_night:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                if len(vehicle_roi.shape) > 2:
                    lab = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl = clahe.apply(l)
                    enhanced_lab = cv2.merge((cl, a, b))
                    enhanced_roi = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    enhanced_roi = clahe.apply(vehicle_roi)
                
                # Try detection again with enhanced image
                if hasattr(process_plate.plate_detector, "detect_plates"):
                    plates = process_plate.plate_detector.detect_plates(
                        enhanced_roi, 
                        confidence=0.25  # Even lower threshold for difficult cases
                    )
                else:
                    detections = process_plate.plate_detector.detect(
                        enhanced_roi, 
                        conf=0.25,
                        classes=[0], 
                        draw=False
                    )
                    
                    plates = []
                    for det in detections:
                        if len(det) >= 4:
                            x1, y1, x2, y2 = map(float, det[:4])
                            plates.append((x1, y1, x2, y2))
            
            # If still no plates, try alternative approach with edge detection
            if not plates:
                h, w = vehicle_roi.shape[:2]
                # Assume plate is in the lower half of the vehicle
                search_area = vehicle_roi[h//3:, :]
                
                # Convert to grayscale
                if len(search_area.shape) > 2:
                    gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
                else:
                    gray = search_area
                
                # Apply bilateral filter to reduce noise while preserving edges
                blurred = cv2.bilateralFilter(gray, 11, 17, 17)
                
                # Find edges
                edges = cv2.Canny(blurred, 30, 200)
                
                # Find contours
                contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Sort contours by area, largest first
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                
                plate_contour = None
                for contour in contours:
                    # Approximate contour
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    # If our approximated contour has four points, it's likely a license plate
                    if len(approx) == 4:
                        plate_contour = approx
                        break
                
                # If we found a plate contour
                if plate_contour is not None:
                    x, y, w, h = cv2.boundingRect(plate_contour)
                    y += h//3  # Adjust for the cropping we did earlier
                    plates = [(x, y, x+w, y+h)]
        
        if not plates:
            # Return placeholder values if no plate was found
            return ((0, 0, 0, 0), vehicle_roi, "")
        
        # Take the largest plate or the one with the highest confidence
        # In this case we're taking the first one
        x1, y1, x2, y2 = plates[0]
        
        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vehicle_roi.shape[1], x2), min(vehicle_roi.shape[0], y2)
        
        # Make a slightly larger crop to ensure we get the whole plate
        padding = 5
        crop_y1 = max(0, int(y1) - padding)
        crop_y2 = min(vehicle_roi.shape[0], int(y2) + padding)
        crop_x1 = max(0, int(x1) - padding)
        crop_x2 = min(vehicle_roi.shape[1], int(x2) + padding)
        
        # Extract the plate region
        plate_img = vehicle_roi[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        if plate_img.size == 0:
            return ((x1, y1, x2, y2), vehicle_roi, "")
        
        # Apply super-resolution to enhance the plate image
        try:
            from src.core.processing.superresolution import enhance_plate_image
            enhanced_plate = enhance_plate_image(plate_img, is_night)
            # OCR to recognize the plate text
            plate_text = recognize_plate(enhanced_plate)
        except ImportError:
            # If super-resolution is not available, use original image
            plate_text = recognize_plate(plate_img)
            enhanced_plate = plate_img
        
        # If that fails, try with original
        if not plate_text:
            plate_text = recognize_plate(plate_img)
            
        # Return the bbox, enhanced plate image, and recognized text
        return ((x1, y1, x2, y2), enhanced_plate, plate_text)
        
    except Exception as e:
        print(f"Error processing plate: {e}")
        import traceback
        traceback.print_exc()
        # Return default values
        return ((0, 0, 0, 0), vehicle_roi, "")