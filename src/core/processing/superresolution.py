import cv2
import numpy as np
import os
import time

# Ruta del directorio de modelos
models_dir = "models"  # Solo el directorio
os.makedirs(models_dir, exist_ok=True)

def enhance_plate(plate_bgr):
    """
    Procesamiento optimizado para placas sin caracteres chinos (como AE670S)
    """
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
        
        # 3. Ecualización de histograma adaptativa con parámetros ajustados
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cl = clahe.apply(gray)
        
        # 4. Umbralización adaptativa para resaltar los caracteres
        # Este método funciona mejor que Canny para este tipo de placas
        thresh = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 5. Operaciones morfológicas para limpiar el ruido
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 6. Convertir de vuelta a BGR
        result = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        
        # 7. Guardar resultado en output/
        # os.makedirs("data/output", exist_ok=True)
        # timestamp = int(time.time())
        # cv2.imwrite(f"data/output/plate_{timestamp}.jpg", result)
        
        return result
        
    except Exception as e:
        print(f"Error en superresolución: {e}")
        return plate_bgr