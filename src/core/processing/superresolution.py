import cv2
import numpy as np
import os
import time

# Ruta del directorio de modelos
models_dir = "models"  # Solo el directorio
os.makedirs(models_dir, exist_ok=True)

def enhance_plate(plate_bgr):
    """
<<<<<<< Updated upstream
    SISTEMA MINIMALISTA 2026 (Deep Learning Friendly):
    
    DESCUBRIMIENTO: PaddleOCR usa internamente CRNN + BiLSTM + CTC.
    Estos modelos de Deep Learning funcionan MEJOR con imágenes a COLOR
    sin procesamiento agresivo. El modelo aprende a extraer características.
    
    Solo hacemos:
    1. Escalado de alta calidad (más píxeles = mejor para el LSTM)
    2. Ligera mejora de contraste (sin destruir colores)
=======
    Procesamiento optimizado para placas - VERSIÓN SIN BINARIZACIÓN.
    Preserva color para mejor OCR con PaddleOCR.
>>>>>>> Stashed changes
    """
    try:
        h, w = plate_bgr.shape[:2]
        if h < 5 or w < 5: 
            return plate_bgr
        
<<<<<<< Updated upstream
        # 1. ESCALADO INTELIGENTE (El tamaño importa para CRNN)
        # Queremos al menos 128px de altura para el modelo
        target_height = 128
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            # Lanczos4 mantiene la nitidez natural sin agregar artefactos
            resized = cv2.resize(plate_bgr, (new_w, new_h), 
                                interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = plate_bgr.copy()
        
        # 2. MEJORA MÍNIMA DE CONTRASTE (Preserva colores 100%)
        # Solo un pequeño boost para que las letras resalten
        # alpha=1.15 (leve aumento de contraste)
        # beta=5 (leve aumento de brillo)
        enhanced = cv2.convertScaleAbs(resized, alpha=1.15, beta=5)
        
        return enhanced
=======
        # 1. Escalar si es muy pequeña (mínimo 150px de ancho para mejor OCR)
        if w < 150:
            scale = 150 / w
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_CUBIC)
        elif w > 400:
            # No escalar demasiado para evitar blur
            scale = 400 / w
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_AREA)
        
        # 2. Reducir ruido preservando bordes (bilateral filter)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Mejorar contraste en espacio LAB (preserva color)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE solo en canal L (luminosidad)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l)
        
        # Recombinar
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. Aumentar nitidez sutilmente
        kernel = np.array([[-0.3,-0.3,-0.3], [-0.3,3.4,-0.3], [-0.3,-0.3,-0.3]])
        result = cv2.filter2D(result, -1, kernel)
        
        return result
>>>>>>> Stashed changes
        
    except Exception as e:
        print(f"Error en enhance_plate minimal: {e}")
        return plate_bgr


<<<<<<< Updated upstream
# `enhance_plate_image` eliminado: era duplicado dead-code.
# El proyecto usa `src.core.processing.resolution_process.enhance_plate_image`
# como única implementación oficial.
=======
def enhance_plate_image(plate_bgr, is_night=False):
    """
    Función wrapper para compatibilidad - PRESERVA COLOR.
    """
    try:
        # Aplicar mejora base (preserva color)
        enhanced = enhance_plate(plate_bgr)
        
        # Si es de noche, aplicar realce adicional EN COLOR
        if is_night:
            # Aumentar brillo y contraste en color
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=25)
            
            # Reducir ruido adicional para noche
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced
            
    except Exception as e:
        print(f"Error en enhance_plate_image: {e}")
        return plate_bgr
>>>>>>> Stashed changes
