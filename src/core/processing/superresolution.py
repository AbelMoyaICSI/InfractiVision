import cv2
import numpy as np
import os
import time

# Ruta del directorio de modelos
models_dir = "models"  # Solo el directorio
os.makedirs(models_dir, exist_ok=True)

def enhance_plate(plate_bgr):
    """
    SISTEMA MINIMALISTA 2026 (Deep Learning Friendly):
    
    DESCUBRIMIENTO: PaddleOCR usa internamente CRNN + BiLSTM + CTC.
    Estos modelos de Deep Learning funcionan MEJOR con imágenes a COLOR
    sin procesamiento agresivo. El modelo aprende a extraer características.
    
    Solo hacemos:
    1. Escalado de alta calidad (más píxeles = mejor para el LSTM)
    2. Ligera mejora de contraste (sin destruir colores)
    """
    try:
        h, w = plate_bgr.shape[:2]
        if h < 5 or w < 5: 
            return plate_bgr
        
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
        
    except Exception as e:
        print(f"Error en enhance_plate minimal: {e}")
        return plate_bgr


# `enhance_plate_image` eliminado: era duplicado dead-code.
# El proyecto usa `src.core.processing.resolution_process.enhance_plate_image`
# como única implementación oficial.
