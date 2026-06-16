"""
🚀 Módulo de Super-Resolución Ultraligera para Placas
Utiliza FSRCNN (Fast Super-Resolution CNN) de OpenCV para mejorar
placas de baja resolución antes del OCR.

Características:
- Modelo de solo 40KB (ultraligero)
- Escala 3x con preservación de bordes
- Optimizado para CPU (no requiere GPU)
- Tiempo de inferencia: ~5-15ms por imagen
"""

import cv2
import numpy as np
from src.path_helper import resource_path

# Singleton para evitar cargar el modelo múltiples veces
_sr_instance = None


class PlateUpscaler:
    """
    Super-Resolución ultraligera para placas de baja resolución.
    Usa FSRCNN x3 de OpenCV DNN.
    """
    
    def __init__(self):
        self.sr = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo FSRCNN de forma segura"""
        try:
            model_path = resource_path("models/FSRCNN_x3.pb")
            
            # Verificar que opencv tiene el módulo dnn_superres
            if not hasattr(cv2, 'dnn_superres'):
                print("⚠️ OpenCV no tiene módulo dnn_superres. Instalando opencv-contrib-python...")
                self.model_loaded = False
                return
            
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("fsrcnn", 3)  # Escala 3x
            self.model_loaded = True
            print("✅ Super-Resolución FSRCNN cargada correctamente (40KB, 3x)")
            
        except Exception as e:
            print(f"⚠️ No se pudo cargar FSRCNN: {e}")
            self.model_loaded = False
    
    def upscale(self, plate_img, min_width=80):
        """
        Mejora la resolución de una placa si es muy pequeña.
        
        Args:
            plate_img: Imagen BGR de la placa
            min_width: Ancho mínimo antes de aplicar SR (default: 80px)
            
        Returns:
            Imagen mejorada o la original si no necesita mejora
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        h, w = plate_img.shape[:2]
        
        # Solo aplicar si la imagen es pequeña
        if w >= min_width:
            return plate_img
        
        # Si el modelo no está cargado, usar bicúbico como fallback
        if not self.model_loaded or self.sr is None:
            print(f"🔍 Fallback bicúbico: {w}px → {w*3}px")
            return cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        try:
            # Aplicar Super-Resolución FSRCNN
            upscaled = self.sr.upsample(plate_img)
            print(f"🚀 FSRCNN SR: {w}x{h}px → {upscaled.shape[1]}x{upscaled.shape[0]}px")
            return upscaled
            
        except Exception as e:
            print(f"⚠️ Error en SR, usando bicúbico: {e}")
            return cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def get_upscaler():
    """
    Obtiene la instancia singleton del upscaler.
    Evita cargar el modelo múltiples veces.
    """
    global _sr_instance
    if _sr_instance is None:
        _sr_instance = PlateUpscaler()
    return _sr_instance


def upscale_plate(plate_img, min_width=80):
    """
    Función de conveniencia para aplicar super-resolución.
    
    Args:
        plate_img: Imagen BGR de la placa
        min_width: Ancho mínimo antes de aplicar SR
        
    Returns:
        Imagen mejorada
    """
    upscaler = get_upscaler()
    return upscaler.upscale(plate_img, min_width)
