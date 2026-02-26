import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.getcwd())

from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.path_helper import resource_path

def visualize_adaptation_pipeline(image_path):
    print(f"🎬 Iniciando Visualizador de Adaptación para: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra la imagen")
        return

    # 1. Cargar imagen
    full_img = cv2.imread(image_path)
    # 2. Inicializar detector y predictor
    detector = PlateDetector(resource_path("models/license_plate_detector.pt"))
    predictor = LPRNetPredictor()
    
    # 3. Pipeline de procesamiento
    # A. Detección YOLO
    detections = detector.detect_plates(full_img, confidence=0.3)
    if detections:
        x1, y1, x2, y2 = [int(v) for v in detections[0]]
        raw_crop = full_img[y1:y2, x1:x2].copy()
    else:
        h, w = full_img.shape[:2]
        raw_crop = full_img[int(h*0.5):h, int(w*0.1):int(w*0.9)].copy()
        
    # B. Autocrop Quirúrgico V10 (Escáner de Energía de Caracteres)
    surgical_crop = predictor.autocrop_plate(raw_crop)
    
    # C. Adaptación Arquitectural V2 (Stretching Directo 94x24)
    adapted_input = predictor.resize_with_padding(surgical_crop, (94, 24))
    
    # D. Inferencia
    plate_text, confidence = predictor.predict(surgical_crop)
    
    # 4. Visualización con Matplotlib
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Pipeline de Sincronización Arquitectural LPRNet\nResultado: {plate_text} (Confiaza: {confidence:.2f})", fontsize=16)
    
    # Subplot 1: Recorte Original YOLO
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
    plt.title(f"1. Recorte YOLO\n({raw_crop.shape[1]}x{raw_crop.shape[0]})")
    plt.axis('off')
    
    # Subplot 2: Escáner de Energía (Al Ras V10)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(surgical_crop, cv2.COLOR_BGR2RGB))
    plt.title(f"2. Escáner de Energía (Abel V10)\n({surgical_crop.shape[1]}x{surgical_crop.shape[0]})")
    plt.axis('off')
    
    # Subplot 3: Input Final Arquitectura (94x24)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(adapted_input, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Stretching Directo (94x24)\nSin barras de fondo")
    # No quitar axis para ver las coordenadas 94x24
    
    # Guardar el mural de validación
    output_path = "data/debug_plates/mural_adaptacion_pixel_perfect.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Mural de validación guardado en: {output_path}")
    
    # Mostrar opcionalmente
    # plt.show()

if __name__ == "__main__":
    target = "data/output/autos/vehicle_A59-183_t3_f303.jpg"
    visualize_adaptation_pipeline(target)
