import cv2
import os
import sys
import numpy as np

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.getcwd())

from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.path_helper import resource_path

def test_surgical_extraction(image_path):
    print(f"🚀 Iniciando prueba de extracción quirúrgica para: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra la imagen en {image_path}")
        return

    # 1. Cargar imagen del vehículo
    vehicle_img = cv2.imread(image_path)
    if vehicle_img is None:
        print("❌ Error: No se pudo cargar la imagen")
        return
    
    # 2. Inicializar Detectores
    detector_model = resource_path("models/license_plate_detector.pt")
    plate_detector = PlateDetector(detector_model)
    predictor = LPRNetPredictor() # Carga LPRNet_CONSENSO_V2.pth por defecto
    
    # 3. Paso 1: Detección con YOLO (Con las mejoras de filtros relajaos)
    print("🔎 Detectando placa con YOLO...")
    detections = plate_detector.detect_plates(vehicle_img, confidence=0.3)
    
    raw_crop = None
    if detections:
        x1, y1, x2, y2 = [int(v) for v in detections[0]]
        print(f"✅ YOLO detectó placa en: [{x1}, {y1}, {x2}, {y2}]")
        raw_crop = vehicle_img[y1:y2, x1:x2].copy()
    else:
        print("⚠️ YOLO no detectó placa. Usando fallback heurístico previo al autocrop...")
        # Fallback heurístico (50% inferior, 80% central)
        h, w = vehicle_img.shape[:2]
        raw_crop = vehicle_img[int(h*0.5):h, int(w*0.1):int(w*0.9)].copy()

    # 4. Paso 2: RECORTE QUIRÚRGICO (Filtro Naranja Abel V4)
    print("✂️ Aplicando Autocrop Quirúrgico V4...")
    surgical_crop = predictor.autocrop_plate(raw_crop)
    
    # 5. Paso 3: Reconocimiento LPRNet (Con Resize + Padding para no aplastar)
    print("🧠 Reconociendo con LPRNet...")
    # predict(img) internamente llama a autocrop_plate y resize_with_padding
    plate_text, confidence = predictor.predict(surgical_crop)
    
    print(f"\n" + "="*40)
    print(f"📊 RESULTADOS DE LA PRUEBA")
    print(f"="*40)
    print(f"📝 Texto Reconocido: {plate_text}")
    print(f"⭐ Confianza: {confidence:.2f}")
    print(f"="*40)
    
    # 6. Guardar Resultados para inspección visual
    output_dir = "data/debug_plates"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path).split('.')[0]
    
    # Guardar el recorte quirúrgico
    surgical_path = os.path.join(output_dir, f"{base_name}_surgical_crop.jpg")
    cv2.imwrite(surgical_path, surgical_crop)
    
    # Guardar el input final que ve el motor (94x24 con adaptación Abel V24)
    input_lprnet = predictor.adapt_for_lprnet(surgical_crop, (94, 24))
    input_path = os.path.join(output_dir, f"{base_name}_lprnet_input.jpg")
    cv2.imwrite(input_path, input_lprnet)
    
    print(f"✅ Recorte quirúrgico guardado en: {surgical_path}")
    print(f"✅ Input final LPRNet guardado en: {input_path}")
    print("="*40)

if __name__ == "__main__":
    target_image = "data/output/autos/vehicle_A59-183_t3_f303.jpg"
    test_surgical_extraction(target_image)
