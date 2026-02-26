import os
import cv2
import sys
import torch
import numpy as np

# Añadir la ruta del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ocr.recognizer import get_lprnet_predictor
from src.core.detection.plate_detector import PlateDetector
from src.path_helper import resource_path

def technical_audit():
    print("🔬 INICIANDO AUDITORÍA TÉCNICA DE INFRACTIVISION")
    print("="*50)
    
    # Rutas
    autos_dir = resource_path("data/output/autos")
    model_path = resource_path("models/license_plate_detector.pt")
    
    if not os.path.exists(autos_dir):
        print(f"❌ No se encontró la carpeta: {autos_dir}")
        return

    # Cargar modelos master
    print("📦 Cargando Motores AI...")
    predictor = get_lprnet_predictor()
    plate_detector = PlateDetector(model_path)
    
    files = [f for f in os.listdir(autos_dir) if f.endswith(('.jpg', '.png'))]
    print(f"🔍 Encontradas {len(files)} capturas para auditar.\n")

    results = []
    
    for filename in files:
        img_path = os.path.join(autos_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        print(f"📁 Analizando: {filename}")
        
        # 1. Detectar placa en el recorte del auto
        plates = plate_detector.detect(img)
        
        if plates:
            px1, py1, px2, py2, pconf = plates[0]
            plate_crop = img[int(py1):int(py2), int(px1):int(px2)]
            
            # 2. Reconocer texto
            text, conf = predictor.predict(plate_crop)
            
            # 3. Clasificar científicamente
            clean_text = text.replace('-', '').replace(' ', '')
            is_valid = len(clean_text) == 6
            reason = "OK (SIIV Válido)" if is_valid else f"NIE (Longitud {len(clean_text)} != 6)"
            
            print(f"   IA Sugiere: {text} | Confianza: {conf:.2f} | Estado: {reason}")
            
            results.append({
                'file': filename,
                'detected': text,
                'conf': conf,
                'valid': is_valid,
                'reason': reason
            })
        else:
            print("   ⚠️ No se detectó placa en esta imagen del auto.")
            results.append({
                'file': filename,
                'detected': "N/A",
                'conf': 0.0,
                'valid': False,
                'reason': "Fallo Detección de Placa"
            })

    # Resumen final
    print("\n" + "="*50)
    print("📊 RESUMEN DE AUDITORÍA")
    total = len(results)
    validos = sum(1 for r in results if r['valid'])
    print(f"✅ Válidos (NID): {validos} / {total} ({validos/total*100:.1f}%)")
    print(f"⚠️ Fallidos (NIE): {total - validos} / {total}")
    print("="*50)

if __name__ == "__main__":
    technical_audit()
