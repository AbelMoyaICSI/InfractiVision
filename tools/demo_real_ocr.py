import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.core.ocr.lprnet_engine import LPRNetPredictor

def run_real_demo():
    print("🚀 Inciando DEMO REAL: OCR sobre capturas de InfractiVision")
    
    predictor = LPRNetPredictor()
    
    # 1. Seleccionar imágenes de placas YA RECORTADAS por InfractiVision
    plate_dir = "data/output/placas"
    
    # Buscamos algunas que parezcan legibles por nombre (Trujillo es 'T')
    targets = [
        "plate_T3K-961.jpg",     # T3K-961
        "plate_ASA-841.jpg",     # ASA-841
        "plate_T1G-837.jpg"      # T1G-837
    ]
    
    # Si no existen, tomamos cualquiera
    existing_targets = [f for f in targets if os.path.exists(os.path.join(plate_dir, f))]
    if not existing_targets:
        existing_targets = [f for f in os.listdir(plate_dir) if f.endswith('.jpg')][:4]

    plt.figure(figsize=(24, 14))
    
    for i, filename in enumerate(existing_targets):
        plate_path = os.path.join(plate_dir, filename)
        plate_img = cv2.imread(plate_path)
        if plate_img is None: continue
        
        print(f"\n📸 Procesando Captura: {filename}")
        
        # EL SECRETO: El motor LPRNet Master hará Autocrop + 94x24 stretching
        # 1. Autocrop interno (Ajuste fino)
        fine_crop = predictor.autocrop_plate(plate_img)
        
        # 2. Inferencia (esto hace el stretching y normalización)
        text, conf = predictor.predict(plate_img)
        
        # 3. Preparar visualización
        master_input = cv2.resize(fine_crop, (94, 24), interpolation=cv2.INTER_LINEAR)
        
        # Layout
        plt.subplot(len(existing_targets), 3, i*3 + 1)
        plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        plt.title(f"1. CAPTURA ORIGINAL\n({plate_img.shape[1]}x{plate_img.shape[0]})")
        plt.axis('off')
        
        plt.subplot(len(existing_targets), 3, i*3 + 2)
        plt.imshow(cv2.cvtColor(fine_crop, cv2.COLOR_BGR2RGB))
        plt.title(f"2. RECORTE FINO (Autocrop)\nDetectando zona de caracteres")
        plt.axis('off')
        
        plt.subplot(len(existing_targets), 3, i*3 + 3)
        plt.imshow(cv2.cvtColor(master_input, cv2.COLOR_BGR2RGB))
        plt.title(f"3. ENTRADA IA (94x24 Stretched)\nRESULTADO: {text} (conf: {conf:.2f})")
        plt.axis('off')
        
        print(f"   ✅ Resultado: {text} | Confianza: {conf:.2f}")

    plt.suptitle("DEMOSTRACIÓN DE PROCESAMIENTO: CAPTURA REAL -> OCR MASTER\n(Validación de Fine Crop y Adaptación 94x24)", 
                 fontsize=26, y=0.98, fontweight='bold', color='navy')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_png = "DEMO_REAL_OCR.png"
    plt.savefig(output_png, dpi=160)
    print(f"\n✨ Demo completada. Imagen guardada en: {output_png}")

if __name__ == "__main__":
    run_real_demo()
