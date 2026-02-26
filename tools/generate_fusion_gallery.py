import cv2
import os
import numpy as np
from src.core.ocr.recognizer import recognize_plate
from src.core.ocr.lprnet_engine import LPRNetPredictor
import matplotlib.pyplot as plt
import re

def run_mass_fusion_validation():
    autos_dir = "data/output/autos"
    output_dir = "data/debug_consenso_visual"
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(autos_dir) if f.endswith(('.png', '.jpg'))]
    
    # 1. Agrupar por Track ID
    groups = {}
    for f in files:
        match = re.search(r'_t(\d+)_', f)
        if match:
            tid = match.group(1)
            groups.setdefault(tid, []).append(f)
            
    # Solo procesar grupos con 3 o más imágenes para que la fusión tenga sentido
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    
    if not groups:
        print("❌ No hay suficientes secuencias para la validación masiva.")
        return

    print(f"🔬 Validando Consenso Visual en {len(groups)} vehículos...")
    engine = LPRNetPredictor()

    for tid, filenames in groups.items():
        print(f"📦 Procesando Vehículo #{tid} ({len(filenames)} tomas)...")
        
        individual_crops = []
        individual_results = []
        
        for fname in filenames:
            img = cv2.imread(os.path.join(autos_dir, fname))
            if img is None: continue
            
            # Recorte quirúrgico individual
            txt, conf, crop = recognize_plate(img, return_processed=True, autocrop=True)
            if crop is not None and crop.size > 0:
                # Normalizar para fusión (300x80)
                crop_hq = cv2.resize(crop, (300, 80), interpolation=cv2.INTER_LANCZOS4)
                individual_crops.append(crop_hq)
                individual_results.append((txt, conf))

        if len(individual_crops) < 3: continue

        # --- FUSIÓN MAESTRA QUIRÚRGICA ---
        # 1. Promediado de mediana
        master_raw = np.median(individual_crops, axis=0).astype(np.uint8)
        
        # 2. SEGUNDO RECORTE (El Recut Final para el Flush Perfecto)
        master_refined = engine.autocrop_plate(master_raw)
        
        # 3. Formato Arquitectónico 94x24
        master_94x24 = cv2.resize(master_refined, (94, 24), interpolation=cv2.INTER_AREA)
        
        # 4. Inferencia LPRNet sobre el Maestro
        m_txt, m_conf = recognize_plate(master_94x24, autocrop=False)
        
        # --- GENERAR MURAL DE COMPARACIÓN ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"CONSENSO VISUAL - VEHÍCULO #{tid}\nRESULTADO MAESTRO: {m_txt} (Conf: {m_conf:.2f})", fontsize=14, fontweight='bold')
        
        # Toma 1
        axes[0,0].imshow(cv2.cvtColor(cv2.resize(individual_crops[0], (94, 24)), cv2.COLOR_BGR2RGB))
        axes[0,0].set_title(f"Toma 1: {individual_results[0][0]} ({individual_results[0][1]:.2f})")
        axes[0,0].axis('off')
        
        # Toma 2
        axes[0,1].imshow(cv2.cvtColor(cv2.resize(individual_crops[1], (94, 24)), cv2.COLOR_BGR2RGB))
        axes[0,1].set_title(f"Toma 2: {individual_results[1][0]} ({individual_results[1][1]:.2f})")
        axes[0,1].axis('off')
        
        # Fusión Bruta (Antes del Recut)
        axes[1,0].imshow(cv2.cvtColor(cv2.resize(master_raw, (94, 24)), cv2.COLOR_BGR2RGB))
        axes[1,0].set_title("Fusión Bruta (Con ruido de bordes)")
        axes[1,0].axis('off')
        
        # Fusión Maestra (Después del Recut + 94x24)
        axes[1,1].imshow(cv2.cvtColor(master_94x24, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title("🏆 PLACA MAESTRA (Surgical Flush Final)")
        axes[1,1].axis('off')
        
        save_path = os.path.join(output_dir, f"consenso_track_{tid}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Galería guardada: {save_path}")

if __name__ == "__main__":
    run_mass_fusion_validation()
