import cv2
import os
import numpy as np
from src.core.ocr.recognizer import recognize_plate
from src.core.ocr.lprnet_engine import LPRNetPredictor
import matplotlib.pyplot as plt

def run_forensic_fusion_report():
    autos_dir = "data/output/autos"
    output_dir = "data/debug_experimento_quirurgico"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Obtener los candidatos MSM-516
    print("🔬 Recolectando evidencia para reporte forense...")
    files = [f for f in os.listdir(autos_dir) if f.endswith(('.png', '.jpg'))]
    target_files = [f for f in files if "_t5_" in f] 
    
    candidates = []
    for fname in target_files:
        img = cv2.imread(os.path.join(autos_dir, fname))
        if img is None: continue
        txt, conf, crop = recognize_plate(img, return_processed=True, autocrop=True, regional_context="Trujillo")
        if txt and len(txt) >= 4:
            candidates.append({'crop': crop, 'conf': conf, 'txt': txt})
    
    candidates.sort(key=lambda x: x['conf'], reverse=True)
    top_3 = candidates[:3]
    
    # Normalizar a 300x80
    originals = [cv2.resize(c['crop'], (300, 80), interpolation=cv2.INTER_LANCZOS4) for c in top_3]
    
    # 2. ALINEACIÓN ECC (PROCESO INTERNO)
    anchor = originals[0]
    height, width = anchor.shape[:2]
    gray_anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
    aligned_crops = [anchor]
    
    for i in range(1, len(originals)):
        current = originals[i]
        gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            (cc, warp_matrix) = cv2.findTransformECC(gray_anchor, gray_current, warp_matrix, cv2.MOTION_AFFINE)
            aligned = cv2.warpAffine(current, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            aligned_crops.append(aligned)
        except:
            aligned_crops.append(current) # Fallback si falla

    # 3. FUSIÓN MAESTRA
    master_raw = np.median(aligned_crops, axis=0).astype(np.uint8)
    
    # 4. RECORTE FINAL CON MARGEN ABEL (3px)
    predictor = LPRNetPredictor()
    master_refined = predictor.autocrop_plate(master_raw)
    
    # --- GENERAR MURAL FORENSE ---
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("REPORTE FORENSE DE FUSIÓN MAESTRA: MSM-516\n(Prueba de Reconstrucción de Evidencia)", fontsize=16, fontweight='bold')
    
    # Fila 1: Originales
    for i in range(3):
        ax = plt.subplot(4, 3, i+1)
        ax.imshow(cv2.cvtColor(originals[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Toma Original #{i+1}\nConf: {top_3[i]['conf']:.2f}")
        ax.axis('off')
        
    # Fila 2: Alineadas (Ver que coinciden)
    for i in range(3):
        ax = plt.subplot(4, 3, i+4)
        ax.imshow(cv2.cvtColor(aligned_crops[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Toma #{i+1} ALINEADA")
        ax.axis('off')

    # Fila 3: Fusión de Evidencia
    ax_fuse = plt.subplot(4, 1, 3)
    # Mostramos la diferencia para que se vea qué se sumó
    diff = cv2.absdiff(aligned_crops[0], aligned_crops[1])
    ax_fuse.imshow(cv2.cvtColor(master_raw, cv2.COLOR_BGR2RGB))
    ax_fuse.set_title("🏆 PLACA MAESTRA (Fusión de Evidencia Limpia)")
    ax_fuse.axis('off')
    
    # Fila 4: Resultado Final
    ax_final = plt.subplot(4, 1, 4)
    ax_final.imshow(cv2.cvtColor(master_refined, cv2.COLOR_BGR2RGB))
    final_txt, final_conf = recognize_plate(master_refined, autocrop=False)
    ax_final.set_title(f"RESUTADO FINAL SIIV: {final_txt} (Confianza: {final_conf:.2f})")
    ax_final.axis('off')

    save_path = os.path.join(output_dir, "forensic_report_msm516.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Reporte Forense generado en: {save_path}")

if __name__ == "__main__":
    run_forensic_fusion_report()
