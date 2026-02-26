import cv2
import os
import numpy as np
import random
from src.core.ocr.recognizer import recognize_plate
from src.core.ocr.lprnet_engine import LPRNetPredictor
import matplotlib.pyplot as plt

def diagnostic_experiment():
    """
    Experimento Diagnóstico: Procesa una muestra aleatoria de imágenes
    para validar el pipeline completo de recorte y OCR.
    """
    autos_dir = "data/output/autos"
    output_dir = "data/debug_experimento_quirurgico"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 Iniciando Experimento Diagnóstico...")
    files = [f for f in os.listdir(autos_dir) if f.endswith(('.png', '.jpg'))]
    
    # Tomar una muestra aleatoria de imágenes que NO sean NIE
    sample_files = [f for f in files if "NIE" not in f]
    random.shuffle(sample_files)
    sample_files = sample_files[:6]  # 6 muestras
    
    predictor = LPRNetPredictor()
    results = []
    
    for fname in sample_files:
        img_path = os.path.join(autos_dir, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ❌ Error cargando: {fname}")
            continue
        
        # Pipeline completo
        txt, conf, crop = recognize_plate(img, return_processed=True, autocrop=True, regional_context="Trujillo")
        
        # Adaptación para visualización (Stretching Abel V24)
        if crop is not None and crop.size > 0:
            crop_94x24 = predictor.adapt_for_lprnet(crop, (94, 24))
        else:
            crop_94x24 = np.zeros((24, 94, 3), dtype=np.uint8)
            crop = np.zeros((50, 200, 3), dtype=np.uint8)
        
        # Extraer placa del nombre del archivo
        plate_from_file = fname.replace("vehicle_", "").split("_")[0]
        
        results.append({
            'file_plate': plate_from_file,
            'detected': txt,
            'conf': conf,
            'file': fname,
            'vehicle': img,
            'crop': crop,
            'final': crop_94x24
        })
        
        match = "✓" if plate_from_file.replace("-","") == txt.replace("-","") else "✗"
        print(f"  {match} Archivo: {plate_from_file} -> Detectado: {txt} (conf: {conf:.2f})")
    
    if not results:
        print("❌ No se encontraron imágenes para diagnóstico.")
        return
    
    # Generar mural de diagnóstico
    n_results = len(results)
    fig, axes = plt.subplots(n_results, 3, figsize=(15, 3.5 * n_results))
    fig.suptitle("DIAGNOSTICO DE PIPELINE OCR - SIIV Trujillo", fontsize=16, fontweight='bold')
    
    if n_results == 1:
        axes = [axes]  # Asegurar que sea iterable
    
    for i, r in enumerate(results):
        # Vehículo original (redimensionado)
        vh = cv2.resize(r['vehicle'], (200, 150))
        axes[i][0].imshow(cv2.cvtColor(vh, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"Archivo: {r['file'][:25]}...", fontsize=9)
        axes[i][0].axis('off')
        
        # Recorte de placa
        if r['crop'] is not None and r['crop'].size > 0:
            crop_display = cv2.resize(r['crop'], (200, 50))
            axes[i][1].imshow(cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title(f"Recorte Autocrop")
        axes[i][1].axis('off')
        
        # Resultado final 94x24
        final_display = cv2.resize(r['final'], (188, 48), interpolation=cv2.INTER_NEAREST)
        axes[i][2].imshow(cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB))
        match_status = "OK" if r['file_plate'].replace("-","") == r['detected'].replace("-","") else "FAIL"
        axes[i][2].set_title(f"[{match_status}] {r['detected']} ({r['conf']:.2f})", fontsize=10)
        axes[i][2].axis('off')
    
    save_path = os.path.join(output_dir, "diagnostic_pipeline.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n🏆 Diagnóstico guardado: {save_path}")

if __name__ == "__main__":
    diagnostic_experiment()
