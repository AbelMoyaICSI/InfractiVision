import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.core.ocr.recognizer import format_siiv_plate

def run_master_dashboard():
    print("🎨 Generando DASHBOARD DE RESULTADOS MASTER...")
    
    predictor = LPRNetPredictor()
    plate_dir = "data/output/placas"
    
    if not os.path.exists(plate_dir):
        print(f"❌ No se encuentra la carpeta {plate_dir}")
        return

    # Listar TODAS las placas y tomar las 12 más recientes o significativas
    files = [f for f in os.listdir(plate_dir) if f.endswith('.jpg')]
    # Intentar priorizar las que tienen nombres de placas reales (6 caracteres o más)
    files.sort(key=lambda x: len(x), reverse=True)
    samples = files[:12] # Mostrar un grid de 3x4

    if not samples:
        print("❌ No hay placas para mostrar.")
        return

    plt.figure(figsize=(22, 16))
    plt.style.use('dark_background')

    for i, filename in enumerate(samples):
        path = os.path.join(plate_dir, filename)
        img = cv2.imread(path)
        if img is None: continue
        
        # 1. Obtener predicción
        text, conf = predictor.predict(img)
        formatted = format_siiv_plate(text)
        
        # 2. Obtener el input visual que ve la IA (94x24)
        fine_crop = predictor.autocrop_plate(img)
        model_input = cv2.resize(fine_crop, (94, 24), interpolation=cv2.INTER_LINEAR)
        
        # --- Dibujar ---
        plt.subplot(4, 3, i + 1)
        
        # Imagen principal (con margen para texto)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Colores dinámicos
        color = '#00FF00' if len(text) >= 6 else '#FFFF00'
        
        # Título con el resultado y confianza
        title_text = f"REAL OCR: {formatted}\n(Conf: {conf:.2f})"
        plt.title(title_text, fontsize=16, fontweight='bold', color=color, pad=10)
        
        # Añadir mini-visualización del input 94x24 en un recuadro
        # Usamos un inset_axes o simplemente dibujamos texto sobre el tamaño
        plt.xlabel(f"Input IA: 94x24 Stretched", fontsize=10, color='gray')
        
        # Eliminar ejes para limpieza
        plt.xticks([])
        plt.yticks([])
        
        # Añadir un borde de color según la confianza
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.suptitle("🚀 INFRACTIVISION: MASTER LPRNet VALIDATION DASHBOARD\n(Sincronización Total: Autocrop + 94x24 Stretched + Master Weights)", 
                 fontsize=28, fontweight='bold', y=0.98, color='white')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    output_file = "MASTER_FINAL_RESULTS.png"
    plt.savefig(output_file, dpi=120, facecolor='#121212')
    print(f"\n✨ DASHBOARD GENERADO EN: {output_file}")
    print("✅ Proceso completado. ¡Revisa la imagen para ver los resultados visuales!")

if __name__ == "__main__":
    run_master_dashboard()
