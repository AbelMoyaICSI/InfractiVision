import os
import cv2
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Añadir la ruta del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ocr.recognizer import get_lprnet_predictor
from src.path_helper import resource_path

def visual_audit(num_samples=12):
    print("🎨 INICIANDO AUDITORÍA VISUAL CIENTÍFICA")
    print("="*50)
    
    # Rutas
    placas_dir = resource_path("data/output/placas")
    
    if not os.path.exists(placas_dir):
        print(f"❌ No se encontró la carpeta de placas: {placas_dir}")
        return

    # Cargar predictor master
    print("📦 Cargando LPRNet Master...")
    predictor = get_lprnet_predictor()
    
    files = [f for f in os.listdir(placas_dir) if f.endswith(('.jpg', '.png'))]
    if not files:
        print("⚠️ No hay placas guardadas para analizar.")
        return
        
    # Mezclar y tomar muestras
    files = files[:num_samples] # Tomar los primeros para la muestra visual
    
    cols = 4
    rows = (len(files) + cols - 1) // cols
    
    plt.figure(figsize=(20, 5 * rows))
    
    for i, filename in enumerate(files):
        img_path = os.path.join(placas_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # Reconocer con el modelo
        text, conf = predictor.predict(img)
        
        # Extraer "verdad" del nombre del archivo (Ground Truth)
        # Formato esperado: plate_ABC123_t7_f1328.jpg
        ground_truth = "???"
        try:
            parts = filename.split('_')
            if len(parts) > 1:
                ground_truth = parts[1].replace('-', '')
        except:
            pass
            
        # CALCULAR PRECISIÓN REAL (NO SOLO CONFIANZA)
        from src.core.ocr.decode_fijo import calculate_character_accuracy
        real_accuracy = calculate_character_accuracy(text, ground_truth)
        
        # Color según precisión REAL (Verde solo si es perfecta)
        color = 'green' if real_accuracy == 100 else ('orange' if real_accuracy >= 60 else 'red')
        
        # Plot
        plt.subplot(rows, cols, i + 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        
        title = f"GT: {ground_truth}\nIA: {text}\nPrecisión: {real_accuracy:.1f}%\nConfianza: {conf*100:.1f}%"
        plt.title(title, color=color, fontsize=9, fontweight='bold')
        plt.axis('off')
        
        print(f"📸 Audit: {filename} -> {text} (Acc: {real_accuracy:.1f}% | Conf: {conf*100:.1f}%)")

    plt.suptitle("AUDITORÍA VISUAL DE RECONOCIMIENTO (LPRNet Master)", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    report_path = resource_path("data/REPORTE_VISUAL_AUDIT.png")
    plt.savefig(report_path, dpi=150)
    print(f"\n✅ REPORTE VISUAL GENERADO: {report_path}")
    
    # Intentar mostrar (si hay entorno gráfico)
    try:
        plt.show()
    except:
        print("ℹ️ No se pudo abrir la ventana de plot (entorno sin GUI), archivo guardado en data/.")

if __name__ == "__main__":
    visual_audit()
