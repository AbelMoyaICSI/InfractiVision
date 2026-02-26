import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.core.ocr.recognizer import format_siiv_plate

def run_precision_gui():
    print("🚀 Lanzando Visor de Precisión Master (Color & Fine Crop)...")
    
    try:
        import matplotlib
        matplotlib.use('TkAgg') 
    except:
        pass

    predictor = LPRNetPredictor()
    plate_dir = "data/output/placas"
    
    if not os.path.exists(plate_dir) or not os.listdir(plate_dir):
        print(f"⚠️ La carpeta {plate_dir} está vacía. Por favor, corre main.py primero.")
        return

    files = [f for f in os.listdir(plate_dir) if f.endswith('.jpg')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(plate_dir, x)), reverse=True)
    samples = files[:6] # 6 placas más recientes

    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title('InfractiVision - Master Precision OCR')
    plt.style.use('dark_background')

    for i, filename in enumerate(samples):
        path = os.path.join(plate_dir, filename)
        img = cv2.imread(path)
        if img is None: continue
        
        # El cerebro: Predicción con el nuevo Super-Autocrop
        text, conf = predictor.predict(img)
        formatted = format_siiv_plate(text)
        
        # Obtener el recorte que hizo la IA internamente para mostrarlo
        fine_crop = predictor.autocrop_plate(img)
        
        # --- Dibujar ---
        plt.subplot(2, 3, i + 1)
        
        # Mostramos el fine crop (lo que realmente leyó la IA)
        plt.imshow(cv2.cvtColor(fine_crop, cv2.COLOR_BGR2RGB))
        
        color = '#00FF00' if len(text) >= 6 else '#FFFF00'
        plt.title(f"DETECCIÓN: {formatted}\nConf: {conf:.2f}", fontsize=16, fontweight='bold', color=color)
        plt.xlabel(f"Original: {img.shape[1]}x{img.shape[0]} | Crop: {fine_crop.shape[1]}x{fine_crop.shape[0]}", color='gray')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("🔬 RESULTADOS DE PRECISIÓN MASTER (FINE CROP + COLOR)\nIgnorando parachoques y logos mediante gradientes verticalizados", 
                 fontsize=22, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("✨ Ventana de precisión abierta.")
    plt.show()

if __name__ == "__main__":
    run_precision_gui()
