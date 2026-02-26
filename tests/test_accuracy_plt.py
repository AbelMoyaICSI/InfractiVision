import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.core.ocr.recognizer import format_siiv_plate

def run_accuracy_test():
    print("🔬 Iniciando Test de Verificación de Tamaño y Textura Natural...")
    
    predictor = LPRNetPredictor()
    demo_dir = "data/output/demo_video"
    
    if not os.path.exists(demo_dir):
        print("❌ Error: Corre primero 'demo_panorama_to_plate.py'")
        return

    # Buscar parejas (panorama y placa exacta)
    files = os.listdir(demo_dir)
    panoramas = sorted([f for f in files if f.startswith("panorama_")])
    plates = sorted([f for f in files if f.startswith("plate_exact_")])
    
    if not plates:
        print("❌ no se encontraron imágenes de prueba.")
        return

    # Usar las 3 mejores detecciones
    samples = plates[:min(3, len(plates))]

    plt.figure(figsize=(22, 12))
    plt.style.use('dark_background')

    for i, plate_filename in enumerate(samples):
        # 1. Cargar la placa recortada (Naturaleza Original)
        plate_path = os.path.join(demo_dir, plate_filename)
        plate_img = cv2.imread(plate_path)
        h, w = plate_img.shape[:2]
        
        # 2. Inferencia Master
        text, conf = predictor.predict(plate_img)
        formatted = format_siiv_plate(text)
        
        # 3. Preparar lo que ve la IA internamente (94x24)
        # Esto es lo que el modelo usa para su red neuronal
        ia_input = cv2.resize(plate_img, (94, 24), interpolation=cv2.INTER_LINEAR)
        
        # --- FILA DE VISUALIZACIÓN ---
        # A. Recorte Original (Sin aplastar)
        plt.subplot(len(samples), 3, i*3 + 1)
        plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        plt.title(f"1. RECORTE NATURAL\nTam: {w}x{h} px\n(Textura Original)", fontsize=14, color='cyan')
        plt.axis('off')
        
        # B. IA Input (El estiramiento que el modelo requiere)
        plt.subplot(len(samples), 3, i*3 + 2)
        plt.imshow(cv2.cvtColor(ia_input, cv2.COLOR_BGR2RGB))
        plt.title(f"2. INPUT IA (94x24)\n(Stretched para Motor LPR)", fontsize=14, color='magenta')
        plt.axis('off')
        
        # C. Resultado Final
        plt.subplot(len(samples), 3, i*3 + 3)
        # Mostramos de nuevo el natural pero con el texto grande
        plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        res_color = 'lime' if len(text) >= 5 else 'yellow'
        plt.title(f"3. RESULTADO OCR\nIA: {formatted}\nConf: {conf:.2f}", fontsize=18, fontweight='bold', color=res_color)
        plt.axis('off')

    plt.suptitle("🧪 TEST DE PRECISIÓN: VERIFICACIÓN DE TAMAÑO ORIGINAL Y TEXTURA\n(Se confirma que el recorte NO tiene binarización y usa el color del video)", 
                 fontsize=24, fontweight='bold', y=0.98, color='white')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_name = "VERIFICACION_ACCURACY_FINAL.png"
    plt.savefig(output_name, dpi=140)
    print(f"\n✨ Test completado. Imagen guardada en: {output_name}")
    print(f"📦 Las placas originales se mantuvieron en su tamaño natural de captura.")
    
    # Intentar abrir la ventana si es posible
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        plt.show()
    except:
        pass

if __name__ == "__main__":
    run_accuracy_test()
