import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.core.detection.plate_detector import PlateDetector

def simulate_old_process(plate_img):
    try:
        h, w = plate_img.shape[:2]
        scale = 200.0 / h
        new_w = int(w * scale)
        resized = cv2.resize(plate_img, (new_w, 200), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binary
    except:
        return plate_img

def run_debug_test():
    print("🚀 Iniciando Test de Comparación Visual: MASTER vs OLD")
    
    predictor = LPRNetPredictor()
    detector = PlateDetector()
    
    sample_dir = "data/output/autos"
    # Escogemos unos específicos con nombres claros
    samples = [
        "vehicle_A3606L_t3_f552.jpg",
        "vehicle_ASA-841_t2_f475.jpg",
        "vehicle_AEG-061_t1_f118.jpg"
    ]
    
    # Verificar que existan, si no cargar los primeros 3
    samples = [f for f in samples if os.path.exists(os.path.join(sample_dir, f))]
    if not samples:
        samples = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')][:3]

    plt.figure(figsize=(24, 12))
    
    for i, filename in enumerate(samples):
        img_path = os.path.join(sample_dir, filename)
        vehicle_img = cv2.imread(img_path)
        if vehicle_img is None: continue
        
        print(f"\n🔍 Probando: {filename} ({vehicle_img.shape})")
        
        # A. Detectar placa (YOLO)
        detections = detector.detect(vehicle_img, conf=0.3)
        if not detections: 
            print(f"⚠️ YOLO no encontró placa en {filename}")
            continue
            
        # Tomar la de mayor confianza
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        x1, y1, x2, y2, score, _ = detections[0]
        
        # Padding generoso para el Fine Crop (30px)
        pad = 30
        yolo_crop = vehicle_img[max(0, y1-pad):min(vehicle_img.shape[0], y2+pad), 
                                max(0, x1-pad):min(vehicle_img.shape[1], x2+pad)]
        
        # B. PROCESO MASTER
        # Queremos ver qué pasa dentro de predict
        fine_crop = predictor.autocrop_plate(yolo_crop)
        master_input = cv2.resize(fine_crop, (94, 24), interpolation=cv2.INTER_LINEAR)
        
        # 3. Normalización exacta
        img_data = master_input.astype('float32')
        img_data = (img_data - 127.5) / 128.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_tensor = torch.from_numpy(img_data).unsqueeze(0).to(predictor.device)
        
        with torch.no_grad():
            logits = predictor.model(img_tensor)
        
        # Debugging logits y preds
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        decoded = predictor.decode(preds)
        
        print(f"   Indices detectados: {preds}")
        print(f"   Resultado: '{decoded}'")
        
        # C. PROCESO VIEJO (Simulación visual)
        old_viz = simulate_old_process(yolo_crop)
        
        # Visualización
        plt.subplot(len(samples), 4, i*4 + 1)
        plt.imshow(cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2RGB))
        plt.title(f"Capture ({yolo_crop.shape[1]}x{yolo_crop.shape[0]})")
        plt.axis('off')
        
        plt.subplot(len(samples), 4, i*4 + 2)
        if len(old_viz.shape) == 2: plt.imshow(old_viz, cmap='gray')
        else: plt.imshow(cv2.cvtColor(old_viz, cv2.COLOR_BGR2RGB))
        plt.title("OLD PROCESS (Binarized)")
        plt.axis('off')
        
        plt.subplot(len(samples), 4, i*4 + 3)
        plt.imshow(cv2.cvtColor(fine_crop, cv2.COLOR_BGR2RGB))
        plt.title(f"MASTER FINE CROP\n{fine_crop.shape[1]}x{fine_crop.shape[0]}")
        plt.axis('off')
        
        plt.subplot(len(samples), 4, i*4 + 4)
        plt.imshow(cv2.cvtColor(master_input, cv2.COLOR_BGR2RGB))
        plt.title(f"94x24 INPUT -> OCR: {decoded}")
        plt.axis('off')
        
    plt.suptitle("VALIDACIÓN MASTER LPRNet: FINE CROP vs OLD BINARY", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_png = "DEBUG_master_comparison_v2.png"
    plt.savefig(output_png, dpi=150)
    print(f"\n✨ Comparativa guardada en: {output_png}")

if __name__ == "__main__":
    run_debug_test()
