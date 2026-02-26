import cv2
import os
import torch
import numpy as np
from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.lprnet_engine import LPRNetPredictor
from src.core.ocr.recognizer import format_siiv_plate

def run_video_to_plate_demo(video_path):
    print(f"🎬 Iniciando Demo: Video -> Panorama -> Recorte Exacto -> LPRNet")
    
    # 1. Cargar Motores
    detector = PlateDetector()
    predictor = LPRNetPredictor()
    
    # Directorio de salida
    out_dir = "data/output/demo_video"
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video en {video_path}")
        return

    frame_count = 0
    detections_found = 0
    max_detections = 5 # Solo tomaremos las primeras 5 para no llenar el disco
    
    print(f"🔍 Procesando video... buscando placas en el panorama.")

    while cap.isOpened() and detections_found < max_detections:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Procesar 1 de cada 10 frames para velocidad
        if frame_count % 10 != 0: continue
        
        # A. DETECCIÓN EN PANORAMA (YOLO)
        results = detector.detect(frame, conf=0.4)
        
        for i, det in enumerate(results):
            x1, y1, x2, y2, score, _ = det
            
            # 1. Tomar captura del PANORAMA (Foto completa)
            panorama_path = os.path.join(out_dir, f"panorama_f{frame_count}_d{i}.jpg")
            cv2.imwrite(panorama_path, frame)
            
            # 2. Recorte Inicial (YOLO Box + pequeño margen)
            pad = 20
            yolo_crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
                              max(0, x1-pad):min(frame.shape[1], x2+pad)]
            
            # 3. RECORTE QUIRÚRGICO MASTER (Solo la placa)
            # Aquí es donde aplicamos el nuevo algoritmo de Sobel
            fine_crop = predictor.autocrop_plate(yolo_crop)
            
            # 4. Reconocimiento
            text, conf = predictor.predict(yolo_crop)
            formatted = format_siiv_plate(text)
            
            # Guardar recortes para que el usuario los vea
            crop_path = os.path.join(out_dir, f"plate_exact_{formatted}_f{frame_count}.jpg")
            cv2.imwrite(crop_path, fine_crop)
            
            print(f"✅ Placa Detectada: {formatted} (Conf: {conf:.2f})")
            print(f"   🖼️ Panorama guardado en: {os.path.basename(panorama_path)}")
            print(f"   ✂️ Recorte Maestro guardado en: {os.path.basename(crop_path)}")
            
            detections_found += 1
            if detections_found >= max_detections: break

    cap.release()
    print(f"\n✨ Demo finalizada. Revisa la carpeta: {out_dir}")
    print(f"📸 Se generaron fotos del panorama completo y los recortes exactos de la placa.")

if __name__ == "__main__":
    video_file = r"C:\Users\Abel\Desktop\InfractiVision\videos\VID4EDIT ‐ Hecho con Clipchamp.mp4"
    run_video_to_plate_demo(video_file)
