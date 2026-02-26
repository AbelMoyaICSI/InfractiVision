import cv2
import numpy as np
import os
import sys
import torch

# Añadir el path del proyecto para importar módulos
sys.path.append(os.getcwd())

from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.lprnet_engine import LPRNetPredictor

def test_ocr_best_moment(video_path, output_folder, roi_points):
    print(f"🚀 Iniciando Test OCR de Oro: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return

    # 1. Cargar Motores de IA
    detector = PlateDetector()
    lpr_engine = LPRNetPredictor()
    lpr_engine.plate_detector = detector # Vincular para autocrop interno
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Mapeo de polígono para PPI
    poly_y_coords = [p[1] for p in roi_points]
    y_min_poly = min(poly_y_coords)
    y_max_poly = max(poly_y_coords)
    poly_depth_range = y_max_poly - y_min_poly
    
    frame_idx = 0
    vehicles_data = {} # {track_id: {'best_pqi': 0, 'data': None}}
    
    # Procesar tramo representativo
    max_test_frames = 1200
    
    print(f"🎥 Procesando {max_test_frames} frames para capturar Momentos de Oro...")
    
    while cap.isOpened() and frame_idx < max_test_frames:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        if frame_idx % 2 != 0: continue # Skip 1 para velocidad
        
        if frame_idx % 100 == 0:
            print(f"--- Frame {frame_idx}/{max_test_frames} ---")

        detections = detector.detect(frame, conf=0.15)
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            
            # Cálculo de PPI
            bumper_y = y2
            ppi = (bumper_y - y_min_poly) / poly_depth_range if poly_depth_range > 0 else 0
            
            # Solo si está en una zona de profundidad razonable
            if 0.1 <= ppi <= 1.8:
                center = ((x1+x2)/2, (y1+y2)/2)
                found_id = None
                
                # Tracking simple por proximidad
                for tid, d in vehicles_data.items():
                    last_pos = d['last_pos']
                    dist = ((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)**0.5
                    if dist < 500:
                        found_id = tid
                        break
                
                if found_id is None:
                    found_id = len(vehicles_data) + 1
                    vehicles_data[found_id] = {
                        'best_pqi': -1,
                        'best_data': None,
                        'last_pos': center
                    }
                
                vehicles_data[found_id]['last_pos'] = center
                
                # Cálculo de PQI (Plate Quality Index): PPI * Confianza
                # Favorece placas grandes y con buena detección
                pqi = ppi * conf
                
                # Actualizar el mejor frame si PPI > 0.85 (Momento de Oro prioritario)
                # O si es el mejor pqi visto hasta ahora
                if pqi > vehicles_data[found_id]['best_pqi']:
                    vehicles_data[found_id]['best_pqi'] = pqi
                    
                    # Captura de datos
                    v_crop = frame[int(max(0, y1-150)):int(min(frame.shape[0], y2+150)), 
                                   int(max(0, x1-150)):int(min(frame.shape[1], x2+150))].copy()
                    
                    # Ejecutar OCR con Autocrop Quirúrgico
                    # return_processed=True nos devuelve el (texto, conf, cropped_img)
                    text, ocr_conf, s_crop = lpr_engine.predict(frame[int(y1):int(y2), int(x1):int(x2)], 
                                                              return_processed=True)
                    
                    # Imagen de entrada LPRNet (94x24) para visualizar la "adaptación"
                    lpr_input = lpr_engine.adapt_for_lprnet(s_crop, (94, 24))
                    
                    vehicles_data[found_id]['best_data'] = {
                        'f': frame_idx,
                        'v_img': v_crop,
                        's_img': s_crop,
                        'lpr_input': lpr_input,
                        'text': text,
                        'ocr_conf': ocr_conf,
                        'ppi': ppi
                    }

    cap.release()
    
    # GENERAR REPORTES VISUALES
    print(f"📊 Generando reportes de lectura para {len(vehicles_data)} vehículos...")
    
    for tid, v in vehicles_data.items():
        data = v['best_data']
        if data is None: continue
        
        # Mural de Validación:
        # Col 1: Vehículo (600x400)
        # Col 2: [Surgical Crop] arriba, [LPRNet Input] abajo, [Texto] al centro
        mural_w = 1000
        mural_h = 500
        mural = np.zeros((mural_h, mural_w, 3), dtype=np.uint8) + 40
        
        # 1. Vehículo
        v_resized = cv2.resize(data['v_img'], (500, 400))
        mural[50:450, 50:550] = v_resized
        
        # 2. Surgical Crop (X-Ray de la placa)
        s_h, s_w = data['s_img'].shape[:2]
        s_aspect = s_w / s_h if s_h > 0 else 1
        new_w = 300
        new_h = int(new_w / s_aspect)
        s_resized = cv2.resize(data['s_img'], (new_w, min(new_h, 100)))
        mural[50:50+s_resized.shape[0], 600:600+new_w] = s_resized
        
        # 3. LPRNet Input (94x24 escalado para ver)
        lpr_vis = cv2.resize(data['lpr_input'], (300, 80), interpolation=cv2.INTER_NEAREST)
        mural[200:280, 600:900] = lpr_vis
        
        # 4. Resultados de Texto
        cv2.putText(mural, f"LECTURA: {data['text']}", (600, 380), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(mural, f"Confianza OCR: {data['ocr_conf']:.2f}", (600, 420), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(mural, f"PPI: {data['ppi']:.2f} (Oro)", (600, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        # Título
        cv2.putText(mural, f"TEST OCR - VEHICULO #{tid}", (50, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        output_path = os.path.join(output_folder, f"ocr_result_v{tid}.png")
        cv2.imwrite(output_path, mural)
        print(f"✅ Reporte guardado: {output_path} -> {data['text']}")

ROI_COLISEO = [
    [2049, 1665],
    [2321, 1600],
    [3633, 1725],
    [3467, 1843]
]

if __name__ == "__main__":
    test_ocr_best_moment("videos/VID2COLISEO.MOV", "tests/resultado_ocr_oro", ROI_COLISEO)
