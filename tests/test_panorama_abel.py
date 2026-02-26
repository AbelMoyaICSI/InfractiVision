import cv2
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.lprnet_engine import LPRNetPredictor

# ── Importar lógica de rectificación v6.3 ──────────────────────────
sys.path.insert(0, os.path.join(os.getcwd(), "tests", "perspective_experiment"))
from auto_rectifier import encontrar_esquinas, aplicar_homografia, order_points
# ───────────────────────────────────────────────────────────────────


def rectificar_placa(plate_roi_bgr):
    """
    Aplica la corrección de perspectiva v6.3 al recorte de placa.
    Devuelve imagen 300x110px o None si falla.
    """
    if plate_roi_bgr is None or plate_roi_bgr.size == 0:
        return None
    try:
        pts, method, score = encontrar_esquinas(plate_roi_bgr)
        if pts is not None:
            warped = aplicar_homografia(plate_roi_bgr, pts)
            return warped
    except Exception as e:
        pass
    return None


def analyze_video_trajectory(video_path, output_folder, roi_points):
    print(f"🚀 Iniciando Test de Panorama + Homografía v6.3: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return

    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📺 Video: {w_vid}x{h_vid}, {fps} FPS, {total_v_frames} frames")

    detector   = PlateDetector()
    lpr_engine = LPRNetPredictor()
    lpr_engine.plate_detector = detector

    os.makedirs(output_folder, exist_ok=True)

    poly_y_coords   = [p[1] for p in roi_points]
    y_min_poly      = min(poly_y_coords)
    y_max_poly      = max(poly_y_coords)
    poly_depth_range = y_max_poly - y_min_poly

    frame_idx    = 0
    vehicles_data = {}
    max_test_frames = 400

    while cap.isOpened() and frame_idx < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"--- Procesando frame {frame_idx}/{max_test_frames} ---")

        if frame_idx % 2 != 0:
            continue

        detections = detector.detect(frame, conf=0.15)
        if detections:
            print(f"F{frame_idx}: {len(detections)} placas detectadas")

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]

            bumper_y = y2
            ppi = (bumper_y - y_min_poly) / poly_depth_range if poly_depth_range > 0 else 0

            print(f"  - Placa en Y={bumper_y:.0f}, PPI={ppi:.2f}")

            if -1.0 <= ppi <= 2.0:
                center   = ((x1+x2)/2, (y1+y2)/2)
                found_id = None

                for tid, data in vehicles_data.items():
                    last_pos = data[-1]['center']
                    dist = ((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)**0.5
                    if dist < 500:
                        found_id = tid
                        break

                if found_id is None:
                    found_id = len(vehicles_data) + 1
                    vehicles_data[found_id] = []

                # Recorte del vehículo (contexto)
                y1_v = int(max(0, y1-100));  y2_v = int(min(frame.shape[0], y2+100))
                x1_v = int(max(0, x1-100));  x2_v = int(min(frame.shape[1], x2+100))
                vehicle_crop = frame[y1_v:y2_v, x1_v:x2_v].copy()

                # Recorte crudo de la placa (YOLO bbox)
                plate_raw = frame[int(y1):int(y2), int(x1):int(x2)].copy()

                # ── HOMOGRAFÍA v6.3 ────────────────────────────────────
                plate_rectified = rectificar_placa(plate_raw)
                # ──────────────────────────────────────────────────────

                # OCR sobre la imagen rectificada (si salió bien) o el raw
                ocr_input = plate_rectified if plate_rectified is not None else plate_raw
                text, ocr_conf, processed = lpr_engine.predict(
                    ocr_input, return_processed=True, autocrop=False
                )

                vehicles_data[found_id].append({
                    'f':           frame_idx,
                    'ppi':         ppi,
                    'score':       conf,
                    'vehicle_img': vehicle_crop,
                    'plate_raw':   plate_raw,
                    'plate_rect':  plate_rectified,   # 300x110 (o None)
                    'text':        text,
                    'ocr_conf':    ocr_conf,
                    'center':      center,
                })

    cap.release()

    print(f"📊 Generando murales para {len(vehicles_data)} vehículos...")

    for tid, frames in vehicles_data.items():
        if len(frames) < 2:
            continue

        indices  = np.linspace(0, len(frames)-1, 6, dtype=int)
        selected = [frames[i] for i in indices]

        # Mural de 3 filas: vehículo | placa raw | placa rectificada
        mural_w = 1200
        mural_h = 520
        mural   = np.zeros((mural_h, mural_w, 3), dtype=np.uint8) + 30
        cell_w  = mural_w // 6

        for i, data in enumerate(selected):
            # ── Fila 1: Vehículo ─────────────────────────────────────
            v_img = cv2.resize(data['vehicle_img'], (cell_w-10, 140))
            mural[15:155, i*cell_w+5:(i+1)*cell_w-5] = v_img

            # ── Fila 2: Placa RAW (YOLO bbox) ────────────────────────
            p_raw = cv2.resize(data['plate_raw'], (cell_w-20, 50))
            mural[165:215, i*cell_w+10:(i+1)*cell_w-10] = p_raw

            # ── Fila 3: Placa RECTIFICADA (Homografía v6.3) ──────────
            if data['plate_rect'] is not None:
                p_rect = cv2.resize(data['plate_rect'], (cell_w-20, 50))
                mural[225:275, i*cell_w+10:(i+1)*cell_w-10] = p_rect
                rect_label = f"RECT: {data['text'] or '--'}"
                label_color = (0, 255, 128)
            else:
                # Rectificación falló → mostrar placeholder
                blank = np.zeros((50, cell_w-20, 3), np.uint8) + 60
                cv2.putText(blank, "NO RECT", (5, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                mural[225:275, i*cell_w+10:(i+1)*cell_w-10] = blank
                rect_label = "NO RECT"
                label_color = (80, 80, 255)

            # ── Datos numéricos ───────────────────────────────────────
            cv2.putText(mural, f"PPI: {data['ppi']:.2f}",
                        (i*cell_w+10, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(mural, f"YOLO: {data['score']:.2f}",
                        (i*cell_w+10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(mural, rect_label,
                        (i*cell_w+10, 355),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1)
            cv2.putText(mural, f"Conf: {data['ocr_conf']:.2f}",
                        (i*cell_w+10, 375),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Marca GOLDEN ─────────────────────────────────────────
            if data['ppi'] > 0.85 and i > 0 and selected[i-1]['ppi'] <= 0.85:
                cv2.rectangle(mural, (i*cell_w, 0), ((i+1)*cell_w, mural_h),
                              (0, 255, 0), 3)
                cv2.putText(mural, "GOLDEN",
                            (i*cell_w+10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Labels de filas ──────────────────────────────────────────
        cv2.putText(mural, "VEHICULO",  (5, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(mural, "YOLO RAW",  (5, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(mural, "HOMOG v6.3",(5, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

        output_path = os.path.join(output_folder, f"trayectoria_vehiculo_{tid}.png")
        cv2.imwrite(output_path, mural)
        print(f"✅ Mural guardado: {output_path} → '{selected[-1]['text']}'")


# ROI REAL de VID2COLISEO
ROI_COLISEO = [
    [2049, 1665],
    [2321, 1600],
    [3633, 1725],
    [3467, 1843]
]

if __name__ == "__main__":
    analyze_video_trajectory(
        "videos/VID2COLISEO.MOV",
        "tests/diagnostico_capturas",
        ROI_COLISEO
    )
