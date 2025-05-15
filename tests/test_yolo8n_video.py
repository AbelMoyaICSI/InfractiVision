# tests/test_yolo8n_video.py

import os
import sys

# 👉 Agrega la carpeta raíz al path para poder importar util.py
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import time
from ultralytics import YOLO
from util import read_license_plate, write_csv

def main():
    # 1) Rutas
    base_dir    = ROOT_DIR
    video_in    = os.path.join(base_dir, "videos", "Traffic IP Camera video.mp4")
    model_lp    = os.path.join(base_dir, "models", "license_plate_detector.pt")
    csv_out     = os.path.join(os.getcwd(), "results.csv")
    video_out   = os.path.join(os.getcwd(), "processed_output.mp4")

    # 2) Carga modelo de placas
    try:
        lp_model = YOLO(model_lp)
    except Exception as e:
        print("❌ Error cargando license_plate_detector.pt:", e, file=sys.stderr)
        sys.exit(1)

    # 3) Abrir vídeo
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_in}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4) Prepara writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    # 5) Resultados para CSV
    results = {}

    # 6) Bucle de inferencia
    frame_idx = 0
    start     = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_idx] = {}
        # inferencia de matrículas
        res = lp_model(frame)[0]
        for *xyxy, score, cls in res.boxes.data.tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            # recorte y OCR
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
            text, txt_score = read_license_plate(thresh)
            if not text:
                continue

            # guardar para CSV
            results[frame_idx][len(results[frame_idx])] = {
                "license_plate": {
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "bbox_score": float(score),
                    "text_score": float(txt_score)
                }
            }

            # dibujar caja y texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 4)
            cv2.putText(
                frame, text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2, (0, 140, 255), 3, cv2.LINE_AA
            )

        out.write(frame)

        # progreso
        pct = int((frame_idx + 1) / total_frames * 100)
        sys.stdout.write(f"\rProcesando video: {pct}%")
        sys.stdout.flush()
        frame_idx += 1

    # 7) Cerrar
    cap.release()
    out.release()
    print(f"\n✅ Vídeo procesado en {time.time() - start:.1f}s")
    print("🎥 Salida en:", video_out)

    # 8) Guardar CSV
    write_csv(results, csv_out)
    print("📄 CSV guardado en:", csv_out)

if __name__ == "__main__":
    main()
