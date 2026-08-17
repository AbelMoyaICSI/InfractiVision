import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.append(os.getcwd())

from src.core.detection.plate_detector import PlateDetector
from src.core.ocr.recognizer import recognize_plate, get_lprnet_predictor
from src.path_helper import resource_path

# ── Homografía v6.3 ────────────────────────────────────────────────
from src.core.processing.auto_rectifier import encontrar_esquinas, aplicar_homografia, order_points
# ──────────────────────────────────────────────────────────────────


def rectificar_placa_roi(plate_roi_bgr):
    """
    Aplica homografía v6.3 al recorte YOLO.
    PASO 0: Añadir padding ANTES de la homografía
            para que los bordes de la placa no queden
            cortados por el bbox ajustado de YOLO.
    PASO 1: encontrar_esquinas sobre imagen con aire
    PASO 2: aplicar_homografia → 300x110 plano
    """
    if plate_roi_bgr is None or plate_roi_bgr.size == 0:
        return None
    try:
        h, w = plate_roi_bgr.shape[:2]

        # ── PADDING PREVIO A LA HOMOGRAFÍA ─────────────────────────
        # El bbox YOLO es muy ajustado: si la placa está en ángulo
        # sus esquinas pueden quedar fuera del recorte → la homografía
        # recorta dígitos. Con aire suficiente, la homografía trabaja
        # con la placa completa y el resultado 300x110 no pierde nada.
        pad_x = int(w * 0.12)   # 12% lateral (≈ 1 carácter de margen)
        pad_y = int(h * 0.18)   # 18% vertical (zona arriba/abajo libre)
        padded = cv2.copyMakeBorder(
            plate_roi_bgr,
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_REPLICATE   # replica el borde → sin franjas negras
        )
        # ────────────────────────────────────────────────────────────

        pts, method, score = encontrar_esquinas(padded)
        if pts is not None:
            return aplicar_homografia(padded, pts)
    except Exception:
        pass
    return None


def strip_header_placa(img_rect, header_frac=0.25):
    """
    Quita la franja 'PERU' (header) del resultado de la homografía.
    El padding lateral YA fue añadido ANTES de la homografía,
    así que aquí solo recortamos la franja superior.
    """
    if img_rect is None:
        return None
    h = img_rect.shape[0]
    cut_y = int(h * header_frac)
    chars_only = img_rect[cut_y:, :]
    if chars_only.shape[0] < 15:
        return img_rect
    return chars_only


class LabForenseV24:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector   = PlateDetector()
        self.predictor  = get_lprnet_predictor()

        # Estado del mejor frame
        self.best_pqi      = 0
        self.best_frame    = None
        self.best_vehicle  = None
        self.best_surgical = None
        self.best_rect     = None          # ← placa rectificada (300x110)
        self.best_text     = "ESPERANDO..."
        self.best_conf     = 0.0
        self.best_method   = ""

        cap = cv2.VideoCapture(video_path)
        self.v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # ROI dinámico adaptativo
        self.roi_points = np.array([
            [int(self.v_w * 0.05), int(self.v_h * 0.98)],
            [int(self.v_w * 0.95), int(self.v_h * 0.98)],
            [int(self.v_w * 0.80), int(self.v_h * 0.35)],
            [int(self.v_w * 0.20), int(self.v_h * 0.35)]
        ], np.int32)

    def calculate_ppi(self, y_bbox):
        y_min = np.min(self.roi_points[:, 1])
        y_max = np.max(self.roi_points[:, 1])
        if y_max == y_min:
            return 0
        return float(np.clip((y_bbox - y_min) / (y_max - y_min), 0, 1))

    def is_inside_roi(self, x, y, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        if 0 <= int(y) < frame_shape[0] and 0 <= int(x) < frame_shape[1]:
            return mask[int(y), int(x)] == 255
        return False

    def draw_dashboard(self, frame_vid, ppi):
        """Dashboard 1280×720 en tiempo real."""
        cv2.polylines(frame_vid, [self.roi_points], True, (0, 255, 255), 3)

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8) + 20

        # ── Panel izquierdo: video en vivo (800×480) ──────────────
        vid_resized = cv2.resize(frame_vid, (800, 480))
        canvas[20:500, 20:820] = vid_resized

        # ── Panel derecho ─────────────────────────────────────────
        xp = 840   # x inicio panel derecho

        # Vehículo (GOLDEN)
        if self.best_vehicle is not None:
            vh = cv2.resize(self.best_vehicle, (400, 200))
            canvas[20:220, xp:xp+400] = vh
            cv2.putText(canvas, "VEHICULO - MOMENTO ORO",
                        (xp, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ── Placa RECTIFICADA (Homografía v6.3) ───────────────────
        rect_y = 230
        cv2.putText(canvas, "PLACA RECTIFICADA (Homografia v6.3)",
                    (xp, rect_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)
        if self.best_rect is not None:
            rect_vis = cv2.resize(self.best_rect, (400, 110), interpolation=cv2.INTER_CUBIC)
            canvas[rect_y:rect_y+110, xp:xp+400] = rect_vis
            # Borde verde para indicar rectificación OK
            cv2.rectangle(canvas, (xp, rect_y), (xp+400, rect_y+110), (0, 255, 128), 2)
        else:
            # Placeholder si no rectificó
            cv2.rectangle(canvas, (xp, rect_y), (xp+400, rect_y+65), (60, 60, 60), -1)
            cv2.putText(canvas, "SIN RECT (bbox directo)",
                        (xp+10, rect_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # ── Placa surgical (autocrop clásico, para comparar) ──────
        surg_y = rect_y + 120
        cv2.putText(canvas, "AUTOCROP CLASICO (referencia)",
                    (xp, surg_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
        if self.best_surgical is not None:
            surg_vis = cv2.resize(self.best_surgical, (400, 60), interpolation=cv2.INTER_CUBIC)
            canvas[surg_y:surg_y+60, xp:xp+400] = surg_vis

        # ── Resultados OCR ────────────────────────────────────────
        info_y = rect_y + 205
        color_conf = (0, 255, 0) if self.best_conf > 0.8 else (0, 255, 255)
        cv2.putText(canvas, f"LECTURA: {self.best_text}",
                    (xp, info_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, color_conf, 2)
        cv2.putText(canvas, f"Conf: {self.best_conf:.2f}   PQI: {self.best_pqi:.3f}",
                    (xp, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ── Barra PPI ─────────────────────────────────────────────
        bar_w = 760
        cv2.rectangle(canvas, (20, 650), (20+bar_w, 675), (40, 40, 40), -1)
        fill_w = int(bar_w * ppi)
        c_bar  = (0, 255, 0) if 0.80 <= ppi <= 0.98 else (0, 150, 255)
        cv2.rectangle(canvas, (20, 650), (20+fill_w, 675), c_bar, -1)
        zona   = "EN ZONA DORADA" if c_bar == (0, 255, 0) else "FUERA DE ZONA"
        cv2.putText(canvas, f"PPI: {ppi:.2f}  ({zona})",
                    (20, 643), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bar, 1)

        # ── Leyenda teclas ────────────────────────────────────────
        cv2.putText(canvas, "Q=Salir   S=Forzar captura",
                    (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        return canvas

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        print("🚀 LabForense + Homografía v6.3 activo. Analizando video...")
        print("   Q = salir  |  S = forzar captura manual")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections  = self.detector.detect(frame, conf=0.25)
            current_ppi = 0.0

            for det in detections:
                x1, y1, x2, y2, conf, _ = det
                in_zone = self.is_inside_roi((x1+x2)/2, (y1+y2)/2, frame.shape)

                if in_zone:
                    current_ppi = self.calculate_ppi(y2)
                    pqi = current_ppi * conf

                    # Momento de Oro: PPI 0.80-0.98 + mejor PQI
                    if 0.80 <= current_ppi <= 0.98 and pqi > self.best_pqi:
                        self.best_pqi = pqi

                        # Vehículo (contexto)
                        m = 150
                        v1 = int(max(0, y1-m));  v2 = int(min(frame.shape[0], y2+m))
                        h1 = int(max(0, x1-m));  h2 = int(min(frame.shape[1], x2+m))
                        self.best_vehicle = frame[v1:v2, h1:h2].copy()

                        # Recorte crudo de la placa
                        plate_raw = frame[int(y1):int(y2), int(x1):int(x2)].copy()

                        # ── PASO 1: HOMOGRAFÍA v6.3 ─────────────────
                        self.best_rect = rectificar_placa_roi(plate_raw)
                        # ── PASO 2: STRIP HEADER (quitar franja PERU) ─
                        # Actuamos sobre la imagen YA rectificada (plana)
                        # para que los caracteres ocupen todo el alto.
                        if self.best_rect is not None:
                            ocr_src = strip_header_placa(self.best_rect)
                        else:
                            ocr_src = plate_raw   # fallback: raw bbox
                        # ── PASO 3: OCR con autocrop sobre imagen plana ─
                        # autocrop=True → YOLO recorta ajustadamente
                        # la imagen YA rectificada, sin perspectiva.
                        txt, o_conf, surg = recognize_plate(
                            ocr_src, return_processed=True, autocrop=True
                        )
                        self.best_text     = txt
                        self.best_conf     = o_conf
                        self.best_surgical = surg

                # Bbox en video
                c_trk = (0, 255, 0) if in_zone else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c_trk, 3)
                cv2.putText(frame, f"{conf:.2f}",
                            (int(x1), int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, c_trk, 2)

            cv2.imshow("INFRACTIVISION — Homografía v6.3 | LIVE",
                       self.draw_dashboard(frame, current_ppi))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and detections:
                print("📸 Captura manual forzada!")
                self.best_pqi = 1.0

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test finalizado.")


if __name__ == "__main__":
    video = "videos/VID2COLISEO.MOV"
    if not os.path.exists(video):
        v_dir = "data/raw/videos"
        if os.path.exists(v_dir):
            videos = [f for f in os.listdir(v_dir) if f.lower().endswith(('.mp4', '.mov'))]
            if videos:
                video = os.path.join(v_dir, videos[0])

    LabForenseV24(video).run()
