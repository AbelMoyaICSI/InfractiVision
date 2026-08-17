import os
import time
import cv2
import numpy as np
from src.core.processing.resolution_process import enhance_plate_image
from src.core.detection.plate_detector import PlateDetector
from src.core.processing.superresolution import enhance_plate
from src.core.ocr.recognizer import recognize_plate
from src.core.processing.plate_ocr_enhancer import enhance_plate_recognition, get_plate_enhancer
from src.path_helper import resource_path

# ── Homografía v6.3 (correccion de perspectiva) ──────────────────────
try:
    from src.core.processing.auto_rectifier import encontrar_esquinas, aplicar_homografia
    _HOMOGRAFIA_OK = True
except Exception as _e:
    _HOMOGRAFIA_OK = False
    print(f"⚠️ Homografía v6.3 no disponible: {_e}")
# ──────────────────────────────────────────────────────────────


def rectificar_perspectiva(plate_raw):
    """
    Pipeline de rectificacion homografica v6.3.
    Orden correcto:
      1. Padding previo al bbox YOLO (12% x, 18% y)  ← da aire a las esquinas
      2. encontrar_esquinas → aplicar_homografia → 300x110 plano
      3. strip header PERU (top 25%)                  ← solo caracteres al OCR
    Retorna imagen lista para LPRNet, o None si falla.
    """
    if not _HOMOGRAFIA_OK or plate_raw is None or plate_raw.size == 0:
        return None
    try:
        h, w = plate_raw.shape[:2]

        # PASO 1: Padding antes de la homografia
        pad_x = int(w * 0.12)
        pad_y = int(h * 0.18)
        padded = cv2.copyMakeBorder(
            plate_raw, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_REPLICATE
        )

        # PASO 2: Homografia v6.3
        pts, method, score = encontrar_esquinas(padded)
        if pts is None:
            return None
        rectified = aplicar_homografia(padded, pts)   # 300x110
        if rectified is None:
            return None

        # PASO 3: Quitar franja PERU (header superior ~25% del alto)
        rh = rectified.shape[0]
        cut_y = int(rh * 0.25)
        chars_only = rectified[cut_y:, :]
        if chars_only.shape[0] < 15:
            return rectified   # Si es muy pequenna, usar completa

        return chars_only

    except Exception as ex:
        print(f"⚠️ rectificar_perspectiva error: {ex}")
        return None

_detector = None

def enhance_plate_night(plate_bgr):
    """
    Versión profesional para optimizar placas nocturnas.
    Aplica técnicas de compensación de luces y realce de contraste de bordes.
    """
    try:
        h, w = plate_bgr.shape[:2]
        if h < 5 or w < 5: return plate_bgr
            
        # 1. Escalamiento de alta calidad
        scale = 3.5 # Un poco más de escalado para más detalle
        resized = cv2.resize(plate_bgr, (int(w * scale), int(h * scale)), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 2. DIGITAL GAIN (MSR) - Tu Brillo de Vegas (0.116)
        img_float = resized.astype(np.float32) + 1.0
        scales = [15, 80, 250]
        msr = np.zeros_like(img_float)
        for sigma in scales:
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            msr += np.log(img_float) - np.log(blur)
        msr = msr / 3.0
        msr_norm = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX)
        wdr_img = cv2.convertScaleAbs(msr_norm, alpha=1.1, beta=15) # Simula tu brillo de Vegas
        
        # 3. FILTRO "VEGAS COLOR" (Niditud máxima)
        lab = cv2.cvtColor(wdr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # A. Reducción de ruido sutil en luminosidad
        denoised_l = cv2.bilateralFilter(l, 7, 50, 50)
        
        # B. Unsharp Masking AGRESIVO (Look Vegas)
        gauss_l = cv2.GaussianBlur(denoised_l, (0, 0), 1.5)
        l_sharpened = cv2.addWeighted(denoised_l, 3.5, gauss_l, -2.5, 0)
        
        # C. CLAHE de color quirúrgico
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))
        final_l = clahe.apply(l_sharpened)
        
        # 4. Profundidad visual
        invGamma = 1.0 / 1.2
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        final_l = cv2.LUT(final_l, table)
        
        # 5. Recombinación de color
        enhanced_lab = cv2.merge([final_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Error en profesional enhance_plate_night: {e}")
        return plate_bgr

import cv2
import numpy as np
import os

def process_plate(vehicle_roi, is_night=False):
    """
    FLUJO CORRECTO DE DOS ETAPAS:
    1. PlateDetector (license_plate_detector.pt) → Encuentra la PLACA dentro del carro
    2. LPRNet → Lee el texto de esa placa recortada
    
    Retorna: ((x1, y1, x2, y2), plate_img, plate_text, confidence)
    """
    if vehicle_roi is None or vehicle_roi.size == 0:
        return ((0,0,0,0), None, "", 0.0)

    try:
        from src.core.ocr.recognizer import get_lprnet_predictor, recognize_plate, calculate_siiv_confidence
        from src.core.detection.plate_detector import PlateDetector
        from src.path_helper import resource_path
        import os

        # ============ ETAPA 1: DETECTAR LA PLACA CON YOLO ESPECIALIZADO ============
        # Cargar el detector de placas (license_plate_detector.pt)
        if not hasattr(process_plate, '_plate_detector') or process_plate._plate_detector is None:
            model_path = resource_path("models/license_plate_detector.pt")
            if os.path.exists(model_path):
                process_plate._plate_detector = PlateDetector(model_path)
                print(f"✅ PlateDetector cargado: {model_path}")
            else:
                process_plate._plate_detector = PlateDetector()  # Usa el path por defecto
        
        detector = process_plate._plate_detector
        
        # Detectar placas dentro del ROI del vehículo
        plate_detections = detector.detect_plates(vehicle_roi, confidence=0.3)
        
        plate_crop = None
        bbox = (0, 0, 0, 0)
        
        if plate_detections:
            # Tomar la primera detección (la de mayor confianza)
            x1, y1, x2, y2 = [int(v) for v in plate_detections[0]]
            
            # Validar coordenadas
            h, w = vehicle_roi.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                # Extraer el recorte EXACTO de la placa
                plate_crop = vehicle_roi[y1:y2, x1:x2].copy()
                bbox = (x1, y1, x2, y2)
                print(f"🎯 PlateDetector encontró placa: {x2-x1}x{y2-y1}px")
        
        # Si no se detectó placa, usar el autocrop como fallback
        if plate_crop is None or plate_crop.size == 0:
            predictor = get_lprnet_predictor()
            plate_crop = predictor.autocrop_plate(vehicle_roi)
            h, w = plate_crop.shape[:2]
            bbox = (0, 0, w, h)
            print(f"⚠️ Usando fallback autocrop: {w}x{h}px")
        
        # ============ ETAPA 2: RECTIFICACIÓN + OCR ============
        # ── PASO A: Homografía v6.3 (padding → perspectiva → strip header) ──
        plate_rectified = rectificar_perspectiva(plate_crop)

        if plate_rectified is not None:
            # Placa rectificada disponible: OCR directo sin autocrop
            ocr_input = plate_rectified
            plate_text, raw_conf = recognize_plate(ocr_input, autocrop=False)
            print(f"📍 Homografía v6.3 OK → '{plate_text}' (conf {raw_conf:.2f})")
        else:
            # Fallback: pipeline original con autocrop quirurgico
            plate_text, raw_conf = recognize_plate(plate_crop, autocrop=True)
            print(f"⚠️ Fallback autocrop → '{plate_text}' (conf {raw_conf:.2f})")

        # Validar con SIIV
        siiv_conf = 0.0
        if plate_text:
            siiv_conf, _ = calculate_siiv_confidence(plate_text, raw_conf)

        print(f"✅ process_plate: '{plate_text}' (SIIV: {siiv_conf:.2f})")
        return (bbox, plate_crop, plate_text, siiv_conf)

    except Exception as e:
        print(f"❌ Error en process_plate: {e}")
        import traceback
        traceback.print_exc()
        return ((0,0,0,0), vehicle_roi, "", 0.0)
