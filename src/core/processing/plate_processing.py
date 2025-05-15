# src/core/processing/plate_processing.py

from src.core.detection.plate_detector import PlateDetector
from src.core.processing.superresolution import enhance_plate
from src.core.ocr.recognizer import recognize_plate

_detector = None

def get_plate_detector():
    global _detector
    if _detector is None:
        _detector = PlateDetector("models/plate_detector.pt")
    return _detector

def process_plate(vehicle_bgr, conf=0.5):
    """
    1) Detecta todas las placas con YOLO,
    2) Aplica superresolución (EDSR),
    3) Ejecuta OCR para extraer el texto.
    Devuelve lista de (bbox, plate_sr, ocr_text, detection_time).
    """
    det = get_plate_detector()
    bboxes = det(vehicle_bgr, conf=conf)
    results = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        plate_crop = vehicle_bgr[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        plate_sr = enhance_plate(plate_crop)
        ocr_text = recognize_plate(plate_sr)
        if not ocr_text:
            continue
        results.append((bbox, plate_sr, ocr_text))
    if not results:
        return None, None, None
    # retornamos la primera detección
    return results[0]
