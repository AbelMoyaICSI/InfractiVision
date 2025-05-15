# src/core/ocr/recognizer.py

import pytesseract
import cv2

def recognize_plate(plate_bgr):
    """
    OCR real usando Tesseract:
    Convierte a escala de grises, umbraliza y extrae texto.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    text = text.strip()
    return text if text else None
