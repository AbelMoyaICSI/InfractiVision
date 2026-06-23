import cv2
import numpy as np
import re
import os
import threading
from src.path_helper import resource_path
from src.core.ocr.lprnet_engine import LPRNetPredictor

# ============================================================================
# CONSTANTES SIIV 2026 (REGLAMENTACIÓN MTC PERÚ)
# ============================================================================

SIIV_REGIONS = {
    'A': {'name': 'Lima/Callao', 'area': 9, 'priority': 'high'},
    'B': {'name': 'Lima/Callao', 'area': 9, 'priority': 'high'},
    'C': {'name': 'Lima/Callao', 'area': 9, 'priority': 'high'},
    'D': {'name': 'Lima/Callao', 'area': 9, 'priority': 'high'},
    'F': {'name': 'Lima/Callao', 'area': 9, 'priority': 'high'},
    'E': {'name': 'Especial (Diplomática/Emergencia)', 'area': 0, 'priority': 'medium'},
    'H': {'name': 'Ancash', 'area': 7, 'priority': 'medium'},
    'L': {'name': 'Loreto', 'area': 4, 'priority': 'medium'},
    'M': {'name': 'Amazonas/Cajamarca/Lambayeque', 'area': 2, 'priority': 'medium'},
    'K': {'name': 'Amazonas/Cajamarca/Lambayeque', 'area': 2, 'priority': 'medium'},
    'P': {'name': 'Tumbes/Piura', 'area': 1, 'priority': 'medium'},
    'S': {'name': 'San Martín', 'area': 3, 'priority': 'medium'},
    'T': {'name': 'La Libertad (TRUJILLO)', 'area': 5, 'priority': 'very_high'},
    'U': {'name': 'Ucayali', 'area': 6, 'priority': 'medium'},
    'V': {'name': 'Arequipa', 'area': 12, 'priority': 'medium'},
    'W': {'name': 'Huánuco/Junín/Pasco', 'area': 8, 'priority': 'medium'},
    'X': {'name': 'Apurímac/Cuzco/Madre de Dios', 'area': 10, 'priority': 'medium'},
    'Y': {'name': 'Ayacucho/Ica/Huancavelica', 'area': 11, 'priority': 'medium'},
    'Z': {'name': 'Moquegua/Puno/Tacna', 'area': 13, 'priority': 'medium'},
    'G': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
    'I': {'name': 'RESERVADO/Ayacucho', 'area': 1, 'priority': 'invalid', 'status': 'reserved'},
    'J': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
    'N': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
    'O': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
    'Q': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
    'R': {'name': 'RESERVADO', 'area': 0, 'priority': 'invalid', 'status': 'reserved'},
}

# ============================================================================
# SINGLETON PARA EL MOTOR LPRNet (PESO MASTER FINAL)
# ============================================================================
_lprnet_predictor = None
_lprnet_lock = threading.Lock()

def get_lprnet_predictor():
    global _lprnet_predictor
    with _lprnet_lock:
        if _lprnet_predictor is None:
            _lprnet_predictor = LPRNetPredictor()
    return _lprnet_predictor

# ============================================================================
# FUNCIÓN PRINCIPAL DE RECONOCIMIENTO (MODO DIRECTO)
# ============================================================================

def recognize_plate(plate_bgr, is_night=False, return_processed=False, autocrop=True, regional_context="Trujillo"):
    """
    RECONOCIMIENTO DIRECTO TRUJILLO SIIV
    Usa el modelo LPRNet MASTER_FINAL entrenado por Abel.
    Retorna (texto_formateado, confianza) o (texto, conf, cropped_img)
    """
    try:
        predictor = get_lprnet_predictor()
        # Inferencia directa con Autocrop y Stretching interno
        if return_processed:
            decoded, confidence, cropped = predictor.predict(plate_bgr, return_processed=True, autocrop=autocrop)
            formatted = format_siiv_plate(decoded, regional_context)
            return formatted, confidence, cropped
        else:
            decoded, confidence = predictor.predict(plate_bgr, autocrop=autocrop)
            formatted = format_siiv_plate(decoded, regional_context)
            return formatted, confidence
            
    except Exception as e:
        print(f"❌ Error en Reconocimiento LPRNet: {e}")
        if return_processed:
            return "", 0.0, plate_bgr
        return "", 0.0

# ============================================================================
# FUNCIONES DE APOYO SIIV (Mantenidas para compatibilidad y UI)
# ============================================================================

def format_siiv_plate(plate_text, regional_context="Trujillo"):
    if not plate_text: return plate_text
    clean = plate_text.replace('-', '').replace(' ', '').upper()
    
    # --- LÓGICA DE INTELIGENCIA REGIONAL (TRUJILLO SIIV) ---
    if len(clean) >= 4:
        first_char = clean[0]
        # En Trujillo, el 90% de infracciones son placas serie 'T'
        if regional_context == "Trujillo" and first_char in ['7', '1', 'Y', 'I']:
            clean = 'T' + clean[1:]
            
    # Formatos SIIV 2010: ABC-123, A1B-234, AB1-234
    if len(clean) == 6:
        # --- HEURÍSTICA DE CARRO (PERÚ) ---
        return f"{clean[:3]}-{clean[3:]}"
    return clean

def validate_siiv_format(plate_text):
    if not plate_text: return False, None, 0.0, ""
    
    # --- LIMPIEZA Y CORRECCIÓN INTELIGENTE ---
    clean = plate_text.replace('-', '').replace(' ', '').upper()
    
    # Si tiene 6 caracteres, aplicamos reglas de oro del MTC Perú
    if len(clean) == 6:
        # 1. La primera posición SIEMPRE es una LETRA (Región)
        if clean[0].isdigit():
            alt = {'7': 'T', '1': 'I', '5': 'S', '2': 'Z', '0': 'O', '8': 'B', '4': 'A'}
            clean = alt.get(clean[0], clean[0]) + clean[1:]
        
        # 2. Las últimas 3 posiciones SIEMPRE son NÚMEROS
        suffix = list(clean[3:])
        for i in range(3):
            if suffix[i].isalpha():
                alt = {'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'T': '7', 'O': '0', 'I': '1', 'L': '1', 'E': '3', 'P': '9', 'A': '4'}
                suffix[i] = alt.get(suffix[i], suffix[i])
        clean = clean[:3] + "".join(suffix)
        
        # 3. IDENTIFICACIÓN DE PATRÓN Y CORRECCIÓN QUIRÚRGICA
        # Formatos válidos: LLL (Particular), LNL (Trujillo/Nuevos), LNN (Antiguos/Otros)
        p1, p2, p3 = clean[0], clean[1], clean[2]
        
        # Si el patrón ya es válido (ej. T71, T7J, TBC), NO TOCAMOS NADA.
        # Esto evita que T70 se convierta en T7P erróneamente.
        current_pattern = f"{'L' if p1.isalpha() else 'N'}{'L' if p2.isalpha() else 'N'}{'L' if p3.isalpha() else 'N'}"
        valid_patterns = ["LLL", "LNL", "LNN"]
        
        if current_pattern not in valid_patterns:
            # Solo corregimos si el patrón es inválido (ej: NNN, NLL, etc.)
            prefix = list(clean[:3])
            
            # La primera SIEMPRE es letra
            if prefix[0].isdigit():
                alt = {'7': 'T', '1': 'I', '5': 'S', '2': 'Z', '0': 'O', '8': 'B', '4': 'A'}
                prefix[0] = alt.get(prefix[0], prefix[0])
            
            # Si el resto es NNL o algo raro, intentamos normalizar a LNN o LNL
            # Pero le damos prioridad a lo que el OCR leyó si tiene sentido
            clean = "".join(prefix) + clean[3:]

    # Regex actualizada para aceptar LNN-NNN (muy común en Trujillo antiguo)
    if re.match(r'^[A-Z]{3}\d{3}$', clean) or re.match(r'^[A-Z]\d[A-Z]\d{3}$', clean) or \
       re.match(r'^[A-Z]\d\d\d{3}$', clean) or re.match(r'^[A-Z]{2}\d{4}$', clean):
        first_letter = clean[0]
        first_letter = clean[0]
        if first_letter in SIIV_REGIONS:
            region = SIIV_REGIONS[first_letter]
            if region.get('status') == 'reserved': return False, 'RESERVED', 0.05, clean
            return True, 'SIIV', 0.9, format_siiv_plate(clean)
    
    return False, None, 0.0, clean

def calculate_siiv_confidence(plate_text, base_confidence=0.5):
    """
    SISTEMA DE DIAGNÓSTICO INTELIGENTE (Protocolo Abel V15)
    Calcula la confianza y asigna una razón amigable para el panel.
    """
    clean = plate_text.replace('-', '').upper()
    details = {
        'valid_siiv': False, 
        'formatted_plate': plate_text,
        'region': 'Desconocida', 
        'priority': 'none', 
        'vehicle_type': 'Desconocido',
        'valid_regional': False,
        'friendly_reason': 'Letras poco claras (Revisión necesaria)'
    }
    
    is_valid, fmt, boost, formatted = validate_siiv_format(clean)
    details['valid_siiv'] = is_valid
    details['formatted_plate'] = formatted
    
    # --- ASIGNACIÓN DE RAZONES AMIGABLES ---
    if is_valid:
        if base_confidence >= 0.85:
            details['friendly_reason'] = "✅ Placa leída correctamente"
        else:
            details['friendly_reason'] = "⚠️ Letras poco claras (Duda razonable)"
    else:
        if len(clean) < 4 or clean == "NIE":
            details['friendly_reason'] = "❌ No se detectó una placa clara (Objeto)"
        elif len(clean) < 6:
            details['friendly_reason'] = "❌ Placa incompleta (Faltan letras)"
        elif base_confidence < 0.30:
            details['friendly_reason'] = "❌ Imagen ilegible (Exceso de brillo/ruido)"
        elif base_confidence < 0.70:
            details['friendly_reason'] = "❌ Imagen muy borrosa para identificación"
        else:
            details['friendly_reason'] = "❌ Formato de placa no reconocido (SIIV)"

    # Datos regionales
    if clean and clean[0] in SIIV_REGIONS:
        region_info = SIIV_REGIONS[clean[0]]
        details['region'] = region_info['name']
        details['priority'] = region_info['priority']
        details['valid_regional'] = region_info['priority'] != 'invalid'
        
    details['vehicle_type'] = get_vehicle_type_by_ending(clean)
    
    return base_confidence, details

def get_vehicle_type_by_ending(plate_text):
    nums = ''.join(c for c in plate_text if c.isdigit())
    if len(nums) < 3: return "Desconocido"
    try:
        last_three = int(nums[-3:])
        if 0 <= last_three <= 599: return 'Particular'
        if 600 <= last_three <= 699: return 'Taxi'
        if 700 <= last_three <= 949: return 'Urbano/Camión'
        if 950 <= last_three <= 969: return 'Interprovincial'
        return 'Pesado/Otros'
    except: return "Otro"

# ============================================================================
# STUBS DE COMPATIBILIDAD (Para no romper otros módulos)
# ============================================================================

def fix_plate_length_and_chars(text):
    return text.upper().replace(' ', '').strip()

def correct_plate_format(text, is_night=False):
    return format_siiv_plate(text)

def enhance_plate_night(img):
    """Mantenemos el filtro Vegas Pro por si se desea mejora visual extra"""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except: return img
