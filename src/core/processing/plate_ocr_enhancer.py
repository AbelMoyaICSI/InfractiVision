"""
Módulo de corrección y mejora de precisión OCR para placas vehiculares
Maneja caracteres confusos y patrones típicos de placas peruanas
"""

import re
import cv2
import numpy as np
from collections import Counter

class PlateOCREnhancer:
    """Mejora la precisión del OCR en placas vehiculares"""
    
    def __init__(self):
        # Diccionario de caracteres confusos y sus correcciones más probables
        self.character_corrections = {
            # Números que se confunden con letras
            '0': ['O', '0'],  # Cero vs O
            '1': ['I', '1', 'L'],  # Uno vs I vs L
            '4': ['A', '4'],  # Cuatro vs A
            '5': ['S', '5'],  # Cinco vs S
            '6': ['G', '6'],  # Seis vs G
            '8': ['B', '8'],  # Ocho vs B
            
            # Letras que se confunden con números
            'O': ['0', 'O'],  # O vs cero
            'I': ['1', 'I', 'L'],  # I vs uno vs L
            'A': ['4', 'A'],  # A vs cuatro
            'S': ['5', 'S'],  # S vs cinco
            'G': ['6', 'G'],  # G vs seis
            'B': ['8', 'B'],  # B vs ocho
            'Z': ['2', 'Z'],  # Z vs dos
            'E': ['3', 'E'],  # E vs tres
            'T': ['7', 'T'],  # T vs siete
            
            # Otros caracteres confusos
            'Q': ['O', '0', 'Q'],
            'D': ['O', '0', 'D'],
            'U': ['V', 'U'],
            'V': ['U', 'V'],
            'M': ['N', 'M'],
            'N': ['M', 'N'],
            'P': ['R', 'P'],
            'R': ['P', 'R'],
            'F': ['E', 'F'],
            'K': ['R', 'K'],
            'W': ['V', 'W'],
            'Y': ['V', 'Y'],
        }
        
        # Patrones SIIV peruanos (Sistema Integral de Identificación Vehicular 2010)
        # Priorizados según la normativa actual
        self.peru_patterns = [
            # FORMATO PRINCIPAL SIIV 2010: 3 letras + 3 números (ABC123)
            r'^[A-Z]{3}[0-9]{3}$',
            # FORMATO SECUNDARIO SIIV 2010: 2 letras + 4 números (AB1234) 
            r'^[A-Z]{2}[0-9]{4}$',
            # FORMATO MIXTO SIIV: letra + número + letra + números (A1B234)
            r'^[A-Z][0-9][A-Z][0-9]{3}$',
            # FORMATO CORTO: 2 letras + 2 números + letra (AB12C)
            r'^[A-Z]{2}[0-9]{2}[A-Z]$',
            # Formatos antiguos (pre-2010) - menor prioridad
            r'^[A-Z]{3}[0-9][A-Z]{2}$',  # ABC1DE
            r'^[A-Z]{3}[0-9][A-Z]$',     # ABC1D
        ]
        
        # Regiones SIIV válidas (primera letra) - EXCLUYE RESERVADAS
        self.siiv_regions = {
            'A', 'B', 'C', 'D', 'F',  # Lima/Callao (prioridad alta)
            'T',  # La Libertad/TRUJILLO (PRIORIDAD MÁXIMA)
            'H', 'L', 'M', 'K', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Otras regiones activas
            'E',  # Especial
        }
        
        # Letras RESERVADAS (NO VÁLIDAS) - NO pueden aparecer como primera letra
        self.reserved_letters = {'G', 'I', 'J', 'N', 'O', 'Q', 'R'}  # Uso futuro/no activo
        
        # Caracteres válidos para placas peruanas
        self.valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.valid_numbers = set('0123456789')
        
    def preprocess_plate_image(self, plate_img, is_night=False):
        """Preprocesa la imagen de la placa para mejorar OCR"""
        try:
            if plate_img is None or plate_img.size == 0:
                return plate_img
            
            # Redimensionar si es muy pequeña
            h, w = plate_img.shape[:2]
            if w < 120 or h < 40:
                scale_factor = max(120/w, 40/h, 2.0)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Convertir a escala de grises
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()
            
            # Mejorar contraste (más agresivo en noche)
            if is_night:
                # CLAHE más fuerte para noche
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                gray = clahe.apply(gray)
                
                # Filtro bilateral para reducir ruido
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
            else:
                # CLAHE suave para día
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Umbralización adaptativa
            threshold = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Operaciones morfológicas para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # También devolver imagen en color mejorada para OCR alternativo
            if len(plate_img.shape) == 3:
                enhanced_color = plate_img.copy()
                if is_night:
                    # Mejorar cada canal de color
                    for i in range(3):
                        enhanced_color[:,:,i] = clahe.apply(enhanced_color[:,:,i])
            else:
                enhanced_color = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            
            return threshold, enhanced_color
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return plate_img, plate_img
    
    def correct_confusing_characters_siiv_aware(self, text, context_position=None):
        """
        Corrige caracteres confusos usando FORMATO SIIV peruano.
        NO aplica correcciones si la placa ya es SIIV válida.
        """
        if not text:
            return text
        
        # PASO 1: Verificar si ya es una placa SIIV válida
        clean = text.upper().replace("-", "").replace(" ", "")
        is_valid, confidence = self.validate_plate_format(clean)
        
        if is_valid and confidence > 0.5:
            # Ya es válida, NO aplicar correcciones
            return clean
        
        # PASO 2: Si NO es válida, aplicar correcciones conscientes del formato
        corrected = list(clean)
        
        # Detectar formato probable
        if len(clean) == 6:
            # Formato ABC123 (3 letras + 3 números)
            if clean[0] in self.siiv_regions or clean[0].isalpha():
                # Posiciones 0-2: deben ser letras
                for i in range(3):
                    if corrected[i].isdigit():
                        # Convertir número a letra
                        if corrected[i] == '1':
                            corrected[i] = 'I'
                        elif corrected[i] == '0':
                            corrected[i] = 'O'
                        elif corrected[i] == '5':
                            corrected[i] = 'S'
                        elif corrected[i] == '7':
                            corrected[i] = 'T'
                        elif corrected[i] == '4':
                            corrected[i] = 'A'
                        elif corrected[i] == '8':
                            corrected[i] = 'B'
                
                # Posiciones 3-5: deben ser números
                for i in range(3, 6):
                    if corrected[i].isalpha():
                        # Convertir letra a número
                        if corrected[i] == 'I':
                            corrected[i] = '1'
                        elif corrected[i] == 'O':
                            corrected[i] = '0'
                        elif corrected[i] == 'S':
                            corrected[i] = '5'
                        elif corrected[i] == 'T':
                            corrected[i] = '7'
                        elif corrected[i] == 'G':
                            corrected[i] = '6'
                        elif corrected[i] == 'B':
                            corrected[i] = '8'
                        elif corrected[i] == 'Z':
                            corrected[i] = '2'
        
        return ''.join(corrected)
    
    def validate_plate_format(self, text):
        """
        Valida si el texto coincide con formatos SIIV peruanos.
        Retorna (is_valid, confidence_score)
        """
        if not text:
            return False, 0.0
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Longitud EXACTA de 6 caracteres para placas peruanas SIIV
        if len(clean_text) != 6:
            print(f"⚠️ ENHANCER: Longitud incorrecta: {len(clean_text)} caracteres (debe ser 6)")
            return False, 0.0
        
        # Verificar si la primera letra es una región SIIV válida
        first_letter = clean_text[0] if clean_text else None
        is_valid_region = first_letter in self.siiv_regions
        is_reserved = first_letter in self.reserved_letters
        
        # RECHAZAR placas con letras RESERVADAS
        if is_reserved:
            print(f"⚠️ ENHANCER: Letra '{first_letter}' es RESERVADA (no válida en Perú)")
            return False, 0.05  # Confianza casi nula
        
        # Validar contra patrones SIIV
        for i, pattern in enumerate(self.peru_patterns):
            if re.match(pattern, clean_text):
                # Calcular confianza según:
                # - Prioridad del patrón (primeros son más comunes)
                # - Si tiene región SIIV válida
                base_confidence = 1.0 - (i * 0.1)  # Disminuye con patrones menos prioritarios
                
                if is_valid_region:
                    # Bonus extra si es región de TRUJILLO
                    if first_letter == 'T':
                        confidence = min(1.0, base_confidence * 1.5)  # +50% para Trujillo
                    else:
                        confidence = min(1.0, base_confidence * 1.2)  # +20% para otras regiones
                else:
                    confidence = base_confidence * 0.7  # Penalización si no es región válida
                
                return True, confidence
        
        return False, 0.0
    
    def enhance_ocr_result(self, raw_text, plate_img=None, is_night=False):
        """
        Mejora el resultado del OCR aplicando correcciones SIIV INTELIGENTES.
        PRIORIDAD 1: Si ya es SIIV válida, NO aplicar correcciones.
        PRIORIDAD 2: Solo corrige si realmente es necesario.
        """
        if not raw_text:
            return raw_text, 0.0
        
        # Limpiar texto inicial
        clean_text = re.sub(r'[^A-Za-z0-9-]', '', raw_text).upper()
        
        if len(clean_text) < 4 or len(clean_text) > 9:
            return clean_text, 0.0
        
        print(f"🔍 ENHANCER: Texto bruto: '{raw_text}' -> Limpio: '{clean_text}'")
        
        # PASO 0: Corregir longitud y caracteres específicos (S→5, 2→7, eliminar 0s)
        from src.core.ocr.recognizer import fix_plate_length_and_chars
        clean_text = fix_plate_length_and_chars(clean_text)
        print(f"   Después fix_length_and_chars: '{clean_text}'")
        
        # PASO 0.5: Corregir letras RESERVADAS (I, G) → T (Trujillo) si es probable
        clean_no_dash = clean_text.replace('-', '')
        if clean_no_dash and len(clean_no_dash) >= 3:
            first_letter = clean_no_dash[0]
            if first_letter in self.reserved_letters:
                # Intentar con 'T' (Trujillo)
                test_with_t = 'T' + clean_no_dash[1:]
                is_t_valid, t_conf = self.validate_plate_format(test_with_t)
                if is_t_valid and t_conf > 0.5:
                    print(f"🔄 ENHANCER: Corrección geográfica: '{clean_text}' → '{test_with_t}' ({first_letter}→T)")
                    clean_text = test_with_t
        
        # PASO 1: Verificar si YA es SIIV válida (SIN correcciones)
        is_valid_raw, conf_raw = self.validate_plate_format(clean_text.replace('-', ''))
        
        if is_valid_raw and conf_raw > 0.5:
            # Ya es válida, NO aplicar correcciones
            print(f"✅ ENHANCER: Placa ya válida, NO corregir: '{clean_text}' (conf: {conf_raw:.2f})")
            
            # Solo formatear con guión si no lo tiene
            from src.core.ocr.recognizer import format_siiv_plate
            formatted = format_siiv_plate(clean_text)
            
            final_confidence = conf_raw * 0.9
            if formatted and formatted[0] == 'T':
                final_confidence = min(1.0, final_confidence * 1.3)
            
            return formatted, final_confidence
        
        # PASO 2: Si NO es válida, intentar correcciones conscientes del formato
        print(f"⚠️ ENHANCER: Placa no válida, aplicando correcciones SIIV...")
        corrected_text = self.correct_confusing_characters_siiv_aware(clean_text, context_position=True)
        
        # Validar después de correcciones
        is_valid, format_confidence = self.validate_plate_format(corrected_text)
        
        if is_valid:
            print(f"✅ ENHANCER: Corrección exitosa: '{clean_text}' -> '{corrected_text}' (conf: {format_confidence:.2f})")
            
            # Formatear con guión
            from src.core.ocr.recognizer import format_siiv_plate
            formatted = format_siiv_plate(corrected_text)
            
            final_confidence = format_confidence * 0.8
            if formatted and formatted[0] == 'T':
                final_confidence = min(1.0, final_confidence * 1.3)
            
            return formatted, final_confidence
        
        # PASO 3: Si aún no es válida, RECHAZAR en lugar de inventar
        # NO usar sugerencias porque genera placas falsas como LG37S
        print(f"❌ ENHANCER: Placa '{corrected_text}' no cumple formato SIIV válido")
        print(f"   Rechazando detección para evitar falsos positivos")
        
        # Retornar vacío para indicar que no es válida
        # Esto evita que se registren placas inventadas
        return "", 0.0

# Instancia global del enhancer
_plate_enhancer = None

def get_plate_enhancer():
    """Obtiene instancia singleton del enhancer"""
    global _plate_enhancer
    if _plate_enhancer is None:
        _plate_enhancer = PlateOCREnhancer()
    return _plate_enhancer

def enhance_plate_recognition(plate_img, raw_ocr_text="", is_night=False):
    """Función principal para mejorar reconocimiento de placas"""
    enhancer = get_plate_enhancer()
    
    try:
        # Preprocesar imagen
        if plate_img is not None and plate_img.size > 0:
            processed_img, color_img = enhancer.preprocess_plate_image(plate_img, is_night)
        else:
            processed_img = plate_img
            color_img = plate_img
        
        # Mejorar resultado de OCR
        enhanced_text, confidence = enhancer.enhance_ocr_result(raw_ocr_text, plate_img, is_night)
        
        return {
            'enhanced_text': enhanced_text,
            'confidence': confidence,
            'processed_image': processed_img,
            'color_image': color_img,
            'original_text': raw_ocr_text
        }
        
    except Exception as e:
        print(f"Error en enhanced_plate_recognition: {e}")
        return {
            'enhanced_text': raw_ocr_text,
            'confidence': 0.0,
            'processed_image': plate_img,
            'color_image': plate_img,
            'original_text': raw_ocr_text
        }
