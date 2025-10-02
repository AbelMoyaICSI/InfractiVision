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
        
        # Patrones típicos de placas peruanas
        self.peru_patterns = [
            # Formato antiguo: 3 letras + 3 números (ABC-123)
            r'^[A-Z]{3}[0-9]{3}$',
            # Formato nuevo: 3 letras + 1 número + 2 letras (ABC1DE)
            r'^[A-Z]{3}[0-9][A-Z]{2}$',
            # Formato taxi: 3 letras + 1 número + 1 letra (ABC1D)
            r'^[A-Z]{3}[0-9][A-Z]$',
            # Formato especial: 2 letras + 4 números (AB1234)
            r'^[A-Z]{2}[0-9]{4}$',
            # Formato moto: 2 letras + 4 números
            r'^[A-Z]{2}[0-9]{4}$',
        ]
        
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
    
    def correct_confusing_characters(self, text, context_position=None):
        """Corrige caracteres confusos basado en contexto de placa peruana"""
        if not text:
            return text
        
        corrected = list(text.upper())
        
        for i, char in enumerate(corrected):
            if char in self.character_corrections:
                # Determinar si debería ser letra o número según posición
                if context_position:
                    if i < 3:  # Primeras 3 posiciones suelen ser letras
                        candidates = [c for c in self.character_corrections[char] if c.isalpha()]
                    elif i == 3:  # Cuarta posición puede ser número
                        candidates = [c for c in self.character_corrections[char] if c.isdigit()]
                    else:  # Posiciones finales dependen del formato
                        candidates = self.character_corrections[char]
                else:
                    candidates = self.character_corrections[char]
                
                # Tomar el candidato más probable
                if candidates:
                    corrected[i] = candidates[0]
        
        return ''.join(corrected)
    
    def validate_plate_format(self, text):
        """Valida si el texto coincide con formatos peruanos"""
        if not text:
            return False, 0
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        for i, pattern in enumerate(self.peru_patterns):
            if re.match(pattern, clean_text):
                return True, i + 1
        
        return False, 0
    
    def suggest_corrections(self, text):
        """Sugiere correcciones para mejorar coincidencia con formatos"""
        if not text:
            return []
        
        suggestions = []
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Generar variaciones corrigiendo caracteres confusos
        def generate_variations(text, pos=0):
            if pos >= len(text):
                return [text]
            
            variations = []
            char = text[pos]
            
            if char in self.character_corrections:
                for replacement in self.character_corrections[char]:
                    new_text = text[:pos] + replacement + text[pos+1:]
                    variations.extend(generate_variations(new_text, pos + 1))
            else:
                variations.extend(generate_variations(text, pos + 1))
            
            return variations
        
        # Limitar variaciones para evitar explosión combinatoria
        if len(clean_text) <= 6:
            variations = generate_variations(clean_text)
            
            for variation in variations[:20]:  # Limitar a 20 variaciones
                is_valid, confidence = self.validate_plate_format(variation)
                if is_valid:
                    suggestions.append((variation, confidence))
        
        # Ordenar por confianza
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]  # Top 5 sugerencias
    
    def enhance_ocr_result(self, raw_text, plate_img=None, is_night=False):
        """Mejora el resultado del OCR aplicando todas las correcciones"""
        if not raw_text:
            return raw_text, 0.0
        
        # Limpiar texto inicial
        clean_text = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        
        if len(clean_text) < 4 or len(clean_text) > 8:
            return clean_text, 0.0
        
        # Aplicar correcciones básicas
        corrected_text = self.correct_confusing_characters(clean_text, context_position=True)
        
        # Validar formato
        is_valid, format_confidence = self.validate_plate_format(corrected_text)
        
        if is_valid:
            return corrected_text, format_confidence * 0.2  # Confianza base
        
        # Si no es válido, buscar sugerencias
        suggestions = self.suggest_corrections(clean_text)
        
        if suggestions:
            best_suggestion, confidence = suggestions[0]
            return best_suggestion, confidence * 0.15  # Confianza reducida por ser sugerencia
        
        # Si nada funciona, devolver texto corregido básicamente
        return corrected_text, 0.1

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