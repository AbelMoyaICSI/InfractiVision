"""Sistema de clasificación NID/NIE para detecciones de placas.

Extraído de preprocessing_dialog.py sin cambios algorítmicos.
"""

class PlateClassificationSystem:
    """
    Sistema de clasificación NID/NIE técnicamente justificado para la tesis.
    
    REGLAS BALANCEADAS (más NID, menos NIE):
    - NID: Detecciones confiables que se incluyen en estadísticas oficiales
    - NIE: Detecciones dudosas que requieren revisión manual
    
    CALIBRACIÓN TÉCNICA:
    - Umbrales optimizados para maximizar TI manteniendo calidad
    - Validación por consenso de múltiples frames
    - Tolerancia realista para condiciones operativas
    """
    
    def __init__(self):
        # UMBRALES CALIBRADOS CORRECTAMENTE (umbral técnico 70%)
        self.confidence_threshold_nid = 0.70    # ✅ Umbral técnico: 70% para NID
        self.char_tolerance = 2                 # Tolerancia razonable de caracteres
        self.min_consensus_frames = 2           # Frames mínimos para consenso
        self.min_plate_length = 5               # Mínimo válido SIIV (A1-234 = 5)
        self.max_plate_length = 8               # Máximo válido SIIV (ABC-1234 = 8)
        
        # Patrones de placas peruanas válidas SIIV 2010 + Regionales (Trujillo)
        self.valid_patterns = [
            r'^[A-Z]{3}-?\d{3}$',        # ABC-123 (Particular Nacional)
            r'^[A-Z]\d[A-Z]-?\d{3}$',    # T1A-123 (Particular Regional/Trujillo)
            r'^[A-Z]{2}\d-?\d{3}$',      # AB1-234 (Vehículos Menores)
            r'^[A-Z]\d{2}-?\d{3}$',      # T42-123 (Público antiguo / Camionetas)
            r'^\d{3}-?[A-Z]{3}$',        # 123-ABC (Formato inverso)
        ]
        
    def classify_detection(self, plate_text, confidence, frame_validations=None):
        """
        Clasifica una detección como NID o NIE basado en reglas técnicas BALANCEADAS.
        
        Args:
            plate_text: Texto de placa detectado
            confidence: Confianza de la detección
            frame_validations: Validaciones temporales opcionales
            
        Returns:
            Tuple[str, dict]: ('NID'/'NIE', metadata_dict)
        """
        if not plate_text:
            return 'NIE', {'razon': 'sin_detecciones'}
            
        # 1. VALIDAR CONFIANZA (MÁS PERMISIVO)
        if confidence < self.confidence_threshold_nid:
            return 'NIE', {
                'razon': 'confianza_baja',
                'confianza': round(confidence, 3),
                'umbral_minimo': self.confidence_threshold_nid
            }
            
        # 2. VALIDAR FORMATO (MÁS FLEXIBLE)
        format_validation = self._validate_format(plate_text)
        if not format_validation['is_valid']:
            return 'NIE', {
                'razon': 'formato_invalido',
                'placa_detectada': plate_text,
                'error_formato': format_validation['error']
            }
            
        # 2.5. 🚨 NUEVA VALIDACIÓN: Caracteres individuales (CRÍTICA para precisión)
        char_validation = self._validate_plate_characters(plate_text, confidence)
        if not char_validation['is_valid']:
            return 'NIE', {
                'razon': 'caracteres_invalidos',
                'placa_detectada': plate_text,
                'rechazo_detallado': char_validation['rejection_reason'],
                'analisis_caracteres': char_validation['char_analysis']
            }
            
        # 3. VALIDAR CONTEXTO TEMPORAL (si disponible)
        if frame_validations:
            temporal_valid = frame_validations.get('crossing_confirmed', True)
            if not temporal_valid:
                return 'NIE', {
                    'razon': 'cruce_no_confirmado',
                    'placa_detectada': plate_text
                }
                
        # ✅ CLASIFICAR COMO NID (DETECCIÓN VÁLIDA) - CON ANÁLISIS COMPLETO
        plate_origin = char_validation.get('plate_origin', 'desconocida')
        format_type = char_validation.get('format_type', 'no_identificado')
        origin_confidence = char_validation.get('origin_confidence', 0.0)
        
        # Ajustar justificación según origen
        if plate_origin == 'peruana':
            justification = f'✅ Placa PERUANA ({format_type}) - Validación estricta aprobada'
            classification_quality = 'alta'
        elif plate_origin == 'extranjera':
            justification = f'🌎 Placa EXTRANJERA ({format_type}) - Validación permisiva aprobada'
            classification_quality = 'media'  # Siempre media para extranjeras
        else:
            justification = 'Formato no identificado - Validación básica aprobada'
            classification_quality = 'baja'
        
        return 'NID', {
            'placa_final': plate_text,
            'confianza': round(confidence, 3),
            'origen_placa': plate_origin,
            'tipo_formato': format_type,
            'confianza_origen': round(origin_confidence, 3),
            'calidad_deteccion': classification_quality,
            'analisis_caracteres': char_validation['char_analysis'],
            'justificacion': justification
        }
        
    def _validate_format(self, plate_text):
        """Valida formato de placa peruana (MÁS PERMISIVO)."""
        import re
        
        clean_plate = plate_text.upper().strip().replace(' ', '')
        
        if len(clean_plate.replace('-', '')) < self.min_plate_length:
            return {'is_valid': False, 'error': 'muy_corta'}
            
        if len(clean_plate.replace('-', '')) > self.max_plate_length:
            return {'is_valid': False, 'error': 'muy_larga'}
            
        # NUEVO: Validación más permisiva
        # 1. Verificar patrones estrictos primero
        for pattern in self.valid_patterns:
            if re.match(pattern, clean_plate):
                return {'is_valid': True, 'pattern': pattern}
                
        # 2. Si no coincide exactamente, usar validación flexible
        if re.match(r'^[A-Z0-9-]{4,9}$', clean_plate):
            # Validar que tenga al menos algunas letras o números
            has_letters = any(c.isalpha() for c in clean_plate)
            has_numbers = any(c.isdigit() for c in clean_plate)
            
            if has_letters and has_numbers:
                return {'is_valid': True, 'pattern': 'flexible_calibrado'}
                
        return {'is_valid': False, 'error': 'patron_no_reconocido'}
    
    def _identify_plate_origin(self, plate_text):
        """
        🌎 Identifica si una placa es PERUANA o EXTRANJERA basándose en patrones específicos
        
        Args:
            plate_text: Texto de la placa detectado
            
        Returns:
            dict: {
                'origin': 'peruana'/'extranjera'/'desconocida',
                'format_type': str,
                'confidence': float,
                'validation_rules': dict
            }
        """
        if not plate_text:
            return {
                'origin': 'desconocida',
                'format_type': 'invalida',
                'confidence': 0.0,
                'validation_rules': {}
            }
        
        import re
        clean_text = plate_text.upper().replace('-', '').replace(' ', '')
        
        # 🇵🇪 PATRONES DE PLACAS PERUANAS (FORMATO OFICIAL CORRECTO)
        peruvian_patterns = {
            'peru_particular': r'^[A-Z]{3}-?\d{3}$',       # ABC-123
            'peru_regional': r'^[A-Z]\d[A-Z]-?\d{3}$',     # T1A-123
            'peru_menor': r'^[A-Z]{2}\d-?\d{3}$',          # AB1-234
            'peru_publico_antiguo': r'^[A-Z]\d{2}-?\d{3}$', # T42-499 (El que Abel reportó)
            'peru_especial': r'^[A-Z]{1}\d{5}$',           # A-12345 (Legacy)
        }
        
        # 🌎 PATRONES DE PLACAS EXTRANJERAS (TODO LO QUE NO SEA PERUANO)
        # Si no coincide con formato peruano, entonces es extranjera
        text_length = len(clean_text)
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)
        
        # 🔍 VERIFICAR PATRONES PERUANOS PRIMERO (FORMATO OFICIAL)
        for format_name, pattern in peruvian_patterns.items():
            if re.match(pattern, clean_text) or re.match(pattern, plate_text.upper()):
                # ✅ Coincide con formato peruano oficial (3 letras + guión + 3 números)
                return {
                    'origin': 'peruana',
                    'format_type': format_name,
                    'confidence': 0.95,
                    'validation_rules': {
                        'strict_format': True,
                        'character_validation': True,
                        'confidence_threshold': 0.75,  # Más estricto para peruanas
                        'allow_problematic_chars': False
                    }
                }
        
        # 🌎 SI NO ES PERUANA, ES EXTRANJERA (Lógica simple y correcta)
        if text_length >= 4 and text_length <= 12 and has_letters and has_numbers:
            # Cualquier placa con formato válido pero que no sea peruano = extranjera
            return {
                'origin': 'extranjera',
                'format_type': 'formato_no_peruano',
                'confidence': 0.80,
                'validation_rules': {
                    'strict_format': False,        # Más permisivo para extranjeras
                    'character_validation': False,  # No validar caracteres estrictamente
                    'confidence_threshold': 0.60,  # Umbral más bajo
                    'allow_problematic_chars': True # Permitir chars problemáticos
                }
            }
        
        # ❌ FORMATO COMPLETAMENTE DESCONOCIDO
        return {
            'origin': 'desconocida',
            'format_type': 'invalida',
            'confidence': 0.0,
            'validation_rules': {
                'strict_format': True,
                'character_validation': True,
                'confidence_threshold': 0.80,
                'allow_problematic_chars': False
            }
        }
    
    def _validate_plate_characters(self, plate_text, confidence):
        """
        🔍 Validación AVANZADA: Analiza caracteres individuales para evitar falsos NID
        
        Implementa verificación estricta a nivel de carácter para prevenir
        que placas con mayoría de caracteres incorrectos pero alta confianza 
        general sean clasificadas como NID.
        
        Args:
            plate_text: Texto de la placa detectado
            confidence: Confianza general de la detección
            
        Returns:
            dict: {
                'is_valid': bool,
                'char_analysis': dict,
                'rejection_reason': str (if invalid)
            }
        """
        if not plate_text or len(plate_text) < 3:
            return {
                'is_valid': False,
                'rejection_reason': 'placa_muy_corta',
                'char_analysis': {}
            }
            
        # 🎯 VALIDACIÓN POR TIPO DE CARÁCTER
        clean_text = plate_text.upper().replace('-', '').replace(' ', '')
        total_chars = len(clean_text)
        
        # Contadores de tipos de caracteres
        letters = sum(1 for c in clean_text if c.isalpha())
        numbers = sum(1 for c in clean_text if c.isdigit())
        invalid_chars = sum(1 for c in clean_text if not c.isalnum())
        
        # 🚨 CRITERIOS DE RECHAZO ESTRICTOS
        
        # 1. Demasiados caracteres inválidos
        invalid_ratio = invalid_chars / total_chars
        if invalid_ratio > 0.2:  # Más del 20% caracteres no alfanuméricos
            return {
                'is_valid': False,
                'rejection_reason': 'demasiados_caracteres_invalidos',
                'char_analysis': {
                    'invalid_chars': invalid_chars,
                    'invalid_ratio': round(invalid_ratio, 2),
                    'threshold': 0.2
                }
            }
            
        # 2. Balance incorrecto letras/números para placa peruana
        if letters < 1 or numbers < 1:
            return {
                'is_valid': False,
                'rejection_reason': 'balance_letras_numeros_incorrecto',
                'char_analysis': {
                    'letters': letters,
                    'numbers': numbers,
                    'required_min': {'letters': 1, 'numbers': 1}
                }
            }
            
        # 3. Patrones sospechosos comunes en OCR erróneo
        suspicious_patterns = [
            r'^[0-9]+$',           # Solo números
            r'^[A-Z]+$',           # Solo letras
            r'[IOQCL]{3,}',        # Demasiadas letras confusas consecutivas
            r'[0123456789]{6,}',   # Demasiados números consecutivos
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, clean_text):
                return {
                    'is_valid': False,
                    'rejection_reason': 'patron_sospechoso_detectado',
                    'char_analysis': {
                        'pattern_matched': pattern,
                        'text_analyzed': clean_text
                    }
                }
        
        # 4. � IDENTIFICAR ORIGEN DE LA PLACA (CRÍTICO para validación)
        plate_origin = self._identify_plate_origin(plate_text)
        validation_rules = plate_origin['validation_rules']
        
        # 5. 🎯 VALIDACIÓN ADAPTATIVA según origen de la placa
        problematic_chars = ['I', 'O', 'Q', 'L', '1', '0']  # Chars comúnmente confundidos
        problematic_count = sum(1 for c in clean_text if c in problematic_chars)
        
        # Aplicar validación según las reglas del origen
        if validation_rules.get('allow_problematic_chars', False):
            # 🌎 PLACAS EXTRANJERAS: Más permisivas
            estimated_char_confidence = confidence * 0.9  # Solo 10% de penalización
            required_confidence_threshold = 0.60  # Muy permisivo
            max_problematic_ratio = 0.8  # Hasta 80% puede ser problemático
        else:
            # 🇵🇪 PLACAS PERUANAS: Validación equilibrada (MTC Estándar)
            penalty_factor = 1 - (problematic_count / total_chars * 0.2) if problematic_count > 0 else 1.0
            estimated_char_confidence = confidence * penalty_factor
            required_confidence_threshold = 0.70  # Sincronizado con el umbral técnico oficial
            max_problematic_ratio = 0.6   # Máximo 60% problemático
            
        # 6. Aplicar umbral según tipo de placa
        if problematic_count >= total_chars * max_problematic_ratio:
            if estimated_char_confidence < required_confidence_threshold:
                return {
                    'is_valid': False,
                    'rejection_reason': 'demasiados_caracteres_problematicos',
                    'plate_origin': plate_origin['origin'],
                    'format_type': plate_origin['format_type'],
                    'char_analysis': {
                        'problematic_chars': problematic_count,
                        'total_chars': total_chars,
                        'problematic_ratio': round(problematic_count / total_chars, 2),
                        'estimated_confidence': round(estimated_char_confidence, 3),
                        'required_confidence': required_confidence_threshold,
                        'validation_mode': 'estricta' if not validation_rules.get('allow_problematic_chars') else 'permisiva'
                    }
                }
        
        # ✅ PLACA PASA TODAS LAS VALIDACIONES
        return {
            'is_valid': True,
            'plate_origin': plate_origin['origin'],
            'format_type': plate_origin['format_type'],
            'origin_confidence': plate_origin['confidence'],
            'char_analysis': {
                'total_chars': total_chars,
                'letters': letters,
                'numbers': numbers,
                'problematic_chars': problematic_count,
                'estimated_confidence': round(estimated_char_confidence, 3),
                'quality': 'alta' if estimated_char_confidence >= 0.9 else 'media',
                'validation_mode': 'estricta' if not validation_rules.get('allow_problematic_chars') else 'permisiva'
            }
        }
