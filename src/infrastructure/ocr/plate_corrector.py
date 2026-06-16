"""SmartPlateCorrector — corrector inteligente de placas (extraído de preprocessing_dialog.py).

Mantiene la lógica algorítmica original sin modificaciones.
"""

class SmartPlateCorrector:
    """
    🚀 Sistema de corrección inteligente OPTIMIZADO con cache y procesamiento rápido
    
    MEJORAS DE RENDIMIENTO:
    - Cache de correcciones previas
    - Procesamiento selectivo por confianza
    - Validación rápida de patrones
    """
    
    def __init__(self):
        # 🚀 CACHE DE CORRECCIONES para evitar reprocesamiento
        self._correction_cache = {}
        self._cache_size_limit = 1000
        
        # ⚡ PROCESAMIENTO SELECTIVO: Solo corregir si confianza < umbral
        self.correction_threshold = 0.75  # Solo procesar si confianza < 75% (más permisivo)
        self.fast_validation = True       # Validación rápida activada
        
        # 📊 MAPAS DE CONFUSIÓN COMÚN optimizados (incluyendo variaciones de Perú)
        self.confusion_map = {
            # Números que se confunden con letras
            '0': ['O', 'D', 'Q', 'U'],
            '1': ['I', 'L', 'T', '7', 'J'],  
            '2': ['Z', 'S', '7'],
            '3': ['E', 'B', '8'],
            '4': ['A', 'H'],
            '5': ['S', 'Z'],
            '6': ['G', 'C', 'B'],
            '7': ['T', 'L', '1', 'F', '2', 'Z'],
            '8': ['B', 'S', '3', '0'],
            '9': ['P', 'R', 'G', '8'],
            
            # Letras que se confunden con números  
            'O': ['0', 'D', 'Q'],
            'I': ['1', 'L', 'J', 'T'],
            'L': ['1', 'I', '7'],
            'S': ['5', '8', '2', 'Z'],
            'G': ['6', '0', '9', 'C'],
            'B': ['8', '3', '6'],
            'T': ['7', '1', 'I', 'Y', 'A'],
            'H': ['N', 'M', '4', 'A', 'K'],
            'N': ['H', 'M', 'W'],
            'M': ['N', 'H', 'W', 'V'],
            'W': ['M', 'V', 'N'],
            'Z': ['2', '7', 'S'],
            'E': ['3', 'F'],
            'P': ['9', 'F', 'R'],
            'K': ['X', 'H', 'A', 'G'], 
            'A': ['4', 'H', 'K', 'X', 'T']
        }
        
        # 🇵🇪 PATRONES VÁLIDOS PERUANOS (SIIV 2010)
        self.peru_patterns = [
            r'^[A-Z]{3}-\d{3}$',    # ABC-123 (Particular Nacional)
            r'^[A-Z]\d[A-Z]-\d{3}$', # T1A-123 (Particular Trujillo/Regional)
            r'^[A-Z]{2}\d-\d{3}$',  # AB1-234 (Vehículos Menores)
            r'^[A-Z]{3}\d{3}$',     # ABC123 (Sin guion)
            r'^[A-Z]\d[A-Z]\d{3}$', # T1A123 (Sin guion)
        ]
        
        # 📋 BASE DE DATOS DE PLACAS CONOCIDAS (pequeña para rapidez)
        self.known_plates = self._load_known_plates()
        
        # === INTELIGENCIA REGIONAL: TRUJILLO (PERÚ) ===
        self.regional_context = "Trujillo"
        self.regional_codes = ["T", "A", "M", "P"] # T (Trujillo), A (Nacional), M (Lima), P (Piura)
        
        # Patrones específicos de Perú (SIIV 2010)
        self.peru_patterns = [
            r'^[A-Z]{3}-\d{3}$',    # ABC-123 (Nacional)
            r'^[TMD]\d[A-Z]-\d{3}$', # T1A-123 (Particular Trujillo)
            r'^[ABCDEFGHJKLNPQRSTVWXYZ]\d[A-Z]-\d{3}$', # Genérico Regional
            r'^[A-Z]{2}\d-\d{3}$',  # AB1-234 (Motos/Menores)
            r'^[A-Z]{3}\d{3}$',     # ABC123 (Sin guion)
            r'^[A-Z]\d[A-Z]\d{3}$', # A1B123
        ]
    
    def _add_to_cache(self, key, result):
        """⚡ Agregar resultado al cache con gestión optimizada"""
        if len(self._correction_cache) >= self._cache_size_limit:
            # Limpiar 20% del cache más antiguo para evitar limpiezas frecuentes
            items_to_remove = max(1, self._cache_size_limit // 5)
            for _ in range(items_to_remove):
                if self._correction_cache:
                    oldest_key = next(iter(self._correction_cache))
                    del self._correction_cache[oldest_key]
        self._correction_cache[key] = result
        
    def get_cache_stats(self):
        """📊 Obtener estadísticas del cache para debugging"""
        return {
            "cache_size": len(self._correction_cache),
            "cache_limit": self._cache_size_limit,
            "cache_usage": f"{len(self._correction_cache)}/{self._cache_size_limit}",
            "fast_validation": self.fast_validation
        }
        
    def correct_plate_smart(self, detected_plate, confidence):
        """
        🚀 VERSIÓN LPRNet Master V2: 
        Aplica reglas estructurales del MTC Perú para corregir confusiones (B/8, O/0, I/1).
        """
        if not detected_plate:
            return detected_plate, confidence, []

        clean = detected_plate.upper().replace(' ', '').replace('-', '')
        
        # SIIV MASTER: Todas las placas vehiculares tienen 6 caracteres
        if len(clean) != 6:
            return clean, confidence, []
            
        corrected = ""
        # REGLA 1: Posición 1 siempre es LETRA (Región T, A, M, etc.)
        c1 = clean[0]
        if c1.isdigit():
            alt = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}
            c1 = alt.get(c1, c1)
        corrected += c1

        # REGLA 2: Posiciones 2 y 3 pueden ser Letras o Números (Formatos LLL, LNL, LLN)
        # Aquí confiamos más en la IA a menos que sea una letra prohibida (Ñ, O, I, Q)
        corrected += clean[1:3]

        # REGLA 3: Posiciones 4, 5 y 6 siempre son NÚMEROS
        for i in range(3, 6):
            char = clean[i]
            if char.isalpha():
                alt = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'S': '5', 'G': '6', 'B': '8', 'T': '7', 'Z': '2', 'E': '3', 'P': '9'}
                corrected += alt.get(char, char)
            else:
                corrected += char
        
        # Formatear con guion SIIV
        formatted = f"{corrected[:3]}-{corrected[3:]}"
        return formatted, confidence, []

    def _correct_by_format_pattern(self, plate):
        """
        🔧 Corrección basada en FORMATO ESPERADO (3 letras + 3 números)
        """
        if not plate or len(plate) < 6:
            return plate
            
        # Remover guiones para análisis
        clean = plate.replace('-', '')
        if len(clean) != 6:
            return plate
            
        corrected = ""
        
        # PRIMEROS 3 CARACTERES: Deben ser LETRAS
        for i in range(3):
            char = clean[i]
            if char.isdigit():
                # Número en posición de letra - convertir
                letter_alternatives = {
                    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', 
                    '4': 'A', '5': 'S', '6': 'G', '7': 'T', 
                    '8': 'B', '9': 'P'
                }
                corrected += letter_alternatives.get(char, char)
            else:
                corrected += char
                
        corrected += '-'  # Añadir guión
        
        # ÚLTIMOS 3 CARACTERES: Deben ser NÚMEROS  
        for i in range(3, 6):
            char = clean[i]
            if char.isalpha():
                # Letra en posición de número - convertir
                number_alternatives = {
                    'O': '0', 'I': '1', 'L': '1', 'S': '5',
                    'G': '6', 'B': '8', 'T': '7', 'Z': '2',
                    'E': '3', 'P': '9'
                }
                corrected += number_alternatives.get(char, char)
            else:
                corrected += char
                
        return corrected

    def _correct_by_proximity(self, plate):
        """
        🎯 Corrección por PROXIMIDAD: Busca combinaciones cercanas válidas
        """
        if not plate:
            return plate
            
        # Generar variaciones inteligentes basadas en confusiones comunes
        variations = [plate]  # Incluir original
        
        # Para cada posición, generar variaciones con caracteres confundibles
        for i, char in enumerate(plate.replace('-', '')):
            if char in self.confusion_map:
                for alternative in self.confusion_map[char]:
                    # Crear nueva variación reemplazando solo este carácter
                    clean_plate = plate.replace('-', '')
                    new_variation = clean_plate[:i] + alternative + clean_plate[i+1:]
                    
                    # Formatear correctamente
                    if len(new_variation) == 6:
                        formatted = f"{new_variation[:3]}-{new_variation[3:]}"
                        variations.append(formatted)
        
        # Evaluar cada variación y seleccionar la mejor
        best_variation = plate
        best_score = self._evaluate_plate_quality(plate)
        
        for variation in variations:
            score = self._evaluate_plate_quality(variation)
            if score > best_score:
                best_score = score
                best_variation = variation
                
        return best_variation

    def _evaluate_plate_quality(self, plate):
        """
        📊 Evalúa la calidad de una placa según múltiples criterios
        """
        if not plate:
            return 0
            
        score = 0
        clean = plate.replace('-', '')
        
        # CRITERIO 1: Formato peruano válido (+40 puntos)
        if self._is_valid_peru_format(plate):
            score += 40
            
        # CRITERIO 2: Balance correcto letras/números (+30 puntos)
        letters = sum(1 for c in clean[:3] if c.isalpha())
        numbers = sum(1 for c in clean[3:] if c.isdigit())
        if letters == 3 and numbers == 3:
            score += 30
            
        # CRITERIO 3: Caracteres NO problemáticos (+20 puntos)
        problematic = ['I', 'O', 'L', '1', '0']  # Comúnmente confundidos
        problematic_count = sum(1 for c in clean if c in problematic)
        if problematic_count <= 2:  # Máximo 2 caracteres problemáticos
            score += 20
            
        # CRITERIO 4: Longitud correcta (+10 puntos)
        if len(clean) == 6:
            score += 10
            
        return score

    def _find_closest_known_plate(self, plate):
        """
        🔍 Busca la placa conocida más similar (SOLO si la base es pequeña)
        """
        if not self.known_plates or len(self.known_plates) > 1000:
            return None  # Evitar búsquedas costosas en bases grandes
            
        best_match = None
        best_similarity = 0.7  # Umbral mínimo
        
        clean_input = plate.replace('-', '').upper()
        
        for known_plate in self.known_plates:
            clean_known = known_plate.replace('-', '').upper()
            similarity = self._calculate_similarity(clean_input, clean_known)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = known_plate
                
        return best_match

    def _calculate_similarity(self, plate1, plate2):
        """
        🧮 Calcula similitud entre dos placas (optimizado)
        """
        if len(plate1) != len(plate2):
            return 0.0
            
        matches = sum(1 for a, b in zip(plate1, plate2) if a == b)
        return matches / len(plate1)

    def _is_valid_peru_format(self, plate):
        """
        ✅ Verifica formato peruano válido
        """
        import re
        if not hasattr(self, 'peru_patterns'):
             self.peru_patterns = [r'^[A-Z]{3}-?\d{3}$', r'^[A-Z]{2}\d-?\d{3}$']
        return any(re.match(pattern, plate) for pattern in self.peru_patterns)

    def generate_variations(self, plate):
        """
        🧬 Genera variaciones inteligentes (O/0, B/8, etc.) 
        para que el sistema de detección de duplicados no falle.
        Fundamental para la coherencia en baja resolución.
        """
        if not plate: return set()
        clean = plate.replace('-', '').upper()
        variations = {plate, clean}
        
        confusions = getattr(self, 'confusion_map', {
            '0': 'OQ', 'O': '0Q', 'B': '83', '8': 'B3', 
            'I': '1L', '1': 'IL', 'Z': '27', '2': 'Z7',
            'S': '5', '5': 'S', 'G': '6', '6': 'G', 'T': '7'
        })
        
        for i, char in enumerate(clean):
            if char in confusions:
                for alt in confusions[char]:
                    var = clean[:i] + alt + clean[i+1:]
                    variations.add(var)
                    # También añadir con guión si tiene longitud 6
                    if len(var) == 6:
                        variations.add(f"{var[:3]}-{var[3:]}")
                        
        return variations

    def _load_known_plates(self):
        """
        📋 Carga base de placas conocidas (implementar según necesidad)
        """
        # OPCIÓN 1: Cargar desde archivo JSON
        try:
            import json
            import os
            
            plates_file = resource_path("data/known_plates.json")
            if os.path.exists(plates_file):
                with open(plates_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ No se pudo cargar base de placas conocidas: {e}")
            
        # OPCIÓN 2: Base hardcodeada pequeña (placas comunes detectadas)
        return [
            "A3K-961", "T3J-538", "A96-8B6", "APH-188", 
            "H1G-421", "T4A-376", "T6D-138", "AV6-190",
            "TGT-947", "ABC-123", "XYZ-456", "DEF-789"
        ]


        
    def _analyze_consensus(self, plate_texts):
        """
        🚀 SUPER-CONSENSO: Analiza consenso CARÁCTER A CARÁCTER para máxima precisión.
        Este método permite que una PC lenta y una rápida lleguen al mismo resultado.
        """
        if len(plate_texts) < self.min_consensus_frames:
            return {'has_consensus': False, 'reason': 'insuficientes_frames'}
            
        # 1. Normalizar todas las placas (quitar guiones y espacios)
        normalized_plates = [p.replace('-', '').replace(' ', '').upper() for p in plate_texts]
        
        # 2. Votación por posición (Votación Temporal de Caracteres)
        # Encontramos la longitud más común
        from collections import Counter
        len_counts = Counter(len(p) for p in normalized_plates)
        target_len = len_counts.most_common(1)[0][0]
        
        # Filtrar placas que no tengan la longitud objetivo (evita ruido)
        valid_normalized = [p for p in normalized_plates if len(p) == target_len]
        if not valid_normalized:
            return {'has_consensus': False, 'reason': 'variación_longitud_excesiva'}
            
        final_chars = []
        for i in range(target_len):
            # Obtener todos los caracteres detectados en esta posición exacta
            chars_at_pos = [p[i] for p in valid_normalized]
            char_votes = Counter(chars_at_pos)
            
            # --- Lógica de Inteligencia Regional (Trujillo) ---
            # Si estamos en la 1ra posición y hay duda, priorizar 'T'
            if i == 0 and 'T' in char_votes and self.regional_context == "Trujillo":
                best_char = 'T'
            else:
                best_char = char_votes.most_common(1)[0][0]
                
            final_chars.append(best_char)
            
        best_text = "".join(final_chars)
        
        # 3. Formatear la placa resultante (ej: ABC123 -> ABC-123)
        if len(best_text) == 6:
            formatted_text = f"{best_text[:3]}-{best_text[3:]}"
        else:
            formatted_text = best_text # Dejar como está para otros formatos
            
        # Calcular porcentaje de confianza del consenso
        consensus_frames = sum(1 for p in normalized_plates if p == best_text)
        
        return {
            'has_consensus': True,
            'best_text': formatted_text,
            'consensus_frames': consensus_frames,
            'total_frames': len(plate_texts)
        }
        min_avg_distance = float('inf')
        
        for candidate in set(plate_texts):
            distances = [self._levenshtein_distance(candidate, other) for other in plate_texts]
            avg_distance = sum(distances) / len(distances)
            
            if avg_distance < min_avg_distance and avg_distance <= self.char_tolerance:
                min_avg_distance = avg_distance
                best_candidate = candidate
                
        return {
            'has_consensus': best_candidate is not None,
            'best_text': best_candidate,
            'consensus_frames': text_counts[best_candidate] if best_candidate else 0
        }
        
    def _validate_format(self, plate_text):
        """Valida formato de placa peruana."""
        import re
        
        if not plate_text or len(plate_text.replace('-', '').replace(' ', '')) < self.min_plate_length:
            return {'is_valid': False, 'error': 'muy_corta'}
            
        if len(plate_text.replace('-', '').replace(' ', '')) > self.max_plate_length:
            return {'is_valid': False, 'error': 'muy_larga'}
            
        # Verificar patrones válidos
        clean_plate = plate_text.upper().strip()
        for pattern in self.valid_patterns:
            if re.match(pattern, clean_plate):
                return {'is_valid': True, 'pattern': pattern}
                
        return {'is_valid': False, 'error': 'patron_no_reconocido'}
        
    def _levenshtein_distance(self, s1, s2):
        """Calcula distancia de edición entre dos strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
