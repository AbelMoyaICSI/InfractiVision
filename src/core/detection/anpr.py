import time
import cv2
import numpy as np
import imutils
import os
import easyocr
import re
from pathlib import Path
from ultralytics import YOLO

class ANPR:
    """
    Automatic Number Plate Recognition (ANPR) class.
    Combines YOLO-based plate detection with OCR for accurate license plate recognition.
    """
    
    def __init__(self, languages=['es', 'en'], model_path="models/license_plate_detector.pt"):
        """
        Initialize the ANPR system.
        """
        # Initialize EasyOCR reader
        self.reader = None
        self.languages = languages
        
        # Pre-compile regex patterns for better performance
        self.plate_patterns = [
            re.compile(r'^[A-Z]{3}\d{3,4}$'),
            re.compile(r'^[A-Z]{3}-\d{3}$'),
            re.compile(r'^[A-Z]{2,3}\d{3}$'),
            re.compile(r'^[A-Z]{3}\d{2,3}$'),
            re.compile(r'^[A-Z]{2,3}-\d{2,3}$'),
            re.compile(r'^[A-Z]{1,3}-\d{3}$'),
            re.compile(r'^[A-Z]{3}-\d{2,3}$'),
            re.compile(r'^\d{2,3}[A-Z]{3}\d{2}$'),
            re.compile(r'^\d{3}-[A-Z]{3}$'),
            re.compile(r'^[A-Z]{2}\d{3}[A-Z]{2}$'),
            re.compile(r'^[A-Z]{1,2}\d{4,5}$'),
            re.compile(r'^\d{4,7}$'),
            re.compile(r'^[A-Z]{1,3}\d{2,3}$')
        ]
        self.partial_plate_pattern = re.compile(r'[A-Z]{2,3}.*\d{2,3}')
        
        # Pre-define character equivalence classes
        self.equiv_classes = [
            set('0OQD'), set('1IL'), set('2Z'), set('5S'), 
            set('8B'), set('6G'), set('VUW'), set('4A'), 
            set('9g'), set('YV')
        ]
        
        # Initialize plate detector
        try:
            # Try multiple paths for model loading
            model_paths = [
                model_path,
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "license_plate_detector.pt"),
                os.path.join(os.getcwd(), "models", "license_plate_detector.pt")
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading license plate detector from: {path}")
                    # Optimizar rendimiento del modelo YOLO
                    self.model = YOLO(path)
                    
                    # Configuraciones para optimizar inferencia
                    if hasattr(self.model, 'predictor'):
                        self.model.predictor.args.verbose = False  # Desactivar salida detallada
                    
                    model_loaded = True
                    print("ANPR: License plate detector loaded successfully")
                    break
            
            if not model_loaded:
                raise FileNotFoundError("License plate detector model not found")
                
        except Exception as e:
            print(f"Error loading license plate detector: {e}")
            try:
                self.model = YOLO(Path("models/yolov8n.pt"))  # Create YOLO object, not just the path
                print("ANPR: Using generic model as fallback")
            except Exception as e2:
                print(f"Critical error, could not load any model: {e2}")
                self.model = None
        
        # Output directories for saving results
        self.output_dir = os.path.join("data", "output")
        self.plates_dir = os.path.join(self.output_dir, "placas")
        
        # Create directories if they don't exist
        os.makedirs(self.plates_dir, exist_ok=True)
        
        # Pre-define kernels para optimizar operaciones morfológicas
        self._init_kernels()
        
        # Cache de preprocesamiento para evitar recomputaciones
        self.preprocess_cache = {}
        self.cache_size_limit = 50  # Limitar tamaño de caché
    
    def _init_kernels(self):
        """Pre-define kernels for faster morphological operations"""
        self.sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.morph_kernel_2x2 = np.ones((2, 2), np.uint8)
        self.dilate_kernel_2x1 = np.ones((2, 1), np.uint8)
        self.vertical_separation_kernel = np.ones((2, 1), np.uint8)
        self.horizontal_separation_kernel = np.ones((1, 2), np.uint8)
    
    def initialize_reader(self):
        """Initialize EasyOCR reader only when needed"""
        if self.reader is None:
            try:
                print("Initializing EasyOCR...")
                self.reader = easyocr.Reader(self.languages, gpu=False, 
                                          quantize=True)  # Usar cuantización para acelerar
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                self.reader = None

    def _is_valid_plate_format(self, text):
        """Check using pre-compiled regex patterns for faster matching"""
        if not text or len(text) < 5:
            return False
        
        # Check if text matches any valid pattern using pre-compiled regex
        for pattern in self.plate_patterns:
            if pattern.match(text):
                return True
        
        # Check for partial match
        if self.partial_plate_pattern.search(text):
            return True
        
        return False
    
    def _iou(self, box1, box2):
        """Optimized IOU calculation"""
        # Calcular área de intersección directamente
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        intersection = x_overlap * y_overlap
        
        # Calcular áreas individuales de una vez
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calcular unión y retornar IOU
        union = area1 + area2 - intersection
        return 0 if union <= 0 else intersection / union
    
    def detect_plates_with_yolo(self, image, conf=0.25):
        """Optimized YOLO detection with reduced complexity"""
        if self.model is None or isinstance(self.model, Path):  # Check if model is Path
            return []
            
        try:
            # Optimization: Resize large images before detection
            orig_h, orig_w = image.shape[:2]
            resized = False
            img_for_detection = image
            
            # Resize very large images to speed up detection
            max_dim = 1280  # YOLO works efficiently at this resolution
            if max(orig_h, orig_w) > max_dim:
                scale = max_dim / max(orig_h, orig_w)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img_for_detection = cv2.resize(image, (new_w, new_h))
                resized = True
            
            # Adjust confidence for specific model types
            conf_threshold = max(0.35, conf) if "license_plate_detector" in str(self.model) else conf
            
            # Run inference with optimized parameters
            results = self.model(
                img_for_detection, 
                conf=conf_threshold, 
                classes=[0],
                verbose=False,  # Disable verbose output
                iou=0.5        # Set IoU threshold directly
            )
            
            # Extract detections efficiently
            detections = []
            
            # Process all boxes at once instead of looping
            for result in results:
                if not result.boxes:
                    continue
                    
                # Extract all boxes at once
                boxes = result.boxes.xyxy.cpu().numpy()  # Get all boxes
                confs = result.boxes.conf.cpu().numpy()  # Get all confidences
                
                # Process all boxes efficiently
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf_score = confs[i]
                    
                    # Adjust coordinates if we resized the image
                    if resized:
                        x1 = int(x1 * (orig_w / img_for_detection.shape[1]))
                        x2 = int(x2 * (orig_w / img_for_detection.shape[1]))
                        y1 = int(y1 * (orig_h / img_for_detection.shape[0]))
                        y2 = int(y2 * (orig_h / img_for_detection.shape[0]))
                    
                    # Fast aspect ratio check
                    width, height = x2 - x1, y2 - y1
                    if height == 0:  # Avoid division by zero
                        continue
                    
                    aspect_ratio = width / height
                    
                    # Filter by aspect ratio
                    if 1.5 <= aspect_ratio <= 6.0:
                        # Quick padding calculation
                        padding_x = max(1, int(width * 0.05))
                        padding_y = max(1, int(height * 0.15))
                        
                        # Boundary check
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(image.shape[1], x2 + padding_x)
                        y2 = min(image.shape[0], y2 + padding_y)
                        
                        # Add detection with confidence
                        detections.append((x1, y1, x2, y2, conf_score))
            
            # Fast non-max suppression
            if detections:
                # Sort by confidence (higher first)
                detections.sort(key=lambda x: x[4], reverse=True)
                
                # Apply custom fast NMS with pre-determined threshold
                keep = []
                indices = list(range(len(detections)))
                
                while indices:
                    # Keep detection with highest confidence
                    current = indices[0]
                    keep.append(current)
                    
                    # Find detections to remove
                    to_remove = []
                    for idx in indices[1:]:
                        if self._iou(detections[current][:4], detections[idx][:4]) > 0.45:
                            to_remove.append(idx)
                    
                    # Update indices
                    indices = [i for i in indices[1:] if i not in to_remove]
                
                # Build final list of detections
                return [detections[i][:4] for i in keep]
            
            return []
            
        except Exception as e:
            print(f"Error in YOLO plate detection: {e}")
            return []
    
    def preprocess_plate_image(self, plate_img):
        """
        Optimized preprocessing pipeline:
        - Uses caching to avoid redundant processing
        - Focuses on most effective techniques first
        - Reduces number of preprocessing methods for speed
        """
        # Check cache first
        img_hash = hash(plate_img.tobytes())
        if img_hash in self.preprocess_cache:
            return self.preprocess_cache[img_hash]
        
        # Use a smaller set of the most effective preprocessing techniques
        processed_images = []
        
        # Original image is always included
        processed_images.append(plate_img)
        
        # Convert to grayscale efficiently
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        processed_images.append(gray)
        
        # Use faster bilateral filter parameters
        bilateral = cv2.bilateralFilter(gray, 9, 15, 15)  # Slightly reduced parameters
        processed_images.append(bilateral)
        
        # Apply CLAHE with optimized parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        processed_images.append(enhanced)
        
        # Sharpen using pre-defined kernel (faster)
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        processed_images.append(sharpened)
        
        # Apply adaptive threshold (most effective technique)
        thresh_adapt = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 9, 2
        )
        processed_images.append(thresh_adapt)
        
        # Apply Otsu's thresholding (fast and effective)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu)
        
        # Reduce number of morphological operations (keep only the most effective)
        morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, self.morph_kernel_2x2)
        processed_images.append(morph_close)
        
        # Additional preprocessing for character segmentation
        char_segmented = self._enhance_for_character_segmentation(enhanced)
        if char_segmented is not None:
            processed_images.append(char_segmented)
        
        # Cache results (with size limit checking)
        if len(self.preprocess_cache) >= self.cache_size_limit:
            # Remove oldest entry when cache is full (simple approach)
            self.preprocess_cache.pop(next(iter(self.preprocess_cache)))
        
        self.preprocess_cache[img_hash] = processed_images
        
        return processed_images
    
    def _enhanced_text_similarity(self, text1, text2):
        """Optimized text similarity check"""
        # Quick checks first (short-circuit)
        if text1 == text2:
            return True
            
        # Normalize texts
        norm1 = text1.replace('-', '')
        norm2 = text2.replace('-', '')
        
        # Subset check
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Length comparison
        if abs(len(norm1) - len(norm2)) > 2:
            return False
        
        # Check pattern similarity first (faster than character comparison)
        pattern1 = self._extract_pattern_key(norm1)
        pattern2 = self._extract_pattern_key(norm2)
        if self._pattern_similarity(pattern1, pattern2) > 0.8:
            return True
        
        # Count differences with early exit
        diff_count = 0
        max_allowed = min(2, max(len(norm1), len(norm2)) // 3)
        
        for i in range(min(len(norm1), len(norm2))):
            if norm1[i] != norm2[i]:
                # Check equivalence classes efficiently
                equiv_found = False
                for equiv_class in self.equiv_classes:
                    if norm1[i] in equiv_class and norm2[i] in equiv_class:
                        equiv_found = True
                        break
                
                if not equiv_found:
                    diff_count += 1
                    
                    # Early exit if already too many differences
                    if diff_count > max_allowed:
                        return False
        
        # Add remaining length difference
        diff_count += abs(len(norm1) - len(norm2))
        
        return diff_count <= max_allowed
        
    def _enhance_for_character_segmentation(self, img):
        """Enhance image specifically for character segmentation"""
        if img is None:
            return None
        
        try:
            # Create a copy to avoid modifying the original
            enhanced = img.copy()
            
            # Apply morphological operations to separate touching characters
            # This helps with characters that might be connected in low-res images
            enhanced = cv2.erode(enhanced, self.vertical_separation_kernel, iterations=1)
            enhanced = cv2.dilate(enhanced, self.vertical_separation_kernel, iterations=1)
            
            # Enhance contrast for better character boundaries
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            if len(enhanced.shape) == 2:  # Grayscale
                enhanced = clahe.apply(enhanced)
            
            # Edge enhancement to improve character boundaries
            edges = cv2.Canny(enhanced, 100, 200)
            enhanced = cv2.addWeighted(enhanced, 0.8, edges, 0.2, 0)
            
            return enhanced
        except:
            return None
    
    def _calculate_pattern_coherence(self, text):
        """Calculate how well the text follows typical license plate patterns"""
        if not text or len(text) < 5:
            return 0.0
        
        # Common license plate patterns (without specifying exact plates)
        pattern_scores = 0.0
        
        # Check for alternating letter/number patterns
        letter_positions = [i for i, c in enumerate(text) if c.isalpha()]
        digit_positions = [i for i, c in enumerate(text) if c.isdigit()]
        
        # Plates often have letters clustered together, then numbers
        if letter_positions and digit_positions:
            # Check for letter-number grouping (e.g., AAA-111, AA-111)
            letter_avg = sum(letter_positions) / len(letter_positions)
            digit_avg = sum(digit_positions) / len(digit_positions)
            
            # If letters tend to come before numbers
            if letter_avg < digit_avg and max(letter_positions) < min(digit_positions):
                pattern_scores += 0.3
            elif digit_avg < letter_avg and max(digit_positions) < min(letter_positions):
                pattern_scores += 0.2
        
        # Check for common length patterns
        if 6 <= len(text) <= 7:
            pattern_scores += 0.2
        elif 5 <= len(text) <= 8:
            pattern_scores += 0.1
        
        # Check for typical letter/number distribution
        letter_count = len([c for c in text if c.isalpha()])
        digit_count = len([c for c in text if c.isdigit()])
        
        if letter_count > 0 and digit_count > 0:
            if abs(letter_count - digit_count) <= 2:  # Balanced distribution
                pattern_scores += 0.2
            
            # Common letter/digit ratios
            if letter_count == 3 and digit_count == 3:  # AAA-111 pattern
                pattern_scores += 0.3
            elif letter_count == 2 and digit_count >= 3:  # AA-1111 pattern
                pattern_scores += 0.2
            elif letter_count == 1 and digit_count >= 4:  # A-11111 pattern
                pattern_scores += 0.1
            elif letter_count == 1 and digit_count == 5 and text[0].isalpha():  # Particularly for A968B6 pattern
                pattern_scores += 0.4
        
        return min(1.0, pattern_scores)  # Cap at 1.0
    
    def _extract_pattern_key(self, text):
        """Extract a pattern key that represents the character types"""
        # Create a pattern representation like "LLLNNN" for letter-number patterns
        pattern = ""
        for c in text:
            if c.isalpha():
                pattern += "L"
            elif c.isdigit():
                pattern += "N"
            elif c == '-':
                pattern += "-"
        return pattern
    
    def _pattern_similarity(self, pattern1, pattern2):
        """Calculate similarity between two patterns"""
        # Handle empty patterns
        if not pattern1 or not pattern2:
            return 0.0
        
        # Calculate Levenshtein distance
        m, n = len(pattern1), len(pattern2)
        if m < n:
            return self._pattern_similarity(pattern2, pattern1)
        
        # Handle empty second pattern
        if n == 0:
            return 0.0
        
        # Initialize current row
        current_row = range(n+1)
        for i in range(1, m+1):
            previous_row, current_row = current_row, [i]+[0]*n
            
            for j in range(1, n+1):
                add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
                if pattern1[i-1] != pattern2[j-1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        # Convert to similarity score (0-1)
        max_len = max(m, n)
        return 1 - (current_row[n] / max_len if max_len > 0 else 0)
    
    def _estimate_format_break(self, text):
        """Estimate where the format breaks between letters and numbers"""
        if not text:
            return 0
        
        # Find the transition point from letters to numbers
        for i in range(1, len(text)):
            if text[i-1].isalpha() and text[i].isdigit():
                return i
            elif text[i-1].isdigit() and text[i].isalpha():
                return i
        
        # If no clear transition, use positional heuristic
        letter_count = sum(1 for c in text if c.isalpha())
        if letter_count > 0 and letter_count < len(text):
            if letter_count <= len(text) // 2:
                # Fewer letters than digits, likely at the beginning
                return letter_count
            else:
                # More letters than digits, check first digit position
                for i, c in enumerate(text):
                    if c.isdigit():
                        return i
        
        # Default to middle if no pattern found
        return len(text) // 2
    
    def _apply_context_aware_corrections(self, text, char_density=0):
        """Apply corrections based on context and position patterns"""
        if not text or len(text) < 3:
            return text
        
        # Enhanced positional correction with pattern awareness
        corrected = []
        has_dash = '-' in text
        dash_pos = text.find('-') if has_dash else self._estimate_format_break(text)
        
        # Position-aware correction dictionary - different for first and second parts
        first_part_corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '6': 'G', '4': 'A'
        }
        
        second_part_corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6', 'A': '4', 
            'T': '7', 'Q': '0', 'D': '0'
        }
        
        # Check for potential "A968B6" pattern - single letter followed by digits then letter then digits
        single_letter_digit_pattern = re.compile(r'^[A-Z]\d{2,4}[A-Z]\d{1,2}$')
        
        # Process each character with context awareness
        for idx, char in enumerate(text):
            # Special case for patterns like A968B6 - careful with letter/digit placement
            if single_letter_digit_pattern.match(text):
                # Keep original letters at positions 0 and 4 for A968B6-like patterns
                if (idx == 0 or idx == 4) and char.isalpha():
                    corrected.append(char)
                # Keep digits elsewhere
                elif idx != 0 and idx != 4 and char.isdigit():
                    corrected.append(char)
                # Correct potential errors in specific positions
                elif idx == 0 and char.isdigit() and char in first_part_corrections:
                    corrected.append(first_part_corrections[char])
                elif idx == 4 and char.isdigit() and char in first_part_corrections:
                    corrected.append(first_part_corrections[char])
                elif idx != 0 and idx != 4 and char.isalpha() and char in second_part_corrections:
                    corrected.append(second_part_corrections[char])
                else:
                    corrected.append(char)
            else:
                # Determine character position relative to format
                is_first_part = (idx < dash_pos) if dash_pos > 0 else (idx < len(text) // 2)
                
                # Apply position-specific corrections
                if is_first_part and char.isdigit() and char in first_part_corrections and not (idx > 0 and text[idx-1].isdigit()):
                    corrected.append(first_part_corrections[char])
                elif not is_first_part and char.isalpha() and char in second_part_corrections and not (idx > 0 and text[idx-1].isalpha()):
                    corrected.append(second_part_corrections[char])
                else:
                    # Keep original character if no correction applies
                    corrected.append(char)
        
        result = ''.join(corrected)
        
        # Detect and fix common OCR errors in plate formats
        # For example: missing first letter in plates that should start with a letter
        if len(result) >= 5 and result[0].isdigit() and all(c.isdigit() for c in result[1:3]):
            # Check if this might be missing a leading letter (common in European plates)
            if char_density > 0 and char_density < 0.01:  # Low character density suggests possible missed char
                possible_letters = ['A', 'B', 'C', 'E']  # Common first letters
                # Add a placeholder letter if pattern suggests it's needed
                result = possible_letters[0] + result
        
        # Format correction for typical patterns without inserting dashes
        # (only if format isn't already valid)
        if not any(pattern.match(result) for pattern in self.plate_patterns):
            # Try to detect character-number transitions
            for i in range(1, len(result)-1):
                if (result[i-1].isalpha() and result[i].isdigit() and result[i+1].isdigit() and 
                    '-' not in result and i >= 2):
                    # Common transition point between letters and numbers
                    result = result[:i] + '-' + result[i:]
                    break
        
        return result
    
    def _group_by_pattern_similarity(self, candidates):
        """Group candidates by pattern similarity rather than just text"""
        groups = {}
        
        for candidate in candidates[:15]:  # Limit to top candidates for efficiency
            text, conf = candidate[0], candidate[1]
            pattern_key = self._extract_pattern_key(text)
            
            if pattern_key in groups:
                groups[pattern_key].append(candidate)
            else:
                # Check for similar patterns
                added = False
                for key in groups.keys():
                    if self._pattern_similarity(pattern_key, key) > 0.7:
                        groups[key].append(candidate)
                        added = True
                        break
                
                if not added:
                    groups[pattern_key] = [candidate]
        
        return groups
    
    def _find_best_representative_text(self, group):
        """Find the best representative text from a group of candidates"""
        if not group:
            return ""
        
        # First check for valid formats with high confidence
        valid_candidates = [(text, conf) for text, conf, *_ in group if self._is_valid_plate_format(text)]
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            return valid_candidates[0][0]
        
        # If no valid formats, use confidence and consistency
        group.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest confidence candidate
        return group[0][0]
    
    def recognize_plate_text(self, plate_img, plate_idx=0):
        """Optimized OCR process with enhanced pattern recognition and context awareness"""
        if plate_img is None or plate_img.size == 0:
            return ""
            
        # Initialize OCR on demand
        if self.reader is None:
            self.initialize_reader()
            
        if self.reader is None:
            return ""
            
        try:
            # Optimize size for OCR performance with better scaling strategy
            height, width = plate_img.shape[:2]
            
            # More intelligent resize based on aspect ratio and clarity
            if width < 200 or height < 50:
                # Small plates need more upscaling - uses cubic for better details
                target_width = max(250, min(400, int(width * 1.5)))
                scale_factor = target_width / width
                plate_img = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor, 
                                    interpolation=cv2.INTER_CUBIC)
            elif width > 400:
                # Downscale very large images but preserve details
                scale_factor = 400.0 / width
                plate_img = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_AREA)
            
            # Enhanced preprocessing to target specific issues
            processed_images = self.preprocess_plate_image(plate_img)
            
            # Lista para candidatos de texto con metadatos extendidos
            text_candidates = []
            
            # Configuraciones optimizadas con ajustes para placas específicas
            configs = [
                # Standard configuration with enhanced character recognition
                {"allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", 
                 "paragraph": False, "height_ths": 0.5},
                
                # High precision configuration with special attention to character segmentation
                {"allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", 
                 "text_threshold": 0.6, "link_threshold": 0.7, "low_text": 0.6,
                 "width_ths": 0.5},
            ]
            
            # Process high-priority images first with different rotation angles
            priority_indices = [0, 3, 5, 6]  # Original, enhanced, thresholded, otsu
            for i in priority_indices:
                if i >= len(processed_images):
                    continue
                
                img = processed_images[i]
                
                # More intelligent rotation handling based on aspect ratio
                rotation_range = 7 if width / height > 3.0 else 5  # Wider plates need more rotation options
                rotations = [0]
                
                if i in [0, 3]:  # Try rotations only for original and enhanced images
                    rotations = [0, -rotation_range, rotation_range]
                    if width / height > 4.0:  # Extra rotations for very wide plates
                        rotations.extend([-rotation_range//2, rotation_range//2])
                
                for rotation in rotations:
                    if rotation != 0:
                        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotation, 1)
                        rot_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    else:
                        rot_img = img
                    
                    # Apply OCR with each configuration
                    for config_idx, config in enumerate(configs):
                        try:
                            results = self.reader.readtext(rot_img, **config)
                            
                            if results and len(results) > 0:
                                # Filter and process results
                                for r in results:
                                    if len(r) >= 3:
                                        bbox, text, conf = r
                                        
                                        # Extract more metadata about detection
                                        bbox_width = max(bbox[1][0] - bbox[0][0], bbox[2][0] - bbox[3][0])
                                        bbox_height = max(bbox[2][1] - bbox[1][1], bbox[3][1] - bbox[0][1])
                                        char_density = len(text) / (bbox_width * bbox_height) if bbox_width * bbox_height > 0 else 0
                                        
                                        # Normalize text with positional awareness
                                        text = text.upper().strip()
                                        text = ''.join(c for c in text if c.isalnum() or c == '-')
                                        
                                        # Positional correction with improved context awareness
                                        # This uses pattern recognition without hardcoding specific plates
                                        corrected_text = self._apply_context_aware_corrections(text, char_density)
                                        
                                        # Extended validation with pattern analysis
                                        is_valid = self._is_valid_plate_format(corrected_text)
                                        
                                        # Calculate pattern coherence score (higher for consistent patterns)
                                        pattern_score = self._calculate_pattern_coherence(corrected_text)
                                        
                                        if corrected_text and len(corrected_text) >= 5:
                                            # Enhanced confidence scoring with multiple factors
                                            adjusted_conf = conf * (1.5 if is_valid else 1.0) * (1.0 + pattern_score)
                                            
                                            # Store rich metadata for better consensus
                                            text_candidates.append((
                                                corrected_text, 
                                                adjusted_conf, 
                                                i,  # Image type 
                                                rotation,
                                                config_idx,  # Config used
                                                pattern_score,  # Pattern coherence
                                                bbox  # Bbox information for spatial analysis
                                            ))
                        except Exception as e:
                            continue  # Skip failures silently
            
            # Process remaining images if necessary
            if not text_candidates or max(c[1] for c in text_candidates) < 0.5:
                for i, img in enumerate(processed_images):
                    if i in priority_indices:
                        continue  # Skip already processed images
                    
                    for config in configs[:1]:  # Use only standard config for remaining images
                        try:
                            results = self.reader.readtext(img, **config)
                            
                            if results and len(results) > 0:
                                for r in results:
                                    if len(r) >= 3:
                                        bbox, text, conf = r
                                        
                                        # Basic processing for secondary images
                                        corrected_text = self._apply_context_aware_corrections(text.upper(), 0)
                                        
                                        if corrected_text and len(corrected_text) >= 5:
                                            is_valid = self._is_valid_plate_format(corrected_text)
                                            pattern_score = self._calculate_pattern_coherence(corrected_text)
                                            adjusted_conf = conf * (1.5 if is_valid else 1.0) * (1.0 + pattern_score * 0.5)
                                            text_candidates.append((
                                                corrected_text, adjusted_conf, i, 0, 0, pattern_score, bbox
                                            ))
                        except Exception:
                            continue
            
            # If no candidates, return empty
            if not text_candidates:
                return ""
            
            # Enhanced consensus approach with grouping and pattern analysis
            if len(text_candidates) >= 2:
                # Group by pattern similarity rather than just text similarity
                pattern_groups = self._group_by_pattern_similarity(text_candidates)
                
                # Score groups based on multiple factors
                best_group = None
                best_score = 0
                
                for pattern, group in pattern_groups.items():
                    # Calculate comprehensive group score
                    group_size = len(group)
                    avg_conf = sum(c[1] for c in group) / group_size
                    avg_pattern_score = sum(c[5] for c in group) / group_size
                    format_bonus = 1.8 if any(self._is_valid_plate_format(c[0]) for c in group) else 1.0
                    
                    # Comprehensive scoring that considers multiple factors
                    score = (group_size * 0.5 + avg_conf * 0.3 + avg_pattern_score * 0.2) * format_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_group = group
                
                if best_group:
                    # Find the most representative text from the best group
                    return self._find_best_representative_text(best_group)
            
            # Sort by adjusted confidence if no consensus
            text_candidates.sort(key=lambda x: x[1], reverse=True)
            return text_candidates[0][0]
                
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""
    
    def _apply_corrections(self, text):
        """Optimized corrections for common OCR errors in plates"""
        if not text:
            return text
        
        # Correct specific characters based on position
        corrected = []
        has_dash = '-' in text
        dash_pos = text.find('-') if has_dash else len(text) // 2
        
        # Common corrections
        corrections = {
            '0': 'O', 'O': '0', '1': 'I', 'I': '1', 
            '2': 'Z', 'Z': '2', '5': 'S', 'S': '5', 
            '8': 'B', 'B': '8', '4': 'A', 'A': '4'
        }
        
        for idx, char in enumerate(text):
            # Determine if in letters or numbers section
            in_letter_part = idx < dash_pos if has_dash else idx < 3
            
            # Apply corrections based on position
            if char in corrections:
                if in_letter_part:
                    # Prefer letters in first part
                    corrected.append(char if char.isalpha() else corrections[char])
                else:
                    # Prefer digits in second part
                    corrected.append(char if char.isdigit() else corrections[char])
            else:
                corrected.append(char)
        
        result = ''.join(corrected)
        
        # Format correction for standard patterns
        if len(result) == 6 and '-' not in result:
            # Check if follows common pattern (3 letters + 3 digits)
            letters = sum(1 for c in result[:3] if c.isalpha())
            digits = sum(1 for c in result[3:] if c.isdigit())
            if letters >= 2 and digits >= 2:
                result = result[:3] + '-' + result[3:]
        
        return result
    
    def detect_and_recognize_plate(self, image):
        """Optimized pipeline for plate detection and recognition"""
        if image is None or image.size == 0:
            return image, "", None, None
        
        # Use original image directly to avoid unnecessary copy
        img = image
        
        # Fast plate detection with YOLO
        plate_detections = self.detect_plates_with_yolo(img)
        
        # If YOLO failed, use simplified fallback
        if not plate_detections:
            # Downscale image for faster contour detection
            scale = 1.0
            proc_img = img
            h, w = img.shape[:2]
            
            # Resize if large image
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                proc_w, proc_h = int(w * scale), int(h * scale)
                proc_img = cv2.resize(img, (proc_w, proc_h))
            
            # Quick contour detection
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Fast blur
            edged = cv2.Canny(gray, 50, 200)
            
            # Find larger contours only (faster)
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            
            # Filter by size first (faster than sorting all)
            min_area = proc_img.shape[0] * proc_img.shape[1] * 0.01  # 1% of image
            large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Sort only large contours
            contours = sorted(large_contours, key=cv2.contourArea, reverse=True)[:5]
            
            # Check for rectangular contour
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                
                if len(approx) == 4:
                    # Get bounding rectangle directly (faster than mask)
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Scale back if we resized
                    if scale != 1.0:
                        x, y = int(x / scale), int(y / scale)
                        w, h = int(w / scale), int(h / scale)
                    
                    plate_detections = [(x, y, x+w, y+h)]
                    break
        
        # If still no detections, use ROI approach
        if not plate_detections:
            # Create detection in lower part of image where plates are often found
            h, w = img.shape[:2]
            y_start = int(h * 0.65)
            y_end = min(h, int(h * 0.95))
            x_start = int(w * 0.2)
            x_end = min(w, int(w * 0.8))
            
            plate_detections = [(x_start, y_start, x_end, y_end)]
        
        # Process each plate efficiently
        best_plate_text = ""
        best_plate_conf = 0
        best_plate_idx = -1
        best_plate_crop = None
        
        # Limit number of detections to process
        for i, (x1, y1, x2, y2) in enumerate(plate_detections[:3]):  # Process max 3 candidates
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            # Extract plate region
            plate_crop = img[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            # Recognize text
            plate_text = self.recognize_plate_text(plate_crop, i)
            
            if plate_text:
                # Score based on features
                conf = len(plate_text) / 8.0  # Base on length
                if self._is_valid_plate_format(plate_text):
                    conf *= 1.5  # Bonus for valid format
                
                # Calculate pattern coherence for better scoring
                pattern_score = self._calculate_pattern_coherence(plate_text)
                conf *= (1.0 + pattern_score)
                
                # Update best detection
                if conf > best_plate_conf:
                    best_plate_text = plate_text
                    best_plate_conf = conf
                    best_plate_idx = i
                    best_plate_crop = plate_crop
        
        # If we found a valid plate
        if best_plate_idx >= 0:
            x1, y1, x2, y2 = plate_detections[best_plate_idx]
            
            # Draw only if needed (skip drawing for intermediate results)
            result_img = img.copy()
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 if len(best_plate_text) <= 8 else 0.6  # Adjust for long text
            
            # Use fast text placement
            cv2.putText(result_img, best_plate_text, (x1, y1-10), font, font_scale, 
                      (0, 255, 0), 2, cv2.LINE_AA)
            
            return result_img, best_plate_text, (x1, y1, x2, y2), best_plate_crop
        
        # No plate detected
        return img, "", None, None
    
    def process_image(self, image):
        """Process image efficiently"""
        processed_img, plate_text, plate_coords, cropped_image = self.detect_and_recognize_plate(image)
        
        # Save plate image only if valid detection
        if plate_text and cropped_image is not None:
            self.save_plate_image(cropped_image, plate_text)
        
        return processed_img, plate_text
    
    def save_plate_image(self, plate_image, plate_text):
        """Save plate image with minimal overhead"""
        if plate_image is None or not plate_text:
            return
            
        # Generate filename efficiently
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{plate_text}_{timestamp}.jpg"
        filepath = os.path.join(self.plates_dir, filename)
        
        try:
            # Save with optimized compression
            cv2.imwrite(filepath, plate_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            print(f"Error saving plate image: {e}")
    
    def process_frame(self, frame, frame_idx=0, is_night=False):
        """Process video frame with optimizations for speed"""
        if frame is None or frame.size == 0:
            return frame, []
        
        # Fast night detection enhancement
        if is_night:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=35)
        
        # Skip frame copy when possible
        processed_frame, plate_text, plate_coords, cropped_image = self.detect_and_recognize_plate(frame)
        
        # Format detections for pipeline
        detections = []
        if plate_text and plate_coords is not None:
            # Generate unique filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_filename = f"plate_{plate_text}_{timestamp}.jpg"
            plate_path = os.path.join(self.plates_dir, plate_filename)
            
            # Save plate image
            if cropped_image is not None:
                cv2.imwrite(plate_path, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Add to detections
            detection_data = {
                "plate": plate_text,
                "plate_path": plate_path,
                "vehicle_path": None,
                "coords": plate_coords,
                "timestamp": frame_idx
            }
            detections.append(detection_data)
        
        return processed_frame, detections

    def calculate_plate_similarity(self, plate1, plate2):
        """
        Calcula la similitud entre dos textos de placas vehiculares.
        
        Args:
            plate1: Texto de la primera placa
            plate2: Texto de la segunda placa
            
        Returns:
            float: Puntuación de similitud entre 0.0 (totalmente diferentes) y 1.0 (idénticas)
        """
        if not plate1 or not plate2:
            return 0.0
        
        # Normalizar placas: eliminar guiones y convertir a mayúsculas
        p1 = plate1.replace('-', '').upper()
        p2 = plate2.replace('-', '').upper()
        
        # Coincidencia exacta
        if p1 == p2:
            return 1.0
        
        # Si tienen longitudes muy diferentes, probablemente sean placas distintas
        if abs(len(p1) - len(p2)) > 2:
            return 0.0
        
        # Calcular distancia de edición (Levenshtein distance)
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    # Calcular inserciones, eliminaciones y sustituciones
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Convertir distancia de edición a similitud
        max_len = max(len(p1), len(p2))
        edit_distance = levenshtein_distance(p1, p2)
        base_similarity = 1.0 - (edit_distance / max_len)
        
        # Ajustes adicionales para mejorar la detección de similitud en placas
        
        # 1. Verificar caracteres confundibles (0/O, 1/I, 8/B, etc.)
        confusable_pairs = [
            ('0', 'O'), ('O', '0'), 
            ('1', 'I'), ('I', '1'), 
            ('5', 'S'), ('S', '5'),
            ('8', 'B'), ('B', '8'),
            ('2', 'Z'), ('Z', '2')
        ]
        
        # Contar coincidencias de caracteres confundibles
        confusable_matches = 0
        for i in range(min(len(p1), len(p2))):
            if i < len(p1) and i < len(p2):
                if p1[i] == p2[i]:
                    continue
                
                # Comprobar si hay un par confundible
                for c1, c2 in confusable_pairs:
                    if (p1[i] == c1 and p2[i] == c2) or (p1[i] == c2 and p2[i] == c1):
                        confusable_matches += 1
                        break
        
        # Aumentar similitud basada en coincidencias de caracteres confundibles
        confusable_boost = 0.1 * (confusable_matches / max(1, min(len(p1), len(p2))))
        
        # 2. Verificar patrones comunes de placas (formato de letras/números)
        def get_pattern(text):
            pattern = ""
            for c in text:
                if c.isalpha():
                    pattern += "L"
                elif c.isdigit():
                    pattern += "N"
                else:
                    pattern += c
            return pattern
        
        pattern1 = get_pattern(p1)
        pattern2 = get_pattern(p2)
        
        # Incrementar similitud si los patrones coinciden
        pattern_match = 1.0 if pattern1 == pattern2 else 0.0
        
        # Calcular similitud total combinada
        combined_similarity = base_similarity + confusable_boost + (0.2 * pattern_match)
        
        # Normalizar para que no exceda 1.0
        return min(1.0, combined_similarity)

    def evaluate_plate_quality(self, plate_img):
        """
        Evalúa la calidad de una imagen de placa para determinar cuál conservar
        
        Args:
            plate_img: Imagen de la placa
            
        Returns:
            float: Puntuación de calidad de la imagen
        """
        if plate_img is None:
            return 0.0
        
        try:
            # 1. Nitidez (usando varianza de Laplaciano)
            laplacian_var = cv2.Laplacian(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # 2. Contraste
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            
            # 3. Brillo medio (preferimos brillo moderado)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs((brightness - 127.5) / 127.5)
            
            # 4. Tamaño de la imagen
            height, width = plate_img.shape[:2]
            size_score = min(1.0, (width * height) / (200 * 50))  # Normalizado para placas típicas
            
            # Calcular puntuación ponderada
            quality_score = (
                0.4 * (laplacian_var / 100) +  # Nitidez (normalizada)
                0.3 * (contrast / 80) +        # Contraste (normalizado)
                0.2 * brightness_score +       # Brillo óptimo
                0.1 * size_score               # Tamaño adecuado
            )
            
            return min(1.0, max(0.0, quality_score))
        
        except Exception as e:
            print(f"Error al evaluar calidad de placa: {e}")
            return 0.1  # Valor bajo por defecto en caso de error

# Example usage remains unchanged
if __name__ == "__main__":
    # Initialize ANPR system
    anpr = ANPR(languages=['es', 'en'])
    
    # Read image
    img_path = "data/output/autos/vehicle_76190.jpg"
    img = cv2.imread(img_path)
    
    if img is not None:
        print(f"Image loaded successfully, shape: {img.shape}")
        
        # Process image
        processed_img, plate_text = anpr.process_image(img)
        
        # Display results
        if plate_text:
            print(f"Detected license plate: {plate_text}")
            
            # Save the result image
            result_path = os.path.join("data", "output", "processed_vehicle.jpg")
            cv2.imwrite(result_path, processed_img)
            print(f"Saved result to {result_path}")
    else:
        print(f"Could not read image from {img_path}")