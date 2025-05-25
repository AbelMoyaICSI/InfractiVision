import cv2
import numpy as np
import os
import time
from pathlib import Path

class SuperResolutionProcessor:
    """
    Super Resolution processor for enhancing license plate images without requiring
    external pre-trained models, designed to be compatible with InfractiVision
    preprocessing pipeline.
    """
    
    def __init__(self, scale_factor=2, denoise_strength=5, sharpen_strength=0.5):
        """
        Initialize the super resolution processor with customizable parameters.
        
        Args:
            scale_factor (int): Upscaling factor (1-4)
            denoise_strength (int): Strength of denoising (0-10)
            sharpen_strength (float): Strength of sharpening (0-1)
        """
        self.scale_factor = max(1, min(4, scale_factor))  # Limit between 1-4
        self.denoise_strength = max(0, min(10, denoise_strength))
        self.sharpen_strength = max(0, min(1, sharpen_strength))
        
        # Configure processing options
        self.use_gpu = False
        try:
            # Check if OpenCV can use CUDA
            cv_build_info = cv2.getBuildInformation()
            self.use_gpu = "NVIDIA CUDA" in cv_build_info and "YES" in cv_build_info.split("NVIDIA CUDA")[1].split("\n")[0]
            
            if self.use_gpu:
                print("CUDA available: GPU acceleration enabled for super-resolution")
        except:
            print("GPU not available, using CPU for processing")
    
    def enhance(self, image):
        """
        Apply multi-step enhancement to the license plate image.
        
        Args:
            image: Input image (numpy array BGR format or path to image)
            
        Returns:
            Enhanced super-resolution image
        """
        # Handle string path input
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            
        if image is None or image.size == 0:
            print("Invalid image input for super-resolution")
            return None
        
        # Save original for final blending
        original = image.copy()
        
        # Create a processing pipeline
        processed = image.copy()
        
        # 1. Preprocessing (normalize brightness/contrast)
        processed = self._preprocess(processed)
        
        # 2. Denoise the image muy suavemente
        processed = self._denoise(processed)
        
        # 3. Mejorar contraste específicamente para OCR - muy conservador
        processed = self._enhance_contrast_for_ocr(processed)
        
        # 4. Mejorar específicamente caracteres de placas
        processed = self._enhance_plate_characters(processed)
        
        # 5. Upscale the image using advanced interpolation
        processed = self._upscale(processed)
        
        # 6. Apply minimal sharpening to enhance details
        processed = self._sharpen(processed)
        
        # 7. Apply post-processing enhancements - mucho más conservador
        processed = self._post_process(processed)
        
        # 8. NUEVO: Mezcla final con la imagen original para preservar textura natural
        if len(original.shape) == 2 and len(processed.shape) > 2:
            original_gray = original
        elif len(original.shape) > 2:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
        
        if len(processed.shape) == 2 and len(original.shape) > 2:
            # Redimensionar si es necesario
            if original_gray.shape != processed.shape:
                original_gray = cv2.resize(original_gray, (processed.shape[1], processed.shape[0]))
            
            # Mezcla final - mucho más conservadora para preservar detalles
            final_result = cv2.addWeighted(processed, 0.7, original_gray, 0.3, 0)
        else:
            final_result = processed
            
        return final_result
    
    def _preprocess(self, image):
        """Prepare the image with very gentle normalization and contrast enhancement"""
        # Convert to grayscale if color (for license plates)
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Guardar original para mezcla posterior
        original = gray.copy()
        
        # NUEVO: Verificar si la imagen necesita mejora de contraste
        min_val, max_val = np.min(gray), np.max(gray)
        contrast_range = max_val - min_val
        
        # Si el contraste ya es bueno, apenas tocamos la imagen
        if contrast_range > 100:
            # Apenas aplicar CLAHE muy suave
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            enhanced = clahe.apply(gray)
            # Mayor peso a la original
            result = cv2.addWeighted(original, 0.7, enhanced, 0.3, 0)
            return result
            
        # Si el contraste es bajo, aplicar mejoras más sustanciales pero cuidadosas
        # Aplicar CLAHE con parámetros muy conservadores
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(2, 2))
        enhanced = clahe.apply(gray)
        
        # Eliminar ruido con filtro muy suave
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Normalizar usando percentiles para evitar valores extremos
        p5 = np.percentile(enhanced, 5)
        p95 = np.percentile(enhanced, 95)
        
        # Asegurar que hay suficiente rango para la normalización
        if p95 - p5 > 10:
            # Normalización más conservadora - sólo estirar parcialmente el histograma
            enhanced = np.clip((enhanced - p5) * 220.0 / (p95 - p5) + 20, 0, 255).astype(np.uint8)
        
        # MUCHO más peso a la imagen original para preservar detalles
        result = cv2.addWeighted(original, 0.6, enhanced, 0.4, 0)
        
        return result
    
    def _enhance_contrast_for_ocr(self, image):
        """Mejorar muy suavemente el contraste para OCR en placas de bajo contraste"""
        # Calcular histograma
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Verificar si la imagen tiene bajo contraste
        hist_norm = hist / (image.shape[0] * image.shape[1])
        cumulative = np.cumsum(hist_norm)
        low_threshold = np.argmax(cumulative > 0.05)
        high_threshold = np.argmax(cumulative > 0.95)
        
        # Para placas, incluso con contraste normal aplicar mejora muy suave
        original = image.copy()
        
        # Aplicar ecualización muy limitada con parámetros muy conservadores
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(image)
        
        # Si el contraste es muy bajo, aplicar mejoras adicionales muy suaves
        if high_threshold - low_threshold < 80:
            # No usar algoritmos agresivos, solo mejora suave de contraste local
            kernel_size = min(image.shape[0] // 4, image.shape[1] // 4, 15)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            if kernel_size >= 3:
                # Mejora local muy suave
                enhanced = cv2.equalizeHist(image)
                # Darle mucho más peso a la original
                enhanced = cv2.addWeighted(original, 0.75, enhanced, 0.25, 0)
        
        # Para todos los casos, mezcla muy conservadora
        return cv2.addWeighted(original, 0.8, enhanced, 0.2, 0)
    
    def _enhance_plate_characters(self, image):
        """Mejora específica para resaltar los caracteres de placas vehiculares"""
        # Esta función se enfoca en mantener la legibilidad de los caracteres
        original = image.copy()
        
        # 1. Detección suave de caracteres
        # Usar bordes más suaves para detectar caracteres
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Umbral adaptativo para detectar caracteres - muy conservador
        _, binary = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Suavizar la máscara para evitar bordes duros
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        char_mask = cv2.GaussianBlur(dilated, (5, 5), 1.5)
        
        # Normalizar la máscara a [0,1]
        char_mask = char_mask / 255.0
        
        # 2. Realizar mejoras selectivas en las áreas de caracteres
        # Mejora del contraste local - muy conservadora
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(2, 2))
        enhanced = clahe.apply(image)
        
        # Aplicar la máscara de forma suave para mezclar
        result = np.zeros_like(image, dtype=np.float32)
        
        # Mezclar: más de la imagen mejorada donde hay caracteres, más de la original en el resto
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                weight = char_mask[i, j] * 0.3  # usar solo 30% del efecto máximo
                result[i, j] = image[i, j] * (1 - weight) + enhanced[i, j] * weight
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Mezcla final - mantener mucho más de la original
        return cv2.addWeighted(original, 0.8, result, 0.2, 0)
    
    def _denoise(self, image):
        """Remove noise very gently while preserving edges and details"""
        original = image.copy()
        
        # Si la fuerza de denoising es muy baja, apenas hacer algo
        if self.denoise_strength <= 3:
            # Solo aplicar un filtro bilateral muy suave
            denoised = cv2.bilateralFilter(image, 5, max(5, self.denoise_strength * 3), 
                                          max(5, self.denoise_strength * 3))
        else:
            # Para niveles medios, usar un enfoque híbrido pero conservador
            # Filtro bilateral para preservar bordes
            bilateral = cv2.bilateralFilter(image, 5, self.denoise_strength * 3, self.denoise_strength * 3)
            
            # Solo para ruido tipo "sal y pimienta", filtro de mediana muy suave
            if self.denoise_strength > 6:
                median = cv2.medianBlur(image, 3)
                # Mezcla con peso hacia bilateral que preserva bordes
                denoised = cv2.addWeighted(bilateral, 0.7, median, 0.3, 0)
            else:
                denoised = bilateral
            
            # Para fuerza muy alta, añadir non-local means pero con mucho peso en la original
            if self.denoise_strength > 8:
                h_value = min(10, self.denoise_strength)
                nlm = cv2.fastNlMeansDenoising(image, None, h_value, 5, 11)
                # Mezcla muy conservadora
                denoised = cv2.addWeighted(denoised, 0.7, nlm, 0.3, 0)
        
        # Mezcla final con la original para preservar detalles
        result = cv2.addWeighted(original, 0.4, denoised, 0.6, 0)
        return result
    
    def _upscale(self, image):
        """Upscale the image with extreme care for detail preservation"""
        h, w = image.shape[:2]
        target_h, target_w = h * self.scale_factor, w * self.scale_factor
        
        # Para placas, preservación de texto es crucial
        # Usar un único método de alta calidad: INTER_LANCZOS4 para texto nítido
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return upscaled
    
    def _sharpen(self, image):
        """Apply very minimal sharpening to enhance details without artifacts"""
        # Si no se requiere nitidez, devolver imagen original
        if self.sharpen_strength <= 0.1:
            return image
            
        original = image.copy()
        
        # Crear una versión suavizada para restar
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Crear máscara de bordes para aplicar nitidez selectiva
        mask = cv2.subtract(image, blurred)
        
        # Ajustar la fuerza pero mantenerla muy baja para evitar ruido
        # Reducir aún más el factor de intensidad
        alpha = self.sharpen_strength * 0.5  # Reducido de 0.8 a 0.5
        
        # Aplicar nitidez de forma muy selectiva
        sharpened = cv2.addWeighted(image, 1.0, mask, alpha, 0)
        
        # Mezclar con la original para suavizar el efecto
        result = cv2.addWeighted(original, 0.6, sharpened, 0.4, 0)
        
        return result
    
    def _post_process(self, image):
        """Apply extremely gentle final processing focused on detail preservation"""
        # Guardar original para mezcla final
        original = image.copy()
        
        # Muy suave eliminación de ruido residual
        denoised = cv2.bilateralFilter(image, 5, 10, 10)
        
        # COMPLETAMENTE NUEVO ENFOQUE: No usar binarización agresiva, sino mejorar contraste local
        # Esto preserva mucho mejor los detalles y las transiciones suaves
        
        # 1. Aplicar CLAHE muy suave para mejorar detalles
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(denoised)
        
        # 2. Detectar bordes para aplicar mejoras selectivas
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # 3. Crear una máscara de bordes diluida
        edge_mask = cv2.GaussianBlur(edges.astype(float) / 255.0, (5, 5), 1.0)
        
        # 4. Aplicar mejoras solo en áreas de texto
        # Donde hay bordes (texto), aumentar contraste ligeramente
        # Donde no hay bordes (fondo), suavizar para reducir ruido
        
        # Crear imagen mejorada solo para áreas de texto
        text_enhanced = enhanced.copy()
        
        # Aumentar contraste local en áreas de texto - muy suavemente
        for i in range(0, enhanced.shape[0]):
            for j in range(0, enhanced.shape[1]):
                if edge_mask[i, j] > 0.2:  # Si es área de texto
                    # Ajuste muy sutil de contraste para mayor legibilidad
                    pixel_val = enhanced[i, j]
                    if pixel_val < 128:
                        # Oscurecer píxeles oscuros muy sutilmente
                        text_enhanced[i, j] = max(0, pixel_val - 5)
                    else:
                        # Aclarar píxeles claros muy sutilmente
                        text_enhanced[i, j] = min(255, pixel_val + 5)
        
        # 5. Combinar con la imagen original - dando MUCHO peso a la original
        # Esto preserva los detalles y solo aplica pequeñas mejoras donde se necesita
        result = cv2.addWeighted(original, 0.85, text_enhanced, 0.15, 0)
        
        return result
    
    def process_plate(self, plate_img, save_path=None):
        """
        Process a license plate image to make it more suitable for OCR.
        
        Args:
            plate_img: Input plate image (numpy array or path)
            save_path: Optional path to save the enhanced image
            
        Returns:
            Enhanced plate image optimized for OCR
        """
        # Make sure we have a proper image
        if isinstance(plate_img, str):
            plate_img = cv2.imread(plate_img)
        
        if plate_img is None or plate_img.size == 0:
            return None
        
        # Guarda la imagen original para combinar al final
        original_img = plate_img.copy()
        
        # Standardize size for better results
        h, w = plate_img.shape[:2]
        aspect_ratio = w / h
        
        # Resize to standard height while maintaining aspect ratio
        standard_height = 100
        new_width = int(standard_height * aspect_ratio)
        resized = cv2.resize(plate_img, (new_width, standard_height))
        
        # Apply super-resolution enhancement
        enhanced = self.enhance(resized)
        
        # NUEVO: Final blending con la imagen original redimensionada
        # Esto ayuda a mantener la naturalidad de la imagen
        original_resized = cv2.resize(original_img, (new_width, standard_height))
        
        # CORREGIDO: Asegurarse de que las imágenes tengan el mismo tamaño y formato
        if len(enhanced.shape) == 2 and len(original_resized.shape) > 2:
            # Si enhanced es gris pero original es color, convertir original a gris
            original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
            
            # VERIFICAR QUE LOS TAMAÑOS COINCIDEN
            if enhanced.shape != original_gray.shape:
                original_gray = cv2.resize(original_gray, (enhanced.shape[1], enhanced.shape[0]))
                
            # Ahora el blending es seguro
            final_enhanced = cv2.addWeighted(enhanced, 0.85, original_gray, 0.15, 0)
        else:
            # Si ambas son del mismo tipo (ambas color o ambas gris)
            try:
                # VERIFICAR Y CORREGIR tamaños para asegurar compatibilidad
                if enhanced.shape[:2] != original_resized.shape[:2]:
                    original_resized = cv2.resize(original_resized, (enhanced.shape[1], enhanced.shape[0]))
                    
                # Si enhanced tiene diferentes canales que original, convertir
                if len(enhanced.shape) != len(original_resized.shape):
                    if len(enhanced.shape) > len(original_resized.shape):
                        # Si enhanced es color y original es gris, convertir original a color
                        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
                    else:
                        # Si enhanced es gris y original es color, convertir enhanced a color
                        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    
                final_enhanced = cv2.addWeighted(enhanced, 0.85, original_resized, 0.15, 0)
            except Exception as e:
                print(f"Error en blending final: {e}")
                final_enhanced = enhanced  # Si hay error, usar solo la mejorada
        
        # Save if requested
        if save_path is not None:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                cv2.imwrite(save_path, final_enhanced)
            except Exception as e:
                print(f"Error al guardar imagen: {e}")
        
        return final_enhanced


# Factory function to create and configure a SuperResolutionProcessor
# based on lighting conditions and customize it
def get_optimal_sr_processor(is_night=False):
    """
    Factory function that returns an optimally configured super resolution processor
    based on lighting conditions.
    
    Args:
        is_night: Boolean indicating if it's a night scene
        
    Returns:
        Configured SuperResolutionProcessor instance
    """
    if is_night:
        # Night scenes - valores EXTREMADAMENTE CONSERVADORES
        return SuperResolutionProcessor(
            scale_factor=2,
            denoise_strength=4,     # MUCHO más bajo - reducido de 6 a 4
            sharpen_strength=0.2    # MUCHO más bajo - reducido de 0.3 a 0.2
        )
    else:
        # Day scenes - valores EXTREMADAMENTE CONSERVADORES
        return SuperResolutionProcessor(
            scale_factor=2,
            denoise_strength=3,     # Reducido de 4 a 3
            sharpen_strength=0.25   # Reducido de 0.4 a 0.25
        )


def enhance_plate_image(plate_img, is_night=False, output_path=None):
    """
    Enhance a license plate image for better OCR recognition.
    Compatible with InfractiVision preprocessing workflow.
    
    Args:
        plate_img: Input plate image (numpy array or path to image)
        is_night: Boolean indicating if it's a night scene
        output_path: Optional path to save the enhanced image
        
    Returns:
        Enhanced license plate image
    """
    # Configure SR parameters based on lighting conditions
    sr_processor = get_optimal_sr_processor(is_night)
    
    # NUEVO: Pre-procesamiento específico para mejorar reconocimiento de caracteres
    if isinstance(plate_img, str):
        plate_img = cv2.imread(plate_img)
    
    if plate_img is not None and plate_img.size > 0:
        # Convertir a escala de grises si es color
        if len(plate_img.shape) > 2:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
            
        # Redimensionar para procesamiento consistente si es muy pequeña
        if gray.shape[0] < 50 or gray.shape[1] < 100:
            aspect = gray.shape[1] / gray.shape[0]
            new_height = 50
            new_width = int(new_height * aspect)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Mejorar la separación entre caracteres
        # Usar umbralización adaptativa para mejorar la definición de caracteres
        if is_night:
            # Para imágenes nocturnas, más agresivo con la ecualización
            gray = cv2.equalizeHist(gray)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 13, 4)
        else:
            # Para imágenes diurnas, más conservador
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
            gray = clahe.apply(gray)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
                                          
        # Operaciones morfológicas para mejorar la separación entre caracteres
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Volver a convertir la imagen procesada
        plate_img = binary
    
    # Process the plate con la imagen pre-procesada
    enhanced_plate = sr_processor.process_plate(plate_img, output_path)
    
    return enhanced_plate


# Utility function to apply super-resolution to all plates in a directory
def enhance_all_plates_in_directory(plates_dir, output_dir=None):
    """
    Batch process all plates in a directory to enhance them with super-resolution.
    
    Args:
        plates_dir: Directory containing plate images
        output_dir: Directory to save enhanced plates (defaults to a subdirectory of plates_dir)
        
    Returns:
        Number of plates processed
    """
    if output_dir is None:
        output_dir = os.path.join(plates_dir, "enhanced")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    plate_files = [f for f in os.listdir(plates_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Process each plate
    processed_count = 0
    for filename in plate_files:
        try:
            input_path = os.path.join(plates_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Skip already processed files
            if os.path.exists(output_path):
                continue
                
            # Process the plate
            plate_img = cv2.imread(input_path)
            if plate_img is not None:
                # NUEVO: Comprobar si la imagen necesita procesamiento
                # Si la imagen ya es clara, hacer procesamiento mínimo
                
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                is_night = gray.mean() < 100  # Umbral para escenas nocturnas
                
                # Métricas para evaluar la calidad de la imagen
                std_dev = np.std(gray)
                blur_index = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # ENFOQUE ULTRA CONSERVADOR: valores mínimos para todas las situaciones
                # Procesamiento extremadamente suave para preservar detalles
                
                if std_dev < 25:  # Contraste muy bajo - necesita ayuda pero con cuidado
                    if is_night:
                        sr_processor = SuperResolutionProcessor(
                            scale_factor=2,
                            denoise_strength=3,     # MUCHO más bajo
                            sharpen_strength=0.15   # EXTREMADAMENTE bajo
                        )
                    else:
                        sr_processor = SuperResolutionProcessor(
                            scale_factor=2,
                            denoise_strength=2,     # EXTREMADAMENTE bajo
                            sharpen_strength=0.2    # EXTREMADAMENTE bajo
                        )
                elif blur_index < 80:  # Imagen borrosa 
                    if is_night:
                        sr_processor = SuperResolutionProcessor(
                            scale_factor=2,
                            denoise_strength=2,     # MUCHO más bajo
                            sharpen_strength=0.3    # Apenas suficiente para compensar el desenfoque
                        )
                    else:
                        sr_processor = SuperResolutionProcessor(
                            scale_factor=2,
                            denoise_strength=2,     # MUCHO más bajo
                            sharpen_strength=0.35   # Apenas suficiente para compensar el desenfoque
                        )
                else:  # Imagen razonablemente buena - procesamiento mínimo
                    sr_processor = SuperResolutionProcessor(
                        scale_factor=2,
                        denoise_strength=2,         # Mínimo
                        sharpen_strength=0.2        # Mínimo
                    )
                
                # Enhance and save
                enhanced = sr_processor.process_plate(plate_img, output_path)
                processed_count += 1
                
                print(f"Processed {filename} - {'night' if is_night else 'day'} mode (std:{std_dev:.1f}, blur:{blur_index:.1f})")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return processed_count