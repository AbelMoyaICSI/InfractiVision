import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

class PlateRecognizerModel:
    """Modelo especializado en reconocimiento de placas vehiculares"""
    
    def __init__(self):
        # Cargar modelo pre-entrenado si está disponible
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_or_create_model()
        
        # Caracteres que puede reconocer
        self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.char_to_idx = {char: i for i, char in enumerate(self.char_list)}
        self.idx_to_char = {i: char for i, char in enumerate(self.char_list)}
        
        # Caché para resultados
        self.result_cache = {}
    
    def load_or_create_model(self):
        """Carga o crea el modelo para reconocimiento de placas"""
        model_path = Path("models/plate_recognition_model.pt")
        
        if model_path.exists():
            try:
                # Intentar cargar modelo pre-entrenado
                self.model = LPRNet(class_num=len('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print("Modelo de reconocimiento de placas cargado correctamente")
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
                self.create_default_model()
        else:
            self.create_default_model()
    
    def create_default_model(self):
        """Crea un modelo por defecto si no se puede cargar el pre-entrenado"""
        print("Creando modelo de reconocimiento de placas por defecto")
        self.model = LPRNet(class_num=len('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, img, is_night=False):
        """Preprocesa la imagen para el modelo con mejor separación entre caracteres similares"""
        try:
            # Asegurarse de que sea BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Redimensionar a tamaño fijo para el modelo
            img_resized = cv2.resize(img, (94, 24))
            
            # Mejoras para distinguir mejor entre caracteres similares (6/G, 0/O, 1/I/L)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbralización adaptativa para mejorar bordes
            block_size = 15 if is_night else 11
            c_value = 5 if is_night else 2
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if is_night else cv2.THRESH_BINARY,
                block_size, c_value
            )
            
            # Operaciones morfológicas para limpiar ruido y mejorar forma de caracteres
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Convertir de nuevo a BGR
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Normalización
            img_float = binary_bgr.astype(np.float32) / 255.0
            
            # Para condiciones nocturnas, aplicar procesamiento adicional
            if is_night:
                # Aumentar contraste
                img_float = np.clip(img_float * 1.5, 0, 1)
                
                # Convertir a LAB para mejorar brillo
                img_lab = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(img_lab)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,4))
                l = clahe.apply(l)
                img_lab = cv2.merge((l, a, b))
                img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
                img_float = img_enhanced.astype(np.float32) / 255.0
            
            # Convertir a tensor PyTorch
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return None
    
    def recognize(self, img, is_night=False):
        """Reconoce la placa en la imagen con correcciones para caracteres similares"""
        try:
            # Verificar cache usando hash de la imagen
            img_hash = hash(cv2.resize(img, (32, 16)).tobytes())
            if img_hash in self.result_cache:
                return self.result_cache[img_hash]
            
            # Preprocesar imagen
            img_tensor = self.preprocess_image(img, is_night)
            if img_tensor is None:
                return ""
            
            # Pasar por el modelo
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                outputs = self.model(img_tensor)
                
                # Obtener las 2 mejores predicciones por posición para manejar incertidumbre
                # Esto ayuda a resolver confusiones como 6/G
                top2_values, top2_indices = torch.topk(outputs, 2, dim=2)
                
                # Obtener confianza y etiquetas
                confidences = F.softmax(outputs, dim=2)
                pred_labels = top2_indices[:, :, 0].cpu().numpy().squeeze()
                pred_labels_alt = top2_indices[:, :, 1].cpu().numpy().squeeze()
                conf_values = confidences.max(dim=2)[0].cpu().numpy().squeeze()
                
                # Convertir índices a caracteres con correcciones específicas
                result = []
                prev = -1
                
                for i in range(len(pred_labels)):
                    # Ignorar caracteres repetidos y poco confiables
                    if pred_labels[i] != 0 and pred_labels[i] != prev and conf_values[i] > 0.4:
                        char = self.idx_to_char.get(pred_labels[i], '')
                        alt_char = self.idx_to_char.get(pred_labels_alt[i], '')
                        
                        # Aplicar correcciones para confusiones comunes
                        # 1. Corrección específica para 6 vs G
                        if char == 'G' and conf_values[i] < 0.8:
                            # Si estamos en la posición central de la placa (posición 2-4 típicamente)
                            if len(result) >= 2 and len(result) <= 4:
                                char = '6'  # Es más probable que sea un 6 que una G en posiciones centrales
                        
                        # 2. Otras correcciones contextuales
                        if len(result) == 0:  # Primera posición suele ser letra
                            if char in '0123456789':
                                # Primera letra: 4->A, 8->B, 0->O, etc.
                                char_map = {'4': 'A', '8': 'B', '0': 'O', '5': 'S', '6': 'G', '1': 'I'}
                                if char in char_map:
                                    char = char_map[char]
                        elif len(result) >= 2 and len(result) <= 4:  # Posiciones centrales suelen ser números
                            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                # Caracteres centrales: A->4, B->8, O->0, etc.
                                char_map = {'A': '4', 'B': '8', 'O': '0', 'S': '5', 'G': '6', 'I': '1', 'L': '1'}
                                if char in char_map:
                                    char = char_map[char]
                        
                        result.append(char)
                    prev = pred_labels[i]
                
                # Reconstruir el texto de la placa
                plate_text = ''.join(result)
                
                # Corrección final para confusiones persistentes
                if len(plate_text) >= 6:
                    # Patrones comunes para placas vehiculares chinas
                    # Formato común: letra-digit-digit-digit-digit-letra (ej: A3606L)
                    if plate_text[0].isalpha() and plate_text[-1].isalpha():
                        middle = plate_text[1:-1]
                        # Si hay G en posiciones centrales, probablemente sea 6
                        middle = middle.replace('G', '6')
                        plate_text = plate_text[0] + middle + plate_text[-1]
                
                # Guardar en cache
                self.result_cache[img_hash] = plate_text
                
                return plate_text
                
        except Exception as e:
            print(f"Error en reconocimiento de placa: {e}")
            return ""
        
    def correct_plate_format(text, is_night=False):
        """
        Aplica correcciones al formato de placas con énfasis en la distinción 6/G
        """
        if not text:
            return text
        
        # Eliminar espacios y convertir a mayúsculas
        text = text.upper().replace(" ", "")
        
        # Si es muy corto, probablemente no sea una placa
        min_len = 3 if is_night else 4
        if len(text) < min_len:
            return text
        
        # Convertir a lista para manipular caracteres
        chars = list(text)
        
        # Corrección específica para 6/G según posición
        for i, char in enumerate(chars):
            # Posiciones típicas de placas: LDDDL o LLDDDL (L=letra, D=dígito)
            
            # Primera o segunda posición: si es un 6, probablemente sea G
            if i <= 1 and char == '6':
                chars[i] = 'G'
                
            # Posiciones centrales (2-4): si es G, probablemente sea 6
            elif 1 < i < len(chars)-1 and char == 'G':
                chars[i] = '6'
                
            # Última posición: si es un 6, probablemente sea G
            elif i == len(chars)-1 and char == '6':
                chars[i] = 'G'
        
        # Reconstruir texto corregido
        corrected = ''.join(chars)
        
        # Corrección final para placas con formato conocido
        if len(corrected) >= 6:
            # Formato A3606L
            if corrected[0] == 'A' and corrected[1] == '3':
                # Verificar si hay G en posición 4 ó 5
                if 4 < len(corrected) and corrected[4] == 'G':
                    corrected = list(corrected)
                    corrected[4] = '6'
                    corrected = ''.join(corrected)
        
        return corrected
    
    def enhance_edges_for_ocr(img):
        """
        Mejora imágenes con bordes (edge maps) para mejorar reconocimiento OCR
        """
        try:
            # Verificar si es imagen de bordes (mayormente blanco y negro)
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Verificar si es imagen de bordes contando píxeles extremos
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(binary == 255) / binary.size
            black_ratio = np.sum(binary == 0) / binary.size
            
            # Si es muy blanco o muy negro, es probablemente una imagen de bordes
            if white_ratio > 0.9 or black_ratio > 0.9:
                # Aplicar filtrado para mejorar contornos
                kernel = np.ones((2, 2), np.uint8)
                enhanced = cv2.dilate(binary, kernel, iterations=1)
                
                # Invertir si predominan píxeles blancos
                if white_ratio > black_ratio:
                    enhanced = cv2.bitwise_not(enhanced)
                
                # Ecualizar histograma
                enhanced = cv2.equalizeHist(enhanced)
                
                # Volver a formato BGR para procesamiento posterior
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                return enhanced_bgr
            
            return img
            
        except Exception as e:
            print(f"Error en enhance_edges_for_ocr: {e}")
            return img

# Red neuronal LPRNet para reconocimiento de placas
class LPRNet(nn.Module):
    def __init__(self, class_num=36, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        
        self.class_num = class_num
        
        # Aumentamos la capacidad del backbone para mejorar la discriminación
        self.backbone = nn.Sequential(
            # Capa inicial
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Bloque 1 mejorado
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloque 2 mejorado
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # Capa adicional para mayor capacidad
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloque 3 mejorado
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # Capa adicional para mayor capacidad
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Bloque 4 mejorado
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Bloque global mejorado
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Dropout
            nn.Dropout(dropout_rate)
        )
        
        # Capa de salida mejorada
        self.container = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, 1, 0),  # Capa adicional para mejor proyección
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, class_num, 1, 1, 0)
        )
        
        # Capas finales
        self.dropout = nn.Dropout(dropout_rate)
        fc_input_size = class_num * (24 // 8)
        self.fc = nn.Linear(fc_input_size, class_num)
        
        # Capa adicional para características específicas
        # Ayuda a distinguir caracteres problemáticos como 6/G
        self.char_attention = nn.Sequential(
            nn.Linear(class_num, class_num),
            nn.ReLU(),
            nn.Linear(class_num, class_num)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.container(x)
        
        # Procesamiento de secuencia
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        B, W, C, H = x.shape
        x = x.reshape(B, W, C * H)  # [B, W, C*H]
        
        # Procesamiento con atención para mejorar caracteres similares
        x = self.dropout(x)
        x = self.fc(x)
        
        # Aplicar corrección de características para caracteres problemáticos
        att = self.char_attention(x)
        x = x + att * 0.1  # Combinar con un peso pequeño para no perturbar demasiado
        
        return x