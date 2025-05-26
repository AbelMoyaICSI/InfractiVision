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
                self.model = LPRNet(
                    class_num=len(self.char_list), 
                    lpr_max_len=8, 
                    phase='test',
                    dropout_rate=0.0  # En inferencia no usamos dropout
                )
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
        self.model = LPRNet(
            class_num=len(self.char_list),
            lpr_max_len=8, 
            phase='test',
            dropout_rate=0.0
        )
        self.model.to(self.device)
        self.model.eval()
        print("Modelo LPRNet creado en modo evaluación")
    
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

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

# Red neuronal LPRNet para reconocimiento de placas
class LPRNet(nn.Module):
    def __init__(self, class_num, lpr_max_len=8, phase='test', dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=128, ch_out=256),   # 8 - Corregido: 128 en lugar de 64
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),  # 16 - Corregido: 256 en lugar de 64
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits