import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from src.path_helper import resource_path

# DICCIONARIO OFICIAL DEL ENTRENAMIENTO (Longitud 35)
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '-'
]

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

class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64), nn.ReLU(),           # 0,1,2
            nn.MaxPool3d((1, 3, 3), (1, 1, 1)),                          # 3
            small_basic_block(64, 128), nn.BatchNorm2d(128), nn.ReLU(),  # 4,5,6
            nn.MaxPool3d((1, 3, 3), (1, 2, 2)),                          # 7
            small_basic_block(128, 256), nn.BatchNorm2d(256), nn.ReLU(), # 8,9,10
            small_basic_block(256, 256), nn.BatchNorm2d(256), nn.ReLU(), # 11,12,13
            nn.MaxPool3d((1, 3, 3), (1, 1, 2)),                          # 14
            nn.Dropout(dropout_rate),                                    # 15
            nn.Conv2d(256, 256, (1, 4)), nn.BatchNorm2d(256), nn.ReLU(), # 16,17,18
            nn.Dropout(dropout_rate),                                    # 19
            nn.Conv2d(256, class_num, (13, 1)),                          # 20
        )
        self.container = nn.Sequential(
            nn.Conv2d(448 + class_num, class_num, kernel_size=(1, 1))    # container.0
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            if isinstance(layer, nn.MaxPool3d): 
                x = layer(x.unsqueeze(2)).squeeze(2)
            elif i == 20: 
                # PARCHE DE INTERPOLACIÓN (NECESARIO PARA TAMAÑOS PEQUEÑOS)
                if x.shape[2] < 13:
                    x = F.interpolate(x, size=(13, x.shape[3]), mode='bilinear', align_corners=True)
                x = layer(x)
            else: 
                x = layer(x)
            
            if i in [2, 6, 13]: 
                keep_features.append(x)

        target_size = x.size()[2:] 
        global_context = []
        for f in keep_features:
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
            f = f / torch.sqrt(torch.mean(torch.pow(f, 2)))
            global_context.append(f)

        x_norm = x / torch.sqrt(torch.mean(torch.pow(x, 2)))
        global_context.append(x_norm)

        x = torch.cat(global_context, 1) # Concatenación de features
        x = self.container(x)
        return torch.mean(x, dim=2)

class LPRNetPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_num = len(CHARS)
        self.model = LPRNet(class_num=self.class_num, dropout_rate=0)
        
        if model_path is None:
            # PRIORIDAD 1: Modelo V4 CORREGIDO (78.86% validación, 4X mejor en difíciles)
            v4_path = resource_path("models/LPRNet_V4_CORREGIDO.pth")
            # PRIORIDAD 2: Modelo V3 Especialista (Fallback si V4 no disponible)
            v3_path = resource_path("models/LPRNet_V3_ESPECIALISTA.pth")
            # PRIORIDAD 3: Modelo CONSENSO_V2 
            v2_path = resource_path("models/LPRNet_CONSENSO_V2.pth")
            # PRIORIDAD 4: Modelo Master Final como reserva
            master_path = resource_path("models/LPRNet_Peru_MASTER_FINAL.pth")
            
            if os.path.exists(v4_path):
                model_path = v4_path
                print("🚀 LPRNet Engine: Usando weights V4_CORREGIDO (78.86%, 4X mejor)")
            elif os.path.exists(v3_path):
                model_path = v3_path
                print("🏆 LPRNet Engine: Usando weights V3_ESPECIALISTA (80% precisión)")
            elif os.path.exists(v2_path):
                model_path = v2_path
                print("📦 LPRNet Engine: Usando weights CONSENSO_V2 (75% precisión)")
            elif os.path.exists(master_path):
                model_path = master_path
                print("📦 LPRNet Engine: Usando weights MASTER_FINAL (fallback)")
            else:
                model_path = None
                print("⚠️ LPRNet Engine: No se encontró ningún archivo de pesos.")
            
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ LPRNet Engine: Pesos cargados exitosamente desde {model_path}")
        else:
            print(f"⚠️ LPRNet Engine: No se encontró ningún archivo de pesos.")
            
        self.model.to(self.device)
        self.model.eval()

        # 🎯 DETECTOR DE PLACA INTEGRADO (YOLO) PARA GUIAR EL RECORTE
        try:
            from src.core.detection.plate_detector import PlateDetector
            plate_model_path = resource_path("models/license_plate_detector.pt")
            self.plate_detector = PlateDetector(plate_model_path)
            print("👁️ LPRNet Engine: Detector YOLO-Plate cargado como guía.")
        except Exception as e:
            self.plate_detector = None
            print(f"⚠️ LPRNet Engine: No se pudo cargar el detector de placas: {e}")

    def decode_greedy(self, logits, threshold=0.15):
        """
        Greedy Decode Nativo (PyTorch):
        Evita problemas de precisión y desbordamiento.
        """
        # logits: (1, 35, 20)
        probs = F.softmax(logits, dim=1).squeeze(0) # (35, 20)
        preds = torch.argmax(probs, dim=0) # (20,)
        confidences = torch.max(probs, dim=0)[0] # (20,)
        
        blank_idx = len(CHARS) - 1
        res = []
        pre_c = -1
        
        for i in range(preds.size(0)):
            c = preds[i].item()
            if c != blank_idx and c != pre_c:
                if confidences[i].item() > threshold:
                    res.append(CHARS[c])
            pre_c = c
            
        return "".join(res)
    
    # decode_fixed_length eliminado (no es seguro para producción por alucinaciones)
    
    def es_placa_valida(self, img_crop):
        """Filtros bypass para asegurar lectura en Lab Forense"""
        return True, "OK"

    def autocrop_plate(self, img):
        """
        RECORTE PRO V24 (Concepto Abel - Cero Distorsión):
        1. PRIORIDAD YOLO: Localiza la placa exacta.
        2. PADDING MAESTRO: 12% Vertical / 10% Horizontal para dar "aire" a las letras.
        3. FALLBACK ÉLITE: Escáner Sobel refinado para evitar recortes de ruedas/sombras.
        """
        if img is None or img.size == 0: 
            return img
        
        try:
            h, w = img.shape[:2]
            plate_roi = None
            
            # =========== MÉTODO 1: YOLO PLATE DETECTOR (PRIORIDAD) ===========
            if hasattr(self, 'plate_detector') and self.plate_detector is not None:
                try:
                    # YOLO 0.40 para noche
                    detections = self.plate_detector.detect_plates(img, confidence=0.40)
                    if detections:
                        best_det = max(detections, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
                        px1, py1, px2, py2 = best_det[:4]
                        
                        pw, ph = px2 - px1, py2 - py1
                        if ph > 0 and 1.8 <= pw/ph <= 6.5:
                            # 🛡️ BUMPER PROTOCOLO MAESTRO (Aire Forense 12%)
                            # Le damos 12% de aire lateral para que el último dígito 
                            # no choque con la pared y se lea completo.
                            pad_x = int(pw * 0.12) 
                            pad_y = int(ph * 0.05) 
                            
                            px1_s, py1_s = max(0, px1 - pad_x), max(0, py1 - pad_y)
                            px2_s, py2_s = min(w, px2 + pad_x), min(h, py2 + pad_y)
                            
                            plate_roi = img[int(py1_s):int(py2_s), int(px1_s):int(px2_s)].copy()
                except: pass
            
            # =========== MÉTODO 2: FALLBACK REFORZADO (SOBEL SIIV) ===========
            if plate_roi is None:
                # 🛠️ FILTRO ANTIRRUEDAS: Ignorar el 15% inferior del recorte del vehículo
                # Las placas raramente están tan pegadas al piso como las ruedas.
                margin_y = int(h * 0.15)
                clean_img = img[:(h - margin_y), :]
                ch, cw = clean_img.shape[:2]

                gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY) if len(clean_img.shape) == 3 else clean_img
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_abs = np.abs(sobel_x).astype(np.uint8)
                _, binary = cv2.threshold(sobel_abs, 45, 255, cv2.THRESH_BINARY) # Umbral un poco más alto
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
                morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                v_proj = np.sum(morphed, axis=0)
                cols = np.where(v_proj > np.max(v_proj) * 0.40)[0]
                if len(cols) > 0:
                    x1, x2 = cols[0], cols[-1]
                    h_proj = np.sum(morphed[:, x1:x2], axis=1)
                    rows = np.where(h_proj > np.max(h_proj) * 0.35)[0]
                    if len(rows) > 0:
                        y1, y2 = rows[0], rows[-1]
                        rw, rh = x2 - x1, y2 - y1
                        # Relación de aspecto estricta para evitar cuadrados (ruedas)
                        if rh > 0 and 2.0 < rw/rh < 6.5:
                            plate_roi = clean_img[y1:y2, x1:x2].copy()
            
            # --- NUEVO: ALINEACIÓN FORENSE V2 (Sin bordes negros destructivos) ---
            if plate_roi is not None and plate_roi.size > 0:
                # Solo alinear si no es una imagen ya rectificada (muy ancha)
                rh, rw = plate_roi.shape[:2]
                if rw/rh < 4.5:
                    plate_roi = self.alinear_placa(plate_roi)
                
            return plate_roi if plate_roi is not None and plate_roi.size > 0 else img
            
        except Exception:
            return img

    def alinear_placa(self, img_plate):
        """
        DESKEWING MAESTRO (Protocolo Abel):
        Detecta el ángulo de la placa y la endereza para que las letras sean verticales.
        """
        try:
            gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
            # Umbral adaptativo para detectar la forma de la placa
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(binary > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Ajuste de ángulo (OpenCV retorna entre 0 y 90)
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            
            # Limitar enderezado a +/- 15 grados para evitar distorsiones locas
            if abs(angle) > 15: return img_plate
            
            (h, w) = img_plate.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_plate, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except:
            return img_plate

    def adapt_for_lprnet(self, img, target_size=(94, 24)):
        """
        ADAPTACIÓN MAESTRO V4: Stretching con protección de bordes.
        """
        if img is None or img.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # 1. Recortar logo PERÚ (9% original, ultra-seguro)
        h, w = img.shape[:2]
        if w/h > 2.0:
            crop_x = int(w * 0.09)
            img = img[:, crop_x:].copy()
        
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    def predict(self, img_bgr, return_processed=False, autocrop=True):
        if img_bgr is None or img_bgr.size == 0:
            return ("", 0.0) if not return_processed else ("", 0.0, img_bgr)

        # Paso 1: Autocrop quirúrgico
        cropped = self.autocrop_plate(img_bgr) if autocrop else img_bgr
        
        # Paso 1.5: VALIDACIÓN PRE-OCR (Bypass para asegurar captura)
        # valido, razon = self.es_placa_valida(cropped)
        # if not valido:
        #    if return_processed: return "", 0.0, cropped
        #    return "", 0.0
        
        # Paso 2: Super-Resolución CONDICIONAL (V4 - Del otro chat)
        # Solo si la placa es muy pequeña (< 80px en dimensión mínima)
        h, w = cropped.shape[:2]
        min_dim = min(h, w)
        
        if min_dim < 80:
            try:
                from src.core.ocr.super_resolution import apply_super_resolution
                cropped = apply_super_resolution(cropped)
                # print(f"✨ SR aplicada (era {w}x{h})")
            except:
                pass  # Si SR no disponible, continuar sin ella
        
        # Paso 3: Adaptación para LPRNet (recorte anti-logo + stretching)
        img_data = self.adapt_for_lprnet(cropped, (94, 24))
        
        # Paso 4: Normalización EXACTA del entrenamiento
        img_prep = img_data.astype('float32')
        img_prep = (img_prep - 127.5) * 0.0078125
        img_prep = np.transpose(img_prep, (2, 0, 1))
        img_tensor = torch.from_numpy(img_prep).unsqueeze(0).to(self.device)
        
        # Paso 5: Inferencia
        with torch.no_grad():
            logits = self.model(img_tensor)
        
        # Paso 6: DECODIFICADOR V4 FINAL
        decoded = self.decode_greedy(logits, threshold=0.15)
        
        # Confianza promedio real nativa
        probs = F.softmax(logits, dim=1)
        conf = probs.max(1)[0].mean().item()
        
        if return_processed:
            return decoded, float(conf), cropped
        return decoded, float(conf)
