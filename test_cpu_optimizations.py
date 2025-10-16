"""
Script de prueba rápida para verificar optimizaciones CPU
"""
import sys
import time
import cv2
import numpy as np

print("=" * 60)
print("🔍 TEST DE OPTIMIZACIONES PARA CPU")
print("=" * 60)

# 1. VERIFICAR HARDWARE
print("\n1️⃣ Verificando hardware...")
import torch
print(f"   PyTorch versión: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("   ⚠️  Ejecutando en CPU (sin GPU)")
else:
    print(f"   ✅ GPU detectada: {torch.cuda.get_device_name(0)}")

# 2. VERIFICAR PaddleOCR
print("\n2️⃣ Verificando PaddleOCR optimizado...")
try:
    from src.core.ocr.recognizer import get_reader
    start = time.time()
    reader = get_reader()
    init_time = time.time() - start
    print(f"   ✅ PaddleOCR inicializado en {init_time:.2f}s")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. VERIFICAR YOLO
print("\n3️⃣ Verificando YOLO optimizado...")
try:
    from src.core.detection.vehicle_detector import VehicleDetector
    print("   Inicializando VehicleDetector...")
    detector = VehicleDetector(model_path="models/yolov8n.pt")
    print(f"   ✅ YOLO configurado:")
    print(f"      - Dispositivo: {detector.device}")
    print(f"      - Tamaño de imagen: {detector.imgsz}px")
    print(f"      - Umbral confianza: {detector.conf_threshold}")
    print(f"      - Score de hardware: {detector.hardware_info['score']}/100")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 4. TEST DE DETECCIÓN CON IMAGEN SINTÉTICA
print("\n4️⃣ Probando detección en imagen sintética...")
try:
    # Crear imagen de prueba (simulando un frame de video)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Añadir formas que simulen vehículos
    cv2.rectangle(test_img, (100, 200), (250, 350), (128, 128, 128), -1)
    cv2.rectangle(test_img, (400, 180), (550, 320), (100, 100, 100), -1)
    
    start = time.time()
    detections = detector.detect(test_img, conf=0.3, draw=False)
    detect_time = time.time() - start
    
    print(f"   ✅ Detección completada en {detect_time*1000:.1f}ms")
    print(f"      - FPS estimado: {1/detect_time:.1f}")
    print(f"      - Detecciones: {len(detections)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 5. RESUMEN Y RECOMENDACIONES
print("\n" + "=" * 60)
print("📊 RESUMEN DE CONFIGURACIÓN")
print("=" * 60)
if not torch.cuda.is_available():
    print("⚠️  Modo CPU activo:")
    print("   - PaddleOCR configurado con enable_mkldnn=True")
    print("   - YOLO usando imgsz=416px (optimizado para CPU)")
    print("   - Umbral de confianza reducido a 0.3")
    print("")
    print("💡 Recomendaciones:")
    print("   - El procesamiento será más lento que con GPU")
    print("   - Esperado: 5-10 FPS en CPU vs 30+ FPS en GPU")
    print("   - La precisión OCR puede ser menor")
    print("   - Ajusta los polígonos si no detecta infracciones")
else:
    print("✅ Modo GPU activo - Rendimiento óptimo")

print("\n🎯 Siguiente paso: Ejecutar main.py y probar con video real")
print("=" * 60)
