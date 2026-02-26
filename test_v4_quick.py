import cv2
import sys
import os

# Añadir el path del proyecto
sys.path.append(os.getcwd())

from src.core.ocr.lprnet_engine import LPRNetPredictor

def test_v4_quick():
    print("🚀 TEST RÁPIDO DEL MODELO V4_CORREGIDO\n")
    
    # Cargar el motor
    lpr_engine = LPRNetPredictor()
    
    # Buscar una imagen de prueba
    test_imgs_dir = "tests/resultado_ocr_oro"
    
    if not os.path.exists(test_imgs_dir):
        print(f"❌ No se encontró la carpeta {test_imgs_dir}")
        return
    
    # Tomar la primera imagen que encuentre
    for fname in os.listdir(test_imgs_dir):
        if fname.endswith('.png'):
            img_path = os.path.join(test_imgs_dir, fname)
            img = cv2.imread(img_path)
            
            print(f"📸 Cargando: {fname}")
            print(f"   Dimensiones: {img.shape}")
            
            # Extraer solo la parte de la placa (mitad superior derecha)
            h, w = img.shape[:2]
            plate_crop = img[50:250, 600:900]
            
            print(f"\n🔍 Ejecutando OCR con V4...")
            text, conf = lpr_engine.predict(plate_crop, autocrop=False)
            
            print(f"\n📊 RESULTADO:")
            print(f"   Texto: {text}")
            print(f"   Confianza: {conf:.3f}")
            print(f"   Largo: {len(text)} caracteres")
            
            if len(text) == 6:
                print(f"   ✅ Longitud correcta!")
            else:
                print(f"   ⚠️ Longitud inesperada (esperado 6)")
            
            break
    
    print("\n✅ Test completado")

if __name__ == "__main__":
    test_v4_quick()
