import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from src.core.ocr.recognizer import recognize_plate, validate_siiv_format

def ocr_test(img):
    # Usamos la función del sistema para ver qué lee la IA
    # Desactivamos autocrop interno para que use exactamente la imagen que le pasamos
    txt, conf = recognize_plate(img, autocrop=False)
    # Validar formato para ver si es NID o NIE
    is_valid, _, _, _ = validate_siiv_format(txt)
    status = "NID" if is_valid else "NIE"
    return f"{txt}\n({status} | {conf*100:.1f}%)"

def run_fine_sr_experiment():
    # 1. Rutas
    models_dir = r"c:\Users\Abel\Desktop\InfractiVision\models"
    sr_model_path = os.path.join(models_dir, "FSRCNN_x3.pb")
    placas_dir = r"c:\Users\Abel\Desktop\InfractiVision\data\output\placas"
    sample_plate = os.path.join(placas_dir, "plate_A02-784.jpg")
    
    if not os.path.exists(sample_plate):
        print("❌ Imagen no encontrada")
        return

    # 2. Preparar Imágenes
    img_orig = cv2.imread(sample_plate)
    h, w = img_orig.shape[:2]
    
    # Reducimos MUCHO para forzar el error (simulando 15-20 metros)
    # Una placa de 100px de ancho es crítica
    target_w = 90
    scale = target_w / w
    img_low = cv2.resize(img_orig, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 3. Super Resolución
    print("🚀 Aplicando FSRCNN x3 y OCR...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(sr_model_path)
    sr.setModel("fsrcnn", 3)
    img_sr = sr.upsample(img_low)
    
    # 4. Bicubic (Tradicional)
    img_bicubic = cv2.resize(img_low, (img_sr.shape[1], img_sr.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 5. TEST DE OCR (El momento de la verdad)
    print("🧠 Ejecutando inferencia LPRNet...")
    res_low = ocr_test(img_low)
    res_bicubic = ocr_test(img_bicubic)
    res_sr = ocr_test(img_sr)
    
    # 6. Gráfica Comparativa con Zoom
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("COMPARATIVA TÉCNICA: IMPACTO DE SR EN LA LECTURA OCR (ABEL V16)", fontsize=18, fontweight='bold', y=0.95)
    
    # Fila 1: Imágenes Completas + Resultado IA
    ax1 = plt.subplot(231)
    ax1.imshow(cv2.cvtColor(img_low, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"1. BAJA RESOLUCIÓN (90px)\nLectura IA: {res_low}", color='orange', fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(232)
    ax2.imshow(cv2.cvtColor(img_bicubic, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"2. REESCALADO TRADICIONAL\nLectura IA: {res_bicubic}", color='red', fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(233)
    ax3.imshow(cv2.cvtColor(img_sr, cv2.COLOR_BGR2RGB))
    ax3.set_title(f"3. SUPER RESOLUCIÓN IA\nLectura IA: {res_sr}", color='green', fontweight='bold')
    ax3.axis('off')
    
    # Fila 2: Zoom Quirúrgico (Letras críticas)
    # Tomamos un pequeño ROI de la primera letra 'A' o 'P'
    zoom_roi = (0, 0, 40, 40) # x, y, w, h aproximado para la primera letra tras el upscale
    
    # Para la imagen low, el ROI es menor
    z_low = img_low[0:15, 0:15]
    z_bicubic = img_bicubic[0:45, 0:45]
    z_sr = img_sr[0:45, 0:45]
    
    ax4 = plt.subplot(234)
    ax4.imshow(cv2.cvtColor(z_low, cv2.COLOR_BGR2RGB))
    ax4.set_title("Detalle: Pixelado", fontsize=10)
    ax4.axis('off')
    
    ax5 = plt.subplot(235)
    ax5.imshow(cv2.cvtColor(z_bicubic, cv2.COLOR_BGR2RGB))
    ax5.set_title("Detalle: Difuso (Blur)", fontsize=10)
    ax5.axis('off')
    
    ax6 = plt.subplot(236)
    ax6.imshow(cv2.cvtColor(z_sr, cv2.COLOR_BGR2RGB))
    ax6.set_title("Detalle: Bordes Definidos (SR)", fontsize=10)
    ax6.axis('off')
    
    # Guardar y Mostrar
    out = r"c:\Users\Abel\Desktop\InfractiVision\COMPARATIVA_OCR_SR_DETALLE.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"✅ Comparativa guardada en: {out}")
    plt.show()

if __name__ == "__main__":
    run_fine_sr_experiment()
