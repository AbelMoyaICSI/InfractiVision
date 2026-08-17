"""
========================================================================
VALIDACIÓN END-TO-END: Rectificación v6.3 → LPRNet OCR
========================================================================
Toma las 7 imágenes RECTIFICADAS (300x110px) y las pasa por LPRNet.
NO usa autocrop ni autocorrección — imagen directa al OCR.
========================================================================
"""
import cv2
import numpy as np
import os
import sys
import glob

project_root = r"c:\Users\Abel\Desktop\InfractiVision"
sys.path.insert(0, project_root)

from src.core.ocr.lprnet_engine import LPRNetPredictor

RESULTS_DIR = os.path.join(project_root, "tests", "perspective_experiment", "results_v6")
PLATES_DIR  = os.path.join(project_root, "tests", "perspective_experiment", "plates")

# Ground truth conocido
GROUND_TRUTH = {
    "perspective1": "AJC123",
    "perspective2": "7FHY345",
    "perspective3": "ABC123",
    "perspective4": "CA4G9B2",
    "perspective5": "ABC123",
    "perspective6": "ABC123",
    "perspective7": "A1B234",
}

print("\n" + "=" * 60)
print(" VALIDACIÓN END-TO-END: Auto Rectifier v6.3 → LPRNet")
print("=" * 60)

# Cargar LPRNet
ocr = LPRNetPredictor()

print("\n" + "-" * 60)
print(f"{'Imagen':<20} {'GT':^10} {'OCR':^12} {'Conf':^8} {'✓/✗':^6}")
print("-" * 60)

rectificadas = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_RECTIFIED.jpg")))

ok = 0
total = 0
resultados = []

for img_path in rectificadas:
    name = os.path.basename(img_path).replace("_RECTIFIED.jpg", "")
    img = cv2.imread(img_path)

    if img is None:
        print(f"{name:<20} {'?':^10} {'NO IMG':^12} {'---':^8} {'❌':^6}")
        continue

    # Pasar directo a OCR, sin autocrop ni transformaciones
    # La imagen ya está rectificada a 300x110px por la homografía
    texto, conf = ocr.predict(img, autocrop=False)

    # Normalizar para comparación
    gt_raw  = GROUND_TRUTH.get(name, "???")
    gt_clean = gt_raw.replace("-", "").upper()
    ocr_clean = texto.replace("-", "").upper()

    match = "✅" if ocr_clean == gt_clean else ("⚠️" if ocr_clean and ocr_clean in gt_clean else "❌")
    if ocr_clean == gt_clean:
        ok += 1
    total += 1

    print(f"{name:<20} {gt_raw:^10} {texto or '(vacío)':^12} {conf:^8.2f} {match:^6}")
    resultados.append((name, gt_raw, texto, conf, match))

print("-" * 60)
print(f"PRECISIÓN: {ok}/{total} = {ok/max(total,1)*100:.0f}%")
print("=" * 60)

# --- REPORTE VISUAL ---
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n = len(resultados)
fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
fig.suptitle(
    "Validación End-to-End: Auto Rectifier v6.3 → LPRNet OCR\n"
    f"Precisión: {ok}/{total} = {ok/max(total,1)*100:.0f}%",
    fontsize=13, fontweight='bold', color='white'
)
fig.patch.set_facecolor('#0d1117')
if n == 1:
    axes = [axes]

for i, (name, gt, texto, conf, match) in enumerate(resultados):
    img_path = os.path.join(RESULTS_DIR, f"{name}_RECTIFIED.jpg")
    img = cv2.imread(img_path)

    for ax in axes[i]:
        ax.set_facecolor('#161b22')

    if img is not None:
        axes[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        axes[i][0].imshow(np.zeros((110, 300, 3), dtype=np.uint8))
    axes[i][0].set_title(f"{name}", color='#aaa', fontsize=10)
    axes[i][0].axis('off')

    # Panel de texto
    axes[i][1].set_xlim(0, 1)
    axes[i][1].set_ylim(0, 1)
    axes[i][1].axis('off')
    color = '#00ff88' if match == '✅' else ('#ffaa00' if match == '⚠️' else '#ff4444')
    axes[i][1].text(0.5, 0.70, f"GT:  {gt}", ha='center', va='center',
                    fontsize=14, color='#aaaaaa', fontfamily='monospace')
    axes[i][1].text(0.5, 0.40, f"OCR: {texto or '(vacío)'}", ha='center', va='center',
                    fontsize=16, color=color, fontfamily='monospace', fontweight='bold')
    axes[i][1].text(0.5, 0.15, f"Confianza: {conf:.2f}  {match}", ha='center', va='center',
                    fontsize=12, color=color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
report_path = os.path.join(RESULTS_DIR, "REPORTE_OCR_v6_3.png")
plt.savefig(report_path, dpi=150, facecolor=fig.get_facecolor())
print(f"\nReporte visual: {report_path}")
plt.show()
