"""
========================================================================
AUTO RECTIFIER v6.3 — CONTORNOS INTELIGENTES + RANSAC (Final)
========================================================================
Pipeline probado: 
  1) YOLO → Bounding Box
  2) ROI con margen → múltiples métodos de umbralización
  3) approxPolyDP → candidatos de 4 esquinas
  4) Scoring: convexidad + ángulos + aspect + centralidad + cobertura
  5) findHomography(RANSAC) → warpPerspective → 300x110px

Mejoras vs v6.2:
  - Score de CENTRALIDAD: los vértices deben cubrir el centro del ROI
  - Score de COBERTURA: mínimo 15% del área del ROI  
  - Validación más estricta de ángulos (55°-125°)
  - Si Contour falla, minAreaRect como expansión controlada
========================================================================
"""

import numpy as np
import cv2
import os
import sys
import glob
import time

project_root = r"c:\Users\Abel\Desktop\InfractiVision"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

OUTPUT_W = 300
OUTPUT_H = 110
STANDARD_WIDTH = 640


def order_points(pts):
    """Ordena 4 puntos: TL→TR→BR→BL."""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def resize_standard(img, target_w=STANDARD_WIDTH):
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale


def aplicar_homografia(imagen, pts_src, dst_w=OUTPUT_W, dst_h=OUTPUT_H):
    src = order_points(pts_src).astype(np.float32)
    dst = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32")
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        H = cv2.getPerspectiveTransform(src, dst)
    if H is None:
        return None
    return cv2.warpPerspective(imagen, H, (dst_w, dst_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def score_quad(pts, roi_h, roi_w):
    """
    Puntúa un cuadrilátero. Cuanto más alto, más probable que sea la placa.
    Considera: convexidad, ángulos, aspect ratio, regularidad, centralidad, cobertura.
    """
    pts = order_points(pts)
    (tl, tr, br, bl) = pts

    # === Convexidad obligatoria ===
    if not cv2.isContourConvex(np.int32(pts)):
        return 0

    # === Ángulos interiores (55°-125°) ===
    for j in range(4):
        v1 = pts[j] - pts[(j+1) % 4]
        v2 = pts[(j+2) % 4] - pts[(j+1) % 4]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ang = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if ang < 55 or ang > 125:
            return 0

    # === Dimensiones ===
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(tl - bl)
    h_right = np.linalg.norm(tr - br)
    w_avg = (w_top + w_bot) / 2
    h_avg = (h_left + h_right) / 2
    if h_avg < 3 or w_avg < 10:
        return 0

    # === Aspect ratio (1.5-5.5 para placas) ===
    aspect = w_avg / h_avg
    if aspect < 1.3 or aspect > 6.0:
        return 0
    # Óptimo: 2.0-3.5
    aspect_score = max(0, 1.0 - abs(aspect - 2.7) / 2.5)

    # === Regularidad de lados opuestos ===
    reg_w = min(w_top, w_bot) / max(w_top, w_bot)
    reg_h = min(h_left, h_right) / max(h_left, h_right)
    reg_score = (reg_w + reg_h) / 2

    # === Cobertura del ROI (15%-85%) ===
    area = cv2.contourArea(np.int32(pts))
    area_ratio = area / (roi_h * roi_w)
    if area_ratio < 0.10 or area_ratio > 0.90:
        return 0
    # Óptimo: 30-60%
    cover_score = max(0, 1.0 - abs(area_ratio - 0.45) / 0.45)

    # === Centralidad: el centro del quad debe estar cerca del centro del ROI ===
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    dx = abs(cx - roi_w / 2) / (roi_w / 2)
    dy = abs(cy - roi_h / 2) / (roi_h / 2)
    central_score = max(0, 1.0 - (dx + dy) / 2)

    return (aspect_score * 0.25 +
            reg_score * 0.20 +
            cover_score * 0.25 +
            central_score * 0.30)


def encontrar_esquinas(roi_bgr):
    """
    Busca las 4 esquinas de la placa usando múltiples métodos.
    Retorna: (pts_4x2_ordered, method, score) o (None, "NONE", 0)
    """
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 15:
        return None, "NONE", 0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)

    all_candidates = []

    # ===== 1. ADAPTIVE THRESHOLD =====
    for blur_k in [5, 3, 7]:
        blur = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
        for block in [11, 15, 21, 7]:
            for c_val in [2, 4]:
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block, c_val)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                quads = _buscar_quads(thresh, h, w)
                for s, p in quads:
                    all_candidates.append((s, p, "Adaptive"))

    # ===== 2. CANNY =====
    for lo, hi in [(30, 90), (50, 150), (80, 200)]:
        edges = cv2.Canny(enhanced, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        quads = _buscar_quads(edges, h, w)
        for s, p in quads:
            all_candidates.append((s, p, "Canny"))

    # ===== 3. BRILLO (LAB L-channel) =====
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    for thr in [130, 150, 170, 110]:
        _, bright = cv2.threshold(lab[:, :, 0], thr, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        quads = _buscar_quads(bright, h, w)
        for s, p in quads:
            all_candidates.append((s, p, "Bright"))

    # ===== 4. COLOR BLANCO (HSV) =====
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    for s_max, v_min in [(60, 150), (80, 120), (50, 170)]:
        mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        quads = _buscar_quads(mask, h, w)
        for s, p in quads:
            all_candidates.append((s, p, "White"))

    # ===== 5. MINAREA RECT del contorno más grande (fallback geométrico) =====
    # Si los contornos exactos fallan, usar el minAreaRect del contorno más grande
    for method_data in [("Adaptive", enhanced), ("Canny", enhanced)]:
        method_name, base = method_data
        if method_name == "Adaptive":
            blur = cv2.GaussianBlur(base, (5, 5), 0)
            mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:
            mask = cv2.Canny(base, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c_max = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c_max) / (h * w) > 0.10:
                rect = cv2.minAreaRect(c_max)
                box = cv2.boxPoints(rect)
                pts = order_points(box.astype(np.float32))
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                s = score_quad(pts, h, w)
                if s > 0:
                    # Penalizar un poco el minAreaRect vs approxPolyDP directo
                    all_candidates.append((s * 0.85, pts, f"MinArea"))

    if not all_candidates:
        return None, "NONE", 0

    # Elegir el MEJOR candidato
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_pts, best_method = all_candidates[0]
    return best_pts, best_method, best_score


def _buscar_quads(binary_mask, img_h, img_w):
    """Busca cuadriláteros válidos en una máscara binaria."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    results = []

    for c in contours:
        area = cv2.contourArea(c)
        if area / (img_h * img_w) < 0.08 or area / (img_h * img_w) > 0.95:
            continue

        peri = cv2.arcLength(c, True)
        for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                pts = order_points(approx.reshape(4, 2).astype("float32"))
                score = score_quad(pts, img_h, img_w)
                if score > 0:
                    results.append((score, pts))
                break

    return results


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

class PlateRectifier:
    def __init__(self, plate_model_path):
        from ultralytics import YOLO
        self.detector = YOLO(plate_model_path)
        print(f"[*] YOLO cargado: {os.path.basename(plate_model_path)}")

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, 0, "NONE", None

        img_std, scale = resize_standard(img, STANDARD_WIDTH)

        results = self.detector(img_std, verbose=False, conf=0.25)
        boxes = []
        for r in results:
            for box in r.boxes:
                xy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes.append((int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3]), conf))
        boxes.sort(key=lambda b: b[4], reverse=True)

        if not boxes:
            print("  [YOLO] No detectó placa")
            return img, None, 0, "NONE", None

        for x1, y1, x2, y2, det_conf in boxes[:3]:
            print(f"  [YOLO] Box=({x1},{y1})-({x2},{y2}) conf={det_conf:.3f}")

            bw, bh = x2 - x1, y2 - y1
            margin = int(max(bw, bh) * 0.10)
            rx1 = max(0, x1 - margin)
            ry1 = max(0, y1 - margin)
            rx2 = min(img_std.shape[1], x2 + margin)
            ry2 = min(img_std.shape[0], y2 + margin)
            roi = img_std[ry1:ry2, rx1:rx2]

            if roi.shape[0] < 15 or roi.shape[1] < 20:
                continue

            # Encontrar esquinas
            pts_local, method, score = encontrar_esquinas(roi)

            if pts_local is not None:
                pts_std = pts_local.copy()
                pts_std[:, 0] += rx1
                pts_std[:, 1] += ry1
                pts_orig = pts_std / scale

                warped = aplicar_homografia(img, pts_orig)
                if warped is not None:
                    print(f"  [OK] {method} score={score:.3f}")
                    return img, pts_orig, det_conf, f"YOLO+{method}", warped

            # Fallback: bbox
            pts_box = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype="float32") / scale
            warped = aplicar_homografia(img, pts_box)
            if warped is not None:
                print(f"  [FALLBACK] Bbox")
                return img, pts_box, det_conf * 0.3, "YOLO-Box", warped

        return img, None, 0, "NONE", None


# =====================================================================
# MAIN
# =====================================================================

def main():
    plate_model = os.path.join(project_root, "models", "license_plate_detector.pt")
    plates_dir = os.path.join(project_root, "tests", "perspective_experiment", "plates")
    output_dir = os.path.join(project_root, "tests", "perspective_experiment", "results_v6")
    os.makedirs(output_dir, exist_ok=True)

    rectifier = PlateRectifier(plate_model)
    images = sorted(glob.glob(os.path.join(plates_dir, "*.jpg")))
    results = []

    print("\n" + "=" * 65)
    print("AUTO RECTIFIER v6.3 — CONTORNOS INTELIGENTES + RANSAC")
    print("Múltiples métodos → scoring (centralidad+cobertura+aspect) → RANSAC")
    print("=" * 65)

    total_time = 0

    for img_path in images:
        name = os.path.basename(img_path)
        print(f"\n--- {name} ---")

        t0 = time.time()
        img, pts, conf, method, warped = rectifier.process(img_path)
        elapsed = time.time() - t0
        total_time += elapsed

        if pts is not None and warped is not None:
            marked = img.copy()
            rect_pts = order_points(pts)
            cv2.polylines(marked, [np.int32(rect_pts)], True, (0, 255, 0), 3)
            for i, p in enumerate(rect_pts):
                labels = ["TL", "TR", "BR", "BL"]
                cv2.circle(marked, tuple(np.int32(p)), 8, (0, 0, 255), -1)
                cv2.putText(marked, labels[i], tuple(np.int32(p) + [5, -10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            base = os.path.splitext(name)[0]
            cv2.imwrite(os.path.join(output_dir, f"{base}_RECTIFIED.jpg"), warped)
            cv2.imwrite(os.path.join(output_dir, f"{base}_MARKED.jpg"), marked)
            print(f"  ✅ [{method}] t={elapsed:.2f}s → {OUTPUT_W}x{OUTPUT_H}")
            results.append((name, img, warped, marked, conf, method, elapsed))
        else:
            print(f"  ❌ NO DETECTADA t={elapsed:.2f}s")
            results.append((name, img, None, None, 0, "NONE", elapsed))

    avg = total_time / max(len(images), 1)
    print(f"\n{'='*65}")
    print(f"TOTAL: {total_time:.1f}s | PROMEDIO: {avg:.2f}s/imagen")
    print(f"{'='*65}")

    # --- REPORTE ---
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(20, 5 * n))
    fig.suptitle(
        "AUTO RECTIFIER v6.3 — CONTORNOS INTELIGENTES + RANSAC\n"
        "Scoring: centralidad + cobertura + aspect + regularidad → RANSAC → 300x110",
        fontsize=13, fontweight='bold', color='white'
    )
    fig.patch.set_facecolor('#0d1117')
    if n == 1:
        axes = [axes]

    for i, (name, orig, warp, mark, conf, method, elapsed) in enumerate(results):
        for ax in axes[i]:
            ax.set_facecolor('#161b22')

        axes[i][0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"ORIGINAL: {name}", color='#e94560', fontweight='bold', fontsize=10)
        axes[i][0].axis('off')

        if mark is not None:
            axes[i][1].imshow(cv2.cvtColor(mark, cv2.COLOR_BGR2RGB))
            axes[i][1].set_title(f"[{method}] t={elapsed:.2f}s",
                                 color='#00d4ff', fontweight='bold', fontsize=10)
        else:
            axes[i][1].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            axes[i][1].set_title("NO DETECTADO", color='red', fontweight='bold')
        axes[i][1].axis('off')

        if warp is not None:
            axes[i][2].imshow(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
            axes[i][2].set_title(f"RECTIFICADA ✓ ({OUTPUT_W}x{OUTPUT_H})",
                                 color='#00ff88', fontweight='bold', fontsize=10)
        else:
            blank = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
            cv2.putText(blank, "FALLO", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            axes[i][2].imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
            axes[i][2].set_title("FALLO", color='red', fontweight='bold')
        axes[i][2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    report = os.path.join(output_dir, "REPORTE_v6_3.png")
    plt.savefig(report, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nReporte: {report}")
    plt.show()


if __name__ == "__main__":
    main()
