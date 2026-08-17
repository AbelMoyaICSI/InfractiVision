"""Debug v3: Blackhat con kernel de cierre más controlado."""
import cv2, numpy as np, os, glob, sys
sys.path.insert(0, r"c:\Users\Abel\Desktop\InfractiVision")

plates_dir = r"c:\Users\Abel\Desktop\InfractiVision\tests\perspective_experiment\plates"
out = r"c:\Users\Abel\Desktop\InfractiVision\tests\perspective_experiment\results_v6\debug"
os.makedirs(out, exist_ok=True)

from ultralytics import YOLO
detector = YOLO(r"c:\Users\Abel\Desktop\InfractiVision\models\license_plate_detector.pt")

for img_path in sorted(glob.glob(os.path.join(plates_dir, "*.jpg"))):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    
    results = detector(img, verbose=False, conf=0.25)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        continue
    
    x1, y1, x2, y2 = boxes[0]
    m = int(max(x2-x1, y2-y1) * 0.08)
    roi = img[max(0,y1-m):min(img.shape[0],y2+m), max(0,x1-m):min(img.shape[1],x2+m)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = roi.shape[:2]
    
    print(f"\n{name} ROI={w}x{h}:")
    
    # Blackhat con kernel pequeño
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Cierre CONTROLADO: solo unir letras cercanas, NO todo
    # Kernel ancho para horizontal, pero estrecho verticalmente
    for close_w in [20, 30, 40]:
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k, iterations=1)
        
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) / (h*w) > 0.03]
        
        if not cnts:
            print(f"  close_w={close_w}: no cnts")
            continue
        
        # Buscar contorno con MEJOR aspect ratio de texto (2.0-8.0)
        best = None
        best_score = 0
        for c in cnts:
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            if rh < 5:
                continue
            asp = rw / rh
            area_r = cv2.contourArea(c) / (h * w)
            
            # Score: aspect cercano a 4-5 (texto de 6 chars) + area grande
            score = max(0, 1 - abs(asp - 4.5) / 5.0) * 0.5 + min(1, area_r * 5) * 0.5
            if score > best_score:
                best_score = score
                best = (c, rect, asp, area_r)
        
        if best:
            c, rect, asp, area_r = best
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            
            print(f"  close_w={close_w}: asp={asp:.2f} area={area_r:.3f} "
                  f"rect=({rw:.0f}x{rh:.0f}) angle={angle:.1f} n={len(cnts)}")
            
            vis = roi.copy()
            box = cv2.boxPoints(rect)
            cv2.drawContours(vis, [np.intp(box)], 0, (0, 255, 0), 2)
            
            # Expandir
            pad_w = rw * 1.15
            pad_h = rh * 2.5
            cy_adj = cy - rh * 0.20
            exp = ((cx, cy_adj), (pad_w, pad_h), angle)
            exp_box = cv2.boxPoints(exp)
            cv2.drawContours(vis, [np.intp(exp_box)], 0, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(out, f"{name}_v3_cw{close_w}.jpg"), 
                       np.hstack([
                           cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR),
                           vis
                       ]))

print("\nDone!")
