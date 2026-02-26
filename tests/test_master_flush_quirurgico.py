import cv2
import os
import numpy as np
from src.core.ocr.recognizer import recognize_plate
from src.core.ocr.lprnet_engine import LPRNetPredictor
import matplotlib.pyplot as plt
import re

def run_surgical_fusion_experiment():
    """
    Experimento de Fusión Elite: Busca vehículos con múltiples tomas
    del MISMO track_id y aplica fusión de mediana alineada.
    """
    autos_dir = "data/output/autos"
    output_dir = "data/debug_experimento_quirurgico"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 Agrupando imágenes por Track ID...")
    files = [f for f in os.listdir(autos_dir) if f.endswith(('.png', '.jpg'))]
    
    # Agrupar por track_id
    track_groups = {}
    for fname in files:
        match = re.search(r'_t(\d+)_f', fname)
        if match:
            tid = match.group(1)
            track_groups.setdefault(tid, []).append(fname)
    
    # Filtrar solo tracks con 3+ imágenes para fusión válida
    valid_tracks = {k: v for k, v in track_groups.items() if len(v) >= 3}
    
    if not valid_tracks:
        print("❌ No hay suficientes secuencias para fusión (se necesitan 3+ tomas por vehículo).")
        return
    
    print(f"✅ Encontrados {len(valid_tracks)} vehículos con 3+ tomas.")
    predictor = LPRNetPredictor()
    
    for tid, filenames in list(valid_tracks.items())[:3]:  # Procesar solo los primeros 3
        print(f"\n📦 Procesando Track #{tid} ({len(filenames)} tomas)...")
        
        candidates = []
        for fname in filenames:
            img = cv2.imread(os.path.join(autos_dir, fname))
            if img is None: continue
            
            # 1. Detección YOLO para ROI inicial
            if hasattr(predictor, 'plate_detector') and predictor.plate_detector:
                detections = predictor.plate_detector.detect_plates(img, confidence=0.30)
                if detections:
                    px1, py1, px2, py2 = [int(v) for v in detections[0]]
                    yolo_crop = img[py1:py2, px1:px2].copy()
                    
                    # 2. Recorte Quirúrgico + Predicción Unificada
                    # Usamos return_processed=True para obtener el recorte flush que el modelo realmente ve
                    txt, conf, flush_crop = predictor.predict(yolo_crop, return_processed=True, autocrop=True)
                    
                    if txt and len(txt) >= 4:
                        candidates.append({'crop': flush_crop, 'conf': conf, 'txt': txt, 'file': fname})
        
        if len(candidates) < 2:
            print(f"  ⚠️ Evidencia insuficiente para Track #{tid}")
            continue
            
        # 🎯 FILTRO DE IDENTIDAD: Solo procesar capturas que coincidan con la placa predominante
        from collections import Counter
        all_texts = [c['txt'] for c in candidates]
        most_common_txt, count = Counter(all_texts).most_common(1)[0]
        
        identity_candidates = [c for c in candidates if c['txt'] == most_common_txt]
        
        if len(identity_candidates) < 2:
            print(f"  ⚠️ No hay suficientes capturas de la placa real ({most_common_txt}) para fusionar.")
            continue
            
        identity_candidates.sort(key=lambda x: x['conf'], reverse=True)
        top_candidates = identity_candidates[:3]
        
        print(f"  ✅ Usando {len(top_candidates)} tomas ELITE de la placa {most_common_txt}.")
        for i, c in enumerate(top_candidates):
            h, w = c['crop'].shape[:2]
            print(f"    {i+1}. {c['txt']} (conf: {c['conf']:.2f}) - Tamaño: {w}x{h}")
        
        # Los crops ya vienen con el Escáner de Energía aplicado
        surgical_crops = [c['crop'] for c in top_candidates]

        
        if len(surgical_crops) < 2:
            print(f"  ❌ Recortes quirúrgicos insuficientes para fusión")
            continue
        
        # --- PASO 2: NORMALIZACIÓN POST-QUIRÚRGICA ---
        # Normalizar a 300x80 DESPUÉS del recorte quirúrgico
        base_crops = [cv2.resize(sc, (300, 80), interpolation=cv2.INTER_LANCZOS4) for sc in surgical_crops]
        
        # --- PASO 3: ALINEACIÓN ECC CON VALIDACIÓN ---
        anchor = base_crops[0]
        height, width = anchor.shape[:2]
        gray_anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
        aligned_crops = [anchor]
        alignment_quality = [1.0]  # El ancla tiene calidad perfecta
        
        print(f"  🎯 Alineando frames al ancla (Frame 1)...")
        for i in range(1, len(base_crops)):
            current = base_crops[i]
            gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            try:
                # Ejecutar ECC y obtener correlación
                (cc, warp_matrix) = cv2.findTransformECC(
                    gray_anchor, gray_current, warp_matrix, 
                    cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
                )
                
                # VALIDACIÓN CRÍTICA: Solo aceptar si la correlación es alta
                if cc >= 0.75:
                    aligned = cv2.warpAffine(current, warp_matrix, (width, height), 
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    aligned_crops.append(aligned)
                    alignment_quality.append(cc)
                    print(f"    Frame {i+1}: ✅ Alineado (correlación: {cc:.3f})")
                else:
                    print(f"    Frame {i+1}: ⚠️ Descartado (correlación baja: {cc:.3f})")
            except Exception as e:
                print(f"    Frame {i+1}: ❌ Alineación falló - DESCARTADO")
        
        if len(aligned_crops) < 2:
            print(f"  ❌ No hay suficientes frames bien alineados para fusión")
            continue
        
        print(f"  ✅ Fusionando {len(aligned_crops)} frames con alineación verificada")
        
        # --- PASO 4: FUSIÓN MAESTRA (MEDIANA) ---
        master_raw = np.median(aligned_crops, axis=0).astype(np.uint8)
        
        # --- PASO 5: GENERACIÓN DE OVERLAY DE VALIDACIÓN ---
        # Crear overlay RGB para verificar alineación visualmente
        alignment_overlay = np.zeros_like(anchor)
        if len(aligned_crops) >= 3:
            alignment_overlay[:, :, 0] = cv2.cvtColor(aligned_crops[0], cv2.COLOR_BGR2GRAY)  # Rojo
            alignment_overlay[:, :, 1] = cv2.cvtColor(aligned_crops[1], cv2.COLOR_BGR2GRAY)  # Verde
            alignment_overlay[:, :, 2] = cv2.cvtColor(aligned_crops[2], cv2.COLOR_BGR2GRAY)  # Azul
        elif len(aligned_crops) == 2:
            alignment_overlay[:, :, 0] = cv2.cvtColor(aligned_crops[0], cv2.COLOR_BGR2GRAY)
            alignment_overlay[:, :, 1] = cv2.cvtColor(aligned_crops[1], cv2.COLOR_BGR2GRAY)
        
        # --- PASO 6: RE-RECORTE + STRETCHING ---
        master_refined = predictor.autocrop_plate(master_raw)
        master_94x24 = predictor.adapt_for_lprnet(master_refined, (94, 24))
        
        final_txt, final_conf = predictor.predict(master_refined, autocrop=False)
        
        # Calidad promedio de alineación
        avg_quality = np.mean(alignment_quality)
        
        # Gráfico de Resultados
        fig = plt.figure(figsize=(16, 11))
        best_plate = top_candidates[0]['txt']
        fig.suptitle(f"ELITE FUSION - Track #{tid}\nPlaca Detectada: {best_plate}\nResultado Fusión: {final_txt} (Conf: {final_conf:.2f})\nCalidad Alineación Promedio: {avg_quality:.3f}", 
                     fontsize=14, fontweight='bold')
        
        # Fila 1: Tomas originales
        for i, c in enumerate(top_candidates[:3]):
            plt.subplot(3, 3, i+1)
            plt.imshow(cv2.cvtColor(cv2.resize(c['crop'], (188, 48)), cv2.COLOR_BGR2RGB))
            plt.title(f"Toma {i+1}: {c['txt']} ({c['conf']:.2f})")
            plt.axis('off')
        
        # Fila 2: Overlay de alineación + Fusión
        plt.subplot(3, 3, 4)
        plt.imshow(cv2.resize(alignment_overlay, (188, 48)))
        plt.title(f"🔬 Overlay Alineación\n({len(aligned_crops)} frames)")
        plt.axis('off')
        
        plt.subplot(3, 3, 5)
        plt.imshow(cv2.cvtColor(cv2.resize(master_raw, (188, 48)), cv2.COLOR_BGR2RGB))
        plt.title("Fusión Bruta (Mediana)")
        plt.axis('off')
        
        plt.subplot(3, 3, 6)
        plt.imshow(cv2.cvtColor(cv2.resize(master_refined, (188, 48)), cv2.COLOR_BGR2RGB))
        plt.title("Recorte Refinado")
        plt.axis('off')
        
        # Fila 3: Resultado final
        plt.subplot(3, 3, 8)
        # Ampliar para visualizar mejor
        plt.imshow(cv2.cvtColor(cv2.resize(master_94x24, (188, 48), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
        plt.title(f"🏆 94x24 Final: {final_txt} ({final_conf:.2f})")
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f"elite_fusion_track_{tid}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  🏆 Guardado: {save_path}")

if __name__ == "__main__":
    run_surgical_fusion_experiment()
