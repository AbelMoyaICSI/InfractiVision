"""Post-proceso de evidencias: candidatos -> cuadrante -> placa -> API.

Nuevo flujo (fase ACCION, fuera del live):
  1. El live (`OfficialVideoProcessor`) guarda los TOP-K crops de carro
     completo por infractor (mejor calidad `_quality`), cada uno con su
     `direction` (hacia donde se movia el carro: right/left/unknown).
  2. Aqui, ANTES de evaluar si hay placa, se recorta SOLO el cuadrante
     inferior correspondiente (derecho/izquierdo/completo) con el mismo
     algoritmo del live, y se localiza la placa con el YOLO 1x por candidato.
  3. De las imagenes CON placa se elige la mejor; si NINGUNA trae placa
     segun el modelo, fallback a la de mayor calidad (parametros).
  4. El recorte final queda en `evidence.crop_path` listo para la API de
     Plate Recognizer (`PlateReviewWindow`).

Sin estados pending: todo lo que entra aqui ya es infractor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import time

from src.core.logger import get_logger
from src.domain.entities.plate_evidence import PlateEvidence

log = get_logger("service.plate_review_preparer")

PLATE_MARGIN = 0.5
MIN_PLATE_W = 55
MIN_PLATE_H = 30
PLATE_CONF = 0.40

# Dedup de NIE por texto OCR (fallback): evita filas NIE duplicadas del mismo
# auto cuando YOLO-placas fallo pero la API si leyo el carro completo.
# Se mantiene como NIE: el texto solo es clave de agrupacion.
NIE_DEDUP_MIN_LEN = 5
NIE_DEDUP_MIN_CONF = 0.30


def _quadrant(vehicle: np.ndarray, direction: str = "unknown") -> tuple[np.ndarray, tuple[int, int]]:
    """Mismo algoritmo de cuadrante del live: mitad inferior der/izq/completa."""
    h, w = vehicle.shape[:2]
    if direction == "right":
        return vehicle[h // 2:, w // 2:], (w // 2, h // 2)
    if direction == "left":
        return vehicle[h // 2:, :w // 2], (0, h // 2)
    return vehicle[h // 2:, :], (0, h // 2)


def _get_plate_detector(plate_detector=None):
    """Lazy: el YOLO de placas solo se carga en post-proceso, nunca en live."""
    if plate_detector is not None and getattr(plate_detector, "model", None) is not None:
        return plate_detector
    from src.core.detection.plate_detector import PlateDetector
    det = PlateDetector()
    if getattr(det, "model", None) is None:
        return None
    return det


def _plate_crop_with_margin(vehicle: np.ndarray, local: tuple[int, int, int, int],
                            margin: float = PLATE_MARGIN) -> np.ndarray:
    pad_x = int((local[2] - local[0]) * margin)
    pad_y = int((local[3] - local[1]) * margin)
    return vehicle[
        max(0, local[1] - pad_y):min(vehicle.shape[0], local[3] + pad_y),
        max(0, local[0] - pad_x):min(vehicle.shape[1], local[2] + pad_x),
    ]


def _try_localize_in_image(vehicle: np.ndarray, direction: str,
                           detector, output_dir: Path | None,
                           stem: str) -> tuple[str, list[int], float, float] | None:
    """Cuadrante direccional + YOLO 1x. Retorna (plate_path, bbox, score, segundos) o None.

    El cuarto valor es el wall-clock de la inferencia YOLO de ESTE crop
    (t0 justo antes de `detect`, t1 al terminar), para el TR individual.
    """
    quadrant, (ox, oy) = _quadrant(vehicle, direction)
    _t0 = time.time()
    try:
        plates = detector.detect(quadrant, conf=PLATE_CONF, draw=False)
    except Exception:
        return None
    finally:
        _detect_seconds = time.time() - _t0
    scored: list[tuple[float, list[int]]] = []
    for plate in (plates or []):
        try:
            px1, py1, px2, py2 = map(int, plate[:4])
        except Exception:
            continue
        if px2 <= px1 or py2 <= py1:
            continue
        score = float(plate[4]) if len(plate) > 4 else 0.0
        scored.append((score, [px1 + ox, py1 + oy, px2 + ox, py2 + oy]))
    scored.sort(key=lambda s: s[0], reverse=True)
    for score, local in scored:
        crop = _plate_crop_with_margin(vehicle, tuple(local))
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        if w < MIN_PLATE_W or h < MIN_PLATE_H:
            continue
        if output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            plate_path = output_dir / f"{stem}_plate.jpg"
        else:
            plate_path = Path(f"{stem}_plate.jpg")
        try:
            cv2.imwrite(str(plate_path), crop)
        except Exception as exc:
            log.warning("No se pudo guardar placa %s: %s", plate_path, exc)
            continue
        return str(plate_path), [int(v) for v in local], score, _detect_seconds
    return None


def select_best_with_plate(evidence: PlateEvidence, plate_detector=None,
                           output_dir: str | Path | None = None) -> PlateEvidence:
    """Elige entre los candidatos: mejor CON placa, si no la de mayor calidad.

    Recorre los candidatos en calidad desc; en cada uno recorta el cuadrante
    inferior segun su `direction` y evalua placa. Gana la primera con placa
    viable (mayor calidad de carro). Sin hits => fallback al candidato de
    mayor calidad (carro completo a la API).
    """
    meta = evidence.metadata or {}
    cands = list(meta.get("candidate_crops") or [])
    if not cands:
        return localize_plate_for_evidence(evidence, plate_detector, output_dir)
    cands.sort(key=lambda c: float(c.get("quality", 0)), reverse=True)

    detector = _get_plate_detector(plate_detector)
    if detector is None:
        log.warning("Sin YOLO-placas en post-proceso: track=%s va entero a la API", evidence.track_id)
        _apply_fallback(evidence, cands[0], "sin modelo de placas")
        return evidence

    out_dir = Path(output_dir) if output_dir else None
    for cand in cands:
        path = str(cand.get("path", "") or "")
        if not path or not Path(path).exists():
            continue
        vehicle = cv2.imread(path)
        if vehicle is None or vehicle.size == 0:
            continue
        direction = str(cand.get("direction") or meta.get("direction") or "unknown")
        stem = Path(path).stem
        # Evita colisionar recortes entre candidatos del mismo track.
        stem = f"{stem.rsplit('_c', 1)[0]}_c{cand.get('frame', evidence.frame_index)}"
        hit = _try_localize_in_image(vehicle, direction, detector, out_dir or Path(path).parent, stem)
        if hit is None:
            continue
        plate_path, bbox, score, loc_seconds = hit
        evidence.crop_path = plate_path
        evidence.frame_index = int(cand.get("frame", evidence.frame_index))
        evidence.timestamp_seconds = float(cand.get("timestamp_seconds", evidence.timestamp_seconds))
        evidence.quality_score = float(cand.get("quality", evidence.quality_score))
        evidence.metadata["full_car"] = False
        evidence.metadata["fallback_by_quality"] = False
        evidence.metadata["dedup_eligible"] = False
        evidence.metadata["plate_bbox"] = bbox
        evidence.metadata["plate_score"] = round(score, 4)
        evidence.metadata["plate_localize_seconds"] = round(float(loc_seconds), 4)
        evidence.metadata["selected_candidate_frame"] = int(cand.get("frame", -1))
        evidence.metadata["selected_candidate_direction"] = direction
        evidence.metadata["n_candidates"] = len(cands)
        evidence.metadata.pop("plate_error", None)
        evidence.review_notes = "Placa localizada en post-proceso (cuadrante direccional + YOLO)"
        return evidence

    _apply_fallback(evidence, cands[0], "ningun candidato con placa: mejor por calidad")
    return evidence


def _apply_fallback(evidence: PlateEvidence, best_cand: dict, reason: str) -> None:
    """Fallback por parametros: el candidato de mayor calidad, carro completo.

    El texto OCR que luego devuelva la API sobre este crop SOLO sirve como
    clave de dedup de NIE: la ventana de revision lo conserva como NIE
    (ver `PlateReviewWindow._show_result`). `dedup_eligible=True` lo marca.
    """
    path = str(best_cand.get("path", "") or "")
    if path:
        evidence.crop_path = path
    evidence.frame_index = int(best_cand.get("frame", evidence.frame_index))
    evidence.timestamp_seconds = float(best_cand.get("timestamp_seconds", evidence.timestamp_seconds))
    evidence.quality_score = float(best_cand.get("quality", evidence.quality_score))
    evidence.metadata["full_car"] = True
    evidence.metadata["fallback_by_quality"] = True
    evidence.metadata["dedup_eligible"] = True
    evidence.metadata["selected_candidate_frame"] = int(best_cand.get("frame", -1))
    evidence.metadata["selected_candidate_direction"] = str(
        best_cand.get("direction") or evidence.metadata.get("direction") or "unknown")
    evidence.metadata["n_candidates"] = len(evidence.metadata.get("candidate_crops") or [])
    evidence.metadata["plate_error"] = reason
    if not evidence.review_notes:
        evidence.review_notes = "Vehículo completo (sin placa localizada en candidatos)"


def localize_plate_for_evidence(evidence: PlateEvidence, plate_detector=None,
                                output_dir: str | Path | None = None) -> PlateEvidence:
    """Localiza la placa en UNA imagen (compat: evidencias sin candidatos).

    Usa la `direction` del metadata si existe; si no, cuadrante completo.
    """
    crop_path = str(evidence.crop_path or "")
    if not crop_path or not Path(crop_path).exists():
        evidence.metadata["full_car"] = True
        evidence.metadata["fallback_by_quality"] = True
        evidence.metadata["dedup_eligible"] = True
        evidence.metadata["plate_error"] = "crop no disponible"
        return evidence
    vehicle = cv2.imread(crop_path)
    if vehicle is None or vehicle.size == 0:
        evidence.metadata["full_car"] = True
        evidence.metadata["fallback_by_quality"] = True
        evidence.metadata["dedup_eligible"] = True
        evidence.metadata["plate_error"] = "crop ilegible"
        return evidence

    detector = _get_plate_detector(plate_detector)
    if detector is None:
        log.warning("Sin YOLO-placas en post-proceso: track=%s va entero a la API", evidence.track_id)
        evidence.metadata["full_car"] = True
        evidence.metadata["fallback_by_quality"] = True
        evidence.metadata["dedup_eligible"] = True
        evidence.metadata["plate_error"] = "sin modelo de placas"
        return evidence

    direction = str((evidence.metadata or {}).get("direction") or "unknown")
    out_dir = Path(output_dir) if output_dir else Path(crop_path).parent
    stem = Path(crop_path).stem.replace("_best", "")
    hit = _try_localize_in_image(vehicle, direction, detector, out_dir, stem)
    if hit is None:
        evidence.metadata["full_car"] = True
        evidence.metadata["fallback_by_quality"] = True
        evidence.metadata["dedup_eligible"] = True
        evidence.metadata["plate_error"] = "placa no localizada: carro completo a la API"
        if not evidence.review_notes:
            evidence.review_notes = "Vehículo completo (placa no localizada en cuadrante)"
        return evidence
    plate_path, bbox, score, loc_seconds = hit
    evidence.crop_path = plate_path
    evidence.metadata["plate_localize_seconds"] = round(float(loc_seconds), 4)
    evidence.metadata["full_car"] = False
    evidence.metadata["fallback_by_quality"] = False
    evidence.metadata["dedup_eligible"] = False
    evidence.metadata["plate_bbox"] = bbox
    evidence.metadata["plate_score"] = round(score, 4)
    evidence.metadata["selected_candidate_direction"] = direction
    evidence.metadata.pop("plate_error", None)
    evidence.review_notes = "Placa localizada en post-proceso (cuadrante direccional + YOLO)"
    return evidence


def cleanup_rejected_candidates(evidences: list[PlateEvidence],
                                allowed_dir: str | Path | None = None) -> dict[str, int]:
    """Borra los candidatos NO elegidos tras la validacion (idempotente).

    Conserva por evidencia: `crop_path` final (placa o carro fallback).
    `best.jpg`, video y reporte NO se tocan (el reporte los referencia).
    Solo borra dentro de `allowed_dir` (por defecto, el padre del primer
    candidato) y tolera ausentes: seguro ante doble llamada
    (Exportar + Completado disparan la validacion dos veces).
    Retorna {"removed": n, "kept": m, "errors": k}.
    """
    removed = kept = errors = 0
    for evidence in (evidences or []):
        meta = evidence.metadata or {}
        cands = list(meta.get("candidate_crops") or [])
        if not cands:
            continue
        keep = {str(evidence.crop_path or "")}
        base = Path(allowed_dir) if allowed_dir else None
        if base is None:
            try:
                base = Path(str(cands[0].get("path", ""))).parent
            except Exception:
                base = None
        for cand in cands:
            path = str(cand.get("path", "") or "")
            if not path:
                continue
            if path in keep:
                kept += 1
                continue
            try:
                target = Path(path)
                if base is not None:
                    try:
                        target.resolve().relative_to(base.resolve())
                    except ValueError:
                        errors += 1
                        continue
                if not target.is_file():
                    continue
                target.unlink()
                removed += 1
            except Exception as exc:
                log.warning("No se pudo borrar candidato %s: %s", path, exc)
                errors += 1
    summary = {"removed": removed, "kept": kept, "errors": errors}
    log.info("Limpieza post-validacion: %s", summary)
    return summary


def prepare_evidences_for_review(evidences: list[PlateEvidence],
                                 output_dir: str | Path | None = None,
                                 plate_detector=None) -> list[PlateEvidence]:
    """Prepara todas las evidencias para `PlateReviewWindow`.

    Con candidatos: revisa placa en cada imagen y elige la mejor CON placa
    (fallback a mayor calidad). Sin candidatos: localiza en la unica imagen.
    """
    detector = _get_plate_detector(plate_detector)
    out: list[PlateEvidence] = []
    for evidence in (evidences or []):
        try:
            if (evidence.metadata or {}).get("candidate_crops"):
                out.append(select_best_with_plate(evidence, detector, output_dir))
            else:
                out.append(localize_plate_for_evidence(evidence, detector, output_dir))
        except Exception as exc:
            log.warning("Evidencia track=%s sin localizar: %s", getattr(evidence, "track_id", "?"), exc)
            evidence.metadata["full_car"] = True
            evidence.metadata["fallback_by_quality"] = True
            evidence.metadata["dedup_eligible"] = True
            out.append(evidence)
    return out


def _nie_dedup_key(plate_text: str) -> str:
    """Clave de dedup: placa normalizada (sin guiones/espacios, A-Z0-9)."""
    try:
        from src.infrastructure.ocr.cloud_plate_readers import normalize_plate
        return normalize_plate(plate_text or "")
    except Exception:
        import re as _re
        return _re.sub(r"[^A-Z0-9]", "", (plate_text or "").upper())


def deduplicate_nie_by_plate(
    evidences: list[PlateEvidence],
    min_len: int = NIE_DEDUP_MIN_LEN,
    min_conf: float = NIE_DEDUP_MIN_CONF,
) -> tuple[list[PlateEvidence], list[PlateEvidence]]:
    """Agrupa NIE duplicados por texto OCR reutilizado del fallback.

    Solo actua sobre evidencias NIE (no `validated` con texto): si dos o mas
    comparten la misma placa normalizada (leida de carro completo cuando YOLO
    fallo), conserva la de mayor `(ocr_confidence, quality_score)` y marca el
    resto con `metadata.duplicate_of_track`. Siempre quedan como NIE: nunca
    promueve a NID.

    Reusa el `plate_text` ya obtenido por `PlateReviewWindow` (cero llamadas
    extra a la API). Idempotente: una segunda llamada sobre la lista ya
    filtrada no elimina nada mas.

    Retorna `(kept, duplicates)`.
    """
    items = list(evidences or [])
    if not items:
        return [], []
    groups: dict[str, list[PlateEvidence]] = {}
    passthrough: list[PlateEvidence] = []
    for ev in items:
        # NID validados nunca se fusionan aqui.
        if bool(getattr(ev, "validated", False)) and (getattr(ev, "plate_text", "") or "").strip():
            passthrough.append(ev)
            continue
        key = _nie_dedup_key(getattr(ev, "plate_text", "") or "")
        conf = float(getattr(ev, "ocr_confidence", 0.0) or 0.0)
        if len(key) < min_len or conf < min_conf:
            passthrough.append(ev)
            continue
        groups.setdefault(key, []).append(ev)
    kept: list[PlateEvidence] = list(passthrough)
    duplicates: list[PlateEvidence] = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) <= 1:
            kept.extend(group)
            continue
        ranked = sorted(
            group,
            key=lambda e: (
                -float(getattr(e, "ocr_confidence", 0.0) or 0.0),
                -float(getattr(e, "quality_score", 0.0) or 0.0),
                int(getattr(e, "track_id", 0) or 0),
            ),
        )
        winner = ranked[0]
        try:
            winner.metadata["dedup_key"] = key
            winner.metadata["dedup_group_size"] = len(ranked)
        except Exception:
            pass
        kept.append(winner)
        for dup in ranked[1:]:
            try:
                dup.metadata["duplicate_of_track"] = int(getattr(winner, "track_id", 0) or 0)
                dup.metadata["dedup_key"] = key
                notes = getattr(dup, "review_notes", "") or ""
                suffix = f"Duplicado NIE de track {getattr(winner, 'track_id', '?')} por placa {key} (queda NIE)"
                dup.review_notes = f"{notes} | {suffix}" if notes else suffix
            except Exception:
                pass
            duplicates.append(dup)
    # Orden estable por track para no alterar la UI/reportes.
    kept.sort(key=lambda e: int(getattr(e, "track_id", 0) or 0))
    if duplicates:
        log.info(
            "Dedup NIE por OCR: %d evidencias -> %d unicas (%d duplicadas)",
            len(items), len(kept), len(duplicates),
        )
    return kept, duplicates
