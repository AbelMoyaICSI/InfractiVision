"""
persistence.py — pure standalone JSON save functions, decoupled from Tk and self.player.

Extracted from preprocessing_dialog.py:4160-4465 and 3982-4004.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
from datetime import datetime
from typing import Any


def save_infractions_json(
    infractions: list[dict],
    output_dir: str = "data",
    filename: str = "infracciones.json",
    *,
    avenue_name: str = "Desconocida",
    time_slot: str = "No especificada",
    video_name: str = "desconocido.mp4",
    semaphore_config_id: str = "",
    sistema_version: str = "InfractiVision_v2.0",
) -> str:
    """Save NID (correctly identified) infractions as a stacked JSON."""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    # Load existing
    existing_infractions: list[dict] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "infracciones" in data:
                existing_infractions = data["infracciones"]
            elif isinstance(data, list):
                existing_infractions = data
        except Exception:
            existing_infractions = []

    now = datetime.now()
    nuevas: list[dict] = []

    for inf in infractions:
        plate = inf.get("plate", "")
        if not plate:
            continue

        # Basic plate validation
        clean_plate = plate.replace("-", "").replace(" ", "")
        if len(clean_plate) > 8:
            continue
        if any(bad in plate for bad in ("BOHID", "B OHID", "B-OHID")):
            continue

        # Timestamp calculation
        processing_time = inf.get(
            "time", inf.get("processing_time", inf.get("timestamp", 0))
        )
        if isinstance(processing_time, (int, float)) and processing_time > 0:
            total_seconds = int(processing_time)
            mins, secs = divmod(total_seconds, 60)
            timestamp = f"{mins:02d}:{secs:02d}"
        else:
            frame_number = inf.get("frame", 0)
            fps = inf.get("fps", 30)
            total_seconds = int(frame_number / fps) if frame_number > 0 else 0
            mins, secs = divmod(total_seconds, 60)
            timestamp = f"{mins:02d}:{secs:02d}"

        confidence = inf.get("confidence", 0.0)
        clamped_conf = round(max(0.0, min(1.0, confidence)), 3)

        entry: dict[str, Any] = {
            "placa": plate,
            "fecha": now.strftime("%d/%m/%Y"),
            "hora": now.strftime("%H:%M:%S"),
            "video_timestamp": timestamp,
            "tiempo_video": inf.get("video_duration", "N/A"),
            "ubicacion": avenue_name,
            "franja_horaria": time_slot,
            "tipo": "Semáforo en rojo",
            "estado": "Pendiente",
            "plate_path": inf.get(
                "plate_path",
                os.path.join(output_dir, "output", "placas", f"plate_{plate}.jpg"),
            ),
            "vehicle_path": inf.get(
                "vehicle_path",
                os.path.join(output_dir, "output", "autos", f"vehicle_{plate}.jpg"),
            ),
            "nombre_video": video_name,
            "config_semaforo": semaphore_config_id,
            "clasificacion": inf.get("clasificacion", "NID"),
            "confianza": clamped_conf,
            "tiempo_procesamiento": round(
                inf.get("tiempo_procesamiento", 0), 2
            ),
            "metadata_clasificacion": inf.get("metadata_clasificacion", {}),
            "sistema_version": inf.get("sistema_version", sistema_version),
            "hostname": socket.gethostname(),
            "username": getpass.getuser(),
        }
        if inf.get("modo_nocturno"):
            entry["modo_nocturno"] = True

        nuevas.append(entry)

    # Stack: newest first
    final = nuevas + existing_infractions
    output_data = {"infracciones": final}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return out_path


def save_nie_infractions_json(
    infractions: list[dict],
    output_dir: str = "data",
    filename: str = "nie_infracciones.json",
    *,
    avenue_name: str = "Desconocida",
    time_slot: str = "No especificada",
    video_name: str = "desconocido.mp4",
    semaphore_config_id: str = "",
    sistema_version: str = "InfractiVision_v2.0",
) -> str:
    """Save NIE (incorrectly identified) infractions to a separate stacked JSON."""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    existing: list[dict] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "infracciones" in data:
                existing = data["infracciones"]
            elif isinstance(data, list):
                existing = data
        except Exception:
            existing = []

    now = datetime.now()
    nuevas: list[dict] = []

    for inf in infractions:
        plate = inf.get("plate", "")
        if not plate:
            continue

        clean_plate = plate.replace("-", "").replace(" ", "")
        if len(clean_plate) > 8:
            continue
        if any(bad in plate for bad in ("BOHID", "B OHID", "B-OHID")):
            continue

        confidence = inf.get("confidence", 0.0)
        clamped_conf = round(max(0.0, min(1.0, confidence)), 3)

        entry: dict[str, Any] = {
            "placa": plate,
            "fecha": now.strftime("%d/%m/%Y"),
            "hora": now.strftime("%H:%M:%S"),
            "video_timestamp": inf.get("video_timestamp", "00:00"),
            "tiempo_video": inf.get("video_duration", "N/A"),
            "ubicacion": avenue_name,
            "franja_horaria": time_slot,
            "tipo": "Semáforo en rojo",
            "estado": "Rechazada",
            "plate_path": "",
            "vehicle_path": "",
            "nombre_video": video_name,
            "config_semaforo": semaphore_config_id,
            "clasificacion": "NIE",
            "confianza": clamped_conf,
            "tiempo_procesamiento": round(
                inf.get("tiempo_procesamiento", 0), 2
            ),
            "metadata_clasificacion": {
                "placa_final": plate,
                "confianza": clamped_conf,
                "calidad_deteccion": "baja",
                "justificacion": "No cumple criterios técnicos - Clasificada como NIE",
            },
            "sistema_version": inf.get("sistema_version", sistema_version),
            "hostname": socket.gethostname(),
            "username": getpass.getuser(),
        }
        nuevas.append(entry)

    final = nuevas + existing
    output_data = {"infracciones": final}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return out_path


def save_indicators_json(
    nid_count: int,
    nie_count: int,
    ti_percentage: float,
    tr_individual_minutes: list[float],
    output_dir: str = "data",
    filename: str = "indicadores_rendimiento.json",
    *,
    tr_overall_minutes: float | None = None,
    daily_average: float = 0.0,
    pnp_tr_minutes: float = 7.2,
    pnp_daily: float = 4.23,
) -> str:
    """Save performance indicators to data/indicadores_rendimiento.json."""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    now = datetime.now()

    if tr_overall_minutes is not None:
        sw_min = tr_overall_minutes
    elif tr_individual_minutes:
        sw_min = sum(tr_individual_minutes) / len(tr_individual_minutes)
    else:
        sw_min = 0.0

    tr_reduction = ((pnp_tr_minutes - sw_min) / pnp_tr_minutes * 100) if pnp_tr_minutes else 0
    tr_speedup = (pnp_tr_minutes / sw_min) if sw_min else 0

    output: dict[str, Any] = {
        "fecha_generacion": now.strftime("%d/%m/%Y %H:%M:%S"),
        "dias_analizados": 1,
        "indicadores": {
            "TI": {
                "descripcion": "Tasa de Infracciones detectadas (porcentaje de acierto)",
                "sin_software": {
                    "registros_campo_diarios": pnp_daily,
                    "fuente": "Registros PNP históricos",
                },
                "con_software": {
                    "detecciones_software_diarias": daily_average,
                    "dias_analizados": 1,
                },
                "porcentaje_acierto": round(ti_percentage, 1),
            },
            "TR": {
                "descripcion": "Tiempo de ejecución del sistema por NID detectado",
                "unidad": "minutos de ejecución por NID (min)",
                "sin_software": {
                    "tiempo_promedio_minutos": pnp_tr_minutes,
                    "fuente": "Estimación basada en registros históricos de campo",
                },
                "con_software": {
                    "tiempo_promedio_minutos": round(sw_min, 2),
                    "tiempos_individuales_ocr": [
                        round(t, 2) for t in tr_individual_minutes
                    ],
                    "muestras_analizadas": len(tr_individual_minutes),
                },
                "reduccion_tiempo_porcentual": round(tr_reduction, 1),
                "veces_mas_rapido": round(tr_speedup, 1),
            },
            "NID": {
                "descripcion": "Número de Infracciones Detectadas correctamente",
                "infracciones_hoy": nid_count,
                "promedio_diario": daily_average or nid_count,
                "periodo_analizado": 1,
                "total": nid_count,
            },
            "NIE": {
                "descripcion": "Número de Infracciones Incorrectamente registradas",
                "infracciones_incorrectas": nie_count,
                "total": nie_count,
            },
        },
        "resumen_global": {
            "ti_porcentaje_acierto": f"{round(ti_percentage, 1)}%",
            "tiempo_registro_minutos": f"{sw_min:.2f} min",
            "infracciones_detectadas_hoy": nid_count,
            "nid_total": nid_count,
            "nie_total": nie_count,
            "tir_total": nid_count + nie_count,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return out_path
