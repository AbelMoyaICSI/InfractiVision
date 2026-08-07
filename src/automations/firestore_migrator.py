"""Migración por-video a Firestore en el formato solicitado.

Por cada video procesado (nombre_video) se escribe un documento en la
colección `migraciones/{video_name}` con el siguiente esquema:

    {
      "ti": number, "tr": number, "NID": number, "NIE": number,
      "video-name": string, "fecha": datetime,
      "settings": {"red","green","yellow","polygon"},
      "deteccion": [{"placa","timestamp","confianza","validate"}]
    }

Usa la llave admin `infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json`
(proyecto `infractivision-e8c03`).

Ejecutable:  python -m src.automations.firestore_migrator
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from src.path_helper import resource_path

ADMIN_KEY = resource_path("infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json")
PROJECT_ID = "infractivision-e8c03"

INFRA_FILE = resource_path("data/infracciones.json")
NIE_FILE = resource_path("data/nie_infracciones.json")
INDIC_FILE = resource_path("data/indicadores_rendimiento.json")
TIME_PRESETS_FILE = resource_path("config/time_presets.json")
POLYGON_FILE = resource_path("config/polygon_config.json")


def _load_json_array(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("infracciones", []) if isinstance(data.get("infracciones"), list) else []
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"⚠️ Error leyendo {path}: {e}")
    return []


def _load_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️ Error leyendo {path}: {e}")
        return {}


def _parse_date(value: str) -> datetime | None:
    """Interpreta 'DD/MM/YYYY' (con hora opcional 'DD/MM/YYYY HH:MM:SS')."""
    if not value:
        return None
    value = value.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_video_timestamp(ts: str) -> int:
    """Convierte 'MM:SS' o 'SS' (offset del video) a segundos."""
    if not ts:
        return 0
    parts = ts.replace(",", ":").split(":")
    try:
        segs = [int(p) for p in parts]
    except ValueError:
        return 0
    seconds = 0
    for s in segs:
        seconds = seconds * 60 + s
    return seconds


def _settings_for(video_name: str) -> dict:
    """Construye settings {red, green, yellow, polygon} para un video."""
    presets = _load_json_dict(TIME_PRESETS_FILE)
    polygons = _load_json_dict(POLYGON_FILE)

    cfg = presets.get(video_name, {})
    red = int(cfg.get("red", 0))
    green = int(cfg.get("green", 0))
    yellow = int(cfg.get("yellow", 0))
    polygon = polygons.get(video_name, [])

    return {
        "red": red,
        "green": green,
        "yellow": yellow,
        "polygon": polygon,
    }


def _deteccion_for(video_name: str, infractions: list) -> list:
    """Construye la lista 'deteccion' para un video."""
    detecciones = []
    for inf in infractions:
        placa = inf.get("placa", "")
        fecha = inf.get("fecha", "")
        hora = inf.get("hora", "")
        ts = inf.get("video_timestamp", "")

        if hora:
            try:
                base = _parse_date(fecha) or datetime.now()
                hora_dt = datetime.strptime(hora.strip(), "%H:%M:%S")
                timestamp = base.replace(hour=hora_dt.hour, minute=hora_dt.minute, second=hora_dt.second)
            except ValueError:
                timestamp = _parse_date(fecha)
        else:
            timestamp = _parse_date(fecha)

        if timestamp is None:
            timestamp = datetime.now()

        clasificacion = inf.get("clasificacion", "NID")
        detecciones.append({
            "placa": placa or "NIE",
            "timestamp": timestamp,
            "confianza": float(inf.get("confianza", 0.0) or 0.0),
            "validate": clasificacion == "NID",
        })
    return detecciones


def _build_video_document(video_name: str, infractions: list) -> dict | None:
    if not infractions:
        return None

    nid = sum(1 for i in infractions if i.get("clasificacion", "NID") == "NID")
    nie = sum(1 for i in infractions if i.get("clasificacion", "NID") != "NID")
    total = nid + nie
    ti = (nid / total * 100.0) if total else 0.0

    times = [float(i.get("tiempo_procesamiento", 0) or 0) for i in infractions if (i.get("tiempo_procesamiento") or 0) > 0]
    tr = (sum(times) / len(times) / 60.0) if times else 0.0  # seg -> min

    fecha = _parse_date(infractions[0].get("fecha", "")) or datetime.now()

    # Fallback TR/TI desde indicadores_rendimiento.json si existe coincidencia de video
    indicadores = _load_json_dict(INDIC_FILE)
    if (not times) or tr <= 0:
        tr = 0.0
        try:
            tr_con = indicadores.get("indicadores", {}).get("TR", {}).get("con_software", {})
            tr = float(tr_con.get("tiempo_promedio_minutos", 0) or 0)
        except Exception:
            tr = 0.0

    return {
        "ti": round(ti, 2),
        "tr": round(tr, 4),
        "NID": nid,
        "NIE": nie,
        "video-name": video_name,
        "fecha": fecha,
        "settings": _settings_for(video_name),
        "deteccion": _deteccion_for(video_name, infractions),
    }


def migrate_videos_to_firestore(verbose: bool = True) -> dict:
    """Agrupa las infracciones por video y sube un documento por cada video a Firestore.

    Returns:
        dict con "migrados", "errores" y "documentos".
    """
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not os.path.exists(ADMIN_KEY):
        raise FileNotFoundError(f"No se encontró la llave admin: {ADMIN_KEY}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(ADMIN_KEY)
        firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})

    db = firestore.client()

    infracciones = _load_json_array(INFRA_FILE)
    nie = _load_json_array(NIE_FILE)
    all_infractions = infracciones + nie

    by_video: dict[str, list] = {}
    for inf in all_infractions:
        video = inf.get("nombre_video", "desconocido.mp4")
        by_video.setdefault(video, []).append(inf)

    if verbose:
        print(f"📹 Videos a migrar: {len(by_video)}")

    migrados = 0
    errores = []
    documentos = {}
    for video_name, infractions in by_video.items():
        doc = _build_video_document(video_name, infractions)
        if doc is None:
            continue
        try:
            db.collection("migraciones").document(video_name).set(doc)
            documentos[video_name] = doc
            migrados += 1
            if verbose:
                print(f"  ✔ {video_name}: NID={doc['NID']} NIE={doc['NIE']} TI={doc['ti']}% TR={doc['tr']}min")
        except Exception as e:
            errores.append(f"{video_name}: {e}")
            if verbose:
                print(f"  ✗ {video_name}: {e}")

    if verbose:
        print(f"✅ {migrados} videos migrados a Firestore ({PROJECT_ID})")
    return {"migrados": migrados, "errores": errores, "documentos": documentos}


if __name__ == "__main__":
    try:
        migrate_videos_to_firestore()
    except Exception as e:
        print(f"❌ Error de migración: {e}")
        sys.exit(1)