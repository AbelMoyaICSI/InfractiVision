"""Migración por-video a Firestore en el formato solicitado.

Por cada ejecución de validación se escribe un documento en la
colección `migraciones/{id}` (id único por ejecución; el mismo video puede
repetirse en varios documentos) con el siguiente esquema:

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
import uuid
from datetime import datetime

from src.path_helper import resource_path

ADMIN_KEY = resource_path("infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json")
PROJECT_ID = "infractivision-e8c03"


def _settings_for_from_db(video_name: str) -> dict:
    """Lee settings desde SQLite (video_configs), fallback vacío."""
    try:
        from src.infrastructure.database.app_repository import AppRepository
        repo = AppRepository()
        cfg = repo.get_video_config(video_name)
        if cfg:
            polygon = cfg.get("polygon") or []
            polygon_points = [{"x": float(p[0]), "y": float(p[1])} for p in polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
            return {"red": int(cfg.get("red") or 0), "green": int(cfg.get("green") or 0), "yellow": int(cfg.get("yellow") or 0), "polygon": polygon_points}
    except Exception as e:
        print(f"⚠️ Error leyendo settings desde DB: {e}")
    return {"red": 0, "green": 0, "yellow": 0, "polygon": []}


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


def _combine_fecha_hora(fecha: str, hora: str) -> datetime | None:
    """Combina fecha DD/MM/YYYY + hora HH:MM:SS en un datetime exacto.

    Retorna None si no se puede parsear.
    """
    fecha = (fecha or "").strip()
    hora = (hora or "").strip()
    if hora:
        try:
            base = _parse_date(fecha) or datetime.now()
            hora_dt = datetime.strptime(hora, "%H:%M:%S")
            return base.replace(hour=hora_dt.hour, minute=hora_dt.minute, second=hora_dt.second, microsecond=0)
        except ValueError:
            pass
    if fecha:
        parsed = _parse_date(fecha)
        if parsed is not None:
            return parsed
    return None


def _settings_for(video_name: str) -> dict:
    """Compat: delega a DB."""
    return _settings_for_from_db(video_name)


def _deteccion_for(video_name: str, infractions: list) -> list:
    """Construye la lista 'deteccion' para un video."""
    detecciones = []
    for inf in infractions:
        placa = inf.get("placa", "")
        fecha = inf.get("fecha", "")
        hora = inf.get("hora", "")
        ts = inf.get("video_timestamp", "")

        timestamp = _combine_fecha_hora(fecha, hora) or datetime.now()

        clasificacion = inf.get("clasificacion", "NID")
        detecciones.append({
            "placa": placa or "NIE",
            "timestamp": timestamp,
            "confianza": float(inf.get("confianza", 0.0) or 0.0),
            "validate": clasificacion == "NID",
        })
    return detecciones


def _build_video_document(video_name: str, infractions: list) -> dict | None:
    """Calcula TI/NID/NIE/TR SOLO sobre la sesión pasada (sin fallback JSON)."""
    if not infractions:
        return None
    nid = sum(1 for i in infractions if i.get("clasificacion", "NID") == "NID")
    nie = sum(1 for i in infractions if i.get("clasificacion", "NID") != "NID")
    total = nid + nie
    ti = (nid / total * 100.0) if total else 0.0
    times = [float(i.get("tiempo_procesamiento", 0) or 0) for i in infractions if (i.get("tiempo_procesamiento") or 0) > 0]
    tr = (sum(times) / len(times) / 60.0) if times else 0.0
    # TR fallback desde SQLite indicators si no hay tiempos (misma sesión)
    if tr <= 0:
        try:
            from src.infrastructure.database.app_repository import AppRepository
            indic = AppRepository().get_indicators()
            if indic:
                tr = float(indic.get("indicadores", {}).get("TR", {}).get("con_software", {}).get("tiempo_promedio_minutos", 0) or 0)
        except Exception:
            pass
    # Fecha y hora exacta de la prueba (session): combina fecha + hora de la primera infracción
    first = infractions[0] if infractions else {}
    fecha = _combine_fecha_hora(first.get("fecha", ""), first.get("hora", "")) or datetime.now()
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


def build_session_document(infractions_session: list) -> dict | None:
    """API para sesión actual: agrupa por video y valida coherencia."""
    if not infractions_session:
        return None
    video = infractions_session[0].get("nombre_video", "desconocido.mp4")
    return _build_video_document(video, infractions_session)


def migrate_single_video_to_firestore(video_name: str, infractions_session: list, verbose: bool = True) -> dict:
    """Migra SOLO la sesión actual de un video (SQLite es fuente, sin JSON).

    TI/TR/NID/NIE se calculan exclusivamente sobre `infractions_session`,
    garantizando coincidencia con `indicadores` de la misma sesión.
    """
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not infractions_session:
        return {"migrados": 0, "errores": ["sin infracciones en sesión"], "documentos": {}}
    if not os.path.exists(ADMIN_KEY):
        raise FileNotFoundError(f"No se encontró la llave admin: {ADMIN_KEY}")
    if not firebase_admin._apps:
        cred = credentials.Certificate(ADMIN_KEY)
        firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    db = firestore.client()
    doc = _build_video_document(video_name, infractions_session)
    if doc is None:
        return {"migrados": 0, "errores": ["documento vacío"], "documentos": {}}
    try:
        doc_id = str(uuid.uuid4())
        db.collection("migraciones").document(doc_id).set(doc)
        if verbose:
            print(f"  ✔ {video_name}: NID={doc['NID']} NIE={doc['NIE']} TI={doc['ti']}% TR={doc['tr']}min (sesión)")
        return {"migrados": 1, "errores": [], "documentos": {video_name: doc}}
    except Exception as e:
        if verbose:
            print(f"  ✗ {video_name}: {e}")
        return {"migrados": 0, "errores": [str(e)], "documentos": {}}


def migrate_videos_to_firestore(verbose: bool = True) -> dict:
    """CLI/backfill: migra todo lo almacenado en SQLite (agrupado por video)."""
    import firebase_admin
    from firebase_admin import credentials, firestore
    from src.infrastructure.database.app_repository import AppRepository

    if not os.path.exists(ADMIN_KEY):
        raise FileNotFoundError(f"No se encontró la llave admin: {ADMIN_KEY}")
    if not firebase_admin._apps:
        cred = credentials.Certificate(ADMIN_KEY)
        firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    db = firestore.client()
    repo = AppRepository()
    all_infractions = repo.list_infractions(limit=100000)
    by_video: dict[str, list] = {}
    for inf in all_infractions:
        video = inf.get("nombre_video", "desconocido.mp4")
        by_video.setdefault(video, []).append(inf)
    if verbose:
        print(f"📹 Videos a migrar (SQLite): {len(by_video)}")
    migrados = 0
    errores = []
    documentos = {}
    for video_name, infractions in by_video.items():
        doc = _build_video_document(video_name, infractions)
        if doc is None:
            continue
        try:
            doc_id = str(uuid.uuid4())
            db.collection("migraciones").document(doc_id).set(doc)
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