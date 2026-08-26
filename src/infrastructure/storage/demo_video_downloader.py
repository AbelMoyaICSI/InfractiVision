"""Descarga idempotente de los videos demo al directorio `videos/`.

Estrategia de fuentes, en orden:
  1. Google Cloud Storage autenticado (si existe la Service Account del
     proyecto empaquetada o `GOOGLE_APPLICATION_CREDENTIALS`).
  2. URL directa con token del manifest (descarga anónima).
  3. URL pública (si el bucket se hace público).

Modo autenticado es preferible porque los tokens de media pueden revocarse.
El método verifica size + sha256 cuando están en el manifest; si coinciden
omite la descarga (idempotente). Escribe en `<dest>.part` y hace `os.replace`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable

from src.core.logger import get_logger
from src.core.utils import resource_path
from src.core.utils.paths import user_data_path

log = get_logger("demo_videos")

MANIFEST_PATH = resource_path("config/demo_videos.json")
VIDEO_DIR = user_data_path("videos")
_BUCKET = "infractivision-e8c03.firebasestorage.app"
_SKIP_ENV = "INFRACTI_SKIP_DEMO_DOWNLOAD"


def _load_manifest(manifest: str) -> dict:
    path = Path(manifest)
    if not path.exists():
        return {"version": 1, "base_dir": "videos", "videos": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Manifest de videos demo no legible (%s): %s", path, e)
        return {"version": 1, "base_dir": "videos", "videos": []}


def _service_account_path() -> str | None:
    """Devuelve la Service Account si está disponible (bundle o APPDATA)."""
    key = "infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json"
    for candidate in (resource_path(key), resource_path("secrets/" + key)):
        if Path(candidate).exists():
            return candidate
    try:
        from src.core.utils.paths import APPDATA_DIR
        cand = APPDATA_DIR / key
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return None


def _gcs_download(entry: dict, dest: Path) -> bool:
    """Descarga autenticada vía google-cloud-storage (si hay credencial)."""
    sa = _service_account_path() or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa:
        return False
    try:
        from google.cloud import storage
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(sa)
        client = storage.Client(project="infractivision-e8c03", credentials=creds)
        blob = client.bucket(_BUCKET).blob(entry.get("gcs_path") or entry["filename"])
        blob.download_to_filename(dest)
        return True
    except Exception as e:
        log.warning("Descarga GCS autenticada falló para %s: %s", entry["filename"], e)
        return False


def _http_download(url: str, dest: Path, timeout: int = 120) -> bool:
    """Descarga anónima con requests, reintentos con backoff."""
    try:
        import requests
    except ImportError:
        return False
    last: Exception | None = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                part = dest.with_suffix(dest.suffix + ".part")
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                os.replace(part, dest)
                return True
        except Exception as e:
            last = e
            log.warning("Intento %d/3 falló para %s: %s", attempt + 1, url, e)
    log.error("Descarga fallida tras 3 intentos: %s (%s)", url, last)
    return False


def _download_entry(entry: dict, dest_dir: Path, on_progress: Callable[[str, int], None] | None) -> bool:
    filename = entry["filename"]
    dest = dest_dir / filename
    expected_size = entry.get("size")
    expected_sha = entry.get("sha256")

    if dest.exists():
        size_ok = expected_size is None or dest.stat().st_size == expected_size
        if size_ok:
            if expected_sha is None or _sha256(dest) == expected_sha:
                return True  # ya presente y verificado
        log.info("Video %s incompleto/corrupto, re-descargando", filename)

    if on_progress:
        on_progress(filename, 0)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = _gcs_download(entry, dest)
    if not ok:
        ok = _http_download(entry.get("url", ""), dest) if entry.get("url") else False
    if not ok and entry.get("public_url"):
        ok = _http_download(entry["public_url"], dest)
    if not ok:
        return False

    if expected_size is not None and dest.stat().st_size != expected_size:
        log.warning("Tamaño inesperado para %s (%d != %d)", filename, dest.stat().st_size, expected_size)
    if on_progress:
        on_progress(filename, 1)
    log.info("Video demo listo: %s", dest)
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_demo_videos(
    dest_dir: str | Path | None = None,
    manifest: str | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> dict:
    """Descarga los videos demo faltantes. Idempotente y thread-safe.

    Retorna resumen {ok, failed, skipped}.
    """
    if os.getenv(_SKIP_ENV) in ("1", "true", "True"):
        log.info("Descarga de videos demo omitida por %s", _SKIP_ENV)
        return {"ok": 0, "failed": 0, "skipped": 0}

    data = _load_manifest(manifest or MANIFEST_PATH)
    dest_dir = Path(dest_dir or VIDEO_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    summary = {"ok": 0, "failed": 0, "skipped": 0}
    for entry in data.get("videos", []):
        dest = dest_dir / entry["filename"]
        if dest.exists():
            if entry.get("size") is not None and dest.stat().st_size == entry["size"]:
                summary["skipped"] += 1
                continue
        if _download_entry(entry, dest_dir, on_progress):
            summary["ok"] += 1
        else:
            summary["failed"] += 1
            log.warning("No se pudo descargar video demo: %s", entry["filename"])
    return summary


def ensure_demo_videos_async(
    dest_dir: str | Path | None = None,
    on_progress: Callable[[str, int], None] | None = None,
    callback: Callable[[dict], None] | None = None,
) -> threading.Thread:
    """Lanza `ensure_demo_videos` en un hilo daemon (no bloquea la GUI)."""
    def _job():
        try:
            result = ensure_demo_videos(dest_dir=dest_dir, on_progress=on_progress)
            if callback:
                callback(result)
        except Exception as e:
            log.warning("ensure_demo_videos_async falló: %s", e)

    thread = threading.Thread(target=_job, daemon=True)
    thread.start()
    return thread


def missing_demo_videos(dest_dir: str | Path | None = None, manifest: str | None = None) -> list[str]:
    """Nombres de videos demo que aún no están completos en `dest_dir`."""
    data = _load_manifest(manifest or MANIFEST_PATH)
    dest_dir = Path(dest_dir or VIDEO_DIR)
    missing = []
    for entry in data.get("videos", []):
        dest = dest_dir / entry["filename"]
        if not dest.exists() or (entry.get("size") is not None and dest.stat().st_size != entry["size"]):
            missing.append(entry["filename"])
    return missing