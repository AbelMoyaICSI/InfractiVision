"""Descarga selectiva de modelos AI a `models/` (online ligero).

Los specs PyInstaller ONLINE ya NO bundlean los .pt/.pth/.pb.
Este modulo descarga solo lo necesario, idempotente, con verificacion
sha256+size y reintentos. Inspirado en demo_video_downloader.py pero
para modelos (mucho mas pequenos y criticos: sin ellos no hay inferencia).

Fuentes en orden:
  1. GCS autenticado (Service Account bundleada o GOOGLE_APPLICATION_CREDENTIALS)
  2. URL directa (field "url" en manifest, si existe)
  3. URL publica (field "public_url")

Los fallbacks de LPRNet (V3, CONSENSO, MASTER) solo se descargan si el
V4_CORREGIDO no esta disponible localmente — asi el online no paga 3x2MB
innecesario. El FSRCNN es opcional (super-resolucion).
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

log = get_logger("models")

MANIFEST_PATH = resource_path("config/models_manifest.json")
MODELS_DIR_FROZEN = user_data_path("models")
MODELS_DIR_DEV = Path(resource_path("models")).parent / "models"  # fallback dev
_BUCKET = "infractivision-e8c03.firebasestorage.app"
_SKIP_ENV = "INFRACTI_SKIP_MODEL_DOWNLOAD"


def _resolve_models_dir(dest_dir: str | Path | None) -> Path:
    if dest_dir is not None:
        return Path(dest_dir)
    # En frozen (onefile), resource_path("models") apunta a _MEIPASS/models (solo lectura, efimero).
    # Los modelos descargados deben ir a APPDATA/InfractiVision/models (persistente, escribible).
    # En dev, usamos models/ del proyecto.
    import sys

    if hasattr(sys, "_MEIPASS"):
        p = Path(MODELS_DIR_FROZEN)
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Dev: models/ junto al proyecto (si existe) o APPDATA/models
    dev = Path("models")
    if dev.exists():
        return dev.resolve()
    p = Path(MODELS_DIR_FROZEN)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_manifest(manifest: str | None) -> dict:
    path = Path(manifest or MANIFEST_PATH)
    if not path.exists():
        return {"version": 1, "base_dir": "models", "models": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Manifest de modelos no legible (%s): %s", path, e)
        return {"version": 1, "base_dir": "models", "models": []}


def _service_account_path() -> str | None:
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
    sa = _service_account_path() or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa:
        return False
    try:
        from google.cloud import storage
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(sa)
        client = storage.Client(project="infractivision-e8c03", credentials=creds)
        blob = client.bucket(_BUCKET).blob(entry.get("gcs_path") or entry["filename"])
        # descarga a .part para atomicidad
        part = dest.with_suffix(dest.suffix + ".part")
        blob.download_to_filename(str(part))
        os.replace(part, dest)
        return True
    except Exception as e:
        log.warning("GCS modelos fallo para %s: %s", entry["filename"], e)
        return False


def _http_download(url: str, dest: Path, timeout: int = 180) -> bool:
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
            log.warning("Intento %d/3 modelo %s fallo: %s", attempt + 1, dest.name, e)
    log.error("Descarga modelo fallida %s (%s)", dest.name, last)
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(dest: Path, expected_size: int | None, expected_sha: str | None) -> bool:
    if not dest.exists():
        return False
    if expected_size is not None and dest.stat().st_size != expected_size:
        return False
    if expected_sha is not None:
        try:
            return _sha256(dest) == expected_sha
        except Exception:
            return False
    return True


def _download_entry(entry: dict, dest_dir: Path, on_progress: Callable[[str, int], None] | None) -> bool:
    filename = entry["filename"]
    dest = dest_dir / filename
    expected_size = entry.get("size")
    expected_sha = entry.get("sha256")

    if _is_valid(dest, expected_size, expected_sha):
        return True

    if dest.exists():
        log.info("Modelo %s corrupto/incompleto, re-descargando", filename)

    if on_progress:
        on_progress(filename, 0)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = _gcs_download(entry, dest)
    if not ok and entry.get("url"):
        ok = _http_download(entry["url"], dest)
    if not ok and entry.get("public_url"):
        ok = _http_download(entry["public_url"], dest)

    if not ok:
        return False

    # Verificacion post-descarga (no bloqueante si falla, solo warning)
    if expected_size is not None and dest.stat().st_size != expected_size:
        log.warning("Size inesperado %s %d != %d", filename, dest.stat().st_size, expected_size)
    if expected_sha is not None:
        sha = _sha256(dest)
        if sha != expected_sha:
            log.warning("SHA mismatch %s %s != %s", filename, sha, expected_sha)

    if on_progress:
        on_progress(filename, 1)
    log.info("Modelo listo: %s", dest)
    return True


def _filter_required(entries: list[dict], dest_dir: Path) -> list[dict]:
    """Solo required + fallback si el primario requerido no existe local."""
    required = [e for e in entries if e.get("required")]
    optional = [e for e in entries if not e.get("required")]

    # Si el primario requerido ya esta valido, no descargar sus fallbacks
    result = list(required)
    primary_names = {e["filename"] for e in required}
    for opt in optional:
        fallback_for = opt.get("fallback_for")
        # Si es fallback y el primario ya existe valido, skip
        if fallback_for and fallback_for in primary_names:
            primary_path = dest_dir / fallback_for
            # Si el primario existe (o se descargara), no necesitamos fallback ahora
            # Solo descargamos fallback si el primario falla despues de intentar
            continue
        # FSRCNN y otros opcionales sin fallback_for: no descargar en modo selectivo
        # (se pueden descargar bajo demanda si la app los necesita)
        continue
    return result


def ensure_models(
    dest_dir: str | Path | None = None,
    manifest: str | None = None,
    on_progress: Callable[[str, int], None] | None = None,
    include_optional: bool = False,
) -> dict:
    """Descarga modelos faltantes. Selectivo por defecto (solo required).

    Retorna {ok, failed, skipped, dest_dir}.
    """
    if os.getenv(_SKIP_ENV) in ("1", "true", "True"):
        log.info("Descarga de modelos omitida por %s", _SKIP_ENV)
        return {"ok": 0, "failed": 0, "skipped": 0, "dest_dir": str(dest_dir or "")}

    data = _load_manifest(manifest)
    dest_dir_p = _resolve_models_dir(dest_dir)
    dest_dir_p.mkdir(parents=True, exist_ok=True)

    entries = data.get("models", [])
    if not include_optional:
        entries = _filter_required(entries, dest_dir_p)

    summary = {"ok": 0, "failed": 0, "skipped": 0, "dest_dir": str(dest_dir_p)}
    for entry in entries:
        dest = dest_dir_p / entry["filename"]
        if _is_valid(dest, entry.get("size"), entry.get("sha256")):
            summary["skipped"] += 1
            continue
        # Fallback handling: if primary failed, try its fallbacks
        if _download_entry(entry, dest_dir_p, on_progress):
            summary["ok"] += 1
        else:
            # Si es requerido y fallo, intentar fallback opcional si existe
            fallback_found = False
            for opt in data.get("models", []):
                if opt.get("fallback_for") == entry["filename"]:
                    if _download_entry(opt, dest_dir_p, on_progress):
                        log.info("Fallback %s usado para %s", opt["filename"], entry["filename"])
                        summary["ok"] += 1
                        fallback_found = True
                        break
            if not fallback_found:
                summary["failed"] += 1
                log.warning("No se pudo descargar modelo requerido: %s", entry["filename"])
    return summary


def ensure_models_async(
    dest_dir: str | Path | None = None,
    on_progress: Callable[[str, int], None] | None = None,
    callback: Callable[[dict], None] | None = None,
    include_optional: bool = False,
) -> threading.Thread:
    """Lanza ensure_models en hilo daemon (no bloquea GUI)."""

    def _job():
        try:
            result = ensure_models(dest_dir=dest_dir, on_progress=on_progress, include_optional=include_optional)
            if callback:
                callback(result)
        except Exception as e:
            log.warning("ensure_models_async fallo: %s", e)

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return t


def get_model_path(filename: str, dest_dir: str | Path | None = None) -> str:
    """Resuelve ruta absoluta del modelo: primero bundle (_MEIPASS), luego APPDATA/models, luego dev models/."""
    import sys

    candidates: list[Path] = []
    # 1. Bundle (si el spec viejo aun lo incluye)
    candidates.append(Path(resource_path(f"models/{filename}")))
    # 2. APPDATA persistente (descarga selectiva)
    candidates.append(Path(user_data_path(f"models/{filename}")))
    # 3. Dest_dir explicito
    if dest_dir is not None:
        candidates.append(Path(dest_dir) / filename)
    # 4. Dev models/
    candidates.append(Path("models") / filename)

    for p in candidates:
        if p.exists():
            return str(p)
    # Retorna la ruta esperada en APPDATA (aunque no exista aun) para que el downloader sepa donde ponerlo
    return str(Path(user_data_path(f"models/{filename}")))


def missing_models(dest_dir: str | Path | None = None, manifest: str | None = None, include_optional: bool = False) -> list[str]:
    data = _load_manifest(manifest)
    dest_dir_p = _resolve_models_dir(dest_dir)
    entries = data.get("models", [])
    if not include_optional:
        entries = _filter_required(entries, dest_dir_p)
    missing: list[str] = []
    for e in entries:
        dest = dest_dir_p / e["filename"]
        if not _is_valid(dest, e.get("size"), e.get("sha256")):
            missing.append(e["filename"])
    return missing
