"""Acceso JSON compartido: caché validada por mtime + lock por archivo +
escritura atómica.

Los archivos de datos (infracciones.json, nie_infracciones.json, configs) se
leen y reescriben enteros desde varios hilos (UI, worker de procesamiento,
migración Firestore). Este módulo centraliza el acceso para evitar:
- lecturas redundantes de disco (cache con stat() de validación),
- carreras read-modify-write entre hilos (lock por archivo),
- archivos corruptos por escritura a medias (tmp + rename).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Callable

_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _file_locks_guard:
        return _file_locks.setdefault(path, threading.Lock())


def read_json(path: str, default: object = None) -> object:
    """Lee un JSON con caché validada por mtime (invalida si otro módulo
    reescribió el archivo). Devuelve `default` si no existe o no se puede
    parsear."""
    if default is None:
        default = {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return default
    with _cache_lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = default
    with _cache_lock:
        _cache[path] = (mtime, data)
    return data


def write_json(path: str, data: object) -> None:
    """Escribe JSON atómicamente (tmp + rename) y actualiza la caché."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    with _cache_lock:
        _cache[path] = (mtime, data)


def mutate_json(path: str, fn: Callable[[object], object]) -> object:
    """Aplica `fn(dato_actual)` y guarda, todo bajo lock exclusivo del archivo.

    Previene carreras read-modify-write entre hilos sobre el mismo JSON.
    Devuelve el dato final ya guardado.
    """
    with _get_lock(path):
        data = read_json(path)
        new_data = fn(data)
        write_json(path, new_data)
        return new_data