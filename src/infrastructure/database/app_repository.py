"""Local SQLite storage for the app domain data.

Unifies the legacy JSON persistence into a single SQLite database
(``data/infractions.sqlite``). This module owns the schema and the one-time
migration of the existing JSON files:

    * ``config/{avenue_config,time_presets,polygon_config}.json``
      -> table ``video_configs`` (one row per registered video).
    * ``data/infracciones.json`` (NID) and ``data/nie_infracciones.json`` (NIE)
      -> table ``infractions``.
    * ``data/indicadores_rendimiento.json`` -> table ``indicators``.
    * ``data/historial_migraciones.json`` -> table ``migrations``.

The migration is idempotent: it runs once (tracked in ``meta``) and never
duplicates rows. The JSON files are only read, never modified.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.core.utils import resource_path

log = get_logger("infra.db.app")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "infractions.sqlite"
# Preset (seed) versionable: schema + video_configs con los presets actuales.
# Se usa como bootstrap: si la DB local no existe, se restaura una copia.
PRESET_DB = resource_path("presets/infractions_preset.db")

_SCHEMA_VERSION = "1"
_DATA_MIGRATED_KEY = "data_migrated"

_DDL = [
    # Infracciones (NID + NIE, distinguidas por `clasificacion`)
    """
    CREATE TABLE IF NOT EXISTS infractions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        placa                   TEXT NOT NULL,
        fecha                   TEXT,
        hora                    TEXT,
        video_timestamp         TEXT,
        tiempo_video            TEXT,
        ubicacion               TEXT,
        franja_horaria          TEXT,
        tipo                    TEXT,
        estado                  TEXT,
        plate_path              TEXT,
        vehicle_path            TEXT,
        nombre_video            TEXT,
        config_semaforo         TEXT,
        clasificacion           TEXT NOT NULL DEFAULT 'NID',
        confianza               REAL,
        tiempo_procesamiento    REAL,
        metadata_clasificacion_json TEXT,
        sistema_version         TEXT,
        hostname                TEXT,
        username                TEXT,
        modo_nocturno           INTEGER,
        created_at              TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_infractions_placa ON infractions(placa);",
    "CREATE INDEX IF NOT EXISTS idx_infractions_nombre_video ON infractions(nombre_video);",
    "CREATE INDEX IF NOT EXISTS idx_infractions_clasificacion ON infractions(clasificacion);",
    # Configuración por video registrado (consolida avenue/times/polygon)
    """
    CREATE TABLE IF NOT EXISTS video_configs (
        video_name   TEXT PRIMARY KEY,
        avenue       TEXT DEFAULT '',
        green        REAL,
        yellow       REAL,
        red          REAL,
        time_slot    TEXT DEFAULT '',
        polygon_json TEXT,
        updated_at   TEXT NOT NULL
    );
    """,
    # Reporte global de indicadores (una sola fila, se recalcula al sobrescribir)
    """
    CREATE TABLE IF NOT EXISTS indicators (
        id         INTEGER PRIMARY KEY CHECK (id = 1),
        report_json TEXT,
        updated_at TEXT NOT NULL
    );
    """,
    # Historial acumulativo de migraciones a la nube
    """
    CREATE TABLE IF NOT EXISTS migrations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha        TEXT,
        timestamp    TEXT,
        registros    INTEGER,
        estado       TEXT,
        details_json TEXT
    );
    """,
    # Metadatos de la propia DB (schema_version, flags de migración)
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """,
]

# Columnas de `infractions` (orden fijo; campos legacy + campos de la CA)
_INF_COLUMNS = (
    "placa", "fecha", "hora", "video_timestamp", "tiempo_video", "ubicacion",
    "franja_horaria", "tipo", "estado", "plate_path", "vehicle_path",
    "nombre_video", "config_semaforo", "clasificacion", "confianza",
    "tiempo_procesamiento", "metadata_clasificacion_json", "sistema_version",
    "hostname", "username", "modo_nocturno", "created_at",
)

_VIDEO_COLUMNS = (
    "video_name", "avenue", "green", "yellow", "red", "time_slot",
    "polygon_json", "updated_at",
)

_MIGRATION_COLUMNS = ("fecha", "timestamp", "registros", "estado", "details_json")


class AppRepository:
    """Acceso a la BD SQLite local de datos de la aplicación.

    Patrón de conexión idéntico al ``SQLiteViolationRepository``: conexión
    corta por operación + lock de escritura. WAL habilitado para lecturas
    concurrentes desde los hilos de la GUI.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._ensure_db_from_preset()
        self.ensure_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _ensure_db_from_preset(self) -> None:
        """Restaura una copia del preset si la DB no existe.

        El preset es un seed versionable (schema + `video_configs`); solo se
        copia cuando la base local no existe. Si no hay preset, se continúa y
        `ensure_schema()` creará la DB vacía.
        """
        db = Path(self._db_path)
        if db.exists():
            return
        preset = Path(PRESET_DB)
        if not preset.exists():
            return
        try:
            db.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preset, db)
            log.info("DB restaurada desde preset: %s → %s", preset, db)
        except OSError as e:
            log.warning("No se pudo restaurar la DB desde preset: %s", e)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ─── Schema ───────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            for stmt in _DDL:
                conn.execute(stmt)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", _SCHEMA_VERSION),
            )
            conn.commit()

    def _get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    # ─── Lecturas (verificación + futura integración) ─────────────────────

    def get_video_config(self, video_name: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM video_configs WHERE video_name = ?", (video_name,)
            ).fetchone()
        if row is None:
            return None
        cfg = dict(row)
        cfg["polygon"] = json.loads(cfg.pop("polygon_json")) if cfg.get("polygon_json") else None
        return cfg

    def all_video_configs(self) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM video_configs").fetchall()
        return {r["video_name"]: dict(r) for r in rows}

    def list_infractions(self, limit: int = 100, clasificacion: str | None = None) -> list[dict]:
        q = "SELECT * FROM infractions"
        params: tuple = ()
        if clasificacion:
            q += " WHERE clasificacion = ?"
            params = (clasificacion,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(q, params + (limit,)).fetchall()
        return [dict(r) for r in rows]

    def count_infractions(self, clasificacion: str | None = None) -> int:
        q = "SELECT COUNT(*) AS n FROM infractions"
        params: tuple = ()
        if clasificacion:
            q += " WHERE clasificacion = ?"
            params = (clasificacion,)
        with self._connect() as conn:
            row = conn.execute(q, params).fetchone()
        return int(row["n"])

    def get_indicators(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM indicators WHERE id = 1"
            ).fetchone()
        if row is None or not row["report_json"]:
            return None
        return json.loads(row["report_json"])

    def list_migrations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM migrations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_rows(self, table: str) -> int:
        with self._connect() as conn:
            row = conn.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()
        return int(row["n"])

    # ─── Migración de datos legacy ────────────────────────────────────────

    def migrate_legacy_data(
        self, project_root: str | Path | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Migra los JSON legacy a la BD (idempotente).

        Solo se ejecuta una vez (flag ``data_migrated`` en ``meta``); con
        ``force=True`` se re-ejecuta refrescando ``video_configs`` e insertando
        las infracciones/indicadores/migraciones que aún no existan.
        """
        root = Path(project_root) if project_root else PROJECT_ROOT
        if self._get_meta(_DATA_MIGRATED_KEY) == "1" and not force:
            return {
                "skipped": True,
                "video_configs": self.count_rows("video_configs"),
                "infractions": self.count_rows("infractions"),
                "indicators": self.count_rows("indicators"),
                "migrations": self.count_rows("migrations"),
            }

        now = datetime.now().isoformat()
        summary: dict[str, Any] = {"skipped": False}

        with self._lock, self._connect() as conn:
            # 1) Configuraciones por video (merge de los 3 JSON)
            avenue_cfg = self._read_json(root / "config" / "avenue_config.json", {})
            preset_cfg = self._read_json(root / "config" / "time_presets.json", {})
            polygon_cfg = self._read_json(root / "config" / "polygon_config.json", {})
            video_names = set(avenue_cfg) | set(preset_cfg) | set(polygon_cfg)

            upserted = 0
            for name in video_names:
                preset = preset_cfg.get(name) or {}
                if not isinstance(preset, dict):
                    preset = {}
                row = (
                    name,
                    str(avenue_cfg.get(name, "") or ""),
                    preset.get("green"),
                    preset.get("yellow"),
                    preset.get("red"),
                    preset.get("time_slot", ""),
                    json.dumps(polygon_cfg.get(name, []), ensure_ascii=False),
                    now,
                )
                conn.execute(
                    """
                    INSERT INTO video_configs (video_name, avenue, green, yellow,
                                               red, time_slot, polygon_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_name) DO UPDATE SET
                        avenue = excluded.avenue,
                        green = excluded.green,
                        yellow = excluded.yellow,
                        red = excluded.red,
                        time_slot = excluded.time_slot,
                        polygon_json = excluded.polygon_json,
                        updated_at = excluded.updated_at
                    """,
                    row,
                )
                upserted += 1
            summary["video_configs"] = upserted

            # 2) Infracciones NID + NIE (solo si la tabla está vacía: evita
            #    duplicados si se re-ejecuta con --force tras haber procesado)
            inf_count = 0
            if self.count_rows("infractions") == 0:
                nid_entries = self._read_json(root / "data" / "infracciones.json", [])
                nie_entries = self._read_json(root / "data" / "nie_infracciones.json", [])
                inf_count = self._insert_infractions(
                    conn, nid_entries, default_clasificacion="NID"
                )
                inf_count += self._insert_infractions(
                    conn, nie_entries, default_clasificacion="NIE"
                )
            summary["infractions"] = inf_count

            # 3) Indicadores (reporte global, una sola fila; solo si está vacía)
            ind_count = 0
            if self.count_rows("indicators") == 0:
                indicadores = self._read_json(
                    root / "data" / "indicadores_rendimiento.json", {}
                )
                if isinstance(indicadores, dict) and indicadores:
                    conn.execute(
                        "INSERT INTO indicators(id, report_json, updated_at) VALUES (1, ?, ?)",
                        (json.dumps(indicadores, ensure_ascii=False), now),
                    )
                    ind_count = 1
            summary["indicators"] = ind_count

            # 4) Historial de migraciones (solo si la tabla está vacía)
            mig_count = 0
            if self.count_rows("migrations") == 0:
                history = self._read_json(root / "data" / "historial_migraciones.json", [])
                if isinstance(history, list) and history:
                    for rec in history:
                        if not isinstance(rec, dict):
                            continue
                        known = {c: rec.get(c) for c in _MIGRATION_COLUMNS if c != "details_json"}
                        extra = {k: v for k, v in rec.items() if k not in known}
                        conn.execute(
                            """
                            INSERT INTO migrations (fecha, timestamp, registros, estado, details_json)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                known.get("fecha"),
                                known.get("timestamp"),
                                known.get("registros"),
                                known.get("estado"),
                                json.dumps(extra, ensure_ascii=False) if extra else None,
                            ),
                        )
                        mig_count += 1
            summary["migrations"] = mig_count

            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (_DATA_MIGRATED_KEY, "1"),
            )
            conn.commit()

        log.info("Migración legacy→SQLite completada: %s", summary)
        return summary

    def _insert_infractions(
        self, conn: sqlite3.Connection, data: Any, default_clasificacion: str
    ) -> int:
        entries = data.get("infracciones") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            entries = []
        now = datetime.now().isoformat()
        count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            meta_json = entry.get("metadata_clasificacion")
            if isinstance(meta_json, dict):
                meta_json = json.dumps(meta_json, ensure_ascii=False)
            values = tuple(
                entry.get(col)
                if col != "clasificacion"
                else (entry.get("clasificacion") or default_clasificacion)
                if col == "clasificacion"
                else entry.get(col)
                for col in _INF_COLUMNS
            )
            # created_at y modo_nocturno (bool→int) se normalizan
            values = list(values)
            values[_INF_COLUMNS.index("created_at")] = now
            mi = _INF_COLUMNS.index("modo_nocturno")
            if values[mi] is not None:
                values[mi] = 1 if values[mi] else 0
            conn.execute(
                f"INSERT INTO infractions ({', '.join(_INF_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_INF_COLUMNS))})",
                tuple(values),
            )
            count += 1
        return count

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not Path(path).exists():
            return default
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("No se pudo leer %s: %s", path, e)
            return default


def migrate_legacy_data(
    db_path: str | Path | None = None,
    project_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Conveniencia: migra los datos legacy al sqlite local y devuelve el resumen."""
    repo = AppRepository(db_path or DEFAULT_DB_PATH)
    return repo.migrate_legacy_data(project_root=project_root, force=force)


def create_preset(preset_path: str | Path | None = None) -> Path:
    """Genera el preset (seed) de la BD desde los JSON de `config/`.

    El preset contiene el schema completo y `video_configs` con los presets
    actuales (avenue/times/polygon); las tablas de datos de usuario quedan
    vacías. Es idempotente y regenerable. No toca la DB de producción.
    """
    preset = Path(preset_path) if preset_path else Path(PRESET_DB)
    preset.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()

    avenue_cfg = AppRepository._read_json(
        PROJECT_ROOT / "config" / "avenue_config.json", {}
    )
    preset_cfg = AppRepository._read_json(
        PROJECT_ROOT / "config" / "time_presets.json", {}
    )
    polygon_cfg = AppRepository._read_json(
        PROJECT_ROOT / "config" / "polygon_config.json", {}
    )
    names = set(avenue_cfg) | set(preset_cfg) | set(polygon_cfg)

    with sqlite3.connect(preset) as conn:
        conn.execute("PRAGMA journal_mode=OFF;")
        for stmt in _DDL:
            conn.execute(stmt)
        for name in names:
            p = preset_cfg.get(name) or {}
            if not isinstance(p, dict):
                p = {}
            conn.execute(
                """
                INSERT OR REPLACE INTO video_configs
                    (video_name, avenue, green, yellow, red, time_slot,
                     polygon_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    str(avenue_cfg.get(name, "") or ""),
                    p.get("green"),
                    p.get("yellow"),
                    p.get("red"),
                    p.get("time_slot", ""),
                    json.dumps(polygon_cfg.get(name, []), ensure_ascii=False),
                    now,
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", _SCHEMA_VERSION),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (_DATA_MIGRATED_KEY, "1"),
        )
        conn.commit()

    log.info("Preset generado: %s (%d configs)", preset, len(names))
    return preset