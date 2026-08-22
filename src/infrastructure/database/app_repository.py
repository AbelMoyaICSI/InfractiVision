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

    def add_migration(self, fecha: str, timestamp: str, registros: int, estado: str) -> int:
        """Inserta un registro en el historial de migraciones."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO migrations (fecha, timestamp, registros, estado) "
                "VALUES (?, ?, ?, ?)",
                (fecha, timestamp, registros, estado),
            )
            conn.commit()
            return int(cur.lastrowid)

    def clear_migrations(self) -> None:
        """Vacía la tabla de historial de migraciones."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM migrations")
            conn.commit()

    def count_rows(self, table: str) -> int:
        with self._connect() as conn:
            row = conn.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()
        return int(row["n"])

    # ─── Operaciones de sesión (SQLite como única fuente) ───────────────

    def insert_infractions(self, infractions: list[dict]) -> int:
        """Inserta lote de infracciones (NID/NIE) en `infractions`.

        Deduplica intra-lote por (nombre_video, placa, video_timestamp)
        para evitar stack duplicado cuando se reprocesa el mismo video.
        Retorna cantidad insertada.
        """
        if not infractions:
            return 0
        # Dedup intra-lote
        seen = set()
        deduped = []
        for inf in infractions:
            key = (
                inf.get("nombre_video", ""),
                inf.get("placa", ""),
                inf.get("video_timestamp", ""),
                str(inf.get("tiempo_procesamiento", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(inf)
        now = datetime.now().isoformat()
        inserted = 0
        with self._lock, self._connect() as conn:
            for entry in deduped:
                meta = entry.get("metadata_clasificacion")
                if isinstance(meta, dict):
                    meta = json.dumps(meta, ensure_ascii=False)
                    entry = {**entry, "metadata_clasificacion_json": meta}
                elif "metadata_clasificacion_json" not in entry and "metadata_clasificacion" in entry:
                    entry["metadata_clasificacion_json"] = meta
                values = []
                for col in _INF_COLUMNS:
                    if col == "metadata_clasificacion_json":
                        values.append(entry.get("metadata_clasificacion_json") or entry.get("metadata_clasificacion"))
                    elif col == "created_at":
                        values.append(now)
                    elif col == "modo_nocturno":
                        v = entry.get("modo_nocturno")
                        values.append(1 if v else 0 if v is not None else 0)
                    else:
                        values.append(entry.get(col))
                # Normalizar json string
                idx_meta = _INF_COLUMNS.index("metadata_clasificacion_json")
                if isinstance(values[idx_meta], dict):
                    values[idx_meta] = json.dumps(values[idx_meta], ensure_ascii=False)
                conn.execute(
                    f"INSERT INTO infractions ({', '.join(_INF_COLUMNS)}) VALUES ({', '.join('?' * len(_INF_COLUMNS))})",
                    tuple(values),
                )
                inserted += 1
            conn.commit()
        return inserted

    def delete_by_placa(self, placa: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM infractions WHERE placa = ?", (placa,))
            conn.commit()
            return int(cur.rowcount)

    def clear_infractions(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM infractions")
            conn.commit()
            return int(cur.rowcount)

    def delete_infractions_by_video(self, nombre_video: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM infractions WHERE nombre_video = ?", (nombre_video,))
            conn.commit()
            return int(cur.rowcount)

    def list_infractions_by_video(self, nombre_video: str, limit: int = 10000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM infractions WHERE nombre_video = ? ORDER BY id DESC LIMIT ?",
                (nombre_video, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_indicators(self, report: dict) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO indicators(id, report_json, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET report_json=excluded.report_json, updated_at=excluded.updated_at",
                (json.dumps(report, ensure_ascii=False), now),
            )
            conn.commit()

    def compute_indicators_report(
        self,
        infractions_session: list[dict],
        nombre_video: str | None = None,
        config_semaforo: str | None = None,
    ) -> dict:
        """Genera reporte TI/TR/NID/NIE coherente con la sesión (sin leer JSON)."""
        # Replicar lógica de infractions_management_window.generate_performance_indicators_json
        # pero sin IO, usando solo infractions_session.
        from datetime import datetime as _dt

        if not isinstance(infractions_session, list):
            infractions_session = []
        day_infractions: dict[str, dict] = {}
        nid_count = 0
        nie_count = 0
        for inf in infractions_session:
            fecha = inf.get("fecha", "Sin fecha")
            placa = inf.get("placa", "")
            clas = inf.get("clasificacion", "NID")
            grp = day_infractions.setdefault(fecha, {"total": 0, "placas": {}, "nid": 0, "nie": 0})
            grp["total"] += 1
            if clas == "NID":
                nid_count += 1
                grp["nid"] += 1
            elif clas == "NIE":
                nie_count += 1
                grp["nie"] += 1
            if placa:
                grp["placas"].setdefault(placa, 0)
                grp["placas"][placa] += 1

        pnp_data = {
            "Enero 2023": {"total": 125, "dias": 31},
            "Febrero 2023": {"total": 117, "dias": 28},
            "Marzo 2023": {"total": 137, "dias": 31},
            "Abril 2023": {"total": 129, "dias": 30},
        }
        police_times_min = [7, 6, 5, 10, 8]
        pnp_total = sum(m["total"] for m in pnp_data.values())
        pnp_days = sum(m["dias"] for m in pnp_data.values())
        pnp_daily = pnp_total / pnp_days if pnp_days else 0
        sw_days = len(day_infractions)
        sw_inf = len(infractions_session)
        sw_daily = sw_inf / sw_days if sw_days else 0
        total_detectadas = nid_count + nie_count
        ti_percentage = (nid_count / total_detectadas * 100) if total_detectadas else 0.0
        pnp_sec = (sum(police_times_min) / len(police_times_min) * 60) if police_times_min else 0
        # tiempos individuales desde infractions_session
        times_sec = [float(inf.get("tiempo_procesamiento", 0) or 0) for inf in infractions_session if (inf.get("tiempo_procesamiento") or 0) > 0]
        sw_times_min = [t / 60.0 for t in times_sec]
        sw_min = sum(sw_times_min) / len(sw_times_min) if sw_times_min else 0.0
        pnp_min = pnp_sec / 60.0
        tr_reduction_pct = ((pnp_min - sw_min) / pnp_min * 100) if pnp_min else 0
        tr_speedup = pnp_min / sw_min if sw_min else 0
        nid_today = nid_count
        nie_today = nie_count
        nid_daily_avg = nid_count / sw_days if sw_days > 0 else nid_count
        avenida = infractions_session[0].get("ubicacion", "N/A") if infractions_session else "N/A"
        video_name = nombre_video or (infractions_session[0].get("nombre_video", "desconocido.mp4") if infractions_session else "desconocido.mp4")
        config_id = config_semaforo or (infractions_session[0].get("config_semaforo", "sin-configurar") if infractions_session else "sin-configurar")
        report = {
            "fecha_generacion": _dt.now().strftime("%d/%m/%Y %H:%M:%S"),
            "periodo_analisis": f"{min(day_infractions.keys(), default='N/A')} - {max(day_infractions.keys(), default='N/A')}",
            "dias_analizados": sw_days,
            "ubicacion": avenida,
            "nombre_video": video_name,
            "config_semaforo": config_id,
            "nota": "Datos de la sesión actual de procesamiento, no acumulados históricos",
            "indicadores": {
                "TI": {
                    "descripcion": "Tasa de Infracciones Detectadas (Nivel Diario Agregado)",
                    "unidad": "infracciones por día comparativo (%)",
                    "sin_software": {"registros_campo_diarios": round(pnp_daily, 2), "fuente": "Registros PNP históricos"},
                    "con_software": {"detecciones_software_diarias": round(sw_daily, 2), "dias_analizados": sw_days},
                    "porcentaje_acierto": round(ti_percentage, 2),
                },
                "TR": {
                    "descripcion": "Tiempo de Registro por Infracción Individual",
                    "unidad": "minutos por infracción (min)",
                    "sin_software": {"tiempo_promedio_minutos": round(pnp_min, 2), "fuente": "Estimación basada en registros históricos de campo"},
                    "con_software": {"tiempo_promedio_minutos": round(sw_min, 2), "tiempos_individuales": [round(t, 2) for t in sw_times_min], "muestras_analizadas": len(times_sec)},
                    "reduccion_tiempo_porcentual": round(tr_reduction_pct, 2),
                    "veces_mas_rapido": round(tr_speedup, 2),
                },
                "NID": {"descripcion": "Número de Infracciones Detectadas Correctamente", "unidad": "cantidad válida por día", "infracciones_hoy": nid_today, "promedio_diario": round(nid_daily_avg, 0), "periodo_analizado": f"{sw_days} días", "total": nid_count},
                "NIE": {"descripcion": "Número de Infracciones Incorrectamente Registradas", "unidad": "cantidad no válida por día", "infracciones_incorrectas": nie_count, "total": nie_count},
            },
            "resumen_global": {
                "ti_porcentaje_acierto": f"{ti_percentage:.1f}%",
                "tiempo_registro_minutos": f"{sw_min:.2f} min",
                "infracciones_detectadas_hoy": nid_today,
                "nid_total": nid_count,
                "nie_total": nie_count,
                "tir_total": nid_count + nie_count,
            },
        }
        return report

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