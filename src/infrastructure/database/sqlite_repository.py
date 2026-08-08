"""SQLite repository. Implementa `ViolationRepositoryPort`."""
from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Sequence

from src.core.exceptions import RepositoryError
from src.core.logger import get_logger
from src.domain.entities import Violation, ViolationEvidence
from src.domain.interfaces import ViolationRepositoryPort

log = get_logger("infra.db.sqlite")

_DDL = """
CREATE TABLE IF NOT EXISTS violations (
    id              TEXT PRIMARY KEY,
    plate_text      TEXT NOT NULL,
    plate_confidence REAL NOT NULL,
    vehicle_class_id INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,
    occurred_at     TEXT NOT NULL,
    violation_type  TEXT NOT NULL,
    image_path      TEXT,
    video_path      TEXT,
    ticket_number   TEXT
);
"""


class SQLiteViolationRepository(ViolationRepositoryPort):
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_DDL)
            conn.commit()
        log.info("SQLite repo abierto en %s", db_path)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def save(self, violation: Violation) -> str:
        vid = str(uuid.uuid4())
        ticket = violation.ticket_number or vid[:8].upper()
        ev = violation.evidence or ViolationEvidence(image_path="", video_path=None)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO violations VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        vid,
                        violation.plate_text,
                        violation.plate_confidence,
                        violation.vehicle_class_id,
                        violation.track_id,
                        violation.occurred_at.isoformat(),
                        violation.violation_type,
                        ev.image_path,
                        ev.video_path,
                        ticket,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise RepositoryError(f"SQLite save error: {e}") from e
        return ticket

    def get_by_id(self, violation_id: str) -> Violation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM violations WHERE id = ? OR ticket_number = ?",
                (violation_id, violation_id),
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def list_recent(self, limit: int = 50) -> Sequence[Violation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM violations ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    @staticmethod
    def _row_to_entity(row) -> Violation:
        return Violation(
            plate_text=row[1],
            plate_confidence=row[2],
            vehicle_class_id=row[3],
            track_id=row[4],
            occurred_at=datetime.fromisoformat(row[5]),
            violation_type=row[6],
            evidence=ViolationEvidence(image_path=row[7] or "", video_path=row[8]),
            ticket_number=row[9],
        )
