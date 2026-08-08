"""MySQL repository (intercambiable con SQLite vía DI).

Lazy-import del driver para que el .exe Windows no dependa de mysql-client
si solo se usa SQLite.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Sequence
from urllib.parse import urlparse

from src.core.exceptions import RepositoryError
from src.core.logger import get_logger
from src.domain.entities import Violation, ViolationEvidence
from src.domain.interfaces import ViolationRepositoryPort

log = get_logger("infra.db.mysql")


class MySQLViolationRepository(ViolationRepositoryPort):
    def __init__(self, url: str):
        try:
            import mysql.connector  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RepositoryError(f"mysql-connector no instalado: {e}") from e

        u = urlparse(url)
        self._params = dict(
            host=u.hostname or "localhost",
            port=u.port or 3306,
            user=u.username or "root",
            password=u.password or "",
            database=(u.path or "/infractivision").lstrip("/"),
        )
        self._driver = mysql.connector
        self._lock = threading.Lock()
        self._ensure_schema()
        log.info("MySQL repo conectado a %s", self._params["host"])

    def _connect(self):
        return self._driver.connect(**self._params)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id VARCHAR(64) PRIMARY KEY,
                    plate_text VARCHAR(20) NOT NULL,
                    plate_confidence DOUBLE NOT NULL,
                    vehicle_class_id INT NOT NULL,
                    track_id INT NOT NULL,
                    occurred_at DATETIME NOT NULL,
                    violation_type VARCHAR(40) NOT NULL,
                    image_path TEXT,
                    video_path TEXT,
                    ticket_number VARCHAR(40)
                ) ENGINE=InnoDB
                """
            )
            conn.commit()

    def save(self, violation: Violation) -> str:
        vid = str(uuid.uuid4())
        ticket = violation.ticket_number or vid[:8].upper()
        ev = violation.evidence or ViolationEvidence(image_path="", video_path=None)
        try:
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO violations VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        vid,
                        violation.plate_text,
                        violation.plate_confidence,
                        violation.vehicle_class_id,
                        violation.track_id,
                        violation.occurred_at,
                        violation.violation_type,
                        ev.image_path,
                        ev.video_path,
                        ticket,
                    ),
                )
                conn.commit()
        except Exception as e:
            raise RepositoryError(f"MySQL save error: {e}") from e
        return ticket

    def get_by_id(self, violation_id: str) -> Violation | None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM violations WHERE id=%s OR ticket_number=%s",
                (violation_id, violation_id),
            )
            row = cur.fetchone()
        return self._row_to_entity(row) if row else None

    def list_recent(self, limit: int = 50) -> Sequence[Violation]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM violations ORDER BY occurred_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_entity(r) for r in rows]

    @staticmethod
    def _row_to_entity(row) -> Violation:
        return Violation(
            plate_text=row[1],
            plate_confidence=row[2],
            vehicle_class_id=row[3],
            track_id=row[4],
            occurred_at=row[5] if isinstance(row[5], datetime) else datetime.fromisoformat(str(row[5])),
            violation_type=row[6],
            evidence=ViolationEvidence(image_path=row[7] or "", video_path=row[8]),
            ticket_number=row[9],
        )
