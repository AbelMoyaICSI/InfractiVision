"""Puerto de persistencia. Implementaciones: SQLite, MySQL."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from src.domain.entities import Violation


@runtime_checkable
class ViolationRepositoryPort(Protocol):
    def save(self, violation: Violation) -> str:
        """Persiste la infracción y devuelve el id generado (o ticket_number)."""
        ...

    def get_by_id(self, violation_id: str) -> Violation | None: ...

    def list_recent(self, limit: int = 50) -> Sequence[Violation]: ...
