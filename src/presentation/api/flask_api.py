"""API Flask. Solo conoce los Casos de Uso y devuelve DTOs serializados."""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from flask import Flask, jsonify

from src.application.dto import ViolationDTO
from src.core.logger import get_logger

if TYPE_CHECKING:
    from src.domain.interfaces import ViolationRepositoryPort

log = get_logger("presentation.api")


def create_app(repository: "ViolationRepositoryPort") -> Flask:
    app = Flask("infractivision-api")

    @app.get("/health")
    def health():  # type: ignore[unused-ignore]
        return jsonify({"status": "ok"})

    @app.get("/violations")
    def list_violations():  # type: ignore[unused-ignore]
        items = [
            asdict(ViolationDTO.from_entity(v))
            for v in repository.list_recent(limit=100)
        ]
        # datetime no es JSON-serializable: convertimos a ISO
        for it in items:
            it["occurred_at"] = it["occurred_at"].isoformat()
        return jsonify(items)

    @app.get("/violations/<vid>")
    def get_violation(vid: str):  # type: ignore[unused-ignore]
        v = repository.get_by_id(vid)
        if v is None:
            return jsonify({"error": "not_found"}), 404
        dto = ViolationDTO.from_entity(v)
        payload = asdict(dto)
        payload["occurred_at"] = dto.occurred_at.isoformat()
        return jsonify(payload)

    log.info("Flask API montada (/health, /violations)")
    return app
