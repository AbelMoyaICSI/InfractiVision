"""Persistence for official processing and human-reviewed OCR results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.domain.entities.plate_evidence import PlateEvidence


class ReportRepository:
    def save_processing(self, path: str | Path, payload: dict) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def export_validated(self, directory: str | Path, evidences: list[PlateEvidence]) -> tuple[Path, Path]:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        rows = [e.to_dict() for e in evidences if e.validated and e.plate_text]
        json_path = target_dir / "reporte_placas_validadas.json"
        csv_path = target_dir / "reporte_placas_validadas.csv"
        json_path.write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        fields = list(rows[0].keys()) if rows else ["video", "vehicle_id", "plate", "validated"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return json_path, csv_path
