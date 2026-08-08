"""Shared accuracy calculations for cloud OCR evaluators."""
from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize_plate(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def format_plate(value: str | None) -> str:
    clean = normalize_plate(value)
    return f"{clean[:3]}-{clean[3:]}" if len(clean) == 6 else clean


def _filename_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_ground_truth(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("videos_verdad", [])


def infer_video(filename: str, entries: list[dict]) -> dict | None:
    key = _filename_key(Path(filename).stem)
    matches = []
    for entry in entries:
        video_key = _filename_key(Path(entry.get("path_name", "")).stem)
        raw_stem = Path(entry.get("path_name", "")).stem
        aliases = [video_key]
        for separator in (" ‐ ", " - ", "_", " "):
            if separator in raw_stem:
                prefix = _filename_key(raw_stem.split(separator, 1)[0])
                if prefix:
                    aliases.append(prefix)
        for alias in aliases:
            if alias and (alias in key or key.startswith(alias)):
                matches.append((len(alias), entry))
                break
    return max(matches, key=lambda item: item[0])[1] if matches else None


def similarity(predicted: str, expected: str) -> float:
    if not predicted or not expected:
        return 0.0
    return SequenceMatcher(None, predicted, expected).ratio()


def evaluate_predictions(predictions: list[dict], entries: list[dict]) -> tuple[list[dict], dict]:
    """Evaluate predictions with one-to-one matching within each video.

    A ground-truth plate can only satisfy one prediction, preventing duplicate
    crops from inflating the precision score.
    """
    rows = []
    used_expected: dict[str, set[str]] = {}
    for prediction in predictions:
        entry = infer_video(prediction["file"], entries)
        expected_values = [normalize_plate(value) for value in (entry or {}).get("cars", [])]
        video_name = (entry or {}).get("path_name", "")
        used = used_expected.setdefault(video_name, set())
        predicted = normalize_plate(prediction.get("plate", ""))
        candidates = [(similarity(predicted, expected), expected) for expected in expected_values if expected not in used]
        candidates.sort(reverse=True)
        best_similarity, best_expected = candidates[0] if candidates else (0.0, "")
        exact = bool(predicted and best_expected and predicted == best_expected)
        if exact:
            used.add(best_expected)
        rows.append({
            **prediction,
            "video": video_name or "UNKNOWN",
            "predicted": format_plate(predicted),
            "expected": format_plate(best_expected),
            "exact_match": exact,
            "similarity": round(best_similarity, 4),
            "ground_truth_count": len(expected_values),
        })

    expected_total = sum(len(entry.get("cars", [])) for entry in entries)
    exact_matches = sum(row["exact_match"] for row in rows)
    detected = sum(bool(row.get("predicted")) for row in rows)
    summary = {
        "predictions": len(rows),
        "detected": detected,
        "no_text": len(rows) - detected,
        "expected_plates": expected_total,
        "exact_matches": exact_matches,
        "precision": exact_matches / len(rows) if rows else 0.0,
        "recall": exact_matches / expected_total if expected_total else 0.0,
        "average_similarity": sum(row["similarity"] for row in rows) / len(rows) if rows else 0.0,
        "video_count": len(entries),
    }
    return rows, summary


def save_evaluation(output_path: str | Path, rows: list[dict], summary: dict) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["file", "predicted", "expected", "exact_match", "similarity"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
