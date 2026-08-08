#!/usr/bin/env python
"""Evaluate Google Vision OCR against tests/verdad.test.json.

Examples:
    python scripts/google_vision_ocr.py --dry-run
    python scripts/google_vision_ocr.py --limit 5
    python scripts/google_vision_ocr.py --input data/output/official/crops
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (ROOT / "data" / "output" / "official" / "crops", ROOT / "data" / "output" / "crops")
DEFAULT_GT = ROOT / "tests" / "verdad.test.json"

sys.path.insert(0, str(ROOT))
from src.application.services.ocr_accuracy_evaluator import evaluate_predictions, load_ground_truth, save_evaluation


def default_input() -> Path:
    return next((path for path in DEFAULT_INPUTS if path.exists()), DEFAULT_INPUTS[-1])


def call_google(image_path: Path, api_key: str, timeout: int = 30, retries: int = 3) -> dict:
    content = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {"requests": [{"image": {"content": content}, "features": [{"type": "TEXT_DETECTION"}]}]}
    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.post(
                "https://vision.googleapis.com/v1/images:annotate",
                params={"key": api_key},
                json=payload,
                timeout=timeout,
            )
            if response.status_code in (429, 500, 502, 503):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            result = response.json().get("responses", [{}])[0]
            if result.get("error"):
                return {"plate": "", "confidence": 0.0, "error": result["error"].get("message", "")}
            annotations = result.get("textAnnotations", [])
            raw = annotations[0].get("description", "") if annotations else ""
            first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
            return {"plate": first_line, "confidence": 0.0, "raw": raw, "confidence_available": False}
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"plate": "", "confidence": 0.0, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Google Vision OCR")
    parser.add_argument("--input", type=Path, default=None, help="Directory with plate crops")
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "google_vision_accuracy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    input_dir = args.input or default_input()
    images = sorted(path for path in input_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No crop images found in {input_dir}")
    entries = load_ground_truth(args.gt_path)
    print(f"Google Vision: {len(images)} images | input={input_dir}")
    if args.dry_run:
        for path in images:
            print(f"  {path.name}")
        return
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY")
    start = time.perf_counter()
    predictions = []
    for index, image in enumerate(images, 1):
        result = call_google(image, api_key)
        predictions.append({"file": image.name, "plate": result.get("plate", ""), "confidence": result.get("confidence", 0.0), "error": result.get("error", ""), "method": "google_vision"})
        print(f"[{index}/{len(images)}] {image.name}: {result.get('plate') or '-'}")
    rows, summary = evaluate_predictions(predictions, entries)
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    json_path, csv_path = save_evaluation(args.output, rows, summary)
    print(f"Exact matches: {summary['exact_matches']}/{summary['predictions']}")
    print(f"Precision: {summary['precision']:.2%} | Recall: {summary['recall']:.2%} | Similarity: {summary['average_similarity']:.2%}")
    print(f"Reports: {json_path} | {csv_path}")


if __name__ == "__main__":
    main()
