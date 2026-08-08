#!/usr/bin/env python
"""Evaluate Plate Recognizer Snapshot API against the project ground truth."""
from __future__ import annotations

import argparse
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


def call_plate_recognizer(image_path: Path, token: str, timeout: int = 45, retries: int = 3) -> dict:
    last_error = ""
    for attempt in range(retries):
        try:
            with image_path.open("rb") as image:
                response = requests.post(
                    "https://api.platerecognizer.com/v1/plate-reader/",
                    headers={"Authorization": f"Token {token}"},
                    data={"regions": "pe"},
                    files={"upload": (image_path.name, image, "image/jpeg")},
                    timeout=timeout,
                )
            if response.status_code in (429, 500, 502, 503):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                return {"plate": "", "confidence": 0.0, "raw": data}
            best = max(results, key=lambda item: float(item.get("score", 0.0)))
            return {"plate": best.get("plate", ""), "confidence": float(best.get("score", 0.0)), "raw": data}
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"plate": "", "confidence": 0.0, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Plate Recognizer OCR")
    parser.add_argument("--input", type=Path, default=None, help="Directory with plate crops")
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "plate_recognizer_accuracy")
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
    print(f"Plate Recognizer: {len(images)} images | input={input_dir}")
    if args.dry_run:
        for path in images:
            print(f"  {path.name}")
        return
    token = os.getenv("PLATE_RECOGNIZER_API_TOKEN")
    if not token:
        raise SystemExit("Missing PLATE_RECOGNIZER_API_TOKEN")
    start = time.perf_counter()
    predictions = []
    for index, image in enumerate(images, 1):
        result = call_plate_recognizer(image, token)
        predictions.append({"file": image.name, "plate": result.get("plate", ""), "confidence": result.get("confidence", 0.0), "error": result.get("error", ""), "method": "plate_recognizer"})
        print(f"[{index}/{len(images)}] {image.name}: {result.get('plate') or '-'} ({result.get('confidence', 0.0):.2f})")
    rows, summary = evaluate_predictions(predictions, entries)
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    json_path, csv_path = save_evaluation(args.output, rows, summary)
    print(f"Exact matches: {summary['exact_matches']}/{summary['predictions']}")
    print(f"Precision: {summary['precision']:.2%} | Recall: {summary['recall']:.2%} | Similarity: {summary['average_similarity']:.2%}")
    print(f"Reports: {json_path} | {csv_path}")


if __name__ == "__main__":
    main()
