"""Optional OCR adapters selected after video processing."""
from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


class PlateRecognizerSnapshotReader:
    method = "plate_recognizer"

    def __init__(self, token: str | None = None, timeout: int = 30,
                 min_interval_seconds: float = 2.0, max_retries: int = 3):
        self.token = token or os.getenv("PLATE_RECOGNIZER_API_TOKEN")
        self.timeout = timeout
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_retries = max(0, int(max_retries))
        self._last_request_at = 0.0

    def _wait_between_requests(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_after(response, fallback: float) -> float:
        value = response.headers.get("Retry-After")
        try:
            return max(0.0, float(value)) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def read(self, image_path: str | Path) -> tuple[str, float]:
        if not self.token:
            raise RuntimeError("Falta PLATE_RECOGNIZER_API_TOKEN")
        response = None
        for attempt in range(self.max_retries + 1):
            self._wait_between_requests()
            with Path(image_path).open("rb") as image:
                response = requests.post(
                    "https://api.platerecognizer.com/v1/plate-reader/",
                    headers={"Authorization": f"Token {self.token}"},
                    data={"regions": "pe"},
                    files={"upload": image},
                    timeout=self.timeout,
                )
            self._last_request_at = time.monotonic()
            if response.status_code != 429:
                break
            if attempt >= self.max_retries:
                response.raise_for_status()
            fallback = min(60.0, 5.0 * (2 ** attempt))
            time.sleep(self._retry_after(response, fallback))

        assert response is not None
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "", 0.0
        best = results[0]
        return normalize_plate(best.get("plate", "")), float(best.get("score", 0.0))


class GoogleVisionReader:
    method = "google_vision"

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_VISION_API_KEY")
        self.timeout = timeout

    def read(self, image_path: str | Path) -> tuple[str, float]:
        if not self.api_key:
            raise RuntimeError("Falta GOOGLE_API_KEY")
        content = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        response = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            params={"key": self.api_key},
            json={"requests": [{"image": {"content": content}, "features": [{"type": "TEXT_DETECTION"}]}]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        annotations = response.json().get("responses", [{}])[0].get("textAnnotations", [])
        raw = annotations[0].get("description", "") if annotations else ""
        return normalize_plate(raw.splitlines()[0] if raw else ""), 0.0
