"""Optional OCR adapters selected after video processing."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import requests

def _load_env_files() -> None:
    """Carga variables de entorno desde los candidatos de `.env`.

    Orden: env real > raíz de proyecto/frozen (_MEIPASS) > APPDATA del usuario.
    Así el token de Plate Recognizer funciona igual en dev y en el exe
    instalado (donde el `.env` viaja empaquetado o se coloca en APPDATA).
    """
    candidates: list[Path] = []
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / ".env")
    try:
        from src.path_helper import resource_path
        candidates.append(Path(resource_path(".env")))
    except Exception:
        pass
    try:
        from src.core.utils.paths import APPDATA_DIR
        candidates.append(APPDATA_DIR / ".env")
        candidates.append(APPDATA_DIR / "plate_recognizer.json")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        for cand in candidates:
            if cand.exists():
                load_dotenv(cand)
    except ImportError:
        pass


_load_env_files()


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


class PlateRecognizerSnapshotReader:
    method = "plate_recognizer"

    def __init__(self, token: str | None = None, timeout: int = 30,
                 min_interval_seconds: float = 2.0, max_retries: int = 3):
        self.token = token or os.getenv("PLATE_RECOGNIZER_API_TOKEN") or self._token_from_appdata()
        self.timeout = timeout
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_retries = max(0, int(max_retries))
        self._last_request_at = 0.0

    @staticmethod
    def _token_from_appdata() -> str | None:
        """Lee el token persistido por el usuario en APPDATA (JSON plano)."""
        try:
            from src.core.utils.paths import APPDATA_DIR
            path = APPDATA_DIR / "plate_recognizer.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("token") or data.get("PLATE_RECOGNIZER_API_TOKEN")
        except Exception:
            return None
        return None

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
