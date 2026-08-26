"""Tests del descargador de videos demo (idempotente y offline-friendly)."""
from pathlib import Path
from unittest import mock

import pytest

from src.infrastructure.storage import demo_video_downloader as dvd


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """Manifest con 2 videos fake y 1 ya presente (solo para omitirlo)."""
    data = {
        "version": 1,
        "base_dir": "videos",
        "videos": [
            {"filename": "a.mp4", "gcs_path": "a.mp4", "size": 3, "sha256": "x",
             "url": "https://example.test/a.mp4", "public_url": "https://public.test/a.mp4"},
            {"filename": "b.mp4", "gcs_path": "b.mp4", "size": 3, "sha256": "y",
             "url": "https://example.test/b.mp4", "public_url": "https://public.test/b.mp4"},
        ],
    }
    m = tmp_path / "manifest.json"
    m.write_text(__import__("json").dumps(data), encoding="utf-8")
    return m


def test_ensure_demo_videos_skips_by_size_and_downloads_missing(tmp_path, monkeypatch, manifest):
    dest = tmp_path / "videos"
    dest.mkdir()
    # 'a.mp4' ya existe con tamaño correcto → debe omitirse (no descargar).
    (dest / "a.mp4").write_bytes(b"abc")

    def fake_http(url, d):
        d.write_bytes(b"abc")
        return True

    monkeypatch.setattr(dvd, "_service_account_path", lambda: None)
    monkeypatch.setattr(dvd, "_gcs_download", lambda entry, d: False)
    http = mock.Mock(side_effect=fake_http)
    monkeypatch.setattr(dvd, "_http_download", http)

    result = dvd.ensure_demo_videos(dest_dir=dest, manifest=str(manifest))

    assert result["skipped"] == 1  # a.mp4 ya presente
    assert result["ok"] == 1       # b.mp4 descargado
    assert (dest / "b.mp4").exists()
    assert http.call_count == 1


def test_ensure_demo_videos_honors_skip_env(tmp_path, monkeypatch, manifest):
    monkeypatch.setenv("INFRACTI_SKIP_DEMO_DOWNLOAD", "1")
    monkeypatch.setattr(dvd, "_gcs_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe llamarse")))
    monkeypatch.setattr(dvd, "_http_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe llamarse")))
    result = dvd.ensure_demo_videos(dest_dir=tmp_path, manifest=str(manifest))
    assert result == {"ok": 0, "failed": 0, "skipped": 0}


def test_missing_demo_videos_reports_incomplete(tmp_path, manifest):
    dest = tmp_path / "videos"
    dest.mkdir()
    (dest / "a.mp4").write_bytes(b"abc")
    missing = dvd.missing_demo_videos(dest_dir=dest, manifest=str(manifest))
    assert missing == ["b.mp4"]


def test_plate_recognizer_token_falls_back_to_appdata(monkeypatch, tmp_path):
    from src.core.utils import paths as paths_mod

    monkeypatch.delenv("PLATE_RECOGNIZER_API_TOKEN", raising=False)
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    (appdata / "plate_recognizer.json").write_text('{"token": "appdata-token"}', encoding="utf-8")
    monkeypatch.setattr(paths_mod, "APPDATA_DIR", appdata)

    from src.infrastructure.ocr.cloud_plate_readers import PlateRecognizerSnapshotReader

    # Probamos el fallback directo (evita contaminación del .env real del repo).
    assert PlateRecognizerSnapshotReader._token_from_appdata() == "appdata-token"