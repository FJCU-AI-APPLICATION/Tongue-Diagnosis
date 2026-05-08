from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend import pipeline as pl
from backend.app import create_app


def _jpeg_bytes(shape=(50, 50, 3)) -> bytes:
    img = np.zeros(shape, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _client(monkeypatch):
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model":"x","temperature":0.0,"max_tokens":100,"top_p":0.9})
    monkeypatch.setattr(pl.client, "run", lambda **_: "DOCTOR")
    app = create_app()
    # Inject a dummy registry into app.state so the route can find one
    app.state.registry = type("R", (), {"heads": [], "detector": None})()
    return TestClient(app)


def test_analyze_returns_200_with_expected_keys(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= {"predictions_block", "heads", "comment", "disclaimer", "timing_ms"}


def test_analyze_400_on_missing_file(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/analyze")
    assert r.status_code in (400, 422)  # FastAPI returns 422 for missing required form


def test_analyze_400_on_corrupt_bytes(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("t.jpg", b"not an image", "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 400
    assert "could not decode" in r.json()["error"]


def test_analyze_413_on_oversize_file(monkeypatch):
    client = _client(monkeypatch)
    big = b"\x00" * (10 * 1024 * 1024 + 1)
    files = {"file": ("t.jpg", big, "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 413
