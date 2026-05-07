"""Tests for POST /api/analyze."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.app import create_app


def _jpeg_bytes() -> bytes:
    img = np.full((40, 40, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _wire_state(app):
    head = MagicMock()
    head.name = "front"
    head.predict.return_value = HeadResult(
        task="front", head_type="single", predictions=[ClassScore("淡紅", 0.85)],
    )
    app.state.registry = Registry(
        detector_enabled=False,
        heads=[head],
        category_map={"front": {"淡紅": "舌色"}},
    )
    app.state.prompt_store = MagicMock()
    app.state.prompt_store.load_current.return_value = "LOCKED"
    app.state.llm_store = MagicMock()
    app.state.llm_store.load_current.return_value = {
        "model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0,
    }
    app.state.llm_client = MagicMock()
    app.state.llm_client.run.return_value = "## 主要中醫體質\n氣虛"


def test_analyze_returns_200_with_expected_keys():
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app)
        r = client.post("/api/analyze", files={"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "user_message" in body and "heads" in body and "comment" in body
    assert body["disclaimer"]
    assert body["heads"][0]["task"] == "front"


def test_analyze_returns_400_when_image_corrupt():
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app)
        r = client.post("/api/analyze", files={"file": ("bad.jpg", b"junk", "image/jpeg")})
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]["error"]


def test_analyze_returns_413_when_image_too_large():
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app)
        huge = b"\xff" * (10 * 1024 * 1024 + 1)
        r = client.post("/api/analyze", files={"file": ("big.jpg", huge, "image/jpeg")})
    assert r.status_code == 413


def test_analyze_returns_503_when_registry_missing():
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app)
        app.state.registry = None
        r = client.post("/api/analyze", files={"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 503
