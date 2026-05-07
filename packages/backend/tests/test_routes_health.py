"""Tests for /health."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from tongue_backend.app import create_app


def test_health_returns_status_ai_version_registry_summary():
    app = create_app()
    with TestClient(app) as client:
        app.state.registry = MagicMock(heads=[1, 2], failed_heads={"x": "boom"})
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "ai_version" in body
    assert body["registry"]["loaded"] == 2
    assert body["registry"]["failed"] == ["x"]


def test_health_when_registry_is_none():
    app = create_app()
    with TestClient(app) as client:
        app.state.registry = None
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["registry"]["loaded"] == 0
