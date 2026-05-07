"""Tests for /api/config/{section} CRUD + reload."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tongue_backend.app import create_app
from tongue_backend.stores.llm_store import LLMStore
from tongue_backend.stores.prompt_store import PromptStore
from tongue_backend.stores.registry_store import RegistryStore


@pytest.fixture
def wired_paths(tmp_path: Path):
    p_default = tmp_path / "system.default.md"
    p_default.write_text("DEFAULT", encoding="utf-8")
    l_default = tmp_path / "llm.default.yaml"
    l_default.write_text("model: x\ntemperature: 0.5\nmax_tokens: 10\ntop_p: 0.9\n", encoding="utf-8")
    r_default = tmp_path / "registry.default.yaml"
    r_default.write_text(
        """
detector: {enabled: false}
heads:
  - name: h
    head_type: single
    arch: resnet50
    weights_uri: local:weights/h.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b]
category_map:
  h: {a: 舌色, b: 舌色}
""",
        encoding="utf-8",
    )
    return tmp_path, p_default, l_default, r_default


def _wire_state(app, p_default, l_default, r_default, tmp_path: Path):
    app.state.prompt_store = PromptStore(p_default, tmp_path / "system.current.md")
    app.state.llm_store = LLMStore(l_default, tmp_path / "llm.current.yaml")
    app.state.registry_store = RegistryStore(r_default, tmp_path / "registry.current.yaml")
    app.state.registry = None


def test_get_prompt_section_returns_content(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        r = client.get("/api/config/prompt")
    assert r.status_code == 200
    assert r.json()["content"] == "DEFAULT"
    assert r.json()["is_default"] is True


def test_put_prompt_persists(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        r = client.put("/api/config/prompt", json={"content": "EDITED"})
        assert r.status_code == 200
        r2 = client.get("/api/config/prompt")
    assert r2.json()["content"] == "EDITED"
    assert r2.json()["is_default"] is False


def test_put_llm_invalid_temperature_returns_422(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        r = client.put(
            "/api/config/llm",
            json={"content": "model: x\ntemperature: 5\nmax_tokens: 10\ntop_p: 0.5\n"},
        )
    assert r.status_code == 422
    assert "temperature" in r.json()["detail"]["error"]


def test_put_registry_invalid_yaml_returns_422(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        r = client.put("/api/config/registry", json={"content": "heads: [::: bad"})
    assert r.status_code == 422


def test_post_reset_restores_default(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        client.put("/api/config/prompt", json={"content": "EDITED"})
        r = client.post("/api/config/prompt/reset")
        assert r.status_code == 200
        r2 = client.get("/api/config/prompt")
    assert r2.json()["content"] == "DEFAULT"


def test_post_registry_reload_returns_loaded_and_failed(wired_paths):
    tmp_path, p_default, l_default, r_default = wired_paths
    app = create_app()
    with TestClient(app) as client:
        _wire_state(app, p_default, l_default, r_default, tmp_path)
        client.put(
            "/api/config/registry",
            json={"content": r_default.read_text(encoding="utf-8")},
        )
        r = client.post("/api/config/registry/reload")
    assert r.status_code == 200
    body = r.json()
    assert "loaded" in body and "failed" in body
    assert "h" in body["failed"]
