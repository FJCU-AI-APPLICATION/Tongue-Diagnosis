"""Tests for the thin httpx API client (mocked transport via respx)."""
from __future__ import annotations

import httpx
import pytest
import respx

from tongue_frontend.api import APIClient


@respx.mock
def test_health_returns_dict():
    respx.get("http://localhost:8000/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "ai_version": "0.2.0"})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.health()
    assert out["status"] == "ok"


@respx.mock
def test_analyze_posts_multipart_and_returns_dict():
    route = respx.post("http://localhost:8000/api/analyze").mock(
        return_value=httpx.Response(200, json={"comment": "OK", "user_message": "x", "heads": []})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.analyze(filename="x.jpg", content=b"fake", content_type="image/jpeg")
    assert route.called
    assert out["comment"] == "OK"


@respx.mock
def test_get_config_section_returns_content():
    respx.get("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"section": "prompt", "content": "X", "is_default": True})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.get_config("prompt")
    assert out["content"] == "X"


@respx.mock
def test_put_config_section_sends_content_body():
    route = respx.put("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"saved": True})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.put_config("prompt", "EDITED")
    assert route.called
    assert out["saved"] is True


@respx.mock
def test_reload_registry_returns_loaded_failed():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(200, json={"loaded": ["front"], "failed": {}})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.reload_registry()
    assert out["loaded"] == ["front"]
