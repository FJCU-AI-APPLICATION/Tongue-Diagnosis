"""Tests for FrontendSettings — env-driven configuration."""

from __future__ import annotations

import pytest

from frontend.settings import FrontendSettings


def test_defaults_when_no_env(monkeypatch):
    # Clear any inherited env so we exercise the documented defaults
    for var in ("TONGUE_BACKEND_URL", "TONGUE_BACKEND_TIMEOUT",
                "GRADIO_SERVER_NAME", "GRADIO_SERVER_PORT"):
        monkeypatch.delenv(var, raising=False)

    s = FrontendSettings(_env_file=None)
    assert s.backend_url == "http://localhost:8000"
    assert s.backend_timeout == 60.0
    assert s.gradio_server_name == "0.0.0.0"
    assert s.gradio_server_port == 7860


def test_tongue_envs_override(monkeypatch):
    monkeypatch.setenv("TONGUE_BACKEND_URL", "http://my-backend:9000")
    monkeypatch.setenv("TONGUE_BACKEND_TIMEOUT", "120")
    s = FrontendSettings()
    assert s.backend_url == "http://my-backend:9000"
    assert s.backend_timeout == 120.0


def test_gradio_envs_override(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8080")
    s = FrontendSettings()
    assert s.gradio_server_name == "127.0.0.1"
    assert s.gradio_server_port == 8080


def test_invalid_port_raises(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_PORT", "not-a-port")
    with pytest.raises(Exception):  # pydantic raises ValidationError
        FrontendSettings()
