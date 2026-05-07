"""Tests for the ADK Gemini client wrapper.

The wrapper is mocked at the ADK boundary; we don't need a live Gemini key.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tongue_backend.llm.client import LLMClient, LLMUnavailableError


def test_run_calls_adk_with_system_user_and_config():
    cfg = {"model": "gemini-2.5-flash", "temperature": 0.5, "max_tokens": 100, "top_p": 0.9}
    client = LLMClient()
    fake_resp = MagicMock(text="OUTPUT")
    with patch.object(client, "_call_adk", return_value=fake_resp) as m:
        out = client.run(system="SYS", user="USR", config=cfg)
    m.assert_called_once_with("SYS", "USR", cfg)
    assert out == "OUTPUT"


def test_run_returns_error_stamped_text_on_network_failure():
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    with patch.object(client, "_call_adk", side_effect=ConnectionError("nope")):
        out = client.run(system="SYS", user="USR", config=cfg)
    assert out.startswith("⚠ 醫師建議產生失敗")
    assert "nope" in out


def test_run_returns_error_when_response_empty():
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    with patch.object(client, "_call_adk", return_value=MagicMock(text="")):
        out = client.run(system="SYS", user="USR", config=cfg)
    assert "未產生回應" in out


def test_run_raises_when_credentials_missing(monkeypatch):
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch.object(client, "_call_adk", side_effect=LLMUnavailableError("no creds")):
        with pytest.raises(LLMUnavailableError):
            client.run(system="SYS", user="USR", config=cfg)
