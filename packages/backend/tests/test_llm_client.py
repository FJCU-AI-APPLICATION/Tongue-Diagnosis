from unittest.mock import MagicMock

import pytest

from tongue_backend.llm import client as llm_client


def test_run_calls_sdk_with_system_user_and_config(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.return_value = "MOCK COMMENT"
    fake_agent_factory = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(llm_client, "_make_agent", fake_agent_factory)

    out = llm_client.run(
        system="SYS",
        user="USR",
        config={"model": "gemini-2.0-flash", "temperature": 0.2, "max_tokens": 1024, "top_p": 0.9},
    )
    assert out == "MOCK COMMENT"
    fake_agent_factory.assert_called_once_with(
        model="gemini-2.0-flash", instructions="SYS",
        temperature=0.2, max_tokens=1024, top_p=0.9,
    )
    fake_agent.run.assert_called_once_with("USR")


def test_run_returns_error_stamp_on_exception(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.side_effect = TimeoutError("upstream timeout")
    monkeypatch.setattr(llm_client, "_make_agent", lambda **_: fake_agent)

    out = llm_client.run(
        system="x", user="y",
        config={"model": "m", "temperature": 0.0, "max_tokens": 1, "top_p": 1.0},
    )
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "upstream timeout" in out


def test_run_returns_error_stamp_on_empty_response(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.return_value = ""
    monkeypatch.setattr(llm_client, "_make_agent", lambda **_: fake_agent)

    out = llm_client.run(
        system="x", user="y",
        config={"model": "m", "temperature": 0.0, "max_tokens": 1, "top_p": 1.0},
    )
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "模型未產生回應" in out
