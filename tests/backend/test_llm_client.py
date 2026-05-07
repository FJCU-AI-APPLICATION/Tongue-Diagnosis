import pytest
from unittest.mock import MagicMock

from backend.llm import client as llm_client
from backend.models import LLMConfig
from backend.stores import secrets_store


def _cfg(**kwargs) -> LLMConfig:
    """Build an LLMConfig with defaults the tests can rely on."""
    return LLMConfig(
        model=kwargs.get("model", "gemini-2.0-flash"),
        temperature=kwargs.get("temperature", 0.2),
        max_tokens=kwargs.get("max_tokens", 1024),
        top_p=kwargs.get("top_p", 0.9),
    )


def _fake_client_returning(text: str) -> MagicMock:
    fake = MagicMock()
    fake.models.generate_content.return_value = MagicMock(text=text)
    return fake


def test_run_calls_sdk_with_system_user_and_config(monkeypatch):
    fake_client = _fake_client_returning("MOCK COMMENT")
    monkeypatch.setattr(llm_client, "_make_client", lambda: fake_client)

    out = llm_client.run(system="SYS", user="USR", config=_cfg())

    assert out == "MOCK COMMENT"
    fake_client.models.generate_content.assert_called_once()
    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.0-flash"
    assert call.kwargs["contents"] == "USR"
    gen_cfg = call.kwargs["config"]
    assert gen_cfg.system_instruction == "SYS"
    assert gen_cfg.temperature == 0.2
    assert gen_cfg.max_output_tokens == 1024
    assert gen_cfg.top_p == 0.9


def test_run_returns_error_stamp_on_exception(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = TimeoutError("upstream timeout")
    monkeypatch.setattr(llm_client, "_make_client", lambda: fake_client)

    out = llm_client.run(system="x", user="y", config=_cfg(model="m", temperature=0.0, max_tokens=1, top_p=1.0))
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "upstream timeout" in out


def test_run_returns_error_stamp_on_empty_response(monkeypatch):
    fake_client = _fake_client_returning("")
    monkeypatch.setattr(llm_client, "_make_client", lambda: fake_client)

    out = llm_client.run(system="x", user="y", config=_cfg(model="m", temperature=0.0, max_tokens=1, top_p=1.0))
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "模型未產生回應" in out


def test_make_client_uses_stored_key(monkeypatch):
    monkeypatch.setattr(secrets_store, "load_api_key", lambda: "STORED_KEY")
    captured = {}

    class FakeGenai:
        def Client(self, api_key):
            captured["api_key"] = api_key
            return MagicMock()

    monkeypatch.setattr(llm_client, "_genai_module", lambda: FakeGenai())
    llm_client._make_client()
    assert captured["api_key"] == "STORED_KEY"


def test_make_client_falls_back_to_google_api_key_env(monkeypatch):
    monkeypatch.setattr(secrets_store, "load_api_key", lambda: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "ENV_GOOGLE")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    captured = {}

    class FakeGenai:
        def Client(self, api_key):
            captured["api_key"] = api_key
            return MagicMock()

    monkeypatch.setattr(llm_client, "_genai_module", lambda: FakeGenai())
    llm_client._make_client()
    assert captured["api_key"] == "ENV_GOOGLE"


def test_make_client_falls_back_to_gemini_api_key_env(monkeypatch):
    monkeypatch.setattr(secrets_store, "load_api_key", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "ENV_GEMINI")
    captured = {}

    class FakeGenai:
        def Client(self, api_key):
            captured["api_key"] = api_key
            return MagicMock()

    monkeypatch.setattr(llm_client, "_genai_module", lambda: FakeGenai())
    llm_client._make_client()
    assert captured["api_key"] == "ENV_GEMINI"


def test_make_client_raises_runtime_error_when_no_key(monkeypatch):
    monkeypatch.setattr(secrets_store, "load_api_key", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        llm_client._make_client()
    assert "尚未設定 Gemini API key" in str(e.value)


def test_run_surfaces_missing_key_error_in_comment_field(monkeypatch):
    """End-to-end: client.run wraps the RuntimeError into the standard ⚠ stamp."""
    monkeypatch.setattr(secrets_store, "load_api_key", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = llm_client.run(system="x", user="y", config=_cfg(model="m", temperature=0.0, max_tokens=1, top_p=1.0))
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "尚未設定 Gemini API key" in out
