from unittest.mock import MagicMock

from tongue_backend.llm import client as llm_client
from tongue_backend.models import LLMConfig


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
