"""Thin wrapper around google-genai — call run(system, user, config) → text."""

from __future__ import annotations

from tongue_backend.models import LLMConfig


ERROR_STAMP = "⚠ 醫師建議產生失敗："


def _make_client():
    """Indirection so tests can monkeypatch without importing google-genai."""
    from google import genai  # type: ignore[import-not-found]

    return genai.Client()


def run(*, system: str, user: str, config: LLMConfig) -> str:
    """Call Gemini with the locked system prompt + user message."""
    from google.genai import types  # type: ignore[import-not-found]

    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        top_p=config.top_p,
    )
    try:
        client = _make_client()
        resp = client.models.generate_content(
            model=config.model,
            contents=user,
            config=gen_config,
        )
        text = resp.text
    except Exception as exc:
        return f"{ERROR_STAMP}{type(exc).__name__}: {exc}"

    if not text or not str(text).strip():
        return f"{ERROR_STAMP}模型未產生回應"
    return str(text)
