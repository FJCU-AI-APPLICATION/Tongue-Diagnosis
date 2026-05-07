"""Thin wrapper around google-genai — call run(system, user, config) → text."""

from __future__ import annotations

import os

from backend.models import LLMConfig
from backend.stores import secrets_store


ERROR_STAMP = "⚠ 醫師建議產生失敗："


def _genai_module():
    """Indirection so tests can monkeypatch without importing google-genai."""
    from google import genai  # type: ignore[import-not-found]

    return genai


def _resolve_api_key() -> str:
    """Stored key takes precedence; fall back to env. Raises if neither is set."""
    key = secrets_store.load_api_key()
    if key:
        return key
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError("尚未設定 Gemini API key")


def _make_client():
    """Build a configured genai client. Raises RuntimeError if no key is available."""
    key = _resolve_api_key()
    return _genai_module().Client(api_key=key)


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
