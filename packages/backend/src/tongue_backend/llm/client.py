"""Thin wrapper around Google ADK — call run(system, user, config) → text."""

from __future__ import annotations

from tongue_backend.models import LLMConfig


ERROR_STAMP = "⚠ 醫師建議產生失敗："


def _make_agent(*, model: str, instructions: str, temperature: float,
                max_tokens: int, top_p: float):
    """Indirection so tests can monkeypatch without importing google-adk."""
    from google.adk.agents import Agent  # type: ignore[import-not-found]

    return Agent(
        model=model,
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


def run(*, system: str, user: str, config: LLMConfig) -> str:
    """Call Gemini through ADK with the locked system prompt + user message."""
    try:
        agent = _make_agent(
            model=config.model,
            instructions=system,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        text = agent.run(user)
    except Exception as exc:
        return f"{ERROR_STAMP}{type(exc).__name__}: {exc}"

    if not text or not str(text).strip():
        return f"{ERROR_STAMP}模型未產生回應"
    return str(text)
