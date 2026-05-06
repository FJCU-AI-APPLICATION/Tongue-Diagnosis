"""Thin wrapper around Google ADK so the rest of the code can call run(system, user, config)."""

from __future__ import annotations

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


def run(*, system: str, user: str, config: dict) -> str:
    try:
        agent = _make_agent(
            model=config["model"],
            instructions=system,
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
            top_p=float(config.get("top_p", 0.9)),
        )
        text = agent.run(user)
    except Exception as exc:
        return f"{ERROR_STAMP}{type(exc).__name__}: {exc}"

    if not text or not str(text).strip():
        return f"{ERROR_STAMP}模型未產生回應"
    return str(text)
