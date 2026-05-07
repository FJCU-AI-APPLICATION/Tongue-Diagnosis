"""ADK Gemini client wrapper.

Encapsulates the boundary so tests can mock at `_call_adk`. The real `_call_adk`
imports `google.adk` lazily so a missing/unauthorised SDK doesn't crash imports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM backend is unreachable due to credentials/config."""


def _retry_count() -> int:
    return int(os.environ.get("LLM_RETRIES", "1"))


@dataclass
class LLMClient:
    """Wraps Google ADK to send (system, user, config) and receive text."""

    def run(self, *, system: str, user: str, config: dict[str, Any]) -> str:
        attempts = _retry_count() + 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                resp = self._call_adk(system, user, config)
                text = (resp.text or "").strip() if resp is not None else ""
                if not text:
                    return "⚠ 醫師建議產生失敗：模型未產生回應"
                return text
            except LLMUnavailableError:
                raise
            except Exception as exc:  # network / 5xx / timeouts
                last_exc = exc
                continue
        msg = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
        return f"⚠ 醫師建議產生失敗：{msg}"

    def _call_adk(self, system: str, user: str, config: dict[str, Any]):
        # Lazy import: keeps imports fast and avoids hard fail when ADK auth missing.
        try:
            from google import adk  # type: ignore
        except Exception as exc:
            raise LLMUnavailableError(f"google-adk import failed: {exc}") from exc

        # NOTE for the implementer: the precise ADK API can vary by version.
        # If `adk.Agent` is the entry point in the installed version, build:
        #   agent = adk.Agent(model=config["model"], system_instruction=system,
        #                     temperature=config["temperature"], top_p=config["top_p"],
        #                     max_output_tokens=config["max_tokens"])
        #   resp = agent.run(user)
        #   return resp  # must expose .text
        # If a newer/older API surface differs, adjust this single method only.
        try:
            agent = adk.Agent(  # type: ignore[attr-defined]
                model=config["model"],
                system_instruction=system,
                temperature=config["temperature"],
                top_p=config.get("top_p", 1.0),
                max_output_tokens=config["max_tokens"],
            )
            return agent.run(user)
        except AttributeError as exc:
            raise LLMUnavailableError(f"unexpected ADK API surface: {exc}") from exc
