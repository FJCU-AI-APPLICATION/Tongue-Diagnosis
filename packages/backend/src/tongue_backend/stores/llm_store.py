"""LLM config: YAML I/O + validation + reset."""

from __future__ import annotations

from pathlib import Path

import yaml

from tongue_backend.stores.paths import LLM_DEFAULT, LLM_CURRENT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


def _ensure_current() -> Path:
    if not LLM_CURRENT.exists():
        LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        LLM_CURRENT.write_text(LLM_DEFAULT.read_text())
    return LLM_CURRENT


def load_current() -> dict:
    text = _ensure_current().read_text()
    return yaml.safe_load(text) or {}


def save(content: str) -> None:
    """Persist a YAML string after validation."""
    parsed = _validate(content)
    LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    LLM_CURRENT.write_text(content)
    return parsed


def reset() -> None:
    LLM_CURRENT.write_text(LLM_DEFAULT.read_text())


def status() -> dict:
    content = _ensure_current().read_text()
    return {
        "content": content,
        "is_default": content == LLM_DEFAULT.read_text(),
        "mtime": LLM_CURRENT.stat().st_mtime,
    }


def _validate(content: str) -> dict:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("LLM config root must be a mapping")
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("model must be a non-empty string")
    temp = data.get("temperature", 0.0)
    if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
        raise ValidationError("temperature must be a number in [0, 2]")
    max_tokens = data.get("max_tokens", 1024)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValidationError("max_tokens must be a positive integer")
    top_p = data.get("top_p", 1.0)
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValidationError("top_p must be in (0, 1]")
    return data
