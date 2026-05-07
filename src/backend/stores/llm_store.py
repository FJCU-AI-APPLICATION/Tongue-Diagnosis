"""LLM config: YAML I/O + validation + reset, returning typed ``LLMConfig``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from backend.models import ConfigStatus, LLMConfig
from backend.stores.paths import LLM_CURRENT, LLM_DEFAULT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


def _ensure_current() -> Path:
    if not LLM_CURRENT.exists():
        LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        LLM_CURRENT.write_text(LLM_DEFAULT.read_text())
    return LLM_CURRENT


def load_current() -> LLMConfig:
    return _validate(_ensure_current().read_text())


def save(content: str) -> LLMConfig:
    """Persist a YAML string after validation. Returns the parsed config."""
    cfg = _validate(content)
    LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    LLM_CURRENT.write_text(content)
    return cfg


def reset() -> None:
    LLM_CURRENT.write_text(LLM_DEFAULT.read_text())


def status() -> ConfigStatus:
    content = _ensure_current().read_text()
    return ConfigStatus(
        content=content,
        is_default=content == LLM_DEFAULT.read_text(),
        mtime=LLM_CURRENT.stat().st_mtime,
    )


def _validate(content: str) -> LLMConfig:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("LLM config root must be a mapping")
    try:
        return LLMConfig.model_validate(data)
    except PydanticValidationError as e:
        # Surface a single-line, field-prefixed message for the API caller.
        first = e.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        raise ValidationError(f"{loc}: {first['msg']}") from e
