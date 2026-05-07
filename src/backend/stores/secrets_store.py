"""Operator-set Gemini API key — file I/O, fingerprint, validation, live test."""

from __future__ import annotations

import hashlib
import os

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from backend.models import ApiKeyStatus
from backend.stores.paths import GEMINI_API_KEY_FILE, SECRETS_DIR


class ValidationError(ValueError):
    """Raised by save() when the key shape is invalid."""


class _ApiKeyValue(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    value: str = Field(min_length=20, pattern=r"^[A-Za-z0-9_-]+$")


def _validate_format(content: str) -> str:
    """Return the stripped/validated key. Raises ValidationError on bad shape."""
    try:
        return _ApiKeyValue(value=content).value
    except PydanticValidationError as e:
        first = e.errors()[0]
        raise ValidationError(f"金鑰格式不合法: {first['msg']}") from e


def fingerprint(key: str) -> str:
    """First 8 hex chars of sha256(key) — safe to log; useful for 'did my save take?'."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def load_api_key() -> str | None:
    """Return the stored key (trailing whitespace stripped), or None if the file is missing or whitespace-only."""
    if not GEMINI_API_KEY_FILE.exists():
        return None
    raw = GEMINI_API_KEY_FILE.read_text().strip()
    return raw or None


def status() -> ApiKeyStatus:
    key = load_api_key()
    if key is None:
        return ApiKeyStatus(is_set=False, fingerprint=None)
    return ApiKeyStatus(is_set=True, fingerprint=fingerprint(key))


def clear() -> None:
    GEMINI_API_KEY_FILE.unlink(missing_ok=True)


class LiveTestError(RuntimeError):
    """Raised when the live Gemini test call fails."""


def _load_llm_config():
    """Indirection so tests don't need a real llm.yaml on disk."""
    from backend.stores import llm_store

    return llm_store.load_current()


def _make_test_client(key: str):
    """Indirection so tests can replace with a fake without importing google-genai."""
    from google import genai  # type: ignore[import-not-found]

    return genai.Client(api_key=key)


def _live_test(key: str) -> None:
    """Call Gemini with the candidate key + currently configured model.
    Raises LiveTestError on any exception."""
    from google.genai import types  # type: ignore[import-not-found]

    try:
        cfg = _load_llm_config()
        gen_config = types.GenerateContentConfig(max_output_tokens=1)
        client = _make_test_client(key)
        client.models.generate_content(
            model=cfg.model,
            contents="ping",
            config=gen_config,
        )
    except Exception as exc:
        raise LiveTestError(f"{type(exc).__name__}: {exc}") from exc


def save(content: str) -> None:
    """Validate format → live-test against Gemini → write 0o600.

    Raises ``ValidationError`` on format failure, ``LiveTestError`` on Gemini
    failure. The file is NOT written in either failure case.
    """
    candidate = _validate_format(content)
    _live_test(candidate)
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(GEMINI_API_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, candidate.encode())
    finally:
        os.close(fd)
