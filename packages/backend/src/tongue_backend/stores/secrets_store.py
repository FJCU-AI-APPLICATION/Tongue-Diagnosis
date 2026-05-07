"""Operator-set Gemini API key — file I/O, fingerprint, validation, live test."""

from __future__ import annotations

import hashlib

from tongue_backend.models import ApiKeyStatus
from tongue_backend.stores.paths import GEMINI_API_KEY_FILE, SECRETS_DIR


def fingerprint(key: str) -> str:
    """First 8 hex chars of sha256(key) — safe to log; useful for 'did my save take?'."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def load_api_key() -> str | None:
    """Return the stored key (whitespace stripped) or None if missing/empty."""
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
