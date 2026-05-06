"""Read/write the system-prompt 'current' file with reset semantics."""

from __future__ import annotations

from pathlib import Path

from tongue_backend.stores.paths import PROMPT_DEFAULT, PROMPT_CURRENT


def _ensure_current() -> Path:
    """If current is missing, copy from default."""
    if not PROMPT_CURRENT.exists():
        PROMPT_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_CURRENT.write_text(PROMPT_DEFAULT.read_text())
    return PROMPT_CURRENT


def load_current() -> str:
    return _ensure_current().read_text()


def save(content: str) -> None:
    PROMPT_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_CURRENT.write_text(content)


def reset() -> None:
    save(PROMPT_DEFAULT.read_text())


def status() -> dict:
    content = load_current()
    return {
        "content": content,
        "is_default": content == PROMPT_DEFAULT.read_text(),
        "mtime": PROMPT_CURRENT.stat().st_mtime,
    }
