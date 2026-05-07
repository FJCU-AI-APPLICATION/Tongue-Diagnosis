"""Read/write the system-prompt 'current' file with reset semantics."""

from __future__ import annotations

from pathlib import Path

from backend.models import ConfigStatus
from backend.stores.paths import PROMPT_CURRENT, PROMPT_DEFAULT


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


def status() -> ConfigStatus:
    content = load_current()
    return ConfigStatus(
        content=content,
        is_default=content == PROMPT_DEFAULT.read_text(),
        mtime=PROMPT_CURRENT.stat().st_mtime,
    )
