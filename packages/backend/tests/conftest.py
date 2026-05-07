"""Shared fixtures for backend tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_default_prompt(tmp_path: Path) -> Path:
    p = tmp_path / "system.default.md"
    p.write_text("DEFAULT PROMPT", encoding="utf-8")
    return p


@pytest.fixture
def fake_current_prompt(tmp_path: Path) -> Path:
    return tmp_path / "system.current.md"
