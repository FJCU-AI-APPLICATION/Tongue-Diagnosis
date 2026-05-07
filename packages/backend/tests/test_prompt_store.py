"""Tests for PromptStore: load_current, save, reset, first-boot copy."""
from __future__ import annotations

from pathlib import Path

import pytest

from tongue_backend.stores.prompt_store import PromptStore


def test_first_boot_copies_default_to_current(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("hello", encoding="utf-8")
    current = tmp_path / "current.md"
    store = PromptStore(default_path=default, current_path=current)

    text = store.load_current()
    assert text == "hello"
    assert current.read_text(encoding="utf-8") == "hello"


def test_save_writes_to_current(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("default", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("placeholder", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)

    store.save("new prompt")
    assert current.read_text(encoding="utf-8") == "new prompt"


def test_reset_overwrites_current_with_default(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("default", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("user edits", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)

    store.reset()
    assert current.read_text(encoding="utf-8") == "default"


def test_is_default_when_current_matches_default(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("X", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("X", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)
    assert store.is_default() is True

    store.save("Y")
    assert store.is_default() is False
