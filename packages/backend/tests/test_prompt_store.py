from pathlib import Path

import pytest

from tongue_backend.stores import prompt_store


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "system.default.md"
    current = tmp_path / "system.current.md"
    default.write_text("DEFAULT PROMPT")
    monkeypatch.setattr(prompt_store, "PROMPT_DEFAULT", default)
    monkeypatch.setattr(prompt_store, "PROMPT_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    default, current = tmp_paths
    text = prompt_store.load_current()
    assert text == "DEFAULT PROMPT"
    assert current.read_text() == "DEFAULT PROMPT"


def test_save_persists_then_load_returns_same(tmp_paths):
    prompt_store.save("MY EDIT")
    assert prompt_store.load_current() == "MY EDIT"


def test_reset_overwrites_current_with_default(tmp_paths):
    prompt_store.save("MY EDIT")
    prompt_store.reset()
    assert prompt_store.load_current() == "DEFAULT PROMPT"


def test_status_reports_is_default_flag(tmp_paths):
    prompt_store.reset()
    status = prompt_store.status()
    assert status.is_default is True
    prompt_store.save("MY EDIT")
    status = prompt_store.status()
    assert status.is_default is False
    assert status.content == "MY EDIT"
