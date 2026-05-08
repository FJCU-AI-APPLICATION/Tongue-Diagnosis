from pathlib import Path

import pytest

from backend.stores import prompt_store


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "system.default.md"
    current = tmp_path / "system.current.md"
    default.write_text("DEFAULT PROMPT\n{{PREDICTIONS}}\n")
    monkeypatch.setattr(prompt_store, "PROMPT_DEFAULT", default)
    monkeypatch.setattr(prompt_store, "PROMPT_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    default, current = tmp_paths
    text = prompt_store.load_current()
    assert text == "DEFAULT PROMPT\n{{PREDICTIONS}}\n"
    assert current.read_text() == "DEFAULT PROMPT\n{{PREDICTIONS}}\n"


def test_save_persists_then_load_returns_same(tmp_paths):
    prompt_store.save("MY EDIT\n{{PREDICTIONS}}\n")
    assert prompt_store.load_current() == "MY EDIT\n{{PREDICTIONS}}\n"


def test_reset_overwrites_current_with_default(tmp_paths):
    prompt_store.save("MY EDIT\n{{PREDICTIONS}}\n")
    prompt_store.reset()
    assert prompt_store.load_current() == "DEFAULT PROMPT\n{{PREDICTIONS}}\n"


def test_status_reports_is_default_flag(tmp_paths):
    prompt_store.reset()
    status = prompt_store.status()
    assert status.is_default is True
    prompt_store.save("MY EDIT\n{{PREDICTIONS}}\n")
    status = prompt_store.status()
    assert status.is_default is False
    assert status.content == "MY EDIT\n{{PREDICTIONS}}\n"


def test_save_rejects_template_without_placeholder(tmp_paths):
    from backend.llm import prompt
    with pytest.raises(prompt.PromptValidationError):
        prompt_store.save("missing the marker")


def test_save_rejects_template_with_two_placeholders(tmp_paths):
    from backend.llm import prompt
    with pytest.raises(prompt.PromptValidationError):
        prompt_store.save("a {{PREDICTIONS}} b {{PREDICTIONS}} c")


def test_save_does_not_write_file_on_validation_failure(tmp_paths):
    from backend.llm import prompt
    _, current = tmp_paths
    current.unlink(missing_ok=True)
    with pytest.raises(prompt.PromptValidationError):
        prompt_store.save("no marker here")
    assert not current.exists()
