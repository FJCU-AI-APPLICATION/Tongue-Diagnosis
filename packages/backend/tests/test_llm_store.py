"""Tests for LLMStore: validation rules and persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from tongue_backend.stores.llm_store import LLMStore, LLMValidationError


VALID_YAML = """
model: gemini-2.5-flash
temperature: 0.7
max_tokens: 1024
top_p: 0.95
"""


def _store(tmp_path: Path) -> LLMStore:
    default = tmp_path / "default.yaml"
    default.write_text(VALID_YAML, encoding="utf-8")
    current = tmp_path / "current.yaml"
    return LLMStore(default_path=default, current_path=current)


def test_load_returns_dict_with_expected_keys(tmp_path: Path):
    cfg = _store(tmp_path).load_current()
    assert cfg["model"] == "gemini-2.5-flash"
    assert cfg["temperature"] == 0.7


def test_save_validates_temperature_range(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="temperature"):
        s.save("model: x\ntemperature: 3.0\nmax_tokens: 10\ntop_p: 0.5\n")


def test_save_validates_max_tokens_positive(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        s.save("model: x\ntemperature: 0.5\nmax_tokens: 0\ntop_p: 0.5\n")


def test_save_validates_model_non_empty(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="model"):
        s.save("model: ''\ntemperature: 0.5\nmax_tokens: 10\ntop_p: 0.5\n")


def test_save_rejects_yaml_parse_error(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="parse"):
        s.save("model: x\ntemperature: : :\n")


def test_save_persists_valid_yaml(tmp_path: Path):
    s = _store(tmp_path)
    s.save("model: gemini-2.5-flash\ntemperature: 0.2\nmax_tokens: 256\ntop_p: 0.9\n")
    cfg = s.load_current()
    assert cfg["temperature"] == 0.2
    assert cfg["max_tokens"] == 256
