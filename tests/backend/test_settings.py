"""Tests for BackendSettings — env-driven configuration."""

from __future__ import annotations

import pytest

from backend.settings import BackendSettings


def test_defaults_when_no_env():
    s = BackendSettings(_env_file=None)  # ignore any local .env
    assert s.max_upload_mb == 10
    assert s.max_upload_bytes == 10 * 1024 * 1024


def test_max_upload_mb_overridable_via_env(monkeypatch):
    monkeypatch.setenv("TONGUE_MAX_UPLOAD_MB", "25")
    s = BackendSettings()
    assert s.max_upload_mb == 25
    assert s.max_upload_bytes == 25 * 1024 * 1024


def test_alias_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("tongue_max_upload_mb", "7")
    s = BackendSettings()
    assert s.max_upload_mb == 7


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("TONGUE_MAX_UPLOAD_MB", "not-a-number")
    with pytest.raises(Exception):  # pydantic raises ValidationError
        BackendSettings()
