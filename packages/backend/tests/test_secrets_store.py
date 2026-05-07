import hashlib

import pytest

from tongue_backend.models import ApiKeyStatus
from tongue_backend.stores import secrets_store


@pytest.fixture
def tmp_secret(tmp_path, monkeypatch):
    f = tmp_path / "gemini_api_key"
    monkeypatch.setattr(secrets_store, "GEMINI_API_KEY_FILE", f)
    return f


def test_fingerprint_is_first_8_hex_of_sha256():
    key = "AIzaSyD_example_key_value_xyz_123"
    expected = hashlib.sha256(key.encode()).hexdigest()[:8]
    assert secrets_store.fingerprint(key) == expected


def test_status_when_file_missing(tmp_secret):
    assert tmp_secret.exists() is False
    s = secrets_store.status()
    assert s == ApiKeyStatus(is_set=False, fingerprint=None)


def test_status_when_file_whitespace_only(tmp_secret):
    tmp_secret.write_text("   \n  ")
    assert secrets_store.status() == ApiKeyStatus(is_set=False, fingerprint=None)


def test_load_api_key_returns_none_when_missing(tmp_secret):
    assert secrets_store.load_api_key() is None


def test_load_api_key_strips_trailing_whitespace(tmp_secret):
    tmp_secret.write_text("AIzaSyD_example_key_value_xyz_123\n")
    assert secrets_store.load_api_key() == "AIzaSyD_example_key_value_xyz_123"


def test_clear_is_idempotent(tmp_secret):
    secrets_store.clear()  # missing file: no-op
    tmp_secret.write_text("AIzaSyD_example_key_value_xyz_123")
    secrets_store.clear()
    assert tmp_secret.exists() is False
    secrets_store.clear()  # again: still no-op
