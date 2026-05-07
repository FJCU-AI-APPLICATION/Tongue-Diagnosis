import hashlib
from unittest.mock import MagicMock

import pytest

from backend.models import ApiKeyStatus
from backend.stores import secrets_store


@pytest.fixture
def tmp_secret(tmp_path, monkeypatch):
    f = tmp_path / "gemini_api_key"
    monkeypatch.setattr(secrets_store, "GEMINI_API_KEY_FILE", f)
    monkeypatch.setattr(secrets_store, "SECRETS_DIR", tmp_path)
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


def test_status_when_file_has_key(tmp_secret):
    key = "AIzaSyD_example_key_value_xyz_123"
    tmp_secret.write_text(key)
    s = secrets_store.status()
    assert s.is_set is True
    assert s.fingerprint == secrets_store.fingerprint(key)


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


def test_validate_format_strips_and_returns_value():
    assert secrets_store._validate_format("  AIzaSyD_example_key_value_xyz_123  ") == \
        "AIzaSyD_example_key_value_xyz_123"


def test_validate_format_rejects_empty():
    with pytest.raises(secrets_store.ValidationError) as e:
        secrets_store._validate_format("")
    assert "金鑰格式不合法" in str(e.value)


def test_validate_format_rejects_short():
    with pytest.raises(secrets_store.ValidationError) as e:
        secrets_store._validate_format("short")
    assert "金鑰格式不合法" in str(e.value)


def test_validate_format_rejects_bad_chars():
    with pytest.raises(secrets_store.ValidationError) as e:
        secrets_store._validate_format("AIzaSyD_with spaces_and_$ymbols_xyz")
    assert "金鑰格式不合法" in str(e.value)


def test_live_test_calls_genai_with_configured_model(monkeypatch):
    from backend.models import LLMConfig
    cfg = LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=1, top_p=0.9)
    monkeypatch.setattr(secrets_store, "_load_llm_config", lambda: cfg)

    captured: dict = {}

    def fake_make_test_client(key: str):
        captured["key"] = key
        fake = MagicMock()
        fake.models.generate_content.return_value = MagicMock(text="ok")
        captured["client"] = fake
        return fake

    monkeypatch.setattr(secrets_store, "_make_test_client", fake_make_test_client)

    secrets_store._live_test("AIzaSyD_example_key_value_xyz_123")

    assert captured["key"] == "AIzaSyD_example_key_value_xyz_123"
    call = captured["client"].models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["contents"] == "ping"
    assert call.kwargs["config"].max_output_tokens == 1


def test_live_test_raises_LiveTestError_on_genai_exception(monkeypatch):
    from backend.models import LLMConfig
    cfg = LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=1, top_p=0.9)
    monkeypatch.setattr(secrets_store, "_load_llm_config", lambda: cfg)

    fake = MagicMock()
    fake.models.generate_content.side_effect = ValueError("invalid api key")
    monkeypatch.setattr(secrets_store, "_make_test_client", lambda _k: fake)

    with pytest.raises(secrets_store.LiveTestError) as e:
        secrets_store._live_test("AIzaSyD_example_key_value_xyz_123")
    assert "invalid api key" in str(e.value)


def test_live_test_wraps_load_llm_config_failure(monkeypatch):
    def boom():
        raise FileNotFoundError("llm.current.yaml missing")
    monkeypatch.setattr(secrets_store, "_load_llm_config", boom)
    monkeypatch.setattr(secrets_store, "_make_test_client", lambda _k: MagicMock())

    with pytest.raises(secrets_store.LiveTestError) as e:
        secrets_store._live_test("AIzaSyD_example_key_value_xyz_123")
    assert "FileNotFoundError" in str(e.value)
    assert "llm.current.yaml missing" in str(e.value)


def test_save_happy_path_writes_0600(tmp_secret, monkeypatch):
    monkeypatch.setattr(secrets_store, "_live_test", lambda _k: None)
    secrets_store.save("AIzaSyD_example_key_value_xyz_123")
    assert tmp_secret.read_text() == "AIzaSyD_example_key_value_xyz_123"
    mode = tmp_secret.stat().st_mode & 0o777
    assert mode == 0o600


def test_save_rejects_bad_format_without_calling_live_test(tmp_secret, monkeypatch):
    called = {"n": 0}

    def boom(_k):
        called["n"] += 1
    monkeypatch.setattr(secrets_store, "_live_test", boom)

    with pytest.raises(secrets_store.ValidationError):
        secrets_store.save("short")
    assert called["n"] == 0
    assert tmp_secret.exists() is False


def test_save_does_not_persist_when_live_test_fails(tmp_secret, monkeypatch):
    def boom(_k):
        raise secrets_store.LiveTestError("ValueError: bad key")
    monkeypatch.setattr(secrets_store, "_live_test", boom)

    with pytest.raises(secrets_store.LiveTestError):
        secrets_store.save("AIzaSyD_example_key_value_xyz_123")
    assert tmp_secret.exists() is False


def test_save_creates_parent_dir(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "deep" / "secrets"
    monkeypatch.setattr(secrets_store, "SECRETS_DIR", secrets_dir)
    monkeypatch.setattr(secrets_store, "GEMINI_API_KEY_FILE", secrets_dir / "gemini_api_key")
    monkeypatch.setattr(secrets_store, "_live_test", lambda _k: None)

    secrets_store.save("AIzaSyD_example_key_value_xyz_123")
    f = secrets_dir / "gemini_api_key"
    assert f.exists()
    assert (f.stat().st_mode & 0o777) == 0o600
