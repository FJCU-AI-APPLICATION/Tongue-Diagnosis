import pytest

from backend.stores import llm_store


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "llm.default.yaml"
    current = tmp_path / "llm.current.yaml"
    default.write_text(
        "model: gemini-2.0-flash\ntemperature: 0.2\nmax_tokens: 2048\ntop_p: 0.9\n"
    )
    monkeypatch.setattr(llm_store, "LLM_DEFAULT", default)
    monkeypatch.setattr(llm_store, "LLM_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    cfg = llm_store.load_current()
    assert cfg.model == "gemini-2.0-flash"
    assert cfg.temperature == 0.2


def test_save_then_load_round_trip(tmp_paths):
    llm_store.save("model: gemini-2.0-pro\ntemperature: 0.5\nmax_tokens: 1024\ntop_p: 0.9\n")
    cfg = llm_store.load_current()
    assert cfg.model == "gemini-2.0-pro"


def test_save_rejects_invalid_temperature(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("model: x\ntemperature: 99\nmax_tokens: 1024\ntop_p: 0.9\n")
    assert "temperature" in str(e.value)


def test_save_rejects_malformed_yaml(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("model: : :")
    assert "yaml" in str(e.value).lower()


def test_save_rejects_missing_model(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("temperature: 0.2\nmax_tokens: 1024\ntop_p: 0.9\n")
    assert "model" in str(e.value)


def test_reset_restores_default(tmp_paths):
    llm_store.save("model: x\ntemperature: 0.5\nmax_tokens: 1024\ntop_p: 0.9\n")
    llm_store.reset()
    cfg = llm_store.load_current()
    assert cfg.model == "gemini-2.0-flash"
