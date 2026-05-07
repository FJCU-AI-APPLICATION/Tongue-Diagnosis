import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.stores import llm_store, prompt_store, registry_store, secrets_store


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    pd = tmp_path / "system.default.md"; pc = tmp_path / "system.current.md"
    ld = tmp_path / "llm.default.yaml"; lc = tmp_path / "llm.current.yaml"
    rd = tmp_path / "registry.default.yaml"; rc = tmp_path / "registry.current.yaml"
    sf = tmp_path / "gemini_api_key"

    pd.write_text("DEFAULT PROMPT")
    ld.write_text("model: gemini-2.5-flash\ntemperature: 0.2\nmax_tokens: 1024\ntop_p: 0.9\n")
    (tmp_path / "tongue_color.onnx").write_bytes(b"")
    rd.write_text(
        "heads:\n"
        "  - task: 舌色\n    head_type: single\n    onnx_path: tongue_color.onnx\n"
        "    input_size: [224, 224]\n    normalise:\n      mean: [0,0,0]\n      std: [1,1,1]\n"
        "    class_names: ['A','B']\n"
    )

    monkeypatch.setattr(prompt_store, "PROMPT_DEFAULT", pd)
    monkeypatch.setattr(prompt_store, "PROMPT_CURRENT", pc)
    monkeypatch.setattr(llm_store, "LLM_DEFAULT", ld)
    monkeypatch.setattr(llm_store, "LLM_CURRENT", lc)
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", rd)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", rc)
    monkeypatch.setattr(secrets_store, "GEMINI_API_KEY_FILE", sf)
    monkeypatch.setattr(secrets_store, "SECRETS_DIR", tmp_path)

    return TestClient(create_app()), sf


def test_get_api_key_returns_unset_when_missing(app_client):
    client, _ = app_client
    r = client.get("/api/config/api_key")
    assert r.status_code == 200
    assert r.json() == {"is_set": False, "fingerprint": None}


def test_put_api_key_happy_path(app_client, monkeypatch):
    client, sf = app_client
    monkeypatch.setattr(secrets_store, "_live_test", lambda _k: None)

    r = client.put("/api/config/api_key", json={"content": "AIzaSyD_example_key_value_xyz_123"})

    assert r.status_code == 200
    body = r.json()
    assert body["is_set"] is True
    assert body["fingerprint"] is not None and len(body["fingerprint"]) == 8
    assert sf.read_text() == "AIzaSyD_example_key_value_xyz_123"

    # GET reflects state
    r2 = client.get("/api/config/api_key")
    assert r2.json()["is_set"] is True


def test_put_api_key_bad_format(app_client, monkeypatch):
    client, sf = app_client
    called = {"n": 0}
    monkeypatch.setattr(secrets_store, "_live_test", lambda _k: called.update(n=called["n"] + 1))

    r = client.put("/api/config/api_key", json={"content": "short"})

    assert r.status_code == 422
    assert "金鑰格式不合法" in r.json()["error"]
    assert called["n"] == 0
    assert sf.exists() is False


def test_put_api_key_live_test_failure(app_client, monkeypatch):
    client, sf = app_client

    def boom(_k):
        raise secrets_store.LiveTestError("ValueError: invalid api key")
    monkeypatch.setattr(secrets_store, "_live_test", boom)

    r = client.put("/api/config/api_key", json={"content": "AIzaSyD_example_key_value_xyz_123"})

    assert r.status_code == 422
    assert "金鑰測試失敗" in r.json()["error"]
    assert "invalid api key" in r.json()["error"]
    assert sf.exists() is False


def test_delete_api_key_idempotent(app_client, monkeypatch):
    client, sf = app_client
    monkeypatch.setattr(secrets_store, "_live_test", lambda _k: None)

    # First delete on missing file
    r = client.delete("/api/config/api_key")
    assert r.status_code == 200
    assert r.json() == {"is_set": False, "fingerprint": None}

    # Save then delete
    client.put("/api/config/api_key", json={"content": "AIzaSyD_example_key_value_xyz_123"})
    assert sf.exists()
    r = client.delete("/api/config/api_key")
    assert r.status_code == 200
    assert r.json()["is_set"] is False
    assert sf.exists() is False
