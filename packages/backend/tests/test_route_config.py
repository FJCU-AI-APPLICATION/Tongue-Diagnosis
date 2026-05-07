import pytest
from fastapi.testclient import TestClient

from tongue_backend.app import create_app
from tongue_backend.stores import llm_store, prompt_store, registry_store


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    # Redirect all default/current paths into tmp_path
    pd = tmp_path / "system.default.md"; pc = tmp_path / "system.current.md"
    ld = tmp_path / "llm.default.yaml"; lc = tmp_path / "llm.current.yaml"
    rd = tmp_path / "registry.default.yaml"; rc = tmp_path / "registry.current.yaml"

    pd.write_text("DEFAULT PROMPT")
    ld.write_text("model: x\ntemperature: 0.2\nmax_tokens: 1024\ntop_p: 0.9\n")
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

    return TestClient(create_app())


def test_get_prompt_returns_default_on_first_read(app_client):
    r = app_client.get("/api/config/prompt")
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "DEFAULT PROMPT"
    assert body["is_default"] is True


def test_put_prompt_persists(app_client):
    r = app_client.put("/api/config/prompt", json={"content": "MY EDIT"})
    assert r.status_code == 200
    r = app_client.get("/api/config/prompt")
    assert r.json()["content"] == "MY EDIT"
    assert r.json()["is_default"] is False


def test_reset_prompt_restores_default(app_client):
    app_client.put("/api/config/prompt", json={"content": "EDIT"})
    r = app_client.post("/api/config/prompt/reset")
    assert r.status_code == 200
    assert app_client.get("/api/config/prompt").json()["content"] == "DEFAULT PROMPT"


def test_put_llm_validates_temperature(app_client):
    bad = "model: x\ntemperature: 99\nmax_tokens: 1024\ntop_p: 0.9\n"
    r = app_client.put("/api/config/llm", json={"content": bad})
    assert r.status_code == 422
    assert "temperature" in r.json()["error"]


def test_put_registry_validates_yaml(app_client):
    r = app_client.put("/api/config/registry", json={"content": "heads: : ::"})
    assert r.status_code == 422


def test_unknown_section_returns_404(app_client):
    r = app_client.get("/api/config/nope")
    assert r.status_code == 404


def test_registry_reload_swaps_app_state(app_client, monkeypatch):
    # Stub the registry loader to avoid loading real ONNX
    from tongue_ai import registry as ai_registry
    monkeypatch.setattr(ai_registry, "_make_onnx_session", lambda _p: object())
    r = app_client.post("/api/config/registry/reload")
    assert r.status_code == 200
    body = r.json()
    assert "loaded" in body and len(body["loaded"]) == 1
