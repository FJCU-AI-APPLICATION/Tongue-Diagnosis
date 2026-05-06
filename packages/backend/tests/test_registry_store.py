import pytest

from tongue_backend.stores import registry_store


VALID_YAML = """
detector: null
heads:
  - task: 舌色
    head_type: single
    onnx_path: tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["A","B"]
"""


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "registry.default.yaml"
    current = tmp_path / "registry.current.yaml"
    default.write_text(VALID_YAML)
    (tmp_path / "tongue_color.onnx").write_bytes(b"")
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", default)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    text = registry_store.load_current_text()
    assert "舌色" in text


def test_save_persists_valid_yaml(tmp_paths):
    registry_store.save(VALID_YAML)
    assert "舌色" in registry_store.load_current_text()


def test_save_rejects_malformed_yaml(tmp_paths):
    with pytest.raises(registry_store.ValidationError):
        registry_store.save("heads: : bad")


def test_save_rejects_missing_keys(tmp_paths):
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save("""
heads:
  - task: 舌色
    head_type: single
""")
    assert "onnx_path" in str(e.value)


def test_save_rejects_missing_onnx_file(tmp_paths):
    bad = VALID_YAML.replace("tongue_color.onnx", "missing.onnx")
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save(bad)
    assert "missing.onnx" in str(e.value)
