from pathlib import Path

import pytest

from backend.stores import registry_store
from backend.stores.paths import REGISTRY_DEFAULT


VALID_ONNX_YAML = """
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


VALID_PYTORCH_YAML = """
heads:
  - task: front
    head_type: single
    weights_uri: local:weights/front.pth
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a, b, c]
"""


@pytest.fixture
def tmp_paths(tmp_path: Path, monkeypatch):
    default = tmp_path / "registry.default.yaml"
    current = tmp_path / "registry.current.yaml"
    default.write_text(VALID_ONNX_YAML)
    (tmp_path / "tongue_color.onnx").write_bytes(b"")
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", default)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", current)
    return default, current


@pytest.fixture
def tmp_paths_pytorch(tmp_path: Path, monkeypatch):
    """Variant fixture where the YAML uses local: weights_uri."""
    default = tmp_path / "registry.default.yaml"
    current = tmp_path / "registry.current.yaml"
    default.write_text(VALID_PYTORCH_YAML)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "front.pth").write_bytes(b"")
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", default)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    text = registry_store.load_current_text()
    assert "舌色" in text


def test_save_persists_valid_onnx_yaml(tmp_paths):
    registry_store.save(VALID_ONNX_YAML)
    assert "舌色" in registry_store.load_current_text()


def test_save_persists_valid_pytorch_yaml_local(tmp_paths_pytorch):
    registry_store.save(VALID_PYTORCH_YAML)
    assert "front" in registry_store.load_current_text()


def test_save_accepts_hf_weights_uri_without_network(tmp_paths):
    """Saving a registry whose head uses `weights_uri: hf:...` must succeed
    without attempting an HF Hub download. Network reachability is deferred
    until the reload step."""
    hf_yaml = """
heads:
  - task: front
    head_type: single
    weights_uri: hf:owner/repo/file.pth
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a, b]
"""
    registry_store.save(hf_yaml)
    assert "hf:owner/repo/file.pth" in registry_store.load_current_text()


def test_save_accepts_shipped_default_registry():
    """Regression guard for the bug where the save validator required
    `onnx_path` even though the runtime loader accepts `weights_uri:` as an
    alternative. The shipped default uses `weights_uri:` and must round-trip
    through `save()` without modification.

    Does not monkeypatch the paths — uses the real shipped default file and
    writes to the real `registry.current.yaml`. Snapshots prior current
    content so the test is non-destructive.
    """
    from backend.stores.paths import REGISTRY_CURRENT

    default_text = REGISTRY_DEFAULT.read_text()
    snapshot = REGISTRY_CURRENT.read_text() if REGISTRY_CURRENT.exists() else None
    try:
        registry_store.save(default_text)
        assert REGISTRY_CURRENT.read_text() == default_text
    finally:
        if snapshot is None:
            REGISTRY_CURRENT.unlink(missing_ok=True)
        else:
            REGISTRY_CURRENT.write_text(snapshot)


def test_save_rejects_malformed_yaml(tmp_paths):
    with pytest.raises(registry_store.ValidationError):
        registry_store.save("heads: : bad")


def test_save_rejects_missing_required_keys(tmp_paths):
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save("""
heads:
  - task: 舌色
    head_type: single
""")
    msg = str(e.value)
    # Loader's contract: any of input_size / normalise / class_names missing
    # — error message names at least one of them. Previously this test
    # asserted on `onnx_path` specifically, which is no longer required when
    # `weights_uri` is supplied.
    assert "input_size" in msg or "normalise" in msg or "class_names" in msg


def test_save_rejects_missing_onnx_file(tmp_paths):
    bad = VALID_ONNX_YAML.replace("tongue_color.onnx", "missing.onnx")
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save(bad)
    assert "missing.onnx" in str(e.value)


def test_save_rejects_missing_local_weights_file(tmp_paths_pytorch):
    bad = VALID_PYTORCH_YAML.replace("front.pth", "missing.pth")
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save(bad)
    assert "missing.pth" in str(e.value)


def test_save_rejects_both_weights_uri_and_onnx_path(tmp_paths):
    body = """
heads:
  - task: confused
    head_type: single
    weights_uri: local:x.pth
    onnx_path: tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a]
"""
    with pytest.raises(registry_store.ValidationError, match="exactly one"):
        registry_store.save(body)


def test_save_rejects_neither_weights_uri_nor_onnx_path(tmp_paths):
    body = """
heads:
  - task: empty
    head_type: single
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a]
"""
    with pytest.raises(registry_store.ValidationError, match="exactly one"):
        registry_store.save(body)


def test_save_rejects_unknown_weights_uri_scheme(tmp_paths):
    body = """
heads:
  - task: weird
    head_type: single
    weights_uri: gs://bucket/file.pth
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a]
"""
    with pytest.raises(registry_store.ValidationError):
        registry_store.save(body)
