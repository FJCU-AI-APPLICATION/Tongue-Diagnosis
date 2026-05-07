from pathlib import Path
from unittest.mock import patch

import pytest

from ai.registry import Registry, RegistryError, load_registry


SAMPLE_ONNX_YAML = """
detector: null
heads:
  - task: 舌色
    head_type: single
    onnx_path: tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["淡","淡紅","微紅","紅","絳","青紫","暗"]
  - task: 舌質
    head_type: multi
    threshold: 0.5
    onnx_path: tongue_body.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["齒痕","胖大","瘦薄","老","嫩","無異常"]
"""


SAMPLE_PYTORCH_YAML = """
detector: null
heads:
  - task: front
    head_type: single
    weights_uri: local:weights/front.pth
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: [a, b, c]
category_map:
  front:
    a: 舌色
    b: 舌色
    c: 舌質
"""


def _write_onnx_yaml(tmp_path: Path, body: str = SAMPLE_ONNX_YAML) -> Path:
    (tmp_path / "tongue_color.onnx").write_bytes(b"")     # placeholder
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    return p


def test_load_registry_two_onnx_heads(tmp_path, monkeypatch):
    yaml_path = _write_onnx_yaml(tmp_path)
    # Stub ort.InferenceSession so we don't actually load ONNX bytes
    import ai.registry as reg
    monkeypatch.setattr(reg, "_make_onnx_session", lambda _path: object())
    registry = load_registry(yaml_path)
    assert isinstance(registry, Registry)
    assert [h.name for h in registry.heads] == ["舌色", "舌質"]
    assert registry.detector is None
    assert registry.category_map == {}


def test_load_registry_missing_required_key(tmp_path, monkeypatch):
    yaml_path = _write_onnx_yaml(
        tmp_path,
        """
heads:
  - task: 舌色
    head_type: single
""",
    )
    import ai.registry as reg
    monkeypatch.setattr(reg, "_make_onnx_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(yaml_path)
    msg = str(e.value)
    assert "input_size" in msg or "normalise" in msg or "class_names" in msg


def test_load_registry_missing_onnx_file(tmp_path, monkeypatch):
    body = SAMPLE_ONNX_YAML.replace("tongue_color.onnx", "missing.onnx")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    import ai.registry as reg
    monkeypatch.setattr(reg, "_make_onnx_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(p)
    assert "missing.onnx" in str(e.value)


def test_load_registry_pytorch_head_via_weights_uri(tmp_path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "front.pth").write_bytes(b"placeholder")
    p = tmp_path / "registry.yaml"
    p.write_text(SAMPLE_PYTORCH_YAML)

    sentinel_session = object()
    with patch(
        "ai.pytorch_session.build_pytorch_session", return_value=sentinel_session
    ) as build_mock:
        registry = load_registry(p)

    assert [h.name for h in registry.heads] == ["front"]
    assert registry.heads[0].session is sentinel_session
    # Resolved path arg passed
    called_path = build_mock.call_args.args[0]
    assert called_path.name == "front.pth"
    assert build_mock.call_args.kwargs == {"num_classes": 3}


def test_load_registry_exposes_category_map(tmp_path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "front.pth").write_bytes(b"")
    p = tmp_path / "registry.yaml"
    p.write_text(SAMPLE_PYTORCH_YAML)

    with patch("ai.pytorch_session.build_pytorch_session", return_value=object()):
        registry = load_registry(p)

    assert registry.category_map == {
        "front": {"a": "舌色", "b": "舌色", "c": "舌質"},
    }


def test_load_registry_rejects_both_weights_uri_and_onnx_path(tmp_path):
    body = """
heads:
  - task: confused
    head_type: single
    weights_uri: local:x.pth
    onnx_path: x.onnx
    input_size: [224, 224]
    normalise: {mean: [0.0,0.0,0.0], std: [1.0,1.0,1.0]}
    class_names: [a]
"""
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    with pytest.raises(RegistryError, match="exactly one"):
        load_registry(p)


def test_load_registry_rejects_neither_weights_uri_nor_onnx_path(tmp_path):
    body = """
heads:
  - task: empty
    head_type: single
    input_size: [224, 224]
    normalise: {mean: [0.0,0.0,0.0], std: [1.0,1.0,1.0]}
    class_names: [a]
"""
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    with pytest.raises(RegistryError, match="exactly one"):
        load_registry(p)
