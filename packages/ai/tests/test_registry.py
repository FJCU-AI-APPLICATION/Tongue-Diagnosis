from pathlib import Path

import pytest

from tongue_ai.registry import Registry, RegistryError, load_registry


SAMPLE_YAML = """
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


def _write_yaml(tmp_path: Path, body: str = SAMPLE_YAML) -> Path:
    (tmp_path / "tongue_color.onnx").write_bytes(b"")     # placeholder
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    return p


def test_load_registry_two_heads(tmp_path, monkeypatch):
    yaml_path = _write_yaml(tmp_path)
    # Stub ort.InferenceSession so we don't actually load ONNX bytes
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    registry = load_registry(yaml_path)
    assert isinstance(registry, Registry)
    assert [h.name for h in registry.heads] == ["舌色", "舌質"]
    assert registry.detector is None


def test_load_registry_missing_required_key(tmp_path, monkeypatch):
    yaml_path = _write_yaml(
        tmp_path,
        """
heads:
  - task: 舌色
    head_type: single
""",
    )
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(yaml_path)
    assert "onnx_path" in str(e.value)


def test_load_registry_missing_onnx_file(tmp_path, monkeypatch):
    body = SAMPLE_YAML.replace("tongue_color.onnx", "missing.onnx")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(p)
    assert "missing.onnx" in str(e.value)
