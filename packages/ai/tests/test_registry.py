"""Tests for the YAML registry loader and validator."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tongue_ai.registry import (
    Registry,
    RegistryError,
    load_registry,
    validate_registry_yaml,
)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _fake_resnet50_state_dict(num_classes: int):
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


VALID_YAML = """
detector:
  enabled: false

heads:
  - name: front
    head_type: single
    arch: resnet50
    weights_uri: local:weights/front.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b, c]

  - name: sublingual
    head_type: single
    arch: resnet50
    weights_uri: local:weights/sub.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [x, y]

category_map:
  front:
    a: 舌色
    b: 舌色
    c: 舌質
  sublingual:
    x: 舌下絡脈
    y: 舌下絡脈
"""


def test_validate_yaml_accepts_valid(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    # No raise
    validate_registry_yaml(p)


def test_validate_yaml_rejects_missing_required_field(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    bad = VALID_YAML.replace("    head_type: single", "")
    _write_yaml(p, bad)
    with pytest.raises(RegistryError, match="head_type"):
        validate_registry_yaml(p)


def test_validate_yaml_rejects_class_name_missing_from_category_map(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    # Drop "c: 舌質" → class c has no v4 mapping
    bad = VALID_YAML.replace("    c: 舌質\n", "")
    _write_yaml(p, bad)
    with pytest.raises(RegistryError, match="category_map.*missing"):
        validate_registry_yaml(p)


def test_load_registry_returns_registry_with_two_heads(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save(_fake_resnet50_state_dict(3), weights_dir / "front.pth")
    torch.save(_fake_resnet50_state_dict(2), weights_dir / "sub.pth")

    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    reg = load_registry(p)
    assert isinstance(reg, Registry)
    assert [h.name for h in reg.heads] == ["front", "sublingual"]
    assert reg.detector_enabled is False
    assert reg.category_map["front"]["a"] == "舌色"


def test_load_registry_partial_failure_records_per_head_error(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    # Only front weights present; sub.pth missing
    torch.save(_fake_resnet50_state_dict(3), weights_dir / "front.pth")
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)

    reg = load_registry(p, raise_on_partial_fail=False)
    assert [h.name for h in reg.heads] == ["front"]
    assert "sublingual" in reg.failed_heads
    assert "not found" in reg.failed_heads["sublingual"]


def test_load_registry_total_failure_raises_when_strict(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    with pytest.raises(RegistryError, match="zero heads"):
        load_registry(p, raise_on_partial_fail=True)
