"""Tests for RegistryStore: persists YAML, validates structure, reload semantics."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tongue_ai.registry import Registry, RegistryError
from tongue_backend.stores.registry_store import RegistryStore


VALID_YAML = """
detector:
  enabled: false
heads:
  - name: head1
    head_type: single
    arch: resnet50
    weights_uri: local:weights/h1.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b]
category_map:
  head1:
    a: 舌色
    b: 舌色
"""


def _fake_state_dict(num_classes: int):
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


def _store(tmp_path: Path) -> RegistryStore:
    default = tmp_path / "registry.default.yaml"
    default.write_text(VALID_YAML, encoding="utf-8")
    current = tmp_path / "registry.current.yaml"
    return RegistryStore(default_path=default, current_path=current)


def test_save_persists_yaml_without_loading_models(tmp_path: Path):
    s = _store(tmp_path)
    s.save(VALID_YAML)
    assert "head1" in s.load_text()


def test_save_rejects_yaml_parse_error(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(RegistryError, match="parse"):
        s.save("heads: [::: bad")


def test_save_rejects_missing_required_field(tmp_path: Path):
    s = _store(tmp_path)
    bad = VALID_YAML.replace("    head_type: single", "")
    with pytest.raises(RegistryError, match="head_type"):
        s.save(bad)


def test_reload_with_missing_weights_returns_failure_list(tmp_path: Path):
    s = _store(tmp_path)
    s.save(VALID_YAML)
    result = s.reload()
    assert "head1" in result.failed_heads


def test_reload_with_valid_weights_returns_loaded_registry(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save(_fake_state_dict(2), weights_dir / "h1.pth")
    s = _store(tmp_path)
    s.save(VALID_YAML)
    reg = s.reload()
    assert isinstance(reg, Registry)
    assert [h.name for h in reg.heads] == ["head1"]
