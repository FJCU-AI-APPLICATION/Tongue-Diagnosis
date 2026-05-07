"""Tests for PyTorchTaskHead — single-label decoding from a fake fc layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import IMAGENET_NORMALISATION, ClassScore, HeadResult


def _fake_resnet_state_dict_for_classes(num_classes: int) -> dict[str, torch.Tensor]:
    """Build a state_dict matching torchvision.models.resnet50 with fc[2048,N]."""
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


def test_load_pytorch_head_accepts_matching_state_dict(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    assert isinstance(head, PyTorchTaskHead)
    assert head.name == "test"
    assert head.class_names == ["a", "b", "c", "d"]


def test_load_pytorch_head_size_mismatch_raises(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(3)  # 3 classes
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],  # 4 names != 3 weight classes
    }
    with pytest.raises(RuntimeError, match="state_dict|size mismatch"):
        load_pytorch_head(cfg, weight_path)


def test_predict_single_returns_top1_with_softmax_score(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    # Force fc bias so class index 2 wins regardless of input
    state["fc.bias"] = torch.tensor([-10.0, -10.0, 100.0, -10.0])
    state["fc.weight"] = torch.zeros((4, 2048))
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    img = np.full((300, 400, 3), 128, dtype=np.uint8)  # BGR mid-grey
    result = head.predict(img)

    assert isinstance(result, HeadResult)
    assert result.task == "test"
    assert result.head_type == "single"
    assert result.error is None
    assert len(result.predictions) == 1
    assert result.predictions[0].label == "c"
    assert result.predictions[0].score == pytest.approx(1.0, abs=1e-3)


def test_predict_records_error_in_head_result_on_failure(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)
    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    bad = np.zeros((0, 0, 3), dtype=np.uint8)  # zero-size image
    result = head.predict(bad)
    assert result.error is not None
    assert result.predictions == []
