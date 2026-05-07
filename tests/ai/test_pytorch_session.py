"""Tests for PyTorchSession + build_pytorch_session."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

from ai.pytorch_session import (
    PyTorchSession,
    build_pytorch_session,
)


def _fake_resnet50_state_dict(num_classes: int) -> dict[str, torch.Tensor]:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


def test_session_exposes_input_named_input():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, 4)
    session = PyTorchSession(model=m, device=torch.device("cpu"))
    inputs = session.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "input"


def test_session_run_returns_logits_with_expected_shape():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, 4)
    m.eval()
    session = PyTorchSession(model=m, device=torch.device("cpu"))
    chw = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out = session.run(None, {"input": chw})
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (1, 4)
    assert out[0].dtype == np.float32


def test_build_pytorch_session_loads_state_dict_and_returns_eval_session(tmp_path: Path):
    state = _fake_resnet50_state_dict(5)
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)
    session = build_pytorch_session(weight_path, num_classes=5)
    assert isinstance(session, PyTorchSession)
    assert session.model.training is False  # eval() was called
    # End-to-end run yields the right shape
    chw = np.zeros((1, 3, 224, 224), dtype=np.float32)
    (logits,) = session.run(None, {"input": chw})
    assert logits.shape == (1, 5)


def test_build_pytorch_session_class_count_mismatch_raises(tmp_path: Path):
    state = _fake_resnet50_state_dict(3)  # 3 classes in weights
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)
    with pytest.raises(RuntimeError, match="state_dict|size mismatch"):
        build_pytorch_session(weight_path, num_classes=4)  # 4 != 3


def test_session_decode_via_existing_taskhead(tmp_path: Path):
    """End-to-end: build a session, plug into TaskHead, get a HeadResult."""
    from ai.task_head import TaskHead
    from ai.types import Normalisation

    # Force class index 2 to win regardless of input
    state = _fake_resnet50_state_dict(4)
    state["fc.bias"] = torch.tensor([-10.0, -10.0, 100.0, -10.0])
    state["fc.weight"] = torch.zeros((4, 2048))
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)
    session = build_pytorch_session(weight_path, num_classes=4)

    head = TaskHead(
        session=session,
        name="test",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        class_names=["a", "b", "c", "d"],
    )
    bgr = np.full((300, 400, 3), 128, dtype=np.uint8)
    result = head.predict(bgr)
    assert result.error is None
    assert result.predictions[0].label == "c"
    assert result.predictions[0].score == pytest.approx(1.0, abs=1e-3)
