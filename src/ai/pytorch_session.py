"""PyTorchSession — a shim that quacks like onnxruntime.InferenceSession.

The existing TaskHead is built around the ORT contract:
    session.get_inputs()[0].name
    session.run(None, {input_name: chw_float32_array}) -> [logits_1xN_array]

This shim wraps a torchvision.models.resnet50 (with the fc layer rewired to
the right number of classes) so PyTorch weights can plug into TaskHead
without TaskHead knowing anything has changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


_INPUT_NAME = "input"


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class _Input:
    name: str


@dataclass
class PyTorchSession:
    """ONNX-session-shaped wrapper around a PyTorch model."""

    model: nn.Module
    device: torch.device

    def get_inputs(self) -> list[_Input]:
        return [_Input(name=_INPUT_NAME)]

    def run(self, _output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        chw = feed[_INPUT_NAME]
        tensor = torch.from_numpy(chw).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return [logits.cpu().numpy()]


def build_pytorch_session(weight_path: Path, num_classes: int) -> PyTorchSession:
    """Load a resnet50 with `fc -> Linear(2048, num_classes)`, return a PyTorchSession."""
    device = _autodetect_device()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return PyTorchSession(model=model, device=device)
