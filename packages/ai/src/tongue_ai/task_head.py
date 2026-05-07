"""PyTorchTaskHead: a single ResNet50 classifier wrapped with preprocessing + decode."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from tongue_ai.preprocessing import bgr_to_rgb, normalise_imagenet, resize_to
from tongue_ai.types import ClassScore, HeadResult, Normalisation


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PyTorchTaskHead:
    name: str
    head_type: Literal["single", "multi"]
    arch: Literal["resnet50"]
    class_names: list[str]
    input_size: tuple[int, int]
    normalise: Normalisation
    threshold: float = 0.5
    device: torch.device = field(default_factory=_autodetect_device)
    model: nn.Module | None = None  # set by factory

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        try:
            rgb = bgr_to_rgb(image_bgr)
            resized = resize_to(rgb, self.input_size)
            arr = normalise_imagenet(resized, self.normalise)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            assert self.model is not None
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        except Exception as exc:
            return HeadResult(
                task=self.name,
                head_type=self.head_type,
                predictions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        if self.head_type == "single":
            idx = int(np.argmax(probs))
            return HeadResult(
                task=self.name,
                head_type="single",
                predictions=[ClassScore(label=self.class_names[idx], score=float(probs[idx]))],
            )
        # multi-label
        kept = [
            ClassScore(label=self.class_names[i], score=float(p))
            for i, p in enumerate(probs)
            if p >= self.threshold
        ]
        kept.sort(key=lambda c: c.score, reverse=True)
        return HeadResult(task=self.name, head_type="multi", predictions=kept)


def load_pytorch_head(cfg: dict[str, Any], weight_path: Path) -> PyTorchTaskHead:
    if cfg["arch"] != "resnet50":
        raise ValueError(f"unsupported arch: {cfg['arch']!r}")
    num_classes = len(cfg["class_names"])
    device = _autodetect_device()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    head = PyTorchTaskHead(
        name=cfg["name"],
        head_type=cfg["head_type"],
        arch=cfg["arch"],
        class_names=list(cfg["class_names"]),
        input_size=tuple(cfg["input_size"]),
        normalise=cfg["normalise"],
        threshold=cfg.get("threshold", 0.5),
        device=device,
        model=model,
    )
    return head
