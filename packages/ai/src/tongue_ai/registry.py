"""Registry loader — turns a YAML config into a list of TaskHead objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tongue_ai.task_head import TaskHead
from tongue_ai.types import Normalisation


class RegistryError(ValueError):
    """Raised when a registry YAML is malformed."""


@dataclass
class Registry:
    heads: list[TaskHead]
    detector: object | None  # detector callable; None if disabled


REQUIRED_HEAD_KEYS = {"task", "head_type", "onnx_path", "input_size", "normalise", "class_names"}


def _make_session(onnx_path: Path):
    """Indirection so tests can monkeypatch without importing onnxruntime."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path))


def load_registry(yaml_path: str | Path) -> Registry:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise RegistryError(f"Registry YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(data, dict):
        raise RegistryError("registry root must be a mapping")

    heads_data = data.get("heads")
    if not isinstance(heads_data, list) or not heads_data:
        raise RegistryError("registry must define a non-empty 'heads' list")

    base_dir = yaml_path.parent
    heads = [_build_head(item, base_dir) for item in heads_data]

    detector = None  # POC: detector wiring deferred (real detector models added later)
    return Registry(heads=heads, detector=detector)


def _build_head(raw: Any, base_dir: Path) -> TaskHead:
    if not isinstance(raw, dict):
        raise RegistryError(f"head entry must be a mapping, got {type(raw).__name__}")
    missing = REQUIRED_HEAD_KEYS - raw.keys()
    if missing:
        raise RegistryError(f"head '{raw.get('task','?')}' missing keys: {sorted(missing)}")

    onnx_path = (base_dir / raw["onnx_path"]).resolve()
    if not onnx_path.exists():
        raise RegistryError(f"onnx_path not found on disk: {onnx_path} (config: {raw['onnx_path']})")

    input_size = tuple(raw["input_size"])
    if len(input_size) != 2:
        raise RegistryError(f"input_size must be [H, W], got {raw['input_size']}")

    norm = raw["normalise"]
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        raise RegistryError(f"head '{raw['task']}': normalise must define mean and std")

    class_names = list(raw["class_names"])
    if not class_names:
        raise RegistryError(f"head '{raw['task']}': class_names must be non-empty")

    head_type = raw["head_type"]
    if head_type not in {"single", "multi"}:
        raise RegistryError(f"head '{raw['task']}': head_type must be 'single' or 'multi'")

    return TaskHead(
        session=_make_session(onnx_path),
        name=raw["task"],
        head_type=head_type,
        input_size=input_size,
        normalise=Normalisation(mean=norm["mean"], std=norm["std"]),
        class_names=class_names,
        threshold=float(raw.get("threshold", 0.5)),
        already_probs=bool(raw.get("already_probs", False)),
    )
