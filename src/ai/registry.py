"""Registry loader — turns a YAML config into a list of TaskHead objects.

Two head backends are supported, dispatched by the per-head YAML key:

  weights_uri: hf:owner/repo/file.pth  → PyTorchSession (via WeightSource)
  weights_uri: local:relative/path.pth → PyTorchSession (local file)
  onnx_path:   ../../ai/models/x.onnx  → ORT InferenceSession (existing)

A head MUST supply exactly one of those two keys.

The optional top-level `category_map` is a `dict[head_name, dict[class, v4_category]]`
used by the user-message builder to split composite-head predictions back into
v4-schema bullets. It is preserved verbatim and not validated for class coverage
(consumers tolerate orphan / missing entries).

Two entry points:

  ``validate_yaml(content, base_dir)`` — shape-only validation, no network,
    no session construction. Suitable for save-time validation by editors
    that want to reject typos without paying for HF downloads. ``local:``
    URIs and ``onnx_path`` entries are still checked for on-disk existence.

  ``load_registry(yaml_path)`` — full load. Calls ``validate_yaml`` then
    resolves weights (may hit HF Hub) and constructs ONNX / PyTorch sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai.task_head import TaskHead
from ai.types import Normalisation
from ai.weights import parse_weights_uri


class RegistryError(ValueError):
    """Raised when a registry YAML is malformed."""


@dataclass
class Registry:
    heads: list[TaskHead]
    detector: object | None  # detector callable; None if disabled
    category_map: dict[str, dict[str, str]] = field(default_factory=dict)


_COMMON_REQUIRED_KEYS = {"task", "head_type", "input_size", "normalise", "class_names"}


def _make_onnx_session(onnx_path: Path):
    """Indirection so tests can monkeypatch without importing onnxruntime."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path))


def validate_yaml(content: str, base_dir: Path) -> dict[str, Any]:
    """Parse + shape-check a registry YAML. No network, no session construction.

    Returns the parsed dict on success. Raises :class:`RegistryError` with a
    message that names the offending head when something is wrong.

    The contract here matches what :func:`load_registry` accepts, so anything
    that passes this function will not fail load-time *shape* validation. HF
    Hub downloads and ONNX/PyTorch session construction are deferred to load.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise RegistryError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise RegistryError("registry root must be a mapping")

    heads_data = data.get("heads")
    if not isinstance(heads_data, list) or not heads_data:
        raise RegistryError("registry must define a non-empty 'heads' list")
    for raw in heads_data:
        _validate_head_shape(raw, base_dir)

    category_map = data.get("category_map")
    if category_map is not None and not isinstance(category_map, dict):
        raise RegistryError("'category_map' must be a mapping if provided")

    return data


def load_registry(yaml_path: str | Path) -> Registry:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise RegistryError(f"Registry YAML not found: {yaml_path}")

    base_dir = yaml_path.parent
    data = validate_yaml(yaml_path.read_text(), base_dir)

    heads = [_build_head(item, base_dir) for item in data["heads"]]
    category_map = data.get("category_map") or {}

    detector = None  # POC: detector wiring deferred (real detector models added later)
    return Registry(heads=heads, detector=detector, category_map=category_map)


def _validate_head_shape(raw: Any, base_dir: Path) -> None:
    """All checks a head must pass — except weight/session construction.

    ``local:`` URIs and ``onnx_path`` entries are resolved against ``base_dir``
    and required to exist on disk. ``hf:`` URIs are not contacted.
    """
    if not isinstance(raw, dict):
        raise RegistryError(f"head entry must be a mapping, got {type(raw).__name__}")

    common_missing = _COMMON_REQUIRED_KEYS - raw.keys()
    if common_missing:
        raise RegistryError(
            f"head '{raw.get('task','?')}' missing keys: {sorted(common_missing)}"
        )

    has_weights_uri = "weights_uri" in raw
    has_onnx_path = "onnx_path" in raw
    if has_weights_uri == has_onnx_path:
        raise RegistryError(
            f"head '{raw['task']}': must supply exactly one of 'weights_uri' or 'onnx_path'"
        )

    input_size = raw["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise RegistryError(
            f"head '{raw['task']}': input_size must be [H, W], got {input_size!r}"
        )

    norm = raw["normalise"]
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        raise RegistryError(
            f"head '{raw['task']}': normalise must define mean and std"
        )

    class_names = raw["class_names"]
    if not isinstance(class_names, list) or not class_names:
        raise RegistryError(
            f"head '{raw['task']}': class_names must be a non-empty list"
        )

    head_type = raw["head_type"]
    if head_type not in {"single", "multi"}:
        raise RegistryError(
            f"head '{raw['task']}': head_type must be 'single' or 'multi'"
        )

    if has_onnx_path:
        onnx_path = (base_dir / raw["onnx_path"]).resolve()
        if not onnx_path.exists():
            raise RegistryError(
                f"onnx_path not found on disk: {onnx_path} (config: {raw['onnx_path']})"
            )
    else:
        # weights_uri: validate scheme + (for `local:`) on-disk existence.
        # `hf:` URIs are intentionally not resolved here — that's load-time.
        uri = raw["weights_uri"]
        try:
            scheme, ref = parse_weights_uri(uri)
        except ValueError as e:
            raise RegistryError(f"head '{raw['task']}': {e}") from e
        if scheme == "local":
            local_path = (base_dir / ref).resolve()
            if not local_path.is_file():
                raise RegistryError(
                    f"local weights not found: {local_path} "
                    f"(uri={uri!r})"
                )


def _build_head(raw: dict[str, Any], base_dir: Path) -> TaskHead:
    """Shape has already been validated by ``validate_yaml``; here we only
    resolve weights and construct the backing session."""
    input_size = tuple(raw["input_size"])
    norm = raw["normalise"]
    class_names = list(raw["class_names"])

    if "onnx_path" in raw:
        onnx_path = (base_dir / raw["onnx_path"]).resolve()
        session: Any = _make_onnx_session(onnx_path)
    else:
        from ai.pytorch_session import build_pytorch_session
        from ai.weights import WeightSource

        weight_path = WeightSource(uri=raw["weights_uri"], base_dir=base_dir).resolve()
        session = build_pytorch_session(weight_path, num_classes=len(class_names))

    return TaskHead(
        session=session,
        name=raw["task"],
        head_type=raw["head_type"],
        input_size=input_size,
        normalise=Normalisation(mean=norm["mean"], std=norm["std"]),
        class_names=class_names,
        threshold=float(raw.get("threshold", 0.5)),
        already_probs=bool(raw.get("already_probs", False)),
    )
