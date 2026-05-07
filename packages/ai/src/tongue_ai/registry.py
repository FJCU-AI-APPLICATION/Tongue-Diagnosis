"""Registry: YAML loader + validator for the inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import IMAGENET_NORMALISATION, Normalisation
from tongue_ai.weights import WeightFetchError, WeightSource


REQUIRED_HEAD_FIELDS = {
    "name",
    "head_type",
    "arch",
    "weights_uri",
    "input_size",
    "normalise",
    "class_names",
}

_NORMALISATIONS: dict[str, Normalisation] = {
    "imagenet": IMAGENET_NORMALISATION,
}


class RegistryError(RuntimeError):
    """Raised on registry validation or load failures."""


@dataclass
class Registry:
    detector_enabled: bool
    heads: list[PyTorchTaskHead]
    category_map: dict[str, dict[str, str]]
    failed_heads: dict[str, str] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"yaml parse error: {exc}") from exc


def validate_registry_yaml(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if "heads" not in data or not isinstance(data["heads"], list) or not data["heads"]:
        raise RegistryError("registry must contain a non-empty 'heads' list")
    cmap = data.get("category_map") or {}
    if not isinstance(cmap, dict):
        raise RegistryError("'category_map' must be a mapping if provided")

    for head in data["heads"]:
        missing = REQUIRED_HEAD_FIELDS - set(head)
        if missing:
            raise RegistryError(
                f"head {head.get('name', '?')!r} missing required fields: {sorted(missing)}"
            )
        if head["normalise"] not in _NORMALISATIONS:
            raise RegistryError(
                f"head {head['name']!r}: unknown normalise key {head['normalise']!r}"
            )
        head_map = cmap.get(head["name"], {})
        missing_classes = [c for c in head["class_names"] if c not in head_map]
        if missing_classes:
            raise RegistryError(
                f"category_map missing entries for head {head['name']!r}: {missing_classes}"
            )
    return data


def load_registry(path: Path, *, raise_on_partial_fail: bool = True) -> Registry:
    data = validate_registry_yaml(path)
    base_dir = path.parent

    loaded: list[PyTorchTaskHead] = []
    failed: dict[str, str] = {}

    for cfg in data["heads"]:
        full_cfg = {
            **cfg,
            "normalise": _NORMALISATIONS[cfg["normalise"]],
        }
        try:
            src = WeightSource(uri=cfg["weights_uri"], base_dir=base_dir)
            weight_path = src.resolve()
            head = load_pytorch_head(full_cfg, weight_path)
            loaded.append(head)
        except (WeightFetchError, RuntimeError, ValueError) as exc:
            failed[cfg["name"]] = f"{type(exc).__name__}: {exc}"

    if not loaded and raise_on_partial_fail:
        details = "; ".join(f"{n}: {e}" for n, e in failed.items()) or "no heads configured"
        raise RegistryError(f"zero heads loaded — {details}")

    detector_enabled = bool((data.get("detector") or {}).get("enabled", False))

    return Registry(
        detector_enabled=detector_enabled,
        heads=loaded,
        category_map=data.get("category_map", {}),
        failed_heads=failed,
    )
