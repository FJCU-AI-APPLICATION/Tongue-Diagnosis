"""Registry config: YAML I/O + validation + reset.

Note: this module persists the YAML and validates it without loading any ONNX
sessions. The expensive reload-into-process step lives in app.state.registry
and is exercised via app.reload_registry() (Task 18).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tongue_backend.stores.paths import REGISTRY_DEFAULT, REGISTRY_CURRENT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


REQUIRED_HEAD_KEYS = {"task", "head_type", "onnx_path", "input_size", "normalise", "class_names"}


def _ensure_current() -> Path:
    if not REGISTRY_CURRENT.exists():
        REGISTRY_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_CURRENT.write_text(REGISTRY_DEFAULT.read_text())
    return REGISTRY_CURRENT


def load_current_text() -> str:
    return _ensure_current().read_text()


def save(content: str) -> None:
    _validate(content)
    REGISTRY_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_CURRENT.write_text(content)


def reset() -> None:
    REGISTRY_CURRENT.write_text(REGISTRY_DEFAULT.read_text())


def status() -> dict:
    content = load_current_text()
    return {
        "content": content,
        "is_default": content == REGISTRY_DEFAULT.read_text(),
        "mtime": REGISTRY_CURRENT.stat().st_mtime,
    }


def _validate(content: str) -> None:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("registry root must be a mapping")
    heads = data.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValidationError("registry must define a non-empty 'heads' list")

    base_dir = REGISTRY_CURRENT.parent
    for raw in heads:
        if not isinstance(raw, dict):
            raise ValidationError(f"head entry must be a mapping, got {type(raw).__name__}")
        missing = REQUIRED_HEAD_KEYS - raw.keys()
        if missing:
            raise ValidationError(
                f"head '{raw.get('task','?')}' missing keys: {sorted(missing)}"
            )
        onnx_path = (base_dir / raw["onnx_path"]).resolve()
        if not onnx_path.exists():
            raise ValidationError(f"onnx_path not found on disk: {raw['onnx_path']}")
        if raw["head_type"] not in {"single", "multi"}:
            raise ValidationError(
                f"head '{raw['task']}': head_type must be 'single' or 'multi'"
            )
        if not list(raw.get("class_names") or []):
            raise ValidationError(f"head '{raw['task']}': class_names must be non-empty")
