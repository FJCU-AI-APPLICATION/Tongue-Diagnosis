"""Registry config: YAML I/O + validation + reset.

Validation delegates to :func:`ai.registry.validate_yaml` so save-time and
load-time checks share one contract. The expensive load step (HF downloads,
torch/ONNX session construction) lives in ``app.state.registry`` and is
exercised via ``POST /api/config/registry/reload``.
"""

from __future__ import annotations

from ai.registry import RegistryError, validate_yaml

from backend.models import ConfigStatus
from backend.stores.paths import REGISTRY_CURRENT, REGISTRY_DEFAULT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


def _ensure_current():
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


def status() -> ConfigStatus:
    content = load_current_text()
    return ConfigStatus(
        content=content,
        is_default=content == REGISTRY_DEFAULT.read_text(),
        mtime=REGISTRY_CURRENT.stat().st_mtime,
    )


def _validate(content: str) -> None:
    try:
        validate_yaml(content, base_dir=REGISTRY_CURRENT.parent)
    except RegistryError as e:
        raise ValidationError(str(e)) from e
