"""GET/PUT/RESET /api/config/{section} + POST /api/config/registry/reload."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.llm import prompt
from backend.models import ConfigStatus, OkResponse, PutBody, ReloadResult
from backend.stores import llm_store, prompt_store, registry_store


router = APIRouter(prefix="/api/config", tags=["config"])


# Map {section} → (status_fn, save_fn, reset_fn)
_SECTIONS = {
    "prompt":   (prompt_store.status,   prompt_store.save,   prompt_store.reset),
    "llm":      (llm_store.status,      llm_store.save,      llm_store.reset),
    "registry": (registry_store.status, registry_store.save, registry_store.reset),
}


def _resolve(section: str):
    if section not in _SECTIONS:
        raise HTTPException(status_code=404, detail={"error": f"unknown section: {section}"})
    return _SECTIONS[section]


@router.post("/registry/reload", response_model=ReloadResult)
def reload_registry(request: Request) -> ReloadResult:
    """Re-init ONNX sessions in-process. All-or-rollback on total failure."""
    from ai import RegistryError, load_registry

    previous = getattr(request.app.state, "registry", None)
    # Ensure the 'current' YAML exists (copies from default if missing).
    # Needed when startup didn't run (e.g. TestClient used without `with`).
    registry_store.load_current_text()
    # Use registry_store.REGISTRY_CURRENT so monkeypatching in tests is respected.
    registry_current = registry_store.REGISTRY_CURRENT
    try:
        new_registry = load_registry(registry_current)
    except RegistryError as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})

    request.app.state.registry = new_registry
    return ReloadResult(
        loaded=[h.name for h in new_registry.heads],
        failed=[],
        previous_kept=previous is not None,
    )


@router.get("/{section}", response_model=ConfigStatus)
def get_section(section: str) -> ConfigStatus:
    status_fn, _, _ = _resolve(section)
    return status_fn()


@router.put("/{section}", response_model=OkResponse)
def put_section(section: str, body: PutBody) -> OkResponse:
    _, save_fn, _ = _resolve(section)
    try:
        save_fn(body.content)
    except (
        llm_store.ValidationError,
        registry_store.ValidationError,
        prompt.PromptValidationError,
    ) as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
    return OkResponse()


@router.post("/{section}/reset", response_model=OkResponse)
def reset_section(section: str) -> OkResponse:
    _, _, reset_fn = _resolve(section)
    reset_fn()
    return OkResponse()
