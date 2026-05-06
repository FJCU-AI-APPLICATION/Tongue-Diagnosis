"""GET/PUT/RESET /api/config/{section} + POST /api/config/registry/reload."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from tongue_backend.stores import llm_store, prompt_store, registry_store


router = APIRouter(prefix="/api/config", tags=["config"])


class _PutBody(BaseModel):
    content: str


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


@router.post("/registry/reload")
def reload_registry(request: Request):
    """Re-init ONNX sessions in-process. All-or-rollback on total failure."""
    from tongue_ai import load_registry, RegistryError

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
    return {
        "loaded": [h.name for h in new_registry.heads],
        "failed": [],
        "previous_kept": previous is not None,
    }


@router.get("/{section}")
def get_section(section: str):
    status_fn, _, _ = _resolve(section)
    return status_fn()


@router.put("/{section}")
def put_section(section: str, body: _PutBody):
    _, save_fn, _ = _resolve(section)
    try:
        save_fn(body.content)
    except (llm_store.ValidationError, registry_store.ValidationError) as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
    return {"ok": True}


@router.post("/{section}/reset")
def reset_section(section: str):
    _, _, reset_fn = _resolve(section)
    reset_fn()
    return {"ok": True}
