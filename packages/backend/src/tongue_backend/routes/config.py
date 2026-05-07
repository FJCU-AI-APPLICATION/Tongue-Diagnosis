"""/api/config/{section} CRUD + reload (prompt | llm | registry)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from tongue_ai.registry import RegistryError

from tongue_backend.stores.llm_store import LLMValidationError


router = APIRouter(prefix="/api/config", tags=["config"])


Section = Literal["prompt", "llm", "registry"]


class _ContentBody(BaseModel):
    content: str


def _store(request: Request, section: Section):
    if section == "prompt":
        return request.app.state.prompt_store
    if section == "llm":
        return request.app.state.llm_store
    if section == "registry":
        return request.app.state.registry_store
    raise HTTPException(status_code=404, detail={"error": f"unknown section {section!r}"})


def _read_text(store, section: Section) -> str:
    if section == "registry":
        return store.load_text()
    if section == "prompt":
        return store.load_current()
    if section == "llm":
        # llm_store.load_current returns dict; we want the raw YAML
        return store.current_path.read_text(encoding="utf-8") if store.current_path.exists() else store.default_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail={"error": f"unknown section {section!r}"})


@router.get("/{section}")
async def get_section(section: Section, request: Request) -> dict:
    store = _store(request, section)
    content = _read_text(store, section)
    return {
        "section": section,
        "content": content,
        "is_default": store.is_default(),
    }


@router.put("/{section}")
async def put_section(section: Section, body: _ContentBody, request: Request) -> dict:
    store = _store(request, section)
    try:
        store.save(body.content)
    except LLMValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
    except RegistryError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
    return {"section": section, "saved": True, "is_default": store.is_default()}


@router.post("/{section}/reset")
async def reset_section(section: Section, request: Request) -> dict:
    store = _store(request, section)
    store.reset()
    return {"section": section, "reset": True}


@router.post("/registry/reload")
async def reload_registry(request: Request) -> dict:
    store: object = request.app.state.registry_store
    new_registry = store.reload()  # type: ignore[attr-defined]
    if not new_registry.heads:
        # Total fail: keep previous registry; surface the failures
        return {
            "loaded": [],
            "failed": new_registry.failed_heads,
            "rolled_back": True,
        }
    request.app.state.registry = new_registry
    return {
        "loaded": [h.name for h in new_registry.heads],
        "failed": new_registry.failed_heads,
        "rolled_back": False,
    }
