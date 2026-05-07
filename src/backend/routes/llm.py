"""GET /api/llm/models — list Gemini models the configured key can use."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.llm import client as llm_client
from backend.models import LLMModelsResponse


router = APIRouter(prefix="/api/llm", tags=["llm"])


def _bare_id(name: str) -> str:
    """Strip the ``models/`` prefix Google returns so the result drops straight
    into ``llm.yaml``'s ``model:`` field."""
    return name.removeprefix("models/")


@router.get("/models", response_model=LLMModelsResponse)
def list_models() -> LLMModelsResponse:
    try:
        client = llm_client._make_client()
    except RuntimeError as e:
        # No API key set yet — surface as 412 Precondition Failed so the
        # frontend can disambiguate from a network error.
        raise HTTPException(status_code=412, detail={"error": str(e)})

    try:
        page = client.models.list()
    except Exception as e:
        raise HTTPException(
            status_code=502, detail={"error": f"Gemini list failed: {type(e).__name__}: {e}"}
        )

    ids: list[str] = []
    for m in page:
        actions = getattr(m, "supported_actions", None) or getattr(
            m, "supported_generation_methods", None
        ) or []
        if "generateContent" in actions:
            ids.append(_bare_id(m.name))
    ids.sort()
    return LLMModelsResponse(models=ids)
