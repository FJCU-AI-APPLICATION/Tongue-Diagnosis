"""GET /health — basic liveness + AI version + registry status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tongue_ai import __version__ as ai_version


router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    reg = getattr(request.app.state, "registry", None)
    return {
        "status": "ok",
        "ai_version": ai_version,
        "registry_loaded": reg is not None,
        "heads_loaded": [h.name for h in reg.heads] if reg is not None else [],
    }
