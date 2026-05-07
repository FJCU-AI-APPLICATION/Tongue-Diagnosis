"""GET /health — basic liveness + AI version + registry status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ai import __version__ as ai_version

from backend.models import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    reg = getattr(request.app.state, "registry", None)
    return HealthResponse(
        ai_version=ai_version,
        registry_loaded=reg is not None,
        heads_loaded=[h.name for h in reg.heads] if reg is not None else [],
    )
