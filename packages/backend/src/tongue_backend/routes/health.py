"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request

from tongue_ai import __version__ as ai_version


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        registry_summary = {"loaded": 0, "failed": []}
    else:
        registry_summary = {
            "loaded": len(registry.heads),
            "failed": list(registry.failed_heads.keys()),
        }
    return {
        "status": "ok",
        "ai_version": ai_version,
        "registry": registry_summary,
    }
