"""Health check endpoint."""

from fastapi import APIRouter

from tongue_ai import __version__ as ai_version

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "ai_version": ai_version,
    }
