"""GET/PUT/DELETE /api/config/api_key — operator-set Gemini key."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import ApiKeyStatus, PutBody
from backend.stores import secrets_store


router = APIRouter(prefix="/api/config", tags=["api_key"])


@router.get("/api_key", response_model=ApiKeyStatus)
def get_api_key() -> ApiKeyStatus:
    return secrets_store.status()


@router.put("/api_key", response_model=ApiKeyStatus)
def put_api_key(body: PutBody) -> ApiKeyStatus:
    try:
        secrets_store.save(body.content)
    except secrets_store.ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})
    except secrets_store.LiveTestError as e:
        raise HTTPException(status_code=422, detail={"error": f"金鑰測試失敗：{e}"})
    return secrets_store.status()


@router.delete("/api_key", response_model=ApiKeyStatus)
def delete_api_key() -> ApiKeyStatus:
    secrets_store.clear()
    return secrets_store.status()
