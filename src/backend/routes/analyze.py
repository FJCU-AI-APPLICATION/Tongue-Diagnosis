"""POST /api/analyze — tongue diagnosis pipeline."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.models import AnalyzeResponse
from backend.pipeline import ImageDecodeError, analyze
from backend.settings import settings


router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def api_analyze(file: UploadFile, request: Request) -> AnalyzeResponse:
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail={"error": "missing file"})
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"error": "image too large"})

    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "registry unavailable; check /api/config/registry"},
        )

    try:
        return await run_in_threadpool(analyze, data, registry=registry)
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
