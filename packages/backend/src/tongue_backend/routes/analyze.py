"""POST /api/analyze — tongue diagnosis pipeline."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request, UploadFile

from tongue_backend.pipeline import ImageDecodeError, analyze


# Override with env: TONGUE_MAX_UPLOAD_MB=20 → 20 MB cap
MAX_UPLOAD_MB = int(os.environ.get("TONGUE_MAX_UPLOAD_MB", "10"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze")
async def api_analyze(file: UploadFile, request: Request) -> dict:
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail={"error": "missing file"})
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "image too large"})

    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail={"error": "registry unavailable; check /api/config/registry"})

    try:
        return analyze(data, registry=registry)
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
