"""POST /api/analyze: full pipeline (decode → infer → LLM)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile

from tongue_backend.pipeline import analyze as run_pipeline


router = APIRouter(prefix="/api", tags=["analyze"])

MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/analyze")
async def api_analyze(request: Request, file: UploadFile) -> dict:
    if request.app.state.registry is None:
        raise HTTPException(status_code=503, detail={"error": "registry unavailable"})

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "image too large"})
    if not data:
        raise HTTPException(status_code=400, detail={"error": "missing file"})

    try:
        prompt = request.app.state.prompt_store.load_current()
        llm_config = request.app.state.llm_store.load_current()
        resp = run_pipeline(
            image_bytes=data,
            registry=request.app.state.registry,
            prompt=prompt,
            llm_config=llm_config,
            llm_client=request.app.state.llm_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return resp.to_dict()
