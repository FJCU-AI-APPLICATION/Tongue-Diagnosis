"""Tongue analysis endpoints: detect, recognize, classify, and full pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from tongue_ai.detection import detect_tongue
from tongue_ai.recognition import recognize_features
from tongue_ai.classification import classify_tongue

import json

router = APIRouter(prefix="/api", tags=["analyze"])


async def _read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR image."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


@router.post("/detect")
async def api_detect(file: UploadFile) -> dict:
    """Detect tongue region in the uploaded image."""
    image = await _read_image(file)
    return detect_tongue(image)


@router.post("/recognize")
async def api_recognize(file: UploadFile) -> dict:
    """Recognize tongue features in the uploaded image."""
    image = await _read_image(file)
    return recognize_features(image)


@router.post("/classify")
async def api_classify(file: UploadFile) -> dict:
    """Classify tongue condition from the uploaded image."""
    image = await _read_image(file)
    features = recognize_features(image)
    return classify_tongue(features)


@router.post("/analyze")
async def api_analyze(file: UploadFile) -> dict:
    """Run full analysis pipeline: detect → recognize → classify."""
    image = await _read_image(file)
    detection = detect_tongue(image)
    features = recognize_features(image)
    classification = classify_tongue(features)
    return {
        "detection": detection,
        "features": features,
        "classification": classification,
    }


@router.post("/analyze/stream")
async def api_analyze_stream(file: UploadFile) -> StreamingResponse:
    """Run full analysis pipeline with streaming results (SSE)."""
    image = await _read_image(file)

    async def event_stream():
        detection = detect_tongue(image)
        yield f"data: {json.dumps({'step': 'detection', 'result': detection})}\n\n"

        features = recognize_features(image)
        yield f"data: {json.dumps({'step': 'recognition', 'result': features})}\n\n"

        classification = classify_tongue(features)
        yield f"data: {json.dumps({'step': 'classification', 'result': classification})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
