"""Single orchestrator for /api/analyze."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ai import detect_tongue, run_all

from backend.llm import client, predictions, prompt
from backend.models import AnalyzeResponse, LLMConfig, TimingMs
from backend.stores import llm_store, prompt_store


DISCLAIMER = "此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。"
USER_TRIGGER = "請依規則輸出大眾版報告。"


class ImageDecodeError(ValueError):
    """Raised when input bytes cannot be decoded as an image."""


def _load_prompt() -> str:
    return prompt_store.load_current()


def _load_llm_config() -> LLMConfig:
    return llm_store.load_current()


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("could not decode image")
    return img


def analyze(image_bytes: bytes, *, registry) -> AnalyzeResponse:
    """Run the full analysis pipeline. ``registry`` is injected by the caller."""
    t0 = time.perf_counter()
    image = _decode_bgr(image_bytes)
    decode_ms = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    bbox = detect_tongue(image, getattr(registry, "detector", None))
    roi = image if bbox is None else image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    detect_ms = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    heads = run_all(roi, registry)
    infer_ms = int((time.perf_counter() - t2) * 1000)

    category_map = getattr(registry, "category_map", {}) or {}
    block = predictions.render(heads, category_map=category_map or None)

    template = _load_prompt()
    try:
        system = prompt.render(template, block)
    except prompt.PromptValidationError as e:
        comment = f"{client.ERROR_STAMP}{e}"
        llm_ms = 0
    else:
        llm_cfg = _load_llm_config()
        t3 = time.perf_counter()
        comment = client.run(system=system, user=USER_TRIGGER, config=llm_cfg)
        llm_ms = int((time.perf_counter() - t3) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)

    return AnalyzeResponse(
        predictions_block=block,
        heads=heads,
        comment=comment,
        disclaimer=DISCLAIMER,
        category_map=category_map,
        timing_ms=TimingMs(
            decode=decode_ms,
            detect=detect_ms,
            infer=infer_ms,
            llm=llm_ms,
            total=total_ms,
        ),
    )
