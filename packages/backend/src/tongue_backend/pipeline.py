"""Single orchestrator for /api/analyze."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from tongue_ai import detect_tongue, run_all
from tongue_ai.types import HeadResult

from tongue_backend.llm import client, user_message
from tongue_backend.stores import llm_store, prompt_store


DISCLAIMER = "此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。"


class ImageDecodeError(ValueError):
    """Raised when input bytes cannot be decoded as an image."""


# Indirections so tests can monkeypatch the FS reads without touching disk
def _load_prompt() -> str:
    return prompt_store.load_current()


def _load_llm_config() -> dict:
    return llm_store.load_current()


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("could not decode image")
    return img


def _serialise_heads(heads: list[HeadResult]) -> list[dict]:
    return [asdict(h) for h in heads]


def analyze(image_bytes: bytes, *, registry) -> dict[str, Any]:
    """Run the full analysis pipeline. ``registry`` is injected by the caller
    (the route reads ``request.app.state.registry``)."""
    timing: dict[str, int] = {}
    t0 = time.perf_counter()

    image = _decode_bgr(image_bytes)
    timing["decode"] = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    bbox = detect_tongue(image, getattr(registry, "detector", None))
    roi = image if bbox is None else image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    timing["detect"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    heads = run_all(roi, registry)
    timing["infer"] = int((time.perf_counter() - t2) * 1000)

    user_msg = user_message.build(heads)
    system = _load_prompt()
    llm_cfg = _load_llm_config()

    t3 = time.perf_counter()
    comment = client.run(system=system, user=user_msg, config=llm_cfg)
    timing["llm"] = int((time.perf_counter() - t3) * 1000)
    timing["total"] = int((time.perf_counter() - t0) * 1000)

    return {
        "user_message": user_msg,
        "heads": _serialise_heads(heads),
        "comment": comment,
        "disclaimer": DISCLAIMER,
        "timing_ms": timing,
    }
