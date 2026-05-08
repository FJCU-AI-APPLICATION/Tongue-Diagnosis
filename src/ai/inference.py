"""Run all configured task heads on one image, in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai.registry import Registry
from ai.types import HeadResult


_MAX_WORKERS = 16


def run_all(image_bgr: np.ndarray, registry: Registry) -> list[HeadResult]:
    heads = registry.heads
    if not heads:
        return []
    workers = min(len(heads), _MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="head") as pool:
        futures = [pool.submit(_safe_predict, h, image_bgr) for h in heads]
        # Iterate in submission order so results align with registry order
        # regardless of which head finishes first.
        return [f.result() for f in futures]


def _safe_predict(head, image_bgr) -> HeadResult:
    try:
        return head.predict(image_bgr)
    except Exception as exc:
        return HeadResult(
            task=getattr(head, "name", "?"),
            head_type=getattr(head, "head_type", "single"),
            predictions=[],
            error=f"{type(exc).__name__}: {exc}",
        )
