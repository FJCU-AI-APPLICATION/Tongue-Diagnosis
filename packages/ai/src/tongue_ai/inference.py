"""Run all configured task heads on one image."""

from __future__ import annotations

import numpy as np

from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult


def run_all(image_bgr: np.ndarray, registry: Registry) -> list[HeadResult]:
    results: list[HeadResult] = []
    for head in registry.heads:
        try:
            results.append(head.predict(image_bgr))
        except Exception as exc:
            # belt-and-braces: TaskHead.predict already catches, but a buggy head
            # in a future implementation must not break the loop
            results.append(
                HeadResult(
                    task=getattr(head, "name", "?"),
                    head_type=getattr(head, "head_type", "single"),
                    predictions=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
