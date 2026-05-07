"""Tongue ROI detection — pluggable detector abstraction."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ai.types import BBox


Detector = Callable[[np.ndarray], Optional[BBox]]


def detect_tongue(image: np.ndarray, detector: Detector | None) -> BBox | None:
    """Run the configured detector. Returns None when no detector is wired up,
    so the pipeline falls back to using the whole image."""
    if detector is None:
        return None
    return detector(image)
