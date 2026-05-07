"""Tongue ROI detection — currently a pass-through stub.

The trained ResNet50 classifiers accept whole tongue photos, so the detector
is disabled by default. This stub returns a bbox covering the entire image.
"""
from __future__ import annotations

import numpy as np

from tongue_ai.types import BBox


def detect_tongue(image: np.ndarray, *, enabled: bool = False) -> BBox:
    h, w = image.shape[:2]
    return BBox(x=0, y=0, w=w, h=h, confidence=1.0)
