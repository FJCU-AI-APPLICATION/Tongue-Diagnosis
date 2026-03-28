"""Tongue detection: locates tongue region in an image."""

from __future__ import annotations

import numpy as np
import cv2


def detect_tongue(image: np.ndarray) -> dict:
    """Detect tongue region in the input image.

    Parameters
    ----------
    image : np.ndarray
        BGR image as loaded by OpenCV.

    Returns
    -------
    dict
        Bounding box with keys: x, y, w, h, confidence.
    """
    h, w = image.shape[:2]
    # Placeholder: return center crop as "detected" region
    return {
        "x": w // 4,
        "y": h // 4,
        "w": w // 2,
        "h": h // 2,
        "confidence": 0.95,
    }
