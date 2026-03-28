"""Tongue recognition: identifies tongue features and landmarks."""

from __future__ import annotations

import numpy as np


def recognize_features(image: np.ndarray) -> dict:
    """Identify tongue features in the image.

    Returns
    -------
    dict
        Recognised features with keys: coating, color, shape.
    """
    return {
        "coating": "thin_white",
        "color": "pale_red",
        "shape": "normal",
    }
