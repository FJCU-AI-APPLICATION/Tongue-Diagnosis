"""Tests for tongue_ai.detection (pass-through stub)."""
from __future__ import annotations

import numpy as np

from tongue_ai.detection import detect_tongue
from tongue_ai.types import BBox


def test_detect_tongue_returns_whole_image_bbox_when_disabled():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = detect_tongue(img, enabled=False)
    assert bbox == BBox(x=0, y=0, w=640, h=480, confidence=1.0)


def test_detect_tongue_returns_whole_image_bbox_when_no_detector_arg():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = detect_tongue(img)
    assert bbox == BBox(x=0, y=0, w=200, h=100, confidence=1.0)
