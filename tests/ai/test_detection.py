import numpy as np

from ai.detection import detect_tongue
from ai.types import BBox


def test_detect_tongue_with_no_detector_returns_none():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_tongue(img, detector=None) is None


def test_detect_tongue_with_callable_returns_bbox():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    expected = BBox(x=10, y=10, w=50, h=50, confidence=0.9)
    result = detect_tongue(img, detector=lambda _img: expected)
    assert result == expected
