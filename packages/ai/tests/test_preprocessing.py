"""Tests for tongue_ai.preprocessing."""
from __future__ import annotations

import numpy as np
import pytest

from tongue_ai.preprocessing import (
    bgr_to_rgb,
    decode_bgr,
    normalise_imagenet,
    resize_to,
)
from tongue_ai.types import IMAGENET_NORMALISATION


def test_bgr_to_rgb_swaps_channels():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0, 0] == 30  # R first
    assert rgb[0, 0, 1] == 20  # G
    assert rgb[0, 0, 2] == 10  # B


def test_bgr_to_rgb_round_trip_is_identity():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    assert np.array_equal(bgr, bgr_to_rgb(bgr_to_rgb(bgr)))


def test_resize_to_changes_shape():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_to(rgb, (224, 224))
    assert out.shape == (224, 224, 3)


def test_normalise_imagenet_returns_tensor_with_expected_stats():
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)  # mid-grey
    tensor = normalise_imagenet(rgb, IMAGENET_NORMALISATION)
    # Shape: CHW float32
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype.name == "float32"
    # 128/255 = 0.502; (0.502 - 0.485)/0.229 ≈ 0.074 for R channel
    assert tensor[0].mean() == pytest.approx(0.074, abs=0.01)


def test_decode_bgr_reads_jpeg_bytes(tmp_path):
    import cv2
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)
    decoded = decode_bgr(p.read_bytes())
    assert decoded.shape == (10, 10, 3)
    assert decoded.dtype == np.uint8


def test_decode_bgr_raises_on_garbage():
    with pytest.raises(ValueError, match="decode"):
        decode_bgr(b"not an image")
