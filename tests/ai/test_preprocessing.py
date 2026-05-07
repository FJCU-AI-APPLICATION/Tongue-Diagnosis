import numpy as np

from ai.preprocessing import (
    resize_letterbox,
    normalise,
    bgr_to_rgb,
    crop_bbox,
    to_chw_float,
)
from ai.types import BBox, Normalisation


def test_resize_letterbox_keeps_aspect():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = resize_letterbox(img, (224, 224))
    assert out.shape == (224, 224, 3)
    # Letterboxed top/bottom should be black (0)
    assert out[0, 0].tolist() == [0, 0, 0]


def test_normalise_imagenet_zero_mean():
    img = np.full((224, 224, 3), 128, dtype=np.uint8)  # mid-grey
    n = Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out = normalise(img, n)
    assert out.dtype == np.float32
    # Mid-grey ≈ 0.5 → roughly zero-ish after normalisation
    assert abs(out.mean()) < 0.5


def test_bgr_to_rgb_swaps_channels():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 1   # B
    img[..., 2] = 3   # R
    out = bgr_to_rgb(img)
    assert out[0, 0].tolist() == [3, 0, 1]


def test_bgr_to_rgb_round_trip_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
    np.testing.assert_array_equal(img, bgr_to_rgb(bgr_to_rgb(img)))


def test_crop_bbox_returns_correct_region():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    bbox = BBox(x=10, y=20, w=30, h=40)
    out = crop_bbox(img, bbox)
    assert out.shape == (40, 30, 3)


def test_to_chw_float_changes_layout():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    out = to_chw_float(img)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32
