"""Image preprocessing for ONNX inference."""

from __future__ import annotations

import cv2
import numpy as np

from ai.types import BBox, Normalisation


def resize_letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image to ``size`` (H, W) keeping aspect, padding with black."""
    target_h, target_w = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def normalise(image: np.ndarray, params: Normalisation) -> np.ndarray:
    """Convert uint8 [0,255] → float32 normalised by mean/std."""
    arr = image.astype(np.float32) / 255.0
    mean = np.asarray(params.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(params.std, dtype=np.float32).reshape(1, 1, -1)
    return (arr - mean) / std


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Swap channel order BGR↔RGB."""
    return image[..., ::-1].copy()


def crop_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop the image to the given bbox."""
    return image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]


def to_chw_float(image: np.ndarray) -> np.ndarray:
    """HWC → NCHW with batch dim 1, float32."""
    arr = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)
