"""Image preprocessing for the inference pipeline."""
from __future__ import annotations

import cv2
import numpy as np

from tongue_ai.types import Normalisation


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def normalise_imagenet(image_rgb: np.ndarray, n: Normalisation) -> np.ndarray:
    """RGB uint8 HWC → float32 CHW, ImageNet-normalised."""
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - np.array(n.mean, dtype=np.float32)) / np.array(n.std, dtype=np.float32)
    return arr.transpose(2, 0, 1).copy()  # HWC -> CHW
