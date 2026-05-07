"""TaskHead — wraps one ONNX model + its preprocessing + label decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai.preprocessing import (
    bgr_to_rgb,
    normalise,
    resize_letterbox,
    to_chw_float,
)
from ai.types import ClassScore, HeadResult, HeadType, Normalisation


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TaskHead:
    session: Any                 # ort.InferenceSession or compatible mock
    name: str
    head_type: HeadType
    input_size: tuple[int, int]
    normalise: Normalisation
    class_names: list[str]
    threshold: float = 0.5
    already_probs: bool = False  # if True, skip softmax/sigmoid

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        try:
            tensor = self._preprocess(image_bgr)
            input_name = self.session.get_inputs()[0].name
            (logits,) = self.session.run(None, {input_name: tensor})
            return self._decode(logits)
        except Exception as exc:
            return HeadResult(
                task=self.name,
                head_type=self.head_type,
                predictions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = bgr_to_rgb(image_bgr)
        sized = resize_letterbox(rgb, self.input_size)
        floated = normalise(sized, self.normalise)
        return to_chw_float(floated)

    def _decode(self, logits: np.ndarray) -> HeadResult:
        # Take batch index 0
        scores = logits[0]
        if self.head_type == "single":
            probs = scores if self.already_probs else _softmax(scores)
            idx = int(np.argmax(probs))
            return HeadResult(
                task=self.name,
                head_type="single",
                predictions=[
                    ClassScore(label=self.class_names[idx], score=float(probs[idx]))
                ],
            )
        # multi
        probs = scores if self.already_probs else _sigmoid(scores)
        picks = [
            ClassScore(label=self.class_names[i], score=float(probs[i]))
            for i in range(len(self.class_names))
            if probs[i] >= self.threshold
        ]
        return HeadResult(task=self.name, head_type="multi", predictions=picks)
