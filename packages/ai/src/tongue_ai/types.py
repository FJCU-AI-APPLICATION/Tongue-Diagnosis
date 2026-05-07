"""Shared dataclasses for the tongue_ai package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Normalisation:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


IMAGENET_NORMALISATION = Normalisation(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass(frozen=True)
class ClassScore:
    label: str
    score: float


@dataclass
class HeadResult:
    task: str
    head_type: Literal["single", "multi"]
    predictions: list[ClassScore] = field(default_factory=list)
    error: str | None = None
