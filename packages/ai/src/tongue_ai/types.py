"""Shared dataclasses for tongue-ai."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


HeadType = Literal["single", "multi"]


@dataclass(frozen=True)
class ClassScore:
    label: str
    score: float


@dataclass
class HeadResult:
    task: str
    head_type: HeadType
    predictions: list[ClassScore] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class Normalisation:
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
