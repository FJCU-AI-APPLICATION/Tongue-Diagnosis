"""Tests for run_all — iterate registry heads, collect HeadResults."""
from __future__ import annotations

import numpy as np

from tongue_ai.inference import run_all
from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult


class _FakeHead:
    def __init__(self, name: str, return_value: HeadResult):
        self.name = name
        self._rv = return_value
        self.calls = 0

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        self.calls += 1
        return self._rv


def _make_registry(heads):
    return Registry(detector_enabled=False, heads=heads, category_map={})


def test_run_all_returns_one_result_per_head():
    h1 = _FakeHead("front", HeadResult("front", "single", [ClassScore("a", 0.9)]))
    h2 = _FakeHead("sublingual", HeadResult("sublingual", "single", [ClassScore("x", 0.7)]))
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    results = run_all(img, _make_registry([h1, h2]))
    assert [r.task for r in results] == ["front", "sublingual"]
    assert h1.calls == 1 and h2.calls == 1


def test_run_all_preserves_registry_order():
    h1 = _FakeHead("a", HeadResult("a", "single", []))
    h2 = _FakeHead("b", HeadResult("b", "single", []))
    h3 = _FakeHead("c", HeadResult("c", "single", []))
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    results = run_all(img, _make_registry([h3, h1, h2]))
    assert [r.task for r in results] == ["c", "a", "b"]


def test_run_all_continues_after_a_head_raises():
    class _Boom:
        name = "boom"
        def predict(self, _img):
            raise RuntimeError("kaboom")
    h_ok = _FakeHead("ok", HeadResult("ok", "single", [ClassScore("z", 0.5)]))
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    results = run_all(img, _make_registry([_Boom(), h_ok]))

    assert len(results) == 2
    assert results[0].task == "boom"
    assert results[0].error is not None
    assert "kaboom" in results[0].error
    assert results[1].task == "ok"
    assert results[1].error is None
