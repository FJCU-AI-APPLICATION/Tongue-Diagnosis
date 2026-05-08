import time
from dataclasses import dataclass

import numpy as np

from ai.inference import run_all
from ai.registry import Registry
from ai.types import HeadResult, ClassScore


@dataclass
class _StubHead:
    name: str
    head_type: str = "single"
    raises: bool = False

    def predict(self, _image):
        if self.raises:
            raise RuntimeError("kaboom")
        return HeadResult(
            task=self.name,
            head_type=self.head_type,
            predictions=[ClassScore(label="L", score=0.9)],
        )


def test_run_all_returns_one_result_per_head_in_order():
    registry = Registry(
        heads=[_StubHead("舌色"), _StubHead("舌質"), _StubHead("舌苔顏色")],
        detector=None,
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    results = run_all(image, registry)
    assert [r.task for r in results] == ["舌色", "舌質", "舌苔顏色"]


def test_run_all_continues_when_one_head_raises():
    registry = Registry(
        heads=[_StubHead("舌色"), _StubHead("舌質", raises=True), _StubHead("舌苔顏色")],
        detector=None,
    )
    results = run_all(np.zeros((10, 10, 3), dtype=np.uint8), registry)
    assert results[1].error is not None
    assert results[1].predictions == []
    assert results[0].error is None
    assert results[2].error is None


@dataclass
class _SleepHead:
    name: str
    delay: float
    head_type: str = "single"

    def predict(self, _image):
        time.sleep(self.delay)
        return HeadResult(
            task=self.name,
            head_type=self.head_type,
            predictions=[ClassScore(label="L", score=0.9)],
        )


def test_run_all_preserves_registry_order_when_heads_finish_out_of_order():
    registry = Registry(
        heads=[
            _SleepHead("舌色", 0.04),  # finishes last
            _SleepHead("舌質", 0.01),  # finishes first
            _SleepHead("舌苔顏色", 0.02),
        ],
        detector=None,
    )
    results = run_all(np.zeros((10, 10, 3), dtype=np.uint8), registry)
    assert [r.task for r in results] == ["舌色", "舌質", "舌苔顏色"]


def test_run_all_runs_heads_in_parallel():
    delay = 0.08
    registry = Registry(
        heads=[_SleepHead(f"h{i}", delay) for i in range(4)],
        detector=None,
    )
    t0 = time.perf_counter()
    results = run_all(np.zeros((10, 10, 3), dtype=np.uint8), registry)
    elapsed = time.perf_counter() - t0
    # 4 heads × 0.08s sequentially = 0.32s. Parallel should be well under 0.20s.
    assert elapsed < 0.20, f"expected parallel execution, got {elapsed:.3f}s"
    assert len(results) == 4


def test_run_all_with_empty_registry_returns_empty_list():
    registry = Registry(heads=[], detector=None)
    assert run_all(np.zeros((10, 10, 3), dtype=np.uint8), registry) == []
