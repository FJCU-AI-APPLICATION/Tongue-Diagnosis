from dataclasses import dataclass

import numpy as np

from tongue_ai.inference import run_all
from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult, ClassScore


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
