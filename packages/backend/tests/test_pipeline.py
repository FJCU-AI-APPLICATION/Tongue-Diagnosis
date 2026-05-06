from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import pytest

from tongue_ai.types import ClassScore, HeadResult

from tongue_backend import pipeline as pl


def _jpeg_bytes(shape=(100, 100, 3)) -> bytes:
    img = (np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8))
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


@dataclass
class _StubHead:
    name: str
    head_type: str = "single"
    label: str = "L"

    def predict(self, _img):
        return HeadResult(
            task=self.name,
            head_type=self.head_type,
            predictions=[ClassScore(label=self.label, score=0.8)],
        )


@dataclass
class _StubRegistry:
    heads: list
    detector: object | None = None


def test_analyze_returns_full_response_shape(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色", label="淡紅"), _StubHead("舌態", label="無異常")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x", "temperature": 0.0, "max_tokens": 100, "top_p": 0.9})
    monkeypatch.setattr(pl.client, "run", lambda **kw: "## 醫師建議\n你還好")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)

    assert "user_message" in resp
    assert "heads" in resp and len(resp["heads"]) == 2
    assert resp["comment"].startswith("## 醫師建議")
    assert resp["disclaimer"].startswith("此為AI自動生成")
    assert "timing_ms" in resp


def test_analyze_raises_on_corrupt_image_bytes(monkeypatch):
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x"})
    with pytest.raises(pl.ImageDecodeError):
        pl.analyze(b"not a jpeg", registry=_StubRegistry(heads=[]))


def test_analyze_propagates_llm_error_into_comment(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x", "temperature": 0.0, "max_tokens": 100, "top_p": 0.9})
    monkeypatch.setattr(pl.client, "run", lambda **kw: "⚠ 醫師建議產生失敗：upstream")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)
    assert resp["comment"].startswith("⚠ 醫師建議產生失敗")
    assert len(resp["heads"]) == 1  # heads still populated
