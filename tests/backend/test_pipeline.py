from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from ai.types import ClassScore, HeadResult

from backend import pipeline as pl
from backend.models import AnalyzeResponse, LLMConfig


def _jpeg_bytes(shape=(100, 100, 3)) -> bytes:
    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def _stub_llm_config() -> LLMConfig:
    return LLMConfig(model="x", temperature=0.0, max_tokens=100, top_p=0.9)


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
    registry = _StubRegistry(
        heads=[_StubHead("舌色", label="淡紅"), _StubHead("舌態", label="無異常")]
    )
    monkeypatch.setattr(pl, "_load_prompt", lambda: "RULES\n{{PREDICTIONS}}\nFORMAT")
    monkeypatch.setattr(pl, "_load_llm_config", _stub_llm_config)
    monkeypatch.setattr(pl.client, "run", lambda **kw: "## 醫師建議\n你還好")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)

    assert isinstance(resp, AnalyzeResponse)
    assert resp.predictions_block
    assert "舌色" in resp.predictions_block
    assert len(resp.heads) == 2
    assert resp.comment.startswith("## 醫師建議")
    assert resp.disclaimer.startswith("此為AI自動生成")
    assert resp.timing_ms.total >= 0


def test_analyze_raises_on_corrupt_image_bytes(monkeypatch):
    monkeypatch.setattr(pl, "_load_prompt", lambda: "RULES\n{{PREDICTIONS}}\nFORMAT")
    monkeypatch.setattr(pl, "_load_llm_config", _stub_llm_config)
    with pytest.raises(pl.ImageDecodeError):
        pl.analyze(b"not a jpeg", registry=_StubRegistry(heads=[]))


def test_analyze_propagates_llm_error_into_comment(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "RULES\n{{PREDICTIONS}}\nFORMAT")
    monkeypatch.setattr(pl, "_load_llm_config", _stub_llm_config)
    monkeypatch.setattr(pl.client, "run", lambda **kw: "⚠ 醫師建議產生失敗：upstream")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)
    assert resp.comment.startswith("⚠ 醫師建議產生失敗")
    assert len(resp.heads) == 1  # heads still populated


def test_analyze_calls_client_with_filled_system_prompt_and_static_trigger(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色", label="淡紅")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "RULES\n{{PREDICTIONS}}\nFORMAT")
    monkeypatch.setattr(pl, "_load_llm_config", _stub_llm_config)

    captured = {}
    def _capture(**kw):
        captured.update(kw)
        return "OK"
    monkeypatch.setattr(pl.client, "run", _capture)

    pl.analyze(_jpeg_bytes(), registry=registry)

    assert "{{PREDICTIONS}}" not in captured["system"]
    assert "- 舌色：淡紅（0.80）" in captured["system"]
    assert captured["system"].startswith("RULES\n")
    assert captured["system"].endswith("FORMAT")
    assert captured["user"] == pl.USER_TRIGGER


def test_analyze_stamps_comment_when_prompt_template_is_invalid(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色", label="淡紅")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "no marker here")
    monkeypatch.setattr(pl, "_load_llm_config", _stub_llm_config)

    called = []
    monkeypatch.setattr(pl.client, "run", lambda **kw: called.append(kw) or "DOCTOR")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)

    assert called == []  # client.run must NOT be invoked
    assert resp.comment.startswith(pl.client.ERROR_STAMP)
    assert "{{PREDICTIONS}}" in resp.comment
    assert resp.heads[0].task == "舌色"  # heads still populated
