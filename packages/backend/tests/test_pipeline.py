"""Tests for pipeline.analyze: stub registry + stub LLM → asserted response shape."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.pipeline import AnalyzeResponse, analyze, DISCLAIMER


def _jpeg_bytes() -> bytes:
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_registry(predictions_per_head: dict[str, ClassScore | None]) -> Registry:
    heads = []
    for name, pred in predictions_per_head.items():
        h = MagicMock()
        h.name = name
        h.predict.return_value = HeadResult(
            task=name,
            head_type="single",
            predictions=[pred] if pred else [],
            error=None if pred else "no signal",
        )
        heads.append(h)
    return Registry(
        detector_enabled=False,
        heads=heads,
        category_map={
            "front": {"淡紅": "舌色"},
            "sublingual": {"怒張": "舌下絡脈"},
        },
    )


def test_analyze_returns_expected_shape():
    registry = _make_registry({
        "front": ClassScore("淡紅", 0.78),
        "sublingual": ClassScore("怒張", 0.65),
    })
    llm = MagicMock()
    llm.run.return_value = "## 主要中醫體質\n氣虛\n## 警語\n..."

    resp = analyze(
        image_bytes=_jpeg_bytes(),
        registry=registry,
        prompt="LOCKED PROMPT",
        llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
        llm_client=llm,
    )

    assert isinstance(resp, AnalyzeResponse)
    assert resp.disclaimer == DISCLAIMER
    assert "舌色：淡紅" in resp.user_message
    assert "舌下絡脈：怒張" in resp.user_message
    assert resp.comment.startswith("## 主要中醫體質")
    assert resp.category_map["front"]["淡紅"] == "舌色"
    assert resp.timing_ms["total"] >= 0
    llm.run.assert_called_once()
    assert llm.run.call_args.kwargs["system"] == "LOCKED PROMPT"


def test_analyze_continues_when_one_head_errors():
    registry = _make_registry({"front": ClassScore("淡紅", 0.7), "sublingual": None})
    llm = MagicMock()
    llm.run.return_value = "OK"
    resp = analyze(
        image_bytes=_jpeg_bytes(),
        registry=registry,
        prompt="P",
        llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
        llm_client=llm,
    )
    assert any(h.task == "sublingual" and h.error for h in resp.heads)
    assert "舌色：淡紅" in resp.user_message


def test_analyze_raises_value_error_on_undecodable_image():
    registry = _make_registry({"front": ClassScore("淡紅", 0.5)})
    llm = MagicMock()
    with pytest.raises(ValueError, match="decode"):
        analyze(
            image_bytes=b"not an image",
            registry=registry,
            prompt="P",
            llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
            llm_client=llm,
        )
