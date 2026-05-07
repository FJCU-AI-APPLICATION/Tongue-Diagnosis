"""Tests for tongue_ai.types dataclasses."""
from __future__ import annotations

from tongue_ai.types import BBox, ClassScore, HeadResult, Normalisation


def test_class_score_holds_label_and_score():
    cs = ClassScore(label="淡紅", score=0.78)
    assert cs.label == "淡紅"
    assert cs.score == 0.78


def test_head_result_defaults_error_to_none():
    hr = HeadResult(
        task="front",
        head_type="single",
        predictions=[ClassScore(label="淡紅", score=0.78)],
    )
    assert hr.error is None
    assert hr.predictions[0].label == "淡紅"


def test_head_result_with_error_has_empty_predictions():
    hr = HeadResult(task="front", head_type="single", predictions=[], error="boom")
    assert hr.error == "boom"
    assert hr.predictions == []


def test_normalisation_holds_mean_and_std():
    n = Normalisation(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert n.mean == (0.485, 0.456, 0.406)
    assert n.std == (0.229, 0.224, 0.225)


def test_bbox_holds_coordinates():
    b = BBox(x=10, y=20, w=30, h=40, confidence=0.9)
    assert b.x == 10 and b.y == 20 and b.w == 30 and b.h == 40
    assert b.confidence == 0.9
