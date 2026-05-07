"""Golden tests for the composite-head → v4-category user message builder."""
from __future__ import annotations

from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.llm.user_message import build


CATEGORY_MAP = {
    "front": {
        "淡紅": "舌色",
        "胖大": "舌質",
        "偏斜": "舌態",
        "瘀血絲": "舌下絡脈",
        "無異常": "舌質",
    },
    "sublingual": {
        "怒張": "舌下絡脈",
        "曲張": "舌下絡脈",
    },
}


def test_single_front_prediction_renders_one_bullet():
    heads = [HeadResult(task="front", head_type="single", predictions=[ClassScore("淡紅", 0.78)])]
    out = build(heads, CATEGORY_MAP)
    assert "本次舌診判讀結果" in out
    assert "- 舌色：淡紅（0.78）" in out
    assert "請依規則輸出大眾版報告。" in out


def test_two_heads_render_two_bullets():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[ClassScore("胖大", 0.62)]),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.71)]),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "- 舌質：胖大（0.62）" in out
    assert "- 舌下絡脈：怒張（0.71）" in out


def test_cross_head_predictions_merge_under_one_category():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[ClassScore("瘀血絲", 0.51)]),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.72)]),
    ]
    out = build(heads, CATEGORY_MAP)
    # 舌下絡脈 line should contain both classes joined by 、
    assert "舌下絡脈" in out
    assert "瘀血絲（0.51）" in out
    assert "怒張（0.72）" in out
    assert out.count("- 舌下絡脈") == 1


def test_head_with_error_is_skipped():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[], error="boom"),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.6)]),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "boom" not in out
    assert "- 舌下絡脈：怒張（0.60）" in out


def test_no_predictions_renders_no_data_line():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[], error="x"),
        HeadResult(task="sublingual", head_type="single", predictions=[], error="y"),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "（無可用判讀資料）" in out


def test_class_not_in_category_map_is_skipped_silently():
    heads = [HeadResult(task="front", head_type="single", predictions=[ClassScore("UNMAPPED", 0.9)])]
    out = build(heads, CATEGORY_MAP)
    assert "UNMAPPED" not in out
    assert "（無可用判讀資料）" in out
