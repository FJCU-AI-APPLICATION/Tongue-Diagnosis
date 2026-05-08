"""Tests for the analyze view's row rendering.

The 各項判讀 dataframe must show v4 schema categories in Chinese
(舌色 / 舌質 / 舌下絡脈 / ...) — never the composite-head role names
("front" / "sublingual"). Composite heads bridge to v4 categories via
``category_map``, mirroring the backend's ``predictions._render_with_map``.
"""
from __future__ import annotations

from frontend.models import AnalyzeResponse, ClassScore, HeadResult
from frontend.views.analyze import _heads_to_rows


def _hr(task, head_type, preds, error=None):
    return HeadResult(
        task=task,
        head_type=head_type,
        predictions=[ClassScore(label=l, score=s) for l, s in preds],
        error=error,
    )


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


# --- no category_map: per-task registries (task is already Chinese) ---


def test_heads_to_rows_no_category_map_uses_head_task_as_label():
    heads = [_hr("舌色", "single", [("淡紅", 0.78)])]
    rows = _heads_to_rows(heads)
    assert rows == [["舌色", "淡紅 (0.78)"]]


def test_heads_to_rows_no_category_map_errored_head_uses_task_as_label():
    heads = [_hr("舌色", "single", [], error="boom")]
    rows = _heads_to_rows(heads)
    assert rows == [["舌色", "⚠ boom"]]


def test_heads_to_rows_no_category_map_empty_predictions_render_no_marker():
    heads = [_hr("舌色", "single", [])]
    rows = _heads_to_rows(heads)
    assert rows == [["舌色", "(無)"]]


# --- with category_map: composite heads grouped under v4 categories ---


def test_heads_to_rows_with_map_translates_composite_head_task_to_chinese_category():
    heads = [_hr("front", "single", [("淡紅", 0.78)])]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert rows == [["舌色", "淡紅 (0.78)"]]
    assert all("front" not in label for label, _ in rows)


def test_heads_to_rows_with_map_two_heads_emit_two_category_rows():
    heads = [
        _hr("front", "single", [("胖大", 0.62)]),
        _hr("sublingual", "single", [("怒張", 0.71)]),
    ]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert ["舌質", "胖大 (0.62)"] in rows
    assert ["舌下絡脈", "怒張 (0.71)"] in rows


def test_heads_to_rows_with_map_groups_cross_head_predictions_under_one_category():
    heads = [
        _hr("front", "single", [("瘀血絲", 0.51)]),
        _hr("sublingual", "single", [("怒張", 0.72)]),
    ]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    labels = [label for label, _ in rows]
    assert labels.count("舌下絡脈") == 1
    body = next(body for label, body in rows if label == "舌下絡脈")
    assert "瘀血絲 (0.51)" in body
    assert "怒張 (0.72)" in body


def test_heads_to_rows_with_map_drops_orphan_classes():
    heads = [_hr("front", "single", [("UNMAPPED_LABEL", 0.9)])]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert rows == []


def test_heads_to_rows_with_map_unmapped_head_falls_back_to_task():
    """A head missing from category_map shouldn't disappear — show it raw."""
    heads = [_hr("舌色", "single", [("淡紅", 0.78)])]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert rows == [["舌色", "淡紅 (0.78)"]]


def test_heads_to_rows_with_map_errored_head_renders_chinese_categories():
    """Errored composite heads must not leak the English task name."""
    heads = [_hr("front", "single", [], error="boom")]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert rows
    assert all("front" not in label for label, _ in rows)
    assert all("⚠ boom" in body for _, body in rows)


def test_heads_to_rows_with_map_errored_unmapped_head_falls_back_to_task():
    heads = [_hr("舌色", "single", [], error="boom")]
    rows = _heads_to_rows(heads, CATEGORY_MAP)
    assert rows == [["舌色", "⚠ boom"]]


# --- AnalyzeResponse must mirror the backend's category_map field ---


def test_analyze_response_parses_category_map_field():
    body = {
        "predictions_block": "p",
        "heads": [],
        "comment": "OK",
        "disclaimer": "d",
        "category_map": {"front": {"淡紅": "舌色"}},
        "timing_ms": {"decode": 1, "detect": 2, "infer": 3, "llm": 4, "total": 10},
    }
    out = AnalyzeResponse.model_validate(body)
    assert out.category_map == {"front": {"淡紅": "舌色"}}


def test_analyze_response_category_map_defaults_to_empty_dict():
    body = {
        "predictions_block": "p",
        "heads": [],
        "comment": "OK",
        "disclaimer": "d",
        "timing_ms": {"decode": 1, "detect": 2, "infer": 3, "llm": 4, "total": 10},
    }
    out = AnalyzeResponse.model_validate(body)
    assert out.category_map == {}
