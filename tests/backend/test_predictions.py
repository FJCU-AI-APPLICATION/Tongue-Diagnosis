from ai.types import ClassScore, HeadResult

from backend.llm.predictions import render


def _hr(task, head_type, preds, error=None):
    return HeadResult(
        task=task,
        head_type=head_type,
        predictions=[ClassScore(label=l, score=s) for l, s in preds],
        error=error,
    )


def test_render_per_head_emits_one_bullet_per_head():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi",  [("齒痕", 0.84), ("嫩", 0.71)]),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    out = render(heads)
    assert out == "- 舌色：淡紅（0.78）\n- 舌質：齒痕（0.84）、嫩（0.71）\n- 舌態：無異常（0.91）"


def test_render_per_head_skips_errored_heads():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi", [], error="boom"),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    out = render(heads)
    assert "舌質" not in out
    assert "舌色" in out and "舌態" in out


def test_render_empty_list_returns_no_data_sentinel():
    assert render([]) == "- （無可用判讀資料）"


def test_render_no_predictions_on_any_head_returns_sentinel():
    heads = [_hr("舌色", "single", [])]
    assert render(heads) == "- （無可用判讀資料）"


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


def test_render_with_category_map_single_head():
    heads = [_hr("front", "single", [("淡紅", 0.78)])]
    out = render(heads, CATEGORY_MAP)
    assert out == "- 舌色：淡紅（0.78）"
    assert "front" not in out


def test_render_with_category_map_two_heads():
    heads = [
        _hr("front", "single", [("胖大", 0.62)]),
        _hr("sublingual", "single", [("怒張", 0.71)]),
    ]
    out = render(heads, CATEGORY_MAP)
    assert "- 舌質：胖大（0.62）" in out
    assert "- 舌下絡脈：怒張（0.71）" in out


def test_render_with_category_map_merges_cross_head_categories():
    heads = [
        _hr("front", "single", [("瘀血絲", 0.51)]),
        _hr("sublingual", "single", [("怒張", 0.72)]),
    ]
    out = render(heads, CATEGORY_MAP)
    assert out.count("- 舌下絡脈") == 1
    assert "瘀血絲（0.51）" in out
    assert "怒張（0.72）" in out


def test_render_with_category_map_drops_orphan_class():
    heads = [_hr("front", "single", [("UNMAPPED", 0.9)])]
    out = render(heads, CATEGORY_MAP)
    assert "UNMAPPED" not in out
    assert out == "- （無可用判讀資料）"


def test_render_with_category_map_skips_errored_head():
    heads = [
        _hr("front", "single", [], error="boom"),
        _hr("sublingual", "single", [("怒張", 0.6)]),
    ]
    out = render(heads, CATEGORY_MAP)
    assert "boom" not in out
    assert "- 舌下絡脈：怒張（0.60）" in out


def test_render_does_not_emit_header_or_footer():
    heads = [_hr("舌色", "single", [("淡紅", 0.78)])]
    out = render(heads)
    assert "本次舌診判讀結果" not in out
    assert "請依規則輸出大眾版報告" not in out
