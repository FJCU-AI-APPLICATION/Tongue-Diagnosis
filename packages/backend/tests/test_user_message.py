from tongue_ai.types import ClassScore, HeadResult

from tongue_backend.llm.user_message import build


def _hr(task, head_type, preds, error=None):
    return HeadResult(
        task=task,
        head_type=head_type,
        predictions=[ClassScore(label=l, score=s) for l, s in preds],
        error=error,
    )


def test_builds_zh_tw_bullet_list_with_confidences():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi",  [("齒痕", 0.84), ("嫩", 0.71)]),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    msg = build(heads)
    assert msg.startswith("本次舌診判讀結果：")
    assert "- 舌色：淡紅（0.78）" in msg
    assert "- 舌質：齒痕（0.84）、嫩（0.71）" in msg
    assert "- 舌態：無異常（0.91）" in msg
    assert msg.rstrip().endswith("請依規則輸出大眾版報告。")


def test_skips_heads_with_error():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi", [], error="boom"),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    msg = build(heads)
    assert "舌質" not in msg
    assert "舌色" in msg and "舌態" in msg


def test_handles_empty_heads_with_no_data_marker():
    msg = build([])
    assert "（無可用判讀資料）" in msg


# --- category_map mode (composite-head splitter) ----------------------------


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


def test_category_map_single_front_prediction_renders_one_bullet():
    heads = [_hr("front", "single", [("淡紅", 0.78)])]
    msg = build(heads, CATEGORY_MAP)
    assert "- 舌色：淡紅（0.78）" in msg
    assert "front" not in msg  # head name should NOT appear


def test_category_map_two_heads_render_two_bullets():
    heads = [
        _hr("front", "single", [("胖大", 0.62)]),
        _hr("sublingual", "single", [("怒張", 0.71)]),
    ]
    msg = build(heads, CATEGORY_MAP)
    assert "- 舌質：胖大（0.62）" in msg
    assert "- 舌下絡脈：怒張（0.71）" in msg


def test_category_map_cross_head_predictions_merge_under_one_category():
    heads = [
        _hr("front", "single", [("瘀血絲", 0.51)]),
        _hr("sublingual", "single", [("怒張", 0.72)]),
    ]
    msg = build(heads, CATEGORY_MAP)
    assert msg.count("- 舌下絡脈") == 1
    assert "瘀血絲（0.51）" in msg
    assert "怒張（0.72）" in msg


def test_category_map_skips_head_with_error():
    heads = [
        _hr("front", "single", [], error="boom"),
        _hr("sublingual", "single", [("怒張", 0.6)]),
    ]
    msg = build(heads, CATEGORY_MAP)
    assert "boom" not in msg
    assert "- 舌下絡脈：怒張（0.60）" in msg


def test_category_map_orphan_class_is_silently_dropped():
    heads = [_hr("front", "single", [("UNMAPPED", 0.9)])]
    msg = build(heads, CATEGORY_MAP)
    assert "UNMAPPED" not in msg
    assert "（無可用判讀資料）" in msg
