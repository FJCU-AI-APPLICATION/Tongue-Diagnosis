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
