"""Build the per-request zh-TW user message for the locked Gemini prompt.

Composite-head predictions are split back into v4 schema categories via
the static category_map, then grouped so each category renders one bullet.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from tongue_ai.types import ClassScore, HeadResult


HEADER = "本次舌診判讀結果："
FOOTER = "請依規則輸出大眾版報告。"
EMPTY_LINE = "- （無可用判讀資料）"


def build(
    heads: Iterable[HeadResult],
    category_map: dict[str, dict[str, str]],
) -> str:
    grouped: OrderedDict[str, list[ClassScore]] = OrderedDict()
    for head in heads:
        if head.error is not None:
            continue
        head_map = category_map.get(head.task, {})
        for pred in head.predictions:
            cat = head_map.get(pred.label)
            if cat is None:
                continue
            grouped.setdefault(cat, []).append(pred)

    lines: list[str] = [HEADER, ""]
    if not grouped:
        lines.append(EMPTY_LINE)
    else:
        for cat, preds in grouped.items():
            joined = "、".join(f"{p.label}（{p.score:.2f}）" for p in preds)
            lines.append(f"- {cat}：{joined}")
    lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines)
