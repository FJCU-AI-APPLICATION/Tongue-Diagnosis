"""Build the per-request user message for Gemini from HeadResult[]."""

from __future__ import annotations

from tongue_ai.types import HeadResult


HEADER = "本次舌診判讀結果："
FOOTER = "請依規則輸出大眾版報告。"
EMPTY_LINE = "- （無可用判讀資料）"


def build(heads: list[HeadResult]) -> str:
    lines: list[str] = []
    for h in heads:
        if h.error or not h.predictions:
            continue
        rendered = "、".join(f"{p.label}（{p.score:.2f}）" for p in h.predictions)
        lines.append(f"- {h.task}：{rendered}")

    body = "\n".join(lines) if lines else EMPTY_LINE
    return f"{HEADER}\n\n{body}\n\n{FOOTER}"
