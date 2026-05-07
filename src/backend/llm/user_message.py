"""Build the per-request user message for Gemini from HeadResult[].

Two modes, selected by whether ``category_map`` is supplied:

* No category_map (origin behaviour): one bullet per head, keyed by
  ``head.task``. Suitable for per-task heads whose ``task`` is already a
  v4 schema category (e.g. ``舌色``, ``舌質``).

* With category_map: composite-head predictions are split back into v4
  schema categories via ``category_map[head.task][label] -> v4_category``,
  grouped across heads, and emitted as one bullet per category. Required
  for Amanda's two composite heads (``front``, ``sublingual``).
"""

from __future__ import annotations

from collections import OrderedDict

from ai.types import ClassScore, HeadResult


HEADER = "本次舌診判讀結果："
FOOTER = "請依規則輸出大眾版報告。"
EMPTY_LINE = "- （無可用判讀資料）"


def build(
    heads: list[HeadResult],
    category_map: dict[str, dict[str, str]] | None = None,
) -> str:
    if category_map:
        rendered_lines = _build_with_category_map(heads, category_map)
    else:
        rendered_lines = _build_per_head(heads)

    body = "\n".join(rendered_lines) if rendered_lines else EMPTY_LINE
    return f"{HEADER}\n\n{body}\n\n{FOOTER}"


def _build_per_head(heads: list[HeadResult]) -> list[str]:
    out: list[str] = []
    for h in heads:
        if h.error or not h.predictions:
            continue
        rendered = "、".join(f"{p.label}（{p.score:.2f}）" for p in h.predictions)
        out.append(f"- {h.task}：{rendered}")
    return out


def _build_with_category_map(
    heads: list[HeadResult],
    category_map: dict[str, dict[str, str]],
) -> list[str]:
    grouped: OrderedDict[str, list[ClassScore]] = OrderedDict()
    for h in heads:
        if h.error:
            continue
        head_map = category_map.get(h.task, {})
        for pred in h.predictions:
            cat = head_map.get(pred.label)
            if cat is None:
                continue  # orphan class — silently dropped
            grouped.setdefault(cat, []).append(pred)

    out: list[str] = []
    for cat, preds in grouped.items():
        rendered = "、".join(f"{p.label}（{p.score:.2f}）" for p in preds)
        out.append(f"- {cat}：{rendered}")
    return out
