"""Render HeadResult[] as the bullet block that fills {{PREDICTIONS}}.

Two modes (selected by ``category_map``):

* No category_map: one bullet per head, keyed by ``head.task``. Suitable for
  per-task heads whose ``task`` is already a v4 schema category.
* With category_map: composite-head predictions are split back into v4 schema
  categories via ``category_map[head.task][label] -> v4_category``, grouped
  across heads, and emitted as one bullet per category.
"""

from __future__ import annotations

from collections import OrderedDict

from ai.types import ClassScore, HeadResult


EMPTY_LINE = "- （無可用判讀資料）"


def render(
    heads: list[HeadResult],
    category_map: dict[str, dict[str, str]] | None = None,
) -> str:
    if category_map:
        lines = _render_with_map(heads, category_map)
    else:
        lines = _render_per_head(heads)
    return "\n".join(lines) if lines else EMPTY_LINE


def _render_per_head(heads: list[HeadResult]) -> list[str]:
    out: list[str] = []
    for h in heads:
        if h.error or not h.predictions:
            continue
        rendered = "、".join(f"{p.label}（{p.score:.2f}）" for p in h.predictions)
        out.append(f"- {h.task}：{rendered}")
    return out


def _render_with_map(
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
