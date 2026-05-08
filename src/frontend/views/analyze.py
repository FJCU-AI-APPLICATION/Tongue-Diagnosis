"""Tab 1: 舌診分析 — capture/upload + result panel."""

from __future__ import annotations

import io
from collections import OrderedDict

import gradio as gr
import httpx
import numpy as np
from PIL import Image

from frontend import api
from frontend.models import ClassScore, HeadResult
from frontend.settings import settings


def _heads_to_rows(
    heads: list[HeadResult],
    category_map: dict[str, dict[str, str]] | None = None,
) -> list[list[str]]:
    """Render heads as ``[label, body]`` rows for the 各項判讀 dataframe.

    With ``category_map``, composite-head predictions are grouped under their
    v4 schema category in Chinese (mirrors ``backend.llm.predictions``).
    Without it, falls back to ``head.task`` — registries that already use
    Chinese task names render correctly under the same path.
    """
    if not category_map:
        return [_row_for_head(h, h.task) for h in heads]

    grouped: OrderedDict[str, list[ClassScore]] = OrderedDict()
    fallback_rows: list[list[str]] = []
    for h in heads:
        head_map = category_map.get(h.task)
        if head_map is None:
            fallback_rows.append(_row_for_head(h, h.task))
            continue
        if h.error:
            label = " / ".join(_unique_categories(head_map))
            fallback_rows.append([label, f"⚠ {h.error}"])
            continue
        for pred in h.predictions:
            cat = head_map.get(pred.label)
            if cat is None:
                continue  # orphan class — silently dropped, matches backend
            grouped.setdefault(cat, []).append(pred)

    rows = [[cat, "、".join(_format_pred(p) for p in preds)] for cat, preds in grouped.items()]
    rows.extend(fallback_rows)
    return rows


def _row_for_head(h: HeadResult, label: str) -> list[str]:
    if h.error:
        return [label, f"⚠ {h.error}"]
    body = "、".join(_format_pred(p) for p in h.predictions)
    return [label, body or "(無)"]


def _format_pred(p: ClassScore) -> str:
    return f"{p.label} ({p.score:.2f})"


def _unique_categories(head_map: dict[str, str]) -> list[str]:
    seen: dict[str, None] = {}
    for cat in head_map.values():
        seen.setdefault(cat, None)
    return list(seen)


def _to_jpeg_bytes(image: np.ndarray) -> bytes:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _on_analyze(image: np.ndarray | None):
    if image is None:
        return [], "請選擇或拍攝照片", "", "", ""
    try:
        result = api.analyze(_to_jpeg_bytes(image))
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return [], "", f"⚠ 分析失敗：{err}", "", ""
    except httpx.ConnectError:
        return [], "", f"⚠ 無法連線到後端 ({settings.backend_url}) — 請啟動 backend", "", ""
    rows = _heads_to_rows(result.heads, result.category_map)
    timing = result.timing_ms.model_dump()
    timing_str = " · ".join(f"{k}: {v}ms" for k, v in timing.items())
    return rows, result.comment, result.disclaimer, result.predictions_block, timing_str


def build(app: gr.Blocks) -> None:
    """Add the analyze view's components into the current Gradio context.

    `app` is unused here (no view.load needed) but kept for API consistency
    with the editor views. Components are placed directly into whichever
    Tab/Accordion/Blocks the caller is in — no nested ``gr.Blocks()`` wrapper,
    which Gradio 6.14 dislikes when stacked across ≥3 sibling Tabs.
    """
    del app  # unused — kept for API parity
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 舌診影像")
            img = gr.Image(sources=["webcam", "upload"], type="numpy",
                           label="拍攝或上傳", height=320)
            go = gr.Button("分析", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("### 判讀結果")
            heads_table = gr.Dataframe(
                headers=["項目", "預測"],
                interactive=False,
                label="各項判讀",
            )
            comment_md = gr.Markdown(label="醫師建議")
            disclaimer_md = gr.Markdown()
            with gr.Accordion("進階 (debug)", open=False):
                user_msg_box = gr.Textbox(label="判讀區塊（已注入 system prompt）", lines=10)
                timing_box = gr.Textbox(label="耗時 (ms)")

    go.click(
        fn=_on_analyze,
        inputs=[img],
        outputs=[heads_table, comment_md, disclaimer_md, user_msg_box, timing_box],
    )
