"""Tab 1: 舌診分析 — capture/upload + result panel."""

from __future__ import annotations

import io

import gradio as gr
import httpx
import numpy as np
from PIL import Image

from tongue_frontend import api
from tongue_frontend.models import HeadResult
from tongue_frontend.settings import settings


def _heads_to_rows(heads: list[HeadResult]) -> list[list[str]]:
    rows = []
    for h in heads:
        if h.error:
            rows.append([h.task, f"⚠ {h.error}"])
            continue
        preds = "、".join(f"{p.label} ({p.score:.2f})" for p in h.predictions)
        rows.append([h.task, preds or "(無)"])
    return rows


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
    rows = _heads_to_rows(result.heads)
    timing = result.timing_ms.model_dump()
    timing_str = " · ".join(f"{k}: {v}ms" for k, v in timing.items())
    return rows, result.comment, result.disclaimer, result.user_message, timing_str


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
                user_msg_box = gr.Textbox(label="送至 Gemini 的 user message", lines=10)
                timing_box = gr.Textbox(label="耗時 (ms)")

    go.click(
        fn=_on_analyze,
        inputs=[img],
        outputs=[heads_table, comment_md, disclaimer_md, user_msg_box, timing_box],
    )
