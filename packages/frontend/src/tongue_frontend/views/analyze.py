"""Tab 1: 舌診分析 — capture/upload + result panel."""
from __future__ import annotations

import io
import json

import gradio as gr
import numpy as np
from PIL import Image

from tongue_frontend.api import APIClient


def _to_jpeg_bytes(img: np.ndarray) -> bytes:
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _format_heads_table(heads: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for h in heads:
        if h.get("error"):
            rows.append([h["task"], "(error)", h["error"]])
            continue
        labels = "、".join(f"{p['label']}（{p['score']:.2f}）" for p in h.get("predictions", []))
        rows.append([h["task"], labels, ""])
    return rows


def _format_v4_breakdown(category_map: dict, heads: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for h in heads:
        head_map = category_map.get(h["task"], {})
        for p in h.get("predictions", []):
            cat = head_map.get(p["label"])
            if cat is not None:
                rows.append([cat, p["label"], f"{p['score']:.2f}"])
    return rows


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="舌象照片", sources=["upload", "webcam"], type="numpy")
                analyze_btn = gr.Button("分析", variant="primary")
            with gr.Column(scale=2):
                comment_md = gr.Markdown(label="醫師建議")
                disclaimer_md = gr.Markdown(label="警語")
                heads_table = gr.Dataframe(
                    headers=["Head", "Predictions", "Error"], label="原始模型輸出",
                )
                v4_table = gr.Dataframe(
                    headers=["v4 類別", "標籤", "信心度"], label="v4 對應分解",
                )
                with gr.Accordion("進階：傳給 LLM 的訊息 / 計時", open=False):
                    user_msg = gr.Code(label="user_message", language=None)
                    timing_json = gr.JSON(label="timing_ms")

        def _on_analyze(img):
            if img is None:
                return "未提供影像", "", [], [], "", {}
            try:
                jpeg = _to_jpeg_bytes(img)
                result = client.analyze(filename="capture.jpg", content=jpeg, content_type="image/jpeg")
            except Exception as exc:
                return f"後端連線失敗：{exc}", "", [], [], "", {}
            return (
                result.get("comment", ""),
                result.get("disclaimer", ""),
                _format_heads_table(result.get("heads", [])),
                _format_v4_breakdown(result.get("category_map", {}), result.get("heads", [])),
                result.get("user_message", ""),
                result.get("timing_ms", {}),
            )

        analyze_btn.click(
            _on_analyze,
            inputs=[image_input],
            outputs=[comment_md, disclaimer_md, heads_table, v4_table, user_msg, timing_json],
        )
    return tab
