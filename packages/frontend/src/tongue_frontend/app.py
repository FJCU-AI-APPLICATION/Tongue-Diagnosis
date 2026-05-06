"""Gradio app — four tabs: 舌診分析 / 提示詞設定 / LLM 設定 / 模型設定."""

from __future__ import annotations

import os

import gradio as gr

from tongue_frontend.views import analyze, prompt_editor, llm_editor, registry_editor


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Tongue Diagnosis POC") as app:
        gr.Markdown("# 舌診 POC")
        with gr.Tabs():
            with gr.Tab("舌診分析"):
                analyze.build()
            with gr.Tab("提示詞設定"):
                prompt_editor.build()
            with gr.Tab("LLM 設定"):
                llm_editor.build()
            with gr.Tab("模型設定"):
                registry_editor.build()
    return app


if __name__ == "__main__":
    # Honour Gradio's standard env vars so the bind interface and port can be
    # changed without code edits (e.g., GRADIO_SERVER_NAME=127.0.0.1 to
    # restrict to localhost, or GRADIO_SERVER_PORT=8080).
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
