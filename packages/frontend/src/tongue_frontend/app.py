"""Gradio app — four tabs: 舌診分析 / 提示詞設定 / LLM 設定 / 模型設定."""

from __future__ import annotations

import gradio as gr

from tongue_frontend.settings import settings
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
    build_app().launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )
