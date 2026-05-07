"""Gradio four-tab UI for tongue diagnosis."""
from __future__ import annotations

import os

import gradio as gr

from tongue_frontend.api import APIClient
from tongue_frontend.views import (
    analyze as analyze_view,
    llm_editor,
    prompt_editor,
    registry_editor,
)


def build_app() -> gr.Blocks:
    api_base = os.environ.get("TONGUE_API_BASE", "http://localhost:8000")
    client = APIClient(base_url=api_base)

    with gr.Blocks(title="Tongue Diagnosis") as app:
        gr.Markdown("# 舌診分析系統")
        with gr.Tabs():
            with gr.Tab("舌診分析"):
                analyze_view.build_tab(client)
            with gr.Tab("提示詞設定"):
                prompt_editor.build_tab(client)
            with gr.Tab("LLM 設定"):
                llm_editor.build_tab(client)
            with gr.Tab("模型設定"):
                registry_editor.build_tab(client)
    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))


if __name__ == "__main__":
    main()
