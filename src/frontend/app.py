"""Gradio app — two tabs: 舌診分析 / 設定 (three accordions inside).

Gradio 6.14 has a Svelte ``effect_update_depth_exceeded`` regression that
fires on initial render whenever ≥3 sibling ``gr.Tab`` items each contain a
nested ``gr.Blocks()``. The settings tabs are consolidated into one tab with
``gr.Accordion`` sections, and each view's ``build()`` no longer wraps its
components in a nested ``gr.Blocks()``.
"""

from __future__ import annotations

import gradio as gr

from frontend.settings import settings
from frontend.views import analyze, prompt_editor, llm_editor, registry_editor


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Tongue Diagnosis POC") as app:
        gr.Markdown("# 舌診 POC")
        with gr.Tabs():
            with gr.Tab("舌診分析"):
                analyze.build(app)
            with gr.Tab("設定"):
                with gr.Accordion("LLM 設定 (Gemini API key + model)", open=True):
                    llm_editor.build(app)
                with gr.Accordion("提示詞 (System prompt)", open=False):
                    prompt_editor.build(app)
                with gr.Accordion("模型 (Registry)", open=False):
                    registry_editor.build(app)
    return app


def run() -> None:
    """Console-script entry point for `tongue-frontend`."""
    build_app().launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    run()
