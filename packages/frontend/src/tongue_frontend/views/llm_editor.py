"""Tab 3: LLM 設定 — model, temperature, max_tokens, top_p."""

from __future__ import annotations

import gradio as gr
import httpx

from tongue_frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("llm")
    flag = "default" if s.is_default else "custom"
    return s.content, f"狀態：{flag}"


def _save(content: str) -> str:
    try:
        api.put_config("llm", content)
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return f"⚠ 儲存失敗：{err}"
    return "已儲存 — 下次分析將使用新設定"


def _reset() -> tuple[str, str]:
    api.reset_config("llm")
    content, _ = _load()
    return content, "已還原預設值"


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### LLM 設定 (Gemini via ADK)")
        gr.Markdown("以 YAML 編輯：`model`, `temperature` ∈ [0,2], `max_tokens` > 0, `top_p` ∈ (0,1].")
        textbox = gr.Code(language="yaml", label="llm.yaml", lines=10)
        with gr.Row():
            save_btn = gr.Button("儲存", variant="primary")
            reset_btn = gr.Button("還原預設")
            reload_btn = gr.Button("從磁碟重新載入")
        status_box = gr.Markdown()

        view.load(fn=_load, outputs=[textbox, status_box])
        save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
        reset_btn.click(fn=_reset, outputs=[textbox, status_box])
        reload_btn.click(fn=_load, outputs=[textbox, status_box])
    return view
