"""Tab 2: 提示詞設定 — system prompt textarea."""

from __future__ import annotations

import gradio as gr

from tongue_frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("prompt")
    flag = "default" if s.get("is_default") else "custom"
    return s["content"], f"狀態：{flag}"


def _save(content: str) -> str:
    out = api.put_config("prompt", content)
    if "error" in out:
        return f"⚠ 儲存失敗：{out['error']}"
    return "已儲存 — 下次分析將使用新提示詞"


def _reset() -> tuple[str, str]:
    api.reset_config("prompt")
    content, status = _load()
    return content, "已還原預設值"


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### 系統提示詞 (大眾版規則)")
        gr.Markdown("編輯後按「儲存」即可生效；下次「分析」會立刻採用。")
        textbox = gr.Textbox(
            lines=30, label="system prompt", show_copy_button=True
        )
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
