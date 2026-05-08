"""Tab 2: 提示詞設定 — system prompt textarea."""

from __future__ import annotations

import gradio as gr
import httpx

from frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("prompt")
    flag = "default" if s.is_default else "custom"
    return s.content, f"狀態：{flag}"


def _save(content: str) -> str:
    try:
        api.put_config("prompt", content)
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return f"⚠ 儲存失敗：{err}"
    return "已儲存 — 下次分析將使用新提示詞"


def _reset() -> tuple[str, str]:
    api.reset_config("prompt")
    content, _ = _load()
    return content, "已還原預設值"


def build(app: gr.Blocks) -> None:
    """Add the prompt editor into the current Gradio context.

    `app` is the root ``gr.Blocks`` — used for ``app.load`` so initial-state
    hydration fires once when the page loads. No nested ``gr.Blocks()``.
    """
    gr.Markdown("### 系統提示詞 (大眾版規則)")
    gr.Markdown("編輯後按「儲存」即可生效；下次「分析」會立刻採用。")
    gr.Markdown(
        "⚠ 提示詞必須包含 `{{PREDICTIONS}}` 標記**一次**"
        "（會被替換成本次判讀結果）。缺少或重複會導致儲存失敗。"
    )
    textbox = gr.Textbox(
        lines=30, label="system prompt", buttons=["copy"]
    )
    with gr.Row():
        save_btn = gr.Button("儲存", variant="primary")
        reset_btn = gr.Button("還原預設")
        reload_btn = gr.Button("從磁碟重新載入")
    status_box = gr.Markdown()

    app.load(fn=_load, outputs=[textbox, status_box])
    save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
    reset_btn.click(fn=_reset, outputs=[textbox, status_box])
    reload_btn.click(fn=_load, outputs=[textbox, status_box])
