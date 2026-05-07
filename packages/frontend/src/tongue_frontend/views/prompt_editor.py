"""Tab 2: 提示詞設定."""
from __future__ import annotations

import gradio as gr

from tongue_frontend.api import APIClient


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        textarea = gr.Code(label="System Prompt", language="markdown", lines=24)
        status = gr.Markdown()
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reset_btn = gr.Button("Reset to default")
            reload_btn = gr.Button("Reload from disk")

        def _load() -> tuple[str, str]:
            try:
                resp = client.get_config("prompt")
            except Exception as exc:
                return "", f"❌ 讀取失敗：{exc}"
            badge = "（與預設一致）" if resp.get("is_default") else "（已修改）"
            return resp.get("content", ""), f"✅ 已載入 {badge}"

        def _save(content: str) -> str:
            try:
                resp = client.put_config("prompt", content)
            except Exception as exc:
                return f"❌ 儲存失敗：{exc}"
            return "✅ 已儲存" if resp.get("saved") else f"❌ {resp}"

        def _reset() -> tuple[str, str]:
            try:
                client.reset_config("prompt")
            except Exception as exc:
                return "", f"❌ Reset 失敗：{exc}"
            text, _msg = _load()
            return text, "✅ 已恢復預設"

        save_btn.click(_save, inputs=[textarea], outputs=[status])
        reset_btn.click(_reset, outputs=[textarea, status])
        reload_btn.click(_load, outputs=[textarea, status])
        tab.load(_load, outputs=[textarea, status])
    return tab
