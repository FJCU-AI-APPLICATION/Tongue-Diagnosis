"""Tab 4: 模型設定 — Save / Reset / Reload from disk / Apply & Reload Models."""
from __future__ import annotations

import gradio as gr

from tongue_frontend.api import APIClient


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        textarea = gr.Code(label="Registry YAML", language="yaml", lines=24)
        status = gr.Markdown()
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reset_btn = gr.Button("Reset to default")
            reload_btn = gr.Button("Reload from disk")
            apply_btn = gr.Button("Apply & Reload Models", variant="stop")

        def _load() -> tuple[str, str]:
            try:
                resp = client.get_config("registry")
            except Exception as exc:
                return "", f"❌ 讀取失敗：{exc}"
            badge = "（與預設一致）" if resp.get("is_default") else "（已修改）"
            return resp.get("content", ""), f"✅ 已載入 {badge}"

        def _save(content: str) -> str:
            try:
                resp = client.put_config("registry", content)
            except Exception as exc:
                return f"❌ 儲存失敗：{exc}"
            return "✅ 已儲存（尚未生效，請按 Apply & Reload Models）" if resp.get("saved") else f"❌ {resp}"

        def _reset() -> tuple[str, str]:
            try:
                client.reset_config("registry")
            except Exception as exc:
                return "", f"❌ Reset 失敗：{exc}"
            text, _ = _load()
            return text, "✅ 已恢復預設"

        def _apply() -> str:
            try:
                resp = client.reload_registry()
            except Exception as exc:
                return f"❌ Reload 失敗：{exc}"
            loaded = resp.get("loaded", [])
            failed = resp.get("failed", {})
            rolled_back = resp.get("rolled_back", False)
            if rolled_back:
                return f"⚠ 全部失敗，已回滾至前一版本。失敗：{failed}"
            return f"✅ Loaded {len(loaded)} heads. Failed: {failed if failed else '無'}"

        save_btn.click(_save, inputs=[textarea], outputs=[status])
        reset_btn.click(_reset, outputs=[textarea, status])
        reload_btn.click(_load, outputs=[textarea, status])
        apply_btn.click(_apply, outputs=[status])
        tab.load(_load, outputs=[textarea, status])
    return tab
