"""Tab 4: 模型設定 — registry YAML + reload."""

from __future__ import annotations

import gradio as gr
import httpx

from frontend import api


def _err(e: httpx.HTTPStatusError) -> str:
    try:
        return e.response.json().get("error", str(e))
    except Exception:
        return str(e)


def _load() -> tuple[str, str]:
    s = api.get_config("registry")
    flag = "default" if s.is_default else "custom"
    h = api.health()
    heads = "、".join(h.heads_loaded) or "(none)"
    return s.content, f"狀態：{flag}　已載入 {len(h.heads_loaded)} 個 heads：{heads}"


def _save(content: str) -> str:
    try:
        api.put_config("registry", content)
    except httpx.HTTPStatusError as e:
        return f"⚠ 儲存失敗：{_err(e)}"
    return "已儲存 — 點「Apply & Reload Models」生效"


def _reset() -> tuple[str, str]:
    api.reset_config("registry")
    content, _ = _load()
    return content, "已還原預設值"


def _reload() -> str:
    try:
        out = api.reload_registry()
    except httpx.HTTPStatusError as e:
        return f"⚠ 重新載入失敗：{_err(e)}"
    msg = f"已載入 {len(out.loaded)} heads"
    if out.failed:
        msg += f"，失敗 {len(out.failed)}：{out.failed}"
    return msg


def build(app: gr.Blocks) -> None:
    """Add the registry editor into the current Gradio context.

    `app` is the root ``gr.Blocks`` — used for ``app.load`` so initial-state
    hydration fires once when the page loads. No nested ``gr.Blocks()``.
    """
    gr.Markdown("### 模型設定 (registry YAML)")
    gr.Markdown(
        "編輯各 task head 的 ONNX 路徑、輸入大小、normalisation、class_names、threshold。"
        " 「儲存」只會持久化 YAML；要套用必須點「Apply & Reload Models」。"
    )
    textbox = gr.Code(language="yaml", label="registry.yaml", lines=30)
    with gr.Row():
        save_btn = gr.Button("儲存")
        reset_btn = gr.Button("還原預設")
        reload_disk_btn = gr.Button("從磁碟重新載入")
        apply_btn = gr.Button("Apply & Reload Models", variant="primary")
    status_box = gr.Markdown()
    reload_status = gr.Markdown()

    app.load(fn=_load, outputs=[textbox, status_box])
    save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
    reset_btn.click(fn=_reset, outputs=[textbox, status_box])
    reload_disk_btn.click(fn=_load, outputs=[textbox, status_box])
    apply_btn.click(fn=_reload, outputs=[reload_status])
