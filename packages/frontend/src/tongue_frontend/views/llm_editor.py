"""Tab 3: LLM 設定 — Gemini API key + model/temperature/max_tokens/top_p."""

from __future__ import annotations

import gradio as gr
import httpx

from tongue_frontend import api


# --- Gemini API key row ------------------------------------------------------


def _api_key_status_text() -> str:
    s = api.get_api_key_status()
    if s.is_set:
        return f"狀態：已設定 (fingerprint: {s.fingerprint})"
    return "狀態：未設定"


def _save_api_key(content: str) -> tuple[str, str]:
    """Returns (cleared_textbox_value, status_text)."""
    if not content or not content.strip():
        return content, "⚠ 請先輸入金鑰"
    try:
        api.put_api_key(content.strip())
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return content, f"⚠ {err}"
    except httpx.ConnectError:
        return content, "⚠ 無法連線到後端"
    # Clear the textbox on success — never leave the key in the browser.
    return "", _api_key_status_text()


def _clear_api_key() -> tuple[str, str]:
    try:
        api.clear_api_key()
    except httpx.HTTPStatusError as e:
        return "", f"⚠ {e}"
    except httpx.ConnectError:
        return "", "⚠ 無法連線到後端"
    return "", _api_key_status_text()


# --- LLM YAML editor (unchanged behavior) -----------------------------------


def _load_yaml() -> tuple[str, str]:
    s = api.get_config("llm")
    flag = "default" if s.is_default else "custom"
    return s.content, f"狀態：{flag}"


def _save_yaml(content: str) -> str:
    try:
        api.put_config("llm", content)
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return f"⚠ 儲存失敗：{err}"
    return "已儲存 — 下次分析將使用新設定"


def _reset_yaml() -> tuple[str, str]:
    api.reset_config("llm")
    content, _ = _load_yaml()
    return content, "已還原預設值"


# --- View build --------------------------------------------------------------


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### Gemini API key")
        gr.Markdown(
            "在此輸入 Google AI Studio 取得的金鑰；後端會以一次最小呼叫驗證後再儲存。"
            "金鑰永遠不會回傳到瀏覽器。"
        )
        api_key_box = gr.Textbox(
            label="API key",
            type="password",
            placeholder="AIza…",
        )
        with gr.Row():
            api_key_save_btn = gr.Button("儲存", variant="primary")
            api_key_clear_btn = gr.Button("清除")
        api_key_status = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown("### LLM 設定 (Gemini)")
        gr.Markdown("以 YAML 編輯：`model`, `temperature` ∈ [0,2], `max_tokens` > 0, `top_p` ∈ (0,1].")
        textbox = gr.Code(language="yaml", label="llm.yaml", lines=10)
        with gr.Row():
            save_btn = gr.Button("儲存", variant="primary")
            reset_btn = gr.Button("還原預設")
            reload_btn = gr.Button("從磁碟重新載入")
        status_box = gr.Markdown()

        # Wiring — API key row
        view.load(fn=_api_key_status_text, outputs=[api_key_status])
        api_key_save_btn.click(
            fn=_save_api_key,
            inputs=[api_key_box],
            outputs=[api_key_box, api_key_status],
        )
        api_key_clear_btn.click(
            fn=_clear_api_key,
            outputs=[api_key_box, api_key_status],
        )

        # Wiring — YAML editor (unchanged)
        view.load(fn=_load_yaml, outputs=[textbox, status_box])
        save_btn.click(fn=_save_yaml, inputs=[textbox], outputs=[status_box])
        reset_btn.click(fn=_reset_yaml, outputs=[textbox, status_box])
        reload_btn.click(fn=_load_yaml, outputs=[textbox, status_box])

    return view
