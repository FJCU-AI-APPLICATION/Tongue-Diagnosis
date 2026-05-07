"""LLM 設定 — Gemini API key + dynamic model dropdown + structured tuning form."""

from __future__ import annotations

import gradio as gr
import httpx

from frontend import api


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


# --- LLM config form (replaces the YAML editor) -----------------------------

_DEFAULTS = {"model": "", "temperature": 0.2, "max_tokens": 2048, "top_p": 0.9}


def _parse_llm_yaml(yaml_text: str) -> dict[str, str]:
    """Tiny key:value parser — sufficient because llm.yaml is flat scalars.

    Avoids pulling in PyYAML for what is in practice four lines of config.
    """
    out: dict[str, str] = {}
    for line in yaml_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        k, sep, v = s.partition(":")
        if sep:
            out[k.strip()] = v.strip()
    return out


def _form_values(yaml_text: str) -> tuple[str, float, int, float]:
    """Parse YAML into (model, temperature, max_tokens, top_p) form tuple."""
    parsed = _parse_llm_yaml(yaml_text)
    try:
        temperature = float(parsed.get("temperature", _DEFAULTS["temperature"]))
    except ValueError:
        temperature = float(_DEFAULTS["temperature"])
    try:
        max_tokens = int(parsed.get("max_tokens", _DEFAULTS["max_tokens"]))
    except ValueError:
        max_tokens = int(_DEFAULTS["max_tokens"])
    try:
        top_p = float(parsed.get("top_p", _DEFAULTS["top_p"]))
    except ValueError:
        top_p = float(_DEFAULTS["top_p"])
    return parsed.get("model", str(_DEFAULTS["model"])), temperature, max_tokens, top_p


def _build_llm_yaml(model: str, temperature: float, max_tokens: int, top_p: float) -> str:
    return (
        f"model: {model}\n"
        f"temperature: {temperature}\n"
        f"max_tokens: {int(max_tokens)}\n"
        f"top_p: {top_p}\n"
    )


def _refresh_models() -> tuple:
    """Fetch available Gemini models. Returns (Dropdown update, status text)."""
    try:
        resp = api.list_llm_models()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 412:
            return gr.Dropdown(choices=[], allow_custom_value=True), "⚠ 請先設定 API key"
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return gr.Dropdown(choices=[], allow_custom_value=True), f"⚠ {err}"
    except httpx.ConnectError:
        return gr.Dropdown(choices=[], allow_custom_value=True), "⚠ 無法連線到後端"
    return (
        gr.Dropdown(choices=resp.models, allow_custom_value=True),
        f"已載入 {len(resp.models)} 個可用模型",
    )


def _save_form(model: str, temperature: float, max_tokens: int, top_p: float) -> str:
    yaml_text = _build_llm_yaml(model, temperature, max_tokens, top_p)
    try:
        api.put_config("llm", yaml_text)
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", str(e))
        except Exception:
            err = str(e)
        return f"⚠ 儲存失敗：{err}"
    except httpx.ConnectError:
        return "⚠ 無法連線到後端"
    return "已儲存 — 下次分析將使用新設定"


def _load_form() -> tuple[str, str, float, int, float]:
    """Status + form values for current llm.yaml."""
    s = api.get_config("llm")
    flag = "default" if s.is_default else "custom"
    model, temperature, max_tokens, top_p = _form_values(s.content)
    return f"狀態：{flag}", model, temperature, max_tokens, top_p


def _reset_form() -> tuple[str, str, float, int, float]:
    api.reset_config("llm")
    _, model, temperature, max_tokens, top_p = _load_form()
    return "已還原預設值", model, temperature, max_tokens, top_p


# --- View build --------------------------------------------------------------


def build(app: gr.Blocks) -> None:
    """Add the LLM editor (API key + structured form) into the current Gradio context.

    `app` is the root ``gr.Blocks`` — used for ``app.load`` so initial-state
    hydration fires once when the page loads.
    """
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

    # Dynamic model picker — choices populate when 重新整理 is clicked
    # (after a valid key is saved).
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="模型 (model)",
            choices=[],
            allow_custom_value=True,
            interactive=True,
            scale=4,
        )
        refresh_btn = gr.Button("🔄 重新整理", scale=1)
    model_status = gr.Markdown()

    # Structured tuning controls (replaces the previous YAML editor)
    temperature = gr.Slider(
        label="temperature",
        minimum=0, maximum=2, step=0.05,
        value=_DEFAULTS["temperature"],
        info="0 = 嚴謹/穩定；2 = 活潑/不可預測",
    )
    max_tokens = gr.Number(
        label="max_tokens",
        value=_DEFAULTS["max_tokens"],
        precision=0,
        minimum=1,
        info="模型最多輸出的 token 數",
    )
    top_p = gr.Slider(
        label="top_p",
        minimum=0.01, maximum=1.0, step=0.01,
        value=_DEFAULTS["top_p"],
        info="nucleus sampling — 越接近 1 越多樣",
    )

    with gr.Row():
        save_btn = gr.Button("儲存", variant="primary")
        reset_btn = gr.Button("還原預設")
        reload_btn = gr.Button("從磁碟重新載入")
    status_box = gr.Markdown()

    def _initial_load() -> tuple[str, str, str, float, int, float]:
        status, model, t, mt, tp = _load_form()
        return _api_key_status_text(), status, model, t, mt, tp

    app.load(
        fn=_initial_load,
        outputs=[api_key_status, status_box, model_dropdown, temperature, max_tokens, top_p],
    )

    # Wiring — API key row
    api_key_save_btn.click(
        fn=_save_api_key,
        inputs=[api_key_box],
        outputs=[api_key_box, api_key_status],
    )
    api_key_clear_btn.click(
        fn=_clear_api_key,
        outputs=[api_key_box, api_key_status],
    )

    # Wiring — Model dropdown refresh
    refresh_btn.click(
        fn=_refresh_models,
        outputs=[model_dropdown, model_status],
    )

    # Wiring — Tuning form
    save_btn.click(
        fn=_save_form,
        inputs=[model_dropdown, temperature, max_tokens, top_p],
        outputs=[status_box],
    )
    reset_btn.click(
        fn=_reset_form,
        outputs=[status_box, model_dropdown, temperature, max_tokens, top_p],
    )
    reload_btn.click(
        fn=_load_form,
        outputs=[status_box, model_dropdown, temperature, max_tokens, top_p],
    )
