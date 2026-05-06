import io

import httpx
import respx

from tongue_frontend import api


@respx.mock
def test_analyze_posts_multipart_and_returns_json():
    route = respx.post("http://localhost:8000/api/analyze").mock(
        return_value=httpx.Response(200, json={"comment": "OK"})
    )
    out = api.analyze(b"jpeg-bytes", "tongue.jpg")
    assert out == {"comment": "OK"}
    assert route.called


@respx.mock
def test_get_config_returns_status_dict():
    respx.get("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"content": "P", "is_default": True, "mtime": 0})
    )
    out = api.get_config("prompt")
    assert out["content"] == "P"


@respx.mock
def test_put_config_sends_content():
    route = respx.put("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    api.put_config("prompt", "NEW")
    assert route.calls[0].request.read() == b'{"content":"NEW"}'


@respx.mock
def test_reset_config_posts():
    respx.post("http://localhost:8000/api/config/llm/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    out = api.reset_config("llm")
    assert out == {"ok": True}


@respx.mock
def test_reload_registry_posts():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(200, json={"loaded": ["舌色"], "failed": []})
    )
    out = api.reload_registry()
    assert out["loaded"] == ["舌色"]
