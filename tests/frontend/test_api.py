import httpx
import pytest
import respx

from frontend import api
from frontend.models import (
    AnalyzeResponse,
    ConfigStatus,
    ReloadResult,
)


_ANALYZE_BODY = {
    "predictions_block": "msg",
    "heads": [],
    "comment": "OK",
    "disclaimer": "d",
    "timing_ms": {"decode": 1, "detect": 2, "infer": 3, "llm": 4, "total": 10},
}


@respx.mock
def test_analyze_posts_multipart_and_returns_typed_response():
    route = respx.post("http://localhost:8000/api/analyze").mock(
        return_value=httpx.Response(200, json=_ANALYZE_BODY)
    )
    out = api.analyze(b"jpeg-bytes", "tongue.jpg")
    assert isinstance(out, AnalyzeResponse)
    assert out.comment == "OK"
    assert out.timing_ms.total == 10
    assert route.called


@respx.mock
def test_get_config_returns_typed_status():
    respx.get("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"content": "P", "is_default": True, "mtime": 0})
    )
    out = api.get_config("prompt")
    assert isinstance(out, ConfigStatus)
    assert out.content == "P"
    assert out.is_default is True


@respx.mock
def test_put_config_sends_content():
    route = respx.put("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    api.put_config("prompt", "NEW")
    assert route.calls[0].request.read() == b'{"content":"NEW"}'


@respx.mock
def test_put_config_raises_on_422():
    respx.put("http://localhost:8000/api/config/llm").mock(
        return_value=httpx.Response(422, json={"detail": {"error": "bad yaml"}})
    )
    with pytest.raises(httpx.HTTPStatusError):
        api.put_config("llm", "broken: [")


@respx.mock
def test_reset_config_posts():
    route = respx.post("http://localhost:8000/api/config/llm/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    api.reset_config("llm")
    assert route.called


@respx.mock
def test_reload_registry_returns_typed_result():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(
            200, json={"loaded": ["舌色"], "failed": [], "previous_kept": True}
        )
    )
    out = api.reload_registry()
    assert isinstance(out, ReloadResult)
    assert out.loaded == ["舌色"]
    assert out.previous_kept is True


@respx.mock
def test_reload_registry_raises_on_422():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(422, json={"detail": {"error": "missing onnx"}})
    )
    with pytest.raises(httpx.HTTPStatusError):
        api.reload_registry()
