"""Thin httpx client to the FastAPI backend.

Each call returns a typed Pydantic model defined in :mod:`frontend.models`.
Non-2xx responses raise ``httpx.HTTPStatusError`` — callers handle errors via
exception, not a magic ``{"error": ...}`` dict in the success path.
"""

from __future__ import annotations

import httpx

from frontend.models import (
    AnalyzeResponse,
    ApiKeyStatus,
    ConfigStatus,
    HealthResponse,
    LLMModelsResponse,
    ReloadResult,
)
from frontend.settings import settings


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.backend_url, timeout=settings.backend_timeout)


def analyze(image_bytes: bytes, filename: str = "tongue.jpg") -> AnalyzeResponse:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    with _client() as c:
        r = c.post("/api/analyze", files=files)
        r.raise_for_status()
        return AnalyzeResponse.model_validate(r.json())


def get_config(section: str) -> ConfigStatus:
    with _client() as c:
        r = c.get(f"/api/config/{section}")
        r.raise_for_status()
        return ConfigStatus.model_validate(r.json())


def put_config(section: str, content: str) -> None:
    with _client() as c:
        r = c.put(f"/api/config/{section}", json={"content": content})
        r.raise_for_status()


def reset_config(section: str) -> None:
    with _client() as c:
        r = c.post(f"/api/config/{section}/reset")
        r.raise_for_status()


def reload_registry() -> ReloadResult:
    with _client() as c:
        r = c.post("/api/config/registry/reload")
        r.raise_for_status()
        return ReloadResult.model_validate(r.json())


def health() -> HealthResponse:
    with _client() as c:
        r = c.get("/health")
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())


def get_api_key_status() -> ApiKeyStatus:
    with _client() as c:
        r = c.get("/api/config/api_key")
        r.raise_for_status()
        return ApiKeyStatus.model_validate(r.json())


def put_api_key(content: str) -> ApiKeyStatus:
    with _client() as c:
        r = c.put("/api/config/api_key", json={"content": content})
        r.raise_for_status()
        return ApiKeyStatus.model_validate(r.json())


def clear_api_key() -> ApiKeyStatus:
    with _client() as c:
        r = c.delete("/api/config/api_key")
        r.raise_for_status()
        return ApiKeyStatus.model_validate(r.json())


def list_llm_models() -> LLMModelsResponse:
    with _client() as c:
        r = c.get("/api/llm/models")
        r.raise_for_status()
        return LLMModelsResponse.model_validate(r.json())
