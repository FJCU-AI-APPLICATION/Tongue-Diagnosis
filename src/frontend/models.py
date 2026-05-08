"""Pydantic models mirroring the backend wire contract.

The frontend talks to the backend over HTTP, so it owns its own view of the
response schema. Mirroring (rather than importing ``backend.models``)
keeps the frontend dep tree thin — no FastAPI / google-adk on the client —
and makes schema drift surface as a ``ValidationError`` at parse time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ai_version: str
    registry_loaded: bool
    heads_loaded: list[str]


class ConfigStatus(BaseModel):
    """Returned by ``GET /api/config/{section}``."""

    content: str
    is_default: bool
    mtime: float


class ReloadResult(BaseModel):
    """Returned by ``POST /api/config/registry/reload``."""

    loaded: list[str]
    failed: list[str] = []
    previous_kept: bool


class ClassScore(BaseModel):
    label: str
    score: float


class HeadResult(BaseModel):
    task: str
    head_type: str
    predictions: list[ClassScore] = []
    error: str | None = None


class TimingMs(BaseModel):
    decode: int = 0
    detect: int = 0
    infer: int = 0
    llm: int = 0
    total: int = 0


class AnalyzeResponse(BaseModel):
    """Response shape of ``POST /api/analyze``."""

    predictions_block: str
    heads: list[HeadResult]
    comment: str
    disclaimer: str
    category_map: dict[str, dict[str, str]] = Field(default_factory=dict)
    timing_ms: TimingMs


class ApiKeyStatus(BaseModel):
    """Returned by ``/api/config/api_key`` endpoints. Never includes the key."""

    is_set: bool
    fingerprint: str | None = None


class LLMModelsResponse(BaseModel):
    """Returned by ``GET /api/llm/models``."""

    models: list[str]
