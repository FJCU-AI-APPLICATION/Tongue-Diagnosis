"""Pydantic models for backend request/response shapes and config objects.

Internal dicts have been replaced with these typed models so the wire format
is explicit, validated, and discoverable through the FastAPI OpenAPI schema.
The ``HeadResult`` / ``ClassScore`` dataclasses from ``ai.types`` are
serialised by Pydantic v2's built-in stdlib-dataclass support.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai.types import HeadResult


# --- Health -----------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    ai_version: str
    registry_loaded: bool
    heads_loaded: list[str]


# --- LLM config -------------------------------------------------------------


class LLMConfig(BaseModel):
    """The runtime LLM configuration.

    Pydantic enforces the same constraints that ``llm_store._validate`` used
    to enforce by hand:

    * ``temperature`` ∈ [0, 2]
    * ``max_tokens`` > 0
    * ``top_p`` ∈ (0, 1]
    * ``model`` is a non-empty string
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    model: str = Field(min_length=1)
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int = Field(2048, gt=0)
    top_p: float = Field(0.9, gt=0, le=1)


# --- Config CRUD ------------------------------------------------------------


class ConfigStatus(BaseModel):
    """Returned by ``GET /api/config/{section}`` and ``stores.*.status()``."""

    content: str
    is_default: bool
    mtime: float


class PutBody(BaseModel):
    """Request body for ``PUT /api/config/{section}`` and ``PUT /api/config/api_key``."""

    content: str


class OkResponse(BaseModel):
    """Trivial ack returned by ``PUT`` / ``POST .../reset``."""

    ok: bool = True


class ReloadResult(BaseModel):
    """Returned by ``POST /api/config/registry/reload``."""

    loaded: list[str]
    failed: list[str] = Field(default_factory=list)
    previous_kept: bool


# --- Analyze ----------------------------------------------------------------


class TimingMs(BaseModel):
    """Per-stage durations in milliseconds. Zero-defaulted so tests can build
    a partial instance without supplying every field."""

    decode: int = 0
    detect: int = 0
    infer: int = 0
    llm: int = 0
    total: int = 0


class AnalyzeResponse(BaseModel):
    """Response shape of ``POST /api/analyze``.

    ``heads`` is ``list[HeadResult]`` — Pydantic v2 serialises stdlib
    dataclasses out of the box, so the on-the-wire JSON is unchanged.

    ``category_map`` is echoed back from the active registry so the
    frontend can render the v4-category breakdown of composite-head
    predictions without re-implementing the lookup. Empty dict when the
    registry doesn't define a category_map (per-task heads).

    ``predictions_block`` is the rendered bullet block of head predictions;
    after Task 9 it is also what gets substituted into the system prompt's
    ``{{PREDICTIONS}}`` marker.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predictions_block: str
    heads: list[HeadResult]
    comment: str
    disclaimer: str
    category_map: dict[str, dict[str, str]] = Field(default_factory=dict)
    timing_ms: TimingMs


# --- API key (operator-set Gemini key) --------------------------------------


class ApiKeyStatus(BaseModel):
    """Returned by ``/api/config/api_key`` endpoints. Never includes the key."""

    is_set: bool
    fingerprint: str | None = None


# --- LLM model discovery ----------------------------------------------------


class LLMModelsResponse(BaseModel):
    """Returned by ``GET /api/llm/models`` — Gemini model IDs that support
    ``generateContent``. Plain ``id`` is the bare name (e.g. ``gemini-2.5-flash``)
    suitable for the ``model:`` field in ``llm.yaml``; full ``models/...`` form
    is in ``name`` for completeness.
    """

    models: list[str]
