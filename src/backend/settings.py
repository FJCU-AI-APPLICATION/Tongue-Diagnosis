"""Runtime settings for tongue-backend, env-overridable via pydantic-settings.

Override any field by setting its env var (case-insensitive).

Example:
    TONGUE_MAX_UPLOAD_MB=20 uv run uvicorn backend.app:app --port 8000
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Tongue backend service configuration."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    host: str = Field(
        default="127.0.0.1",
        validation_alias="BACKEND_HOST",
        description="Host interface for the `tongue-backend` console script. Default localhost; set 0.0.0.0 to listen on all interfaces.",
    )
    port: int = Field(
        default=8000,
        validation_alias="BACKEND_PORT",
        description="Port for the `tongue-backend` console script.",
    )
    max_upload_mb: int = Field(
        default=10,
        validation_alias="TONGUE_MAX_UPLOAD_MB",
        description="Maximum upload size for /api/analyze, in megabytes.",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


# Module-level singleton — instantiated once at import time so that env vars
# are read before any request hits the route. Tests can monkeypatch this.
settings = BackendSettings()
