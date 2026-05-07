"""Runtime settings for tongue-frontend, env-overridable via pydantic-settings.

Override any field by setting its env var (case-insensitive).

Examples:
    TONGUE_BACKEND_URL=http://my-backend:8000 uv run python -m frontend.app
    GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=8080 uv run python -m frontend.app
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Gradio frontend configuration.

    The fields use two prefixes by convention:

    - ``TONGUE_*``  — POC-specific (where the backend lives, how long to wait)
    - ``GRADIO_*``  — Gradio's own standard env vars (bind interface and port)
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    backend_url: str = Field(
        default="http://localhost:8000",
        validation_alias="TONGUE_BACKEND_URL",
        description="Base URL for the FastAPI backend.",
    )
    backend_timeout: float = Field(
        default=60.0,
        validation_alias="TONGUE_BACKEND_TIMEOUT",
        description="httpx timeout (seconds) for backend calls.",
    )
    gradio_server_name: str = Field(
        default="0.0.0.0",
        validation_alias="GRADIO_SERVER_NAME",
        description="Bind interface — '127.0.0.1' to restrict to localhost.",
    )
    gradio_server_port: int = Field(
        default=7860,
        validation_alias="GRADIO_SERVER_PORT",
        description="Port the Gradio server listens on.",
    )


# Module-level singleton — instantiated once at import time so that env vars
# are read before any view runs. Tests can monkeypatch this.
settings = FrontendSettings()
