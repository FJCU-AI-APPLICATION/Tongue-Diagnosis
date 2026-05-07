"""FastAPI application factory with startup-time registry/store wiring."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tongue_ai.registry import RegistryError

from tongue_backend.llm.client import LLMClient
from tongue_backend.routes.analyze import router as analyze_router
from tongue_backend.routes.config import router as config_router
from tongue_backend.routes.health import router as health_router
from tongue_backend.stores import paths
from tongue_backend.stores.llm_store import LLMStore
from tongue_backend.stores.prompt_store import PromptStore
from tongue_backend.stores.registry_store import RegistryStore


_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prompt_store = PromptStore(paths.PROMPT_DEFAULT, paths.PROMPT_CURRENT)
    app.state.llm_store = LLMStore(paths.LLM_DEFAULT, paths.LLM_CURRENT)
    app.state.registry_store = RegistryStore(paths.REGISTRY_DEFAULT, paths.REGISTRY_CURRENT)
    app.state.llm_client = LLMClient()

    try:
        app.state.registry = app.state.registry_store.reload()
        _log.info(
            "registry loaded: %d heads, %d failed",
            len(app.state.registry.heads),
            len(app.state.registry.failed_heads),
        )
    except RegistryError as exc:
        _log.error("registry load failed at startup: %s", exc)
        app.state.registry = None
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Tongue Diagnosis API", version="0.2.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(config_router)
    return app


app = create_app()
