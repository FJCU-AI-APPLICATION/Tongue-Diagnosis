"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from backend.routes.analyze import router as analyze_router
from backend.routes.api_key import router as api_key_router
from backend.routes.config import router as config_router
from backend.routes.health import router as health_router
from backend.routes.llm import router as llm_router
from backend.stores import llm_store, prompt_store, registry_store
from backend.stores.paths import REGISTRY_CURRENT


log = logging.getLogger("backend")


def create_app() -> FastAPI:
    app = FastAPI(title="Tongue Diagnosis API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(analyze_router)
    # api_key_router must be registered BEFORE config_router — config_router has
    # /{section} catch-all on /api/config, which would otherwise shadow /api/config/api_key.
    app.include_router(api_key_router)
    app.include_router(config_router)
    app.include_router(llm_router)
    app.state.registry = None

    @app.exception_handler(HTTPException)
    async def _flatten_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """If the route raised HTTPException(detail={"error": "..."}), surface
        the dict at top level. Falls back to {"error": <detail>} for str detail."""
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.on_event("startup")
    def _startup() -> None:
        try:
            import torch
            torch.set_num_threads(1)
            log.info("torch threads pinned to 1 (parallel head execution)")
        except Exception as e:
            log.warning("could not pin torch threads: %s", e)
        # Ensure 'current' files exist (copied from default if missing)
        prompt_store.load_current()
        llm_store.load_current()
        registry_store.load_current_text()
        # Try to load registry; on failure, leave app.state.registry = None
        # so /api/analyze returns 503 until the user fixes config + reloads.
        try:
            from ai import load_registry
            app.state.registry = load_registry(REGISTRY_CURRENT)
            log.info("loaded registry: %s heads", len(app.state.registry.heads))
        except Exception as e:
            log.warning("registry not loaded at startup: %s", e)

    return app


# Module-level `app` for `uvicorn backend.app:app`
app = create_app()


def run() -> None:
    """Console-script entry point for `tongue-backend`.

    Honors BACKEND_HOST / BACKEND_PORT env vars (defaults: 127.0.0.1:8000).
    """
    import uvicorn

    from backend.settings import settings

    uvicorn.run("backend.app:app", host=settings.host, port=settings.port)
