"""FastAPI application factory."""

from fastapi import FastAPI

from tongue_backend.routes.health import router as health_router
from tongue_backend.routes.analyze import router as analyze_router


def create_app() -> FastAPI:
    app = FastAPI(title="Tongue Diagnosis API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()
