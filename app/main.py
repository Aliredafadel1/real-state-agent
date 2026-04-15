"""
FastAPI application entrypoint.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.error_handlers import register_error_handlers
from app.routes import router


def create_app() -> FastAPI:
    """Create and configure FastAPI app instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Backend API for AI real estate prediction pipeline.",
    )
    app.include_router(router)
    register_error_handlers(app)
    return app


app = create_app()

