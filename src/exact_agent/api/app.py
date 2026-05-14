"""FastAPI application factory."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from exact_agent.__version__ import __version__
from exact_agent.api.exceptions import register_exception_handlers
from exact_agent.api.middleware import register_middleware
from exact_agent.api.routes import router as api_router
from exact_agent.config import get_settings
from exact_agent.logging_setup import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="EXACT 2026 — Veriscope Agent",
        version=__version__,
        description=(
            "Explainable QA over educational logic and physics queries. "
            "Submission API for the IEEE IJCNN 2026 EXACT challenge."
        ),
    )
    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point (``exact-api``)."""
    settings = get_settings().api
    uvicorn.run(
        "exact_agent.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
