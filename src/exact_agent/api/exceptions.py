"""Mapping of internal errors → HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException


class PipelineError(Exception):
    """Raised by pipelines when they hit an unrecoverable internal state."""


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("validation_error path={} errors={}", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request payload.", "errors": exc.errors()},
        )

    @app.exception_handler(PipelineError)
    async def _pipeline_handler(request: Request, exc: PipelineError) -> JSONResponse:
        logger.exception("pipeline_error path={} message={}", request.url.path, str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Pipeline failure.", "message": str(exc)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
