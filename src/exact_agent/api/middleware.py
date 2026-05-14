"""Request-id + timing middleware for structured logs."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from exact_agent.config import get_settings


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_failed request_id={} path={} elapsed_ms={:.1f}",
                request_id,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "request request_id={} method={} path={} status={} elapsed_ms={:.1f}",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def register_middleware(app: FastAPI) -> None:
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
