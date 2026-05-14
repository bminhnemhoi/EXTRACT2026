"""Structured logging via loguru. JSON output for production, pretty for dev."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from exact_agent.config import get_settings

# Mutable holder avoids the `global` statement (ruff PLW0603).
_state: dict[str, bool] = {"configured": False}


def configure_logging() -> None:
    """Idempotently configure the loguru sink based on settings."""
    if _state["configured"]:
        return

    settings = get_settings().logging
    logger.remove()

    if settings.json_format:
        logger.add(
            sys.stdout,
            level=settings.level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=settings.level,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}"
            ),
            colorize=True,
        )

    _state["configured"] = True


def bind_context(**kwargs: Any) -> Any:
    """Return a contextualized logger (e.g. with `request_id`)."""
    configure_logging()
    return logger.bind(**kwargs)
