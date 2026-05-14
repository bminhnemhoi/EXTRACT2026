"""Render Jinja2 prompt templates from ``configs/prompts/``.

Centralizing prompt I/O here means evaluation, training data prep, and
the live pipeline all read the same template — no copy-paste drift.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from exact_agent.config import get_settings


@lru_cache(maxsize=1)
def _env() -> Environment:
    prompts_dir: Path = get_settings().configs_dir / "prompts"
    if not prompts_dir.exists():
        raise FileNotFoundError(f"prompts directory missing: {prompts_dir}")
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(template_name: str, **kwargs: Any) -> str:
    """Render ``configs/prompts/<template_name>`` with ``kwargs``."""
    template = _env().get_template(template_name)
    return template.render(**kwargs)


def list_templates() -> list[str]:
    return sorted(_env().list_templates())
