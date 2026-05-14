"""Task routing.

The competition test set mixes logic and physics queries into one stream;
we must auto-detect. Rule of thumb agreed with the team:

* If the payload carries any of the configured "premises" keys (default
  ``premises-NL``/``premises_NL``), it is a logic task.
* Otherwise default to physics.

An explicit ``task_type`` on the request always wins. Heuristics are kept
deliberately small and deterministic — anything fancier (e.g. an LLM
classifier) hides bugs and slows tests.
"""

from __future__ import annotations

from collections.abc import Mapping

from exact_agent.config import get_settings
from exact_agent.schemas import PredictRequest, TaskType


def route_task(payload: PredictRequest | Mapping[str, object]) -> TaskType:
    """Return ``"logic"`` or ``"physics"`` for the given payload.

    Accepts either a validated :class:`PredictRequest` or a raw mapping so it
    can be reused in batch scripts and tests without forcing schema validation.
    """
    settings = get_settings().router

    if isinstance(payload, PredictRequest):
        if payload.task_type is not None:
            return payload.task_type
        if payload.premises_NL is not None:
            return "logic"
        return "physics"

    # Mapping path — supports legacy dicts read straight from JSONL.
    explicit = payload.get("task_type")
    if isinstance(explicit, str) and explicit in ("logic", "physics"):
        return explicit  # type: ignore[return-value]

    for key in settings.premises_keys:
        value = payload.get(key)
        if value:
            return "logic"
    return "physics"
