"""Normalize internal pipeline outputs into the public :class:`PredictResponse`.

Why this exists: pipelines emit rich internal dicts (with solver traces,
intermediate symbolic forms, debug fields). The formatter is the single
choke point that:

* drops anything not in the public schema,
* coerces missing-but-known fields into ``None`` rather than absent keys,
* clamps confidence to [0, 1],
* guarantees the response validates against ``PredictResponse``.
"""

from __future__ import annotations

from typing import Any

from exact_agent.schemas import PredictResponse, TaskType


def format_response(
    *,
    answer: str,
    explanation: str,
    cot: list[str] | None = None,
    premises: list[str] | None = None,
    fol: str | None = None,
    confidence: float | None = None,
    task_type: TaskType | None = None,
) -> PredictResponse:
    """Build a :class:`PredictResponse`, sanitizing inputs."""
    if confidence is not None:
        confidence = max(0.0, min(1.0, float(confidence)))

    cot = [step for step in (cot or []) if isinstance(step, str) and step.strip()] or None
    premises = [p for p in (premises or []) if isinstance(p, str) and p.strip()] or None

    return PredictResponse(
        answer=str(answer).strip(),
        explanation=str(explanation).strip(),
        cot=cot,
        premises=premises,
        fol=(fol.strip() if isinstance(fol, str) and fol.strip() else None),
        confidence=confidence,
        task_type=task_type,
    )


def format_from_dict(raw: dict[str, Any], task_type: TaskType | None = None) -> PredictResponse:
    """Convenience wrapper for pipelines that already produced a dict."""
    return format_response(
        answer=raw.get("answer", ""),
        explanation=raw.get("explanation", ""),
        cot=raw.get("cot"),
        premises=raw.get("premises"),
        fol=raw.get("fol"),
        confidence=raw.get("confidence"),
        task_type=task_type,
    )
