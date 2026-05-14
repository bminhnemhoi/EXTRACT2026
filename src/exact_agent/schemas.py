"""Pydantic I/O schemas — the single source of truth for API and pipelines.

The competition submission format (see `debai.md`) requires `answer` and
`explanation`; everything else is optional but contributes to the P3
reasoning-depth score. We accept both hyphenated and underscored variants
of `premises-NL` because the test format is not yet finalized.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["logic", "physics"]


class PredictRequest(BaseModel):
    """Incoming payload. `premises_NL` decides routing to the logic pipeline."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question: str = Field(..., min_length=1, description="The educational query.")
    premises_NL: list[str] | None = Field(
        default=None,
        alias="premises-NL",
        description="Natural-language premises (logic task only).",
    )
    premises_FOL: list[str] | None = Field(
        default=None,
        alias="premises-FOL",
        description="Optional first-order-logic premises if known.",
    )
    claim_FOL: str | None = Field(
        default=None,
        alias="claim-FOL",
        description="Optional FOL form of the question claim (used by Z3 fallback).",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional explicit task type; otherwise inferred by the router.",
    )

    @model_validator(mode="after")
    def _strip_empty_premises(self) -> PredictRequest:
        if self.premises_NL is not None and len(self.premises_NL) == 0:
            self.premises_NL = None
        if self.premises_FOL is not None and len(self.premises_FOL) == 0:
            self.premises_FOL = None
        return self


class PredictResponse(BaseModel):
    """API response. `answer` and `explanation` are required by the rubric."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="Final concise answer.")
    explanation: str = Field(
        ...,
        description="Natural-language justification, ideally derived from a verifier trace.",
    )
    cot: list[str] | None = Field(
        default=None,
        description="Step-by-step reasoning trace.",
    )
    premises: list[str] | None = Field(
        default=None,
        description="Premise IDs used (e.g. ['P1', 'P7']) — only meaningful for logic.",
    )
    fol: str | None = Field(
        default=None,
        description="First-order-logic representation, when available.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Solver-grounded confidence in [0, 1].",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Echo of the routed task type for debugging.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
