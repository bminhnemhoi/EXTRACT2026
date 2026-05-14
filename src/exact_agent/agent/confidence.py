"""Score how trustworthy a pipeline response is, and explain why.

The self-correction loop uses this to decide whether to spend an LLM
revision round on the current draft. We deliberately keep the policy
declarative — ``score_response`` returns a numeric score plus a list of
human-readable reasons, so the orchestrator's decision stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from exact_agent.schemas import PredictResponse, TaskType


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Outcome of grading a draft response."""

    score: float
    """Aggregate confidence in [0, 1]."""

    reasons: tuple[str, ...]
    """Why the score landed where it did. Used by the feedback prompt."""

    needs_revision: bool
    """True when the orchestrator should attempt a self-correction round."""


def score_response(
    response: PredictResponse,
    *,
    task_type: TaskType,
    threshold: float = 0.6,
) -> ConfidenceAssessment:
    """Grade a draft response.

    The signal is intentionally cheap to compute — a few schema-level
    checks plus the solver's own confidence. Anything richer (LLM-judge,
    embedding similarity to gold) belongs to the eval harness, not the
    runtime path.
    """
    reasons: list[str] = []
    score = float(response.confidence) if response.confidence is not None else 0.5

    if not response.answer.strip():
        score = min(score, 0.0)
        reasons.append("answer field is empty")

    if not response.explanation.strip():
        score = min(score, 0.2)
        reasons.append("explanation field is empty")

    if task_type == "logic":
        # An "Unknown" answer with no premise references is an abstention;
        # likely fixable by a translation-and-Z3 retry.
        if response.answer.strip().lower() == "unknown" and not response.premises:
            score = min(score, 0.3)
            reasons.append("logic answer is Unknown with no premise references")
        if response.answer.strip().upper() in {"A", "B", "C", "D"} and not response.premises:
            score = min(score, 0.4)
            reasons.append("multiple-choice answer has no premise references")

    if task_type == "physics":
        # The physics solver tags low confidence (≤0.4) when the verifier
        # raises sanity warnings — that's a strong revision signal.
        if score < 0.5:
            reasons.append("physics solver confidence below 0.5 (verifier warnings)")
        if response.cot is None or len(response.cot) < 2:
            score = min(score, 0.4)
            reasons.append("physics CoT is shorter than 2 steps")

    needs_revision = score < threshold
    if needs_revision and not reasons:
        reasons.append(f"confidence {score:.2f} below threshold {threshold:.2f}")
    return ConfidenceAssessment(
        score=max(0.0, min(1.0, score)),
        reasons=tuple(reasons),
        needs_revision=needs_revision,
    )
