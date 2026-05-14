"""Solver-grounded self-correction loop.

Phase 5 of the plan: when a draft response from a pipeline scores below
the confidence threshold, we ask the LLM to revise it given concrete
verifier feedback. The revised JSON is parsed against
:class:`PredictResponse`. If the revision validates, return it; otherwise
fall back to the original draft (we never make things worse).

Design notes:

* This module wraps the orchestrator output rather than rerunning the
  whole pipeline. The pipelines have already done the deterministic work
  (extraction, classification, computation); the LLM's role is purely to
  improve answer/explanation phrasing or to fill in fields the solver
  could not.
* We cap the loop at ``max_rounds`` (default 2) so a wedged LLM cannot
  stretch the request budget.
* The original CoT trace is appended with revision metadata so the
  reasoning-depth score (P3) reflects the correction work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from exact_agent.agent.confidence import score_response
from exact_agent.llm.prompt_templates import render
from exact_agent.llm.vllm_client import LLMClient, parse_json_completion
from exact_agent.schemas import PredictRequest, PredictResponse, TaskType


@dataclass(frozen=True)
class CorrectionOutcome:
    """Result of one or more self-correction rounds."""

    response: PredictResponse
    rounds_used: int
    initial_score: float
    final_score: float
    accepted: bool
    """``True`` if at least one revision was kept; ``False`` if the original
    draft was returned unchanged (because LLM unavailable, no improvement,
    or LLM produced invalid JSON)."""


class SelfCorrector:
    """Wrap a draft response with up to ``max_rounds`` LLM revision passes.

    The orchestrator is the only caller. Construction parameters mirror the
    ``self_correction`` block of ``configs/app.yaml`` so the API process
    can flip behavior via env vars without touching code.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        max_rounds: int = 2,
        threshold: float = 0.6,
    ) -> None:
        self._llm = llm
        self._max_rounds = max(0, max_rounds)
        self._threshold = threshold

    def maybe_revise(
        self,
        draft: PredictResponse,
        payload: PredictRequest,
        task_type: TaskType,
    ) -> CorrectionOutcome:
        """Run the loop. Returns the (possibly revised) response."""
        initial = score_response(draft, task_type=task_type, threshold=self._threshold)

        if self._llm is None or self._max_rounds == 0 or not initial.needs_revision:
            return CorrectionOutcome(
                response=draft,
                rounds_used=0,
                initial_score=initial.score,
                final_score=initial.score,
                accepted=False,
            )

        current = draft
        current_score = initial.score
        current_assessment = initial
        accepted = False
        round_idx = 0

        while round_idx < self._max_rounds and current_assessment.needs_revision:
            round_idx += 1
            revised = self._attempt_revision(
                payload=payload,
                draft=current,
                feedback="; ".join(current_assessment.reasons) or "low confidence",
            )
            if revised is None:
                break  # LLM call failed; keep current draft
            assessment = score_response(revised, task_type=task_type, threshold=self._threshold)
            if assessment.score <= current_score:
                # The LLM made things worse (or no progress) — stop trying.
                break
            current = revised
            current_score = assessment.score
            current_assessment = assessment
            accepted = True

        if accepted:
            current = _annotate_with_revision_trace(current, rounds=round_idx)

        return CorrectionOutcome(
            response=current,
            rounds_used=round_idx,
            initial_score=initial.score,
            final_score=current_score,
            accepted=accepted,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attempt_revision(
        self,
        *,
        payload: PredictRequest,
        draft: PredictResponse,
        feedback: str,
    ) -> PredictResponse | None:
        assert self._llm is not None
        prompt = render(
            "self_correction.j2",
            question=payload.question,
            premises_nl=payload.premises_NL or [],
            draft_json=draft.model_dump_json(indent=2, exclude_none=True),
            feedback=feedback,
        )
        try:
            raw = self._llm.complete(prompt)
        except Exception:
            return None
        if not raw.strip():
            return None
        try:
            payload_json = parse_json_completion(raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload_json, dict):
            return None
        try:
            return PredictResponse.model_validate({**payload_json, "task_type": draft.task_type})
        except ValidationError:
            return None


def _annotate_with_revision_trace(response: PredictResponse, *, rounds: int) -> PredictResponse:
    """Append a CoT step noting the LLM revision happened."""
    cot = list(response.cot or [])
    cot.append(f"Self-correction loop applied (revisions kept: {rounds}).")
    return response.model_copy(update={"cot": cot})
