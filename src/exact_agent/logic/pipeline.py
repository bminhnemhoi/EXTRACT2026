"""Logic pipeline — wire premise_selector → rule_parser → forward_chainer → answer_verifier.

Day-5: a Z3 entailment fallback runs when the surface chain returns
``Unknown`` and the request carries ``premises-FOL`` plus ``claim-FOL``.
The Z3 verdict overrides the surface one only when Z3 actually decides
the case (Yes / No).
"""

from __future__ import annotations

import re

from exact_agent.agent.output_formatter import format_response
from exact_agent.config import get_settings
from exact_agent.logic.answer_verifier import (
    VerifierResult,
    verify_multiple_choice,
    verify_yes_no,
)
from exact_agent.logic.explanation import render_explanation
from exact_agent.logic.forward_chainer import forward_chain
from exact_agent.logic.premise_selector import select_top_k
from exact_agent.logic.rule_parser import parse_premises
from exact_agent.logic.z3_verifier import verify_with_z3
from exact_agent.schemas import PredictRequest, PredictResponse

_OPTION_RE = re.compile(r"^([A-D])\.\s+(.+)$")


def _detect_question_type(question: str, choices: dict[str, str] | None) -> str:
    """Heuristic question-type detection — used when caller doesn't supply one."""
    if choices or any(
        line.strip().startswith(("A.", "B.", "C.", "D.")) for line in question.splitlines()
    ):
        return "multiple_choice"
    lower = question.lower()
    if "true or false" in lower or "is the following statement true" in lower:
        return "true_false"
    if lower.startswith(("does ", "do ", "is ", "are ", "can ", "will ", "would ", "should ")):
        return "yes_no_unknown"
    return "open"


def _extract_choices(question: str) -> dict[str, str] | None:
    """Pull A/B/C/D options out of the question body."""
    choices: dict[str, str] = {}
    for line in question.splitlines():
        m = _OPTION_RE.match(line.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices or None


def _try_z3_fallback(
    payload: PredictRequest,
    surface: VerifierResult,
) -> VerifierResult | None:
    """Try the Z3 backend when the surface verifier abstained or low-confidence.

    Returns ``None`` if there's no signal to override (no FOL premises, or
    surface already confidently decided), otherwise a fresh
    ``VerifierResult`` to use instead of the surface one.
    """
    fol_premises = payload.premises_FOL
    claim_fol = payload.claim_FOL
    if not fol_premises or not claim_fol:
        return None
    # Don't replace a confident surface answer.
    if surface.answer in {"Yes", "No"} and surface.confidence >= 0.7:
        return None

    z3_result = verify_with_z3(fol_premises, claim_fol)
    if z3_result.verdict == "Unknown":
        return None
    return VerifierResult(
        answer=z3_result.verdict,
        supports=z3_result.supports,
        rationale=f"Z3 fallback: {z3_result.rationale}",
        confidence=0.85,
    )


class LogicPipeline:
    """Orchestrates: premise_selector → rule_parser → forward_chainer → answer_verifier."""

    def __init__(self, top_k: int | None = None) -> None:
        cfg = get_settings().pipelines.logic
        self._top_k = top_k if top_k is not None else cfg.top_k_premises
        self._z3_enabled = cfg.use_z3_fallback

    def run(self, payload: PredictRequest) -> PredictResponse:
        question = payload.question
        premises = payload.premises_NL or []
        choices = _extract_choices(question)

        ranked = select_top_k(question, premises, k=self._top_k)
        rules, facts = parse_premises(premises)
        chain = forward_chain(rules, facts)

        question_type = _detect_question_type(question, choices)

        if question_type == "multiple_choice":
            verifier = verify_multiple_choice(question, choices, chain)
        else:
            verifier = verify_yes_no(question, chain)

        z3_replacement: VerifierResult | None = None
        if self._z3_enabled:
            z3_replacement = _try_z3_fallback(payload, verifier)
            if z3_replacement is not None:
                verifier = z3_replacement

        selected_premise_texts: list[str] = []
        for pid in verifier.supports:
            if 1 <= pid <= len(premises):
                selected_premise_texts.append(premises[pid - 1])

        explanation = render_explanation(question, chain, verifier, selected_premise_texts)
        premises_labels = [f"P{p}" for p in verifier.supports] if verifier.supports else None

        top_labels = ", ".join(rp.label for rp in ranked[:5])
        cot = [
            f"Top premises (by overlap): {top_labels}",
            f"Detected question type: {question_type}",
            f"Forward chain iterations: {chain.iterations}, derived facts: {len(chain.facts)}",
            verifier.rationale,
        ]
        if z3_replacement is not None:
            cot.append("Z3 entailment fallback applied (surface chain returned Unknown).")

        return format_response(
            answer=verifier.answer or "Unknown",
            explanation=explanation,
            cot=cot,
            premises=premises_labels,
            confidence=verifier.confidence,
            task_type="logic",
        )
