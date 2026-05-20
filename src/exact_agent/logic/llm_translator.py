"""Translate an NL question into a FOL claim string for the Z3 backend.

The dataset already provides ``premises_FOL_clean``; what's missing is the
question side. Once we have ``claim_FOL`` we can fully exercise the Day-5
``z3_verifier`` fallback — that's the unlock from this module.

We deliberately keep the LLM scope tiny: input = (question, predicate
vocabulary derived from premises_FOL); output = a single FOL line.

Two surfaces:

* :func:`translate_question_to_fol` — single-shot (backwards compat).
* :func:`translate_question_to_fol_with_verify` (Day-22, E5) — the
  organizer-endorsed neurosymbolic-hybrid loop (Slides 27): LLM proposes
  FOL → parser/vocabulary verify → if reject, *feed the exact reason
  back to the LLM* and regenerate (up to ``max_rounds`` extra rounds).
  The verifier is the ground truth — Z3 Unknown is *not* a rejection
  signal (the gold itself can legitimately be Unknown).
"""

from __future__ import annotations

import re

from exact_agent.llm.prompt_templates import render
from exact_agent.llm.vllm_client import LLMClient
from exact_agent.logic.fol_parser import parse_fol

_FOL_LINE_RE = re.compile(r"[∀∃¬A-Za-z(].+")
_PRED_IN_LINE_RE = re.compile(r"\b([A-Z_][A-Za-z0-9_]+)\s*\(")


class LLMTranslationError(RuntimeError):
    """Raised when the LLM did not return a usable FOL string."""


def translate_question_to_fol(
    question: str,
    premises_fol: list[str],
    client: LLMClient,
    *,
    max_tokens: int = 128,
) -> str:
    """Return a FOL claim string ready to feed ``verify_with_z3``.

    The prompt is the same ``nl_to_fol.j2`` template used for premises,
    with the question text plugged in as the ``premise`` slot. (The
    template asks for "FOL: …" output, which is exactly what we need.)
    A short context block listing predicate names from ``premises_fol``
    nudges the model to reuse the established vocabulary.
    """
    raw = _generate(question, premises_fol, client, max_tokens=max_tokens, feedback=())
    line = _first_fol_line(raw)
    if line is None:
        raise LLMTranslationError(f"no FOL line found; raw={raw[:200]!r}")
    return line


def translate_question_to_fol_with_verify(
    question: str,
    premises_fol: list[str],
    client: LLMClient,
    *,
    max_tokens: int = 128,
    max_rounds: int = 2,
) -> tuple[str, list[str]]:
    """Solver-rejection loop translator (E5).

    Returns ``(accepted_fol_line, trace)``. Trace is a list of human-
    readable strings recording each attempt + its acceptance/rejection
    reason — surfaced into the API ``cot`` so reviewers can see the
    iterative refinement (P3 reward).

    Rejection signals (only these — Z3 verdict is NOT used here):

    * The LLM returns something that has no FOL-looking line.
    * :func:`parse_fol` returns ``None`` (syntactic reject; the parser
      already handles ``∀``, ``∧``, ``→``, comparisons, ground atoms).
    * Every predicate in the proposed claim is novel relative to the
      premise vocabulary (almost certainly a hallucinated translation).

    Raises ``LLMTranslationError`` if all attempts fail.
    """
    if max_rounds < 0:
        raise ValueError("max_rounds must be >= 0")

    vocab = _collect_predicate_vocab(premises_fol)
    feedback: list[str] = []
    trace: list[str] = []

    for attempt in range(max_rounds + 1):
        raw = _generate(
            question,
            premises_fol,
            client,
            max_tokens=max_tokens,
            feedback=tuple(feedback),
        )
        line = _first_fol_line(raw)
        if line is None:
            reason = f"attempt {attempt + 1}: completion contained no FOL-looking line"
            trace.append(reason)
            feedback.append(
                "Your last completion contained no FOL line. Output a single FOL formula."
            )
            continue

        if parse_fol(line, premise_id="CLAIM") is None:
            reason = (
                f"attempt {attempt + 1}: '{line}' did not parse "
                f"(supported: '∀x (body → head)' with ∧ conjunctions, comparisons, "
                f"or a ground/negated atom; ∃ and ∨ are NOT supported)"
            )
            trace.append(reason)
            feedback.append(
                f"Your previous output {line!r} did NOT parse. "
                "Only universal rules ∀x (A(x) ∧ B(x) → C(x)), ground atoms P(c), "
                "or comparisons f(x) >= n are accepted. Do not use ∃ or ∨."
            )
            continue

        claim_preds = set(_PRED_IN_LINE_RE.findall(line))
        if claim_preds and vocab:
            novel = claim_preds - vocab
            if novel == claim_preds:
                reason = (
                    f"attempt {attempt + 1}: '{line}' uses only predicates {sorted(claim_preds)} "
                    f"which are absent from the premise vocabulary {sorted(vocab)}"
                )
                trace.append(reason)
                feedback.append(
                    f"Your last output {line!r} used only predicates {sorted(claim_preds)}, "
                    f"none of which appear in the premise FOL. Reuse one of these "
                    f"predicate names verbatim: {sorted(vocab)}."
                )
                continue

        trace.append(f"attempt {attempt + 1}: '{line}' accepted")
        return line, trace

    raise LLMTranslationError(
        f"all {max_rounds + 1} attempts failed; last reason: "
        f"{trace[-1] if trace else 'unknown'}"
    )


def _generate(
    question: str,
    premises_fol: list[str],
    client: LLMClient,
    *,
    max_tokens: int,
    feedback: tuple[str, ...],
) -> str:
    """Single LLM call. Feedback strings are prepended as guidance for retries."""
    vocab = ", ".join(sorted(_collect_predicate_vocab(premises_fol))) or "none"
    parts = [f"# Predicate vocabulary already used in this problem: {vocab}"]
    if feedback:
        parts.append("# Earlier attempts were rejected — DO NOT repeat the same mistake:")
        for fb in feedback:
            parts.append(f"#   - {fb}")
    parts.append(question.strip())
    contextual = "\n".join(parts)

    prompt = render("nl_to_fol.j2", premise=contextual)
    raw = client.complete(prompt, max_tokens=max_tokens).strip()
    if not raw:
        raise LLMTranslationError("LLM returned empty completion")
    return raw


def _collect_predicate_vocab(premises_fol: list[str]) -> set[str]:
    """Pull predicate names out of the premise FOL strings (rough but useful)."""
    pred_re = re.compile(r"\b([A-Z_][A-Za-z0-9_]+)\s*\(")
    out: set[str] = set()
    for line in premises_fol or []:
        out.update(pred_re.findall(line))
    return out


def _first_fol_line(raw: str) -> str | None:
    """Strip code fences / labels and return the first FOL-looking line."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1].strip("` \n")
    for line in text.splitlines():
        candidate = line.strip()
        if candidate.lower().startswith("fol:"):
            candidate = candidate[4:].strip()
        if not candidate:
            continue
        if _FOL_LINE_RE.match(candidate):
            return candidate
    return None
