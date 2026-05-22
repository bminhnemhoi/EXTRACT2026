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
_PRED_IN_LINE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]+)\s*\(")
# Iter-7: matched _collect_predicate_vocab — allow lowercase snake_case
# predicates that dataset 2026-05-15 actually uses.


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

        # Iter-12: Qualifier-preservation check (the user's "không được bỏ
        # cụm bổ nghĩa như 'honor roll'" rule). For each premise predicate
        # whose naturalized form (snake_case → spaces) appears as a noun
        # phrase in the QUESTION, that predicate MUST appear in the claim.
        # Without this check, "There exists a student on the HONOR ROLL who
        # is ELIGIBLE for a scholarship" gets translated to just
        # `eligible_for_scholarship(student)` and Z3 wrongly says Yes.
        expected_preds = _question_expected_predicates(question, vocab)
        missing_quals = expected_preds - claim_preds
        if missing_quals:
            reason = (
                f"attempt {attempt + 1}: '{line}' is missing qualifier predicate(s) "
                f"{sorted(missing_quals)} that the question explicitly mentions"
            )
            trace.append(reason)
            feedback.append(
                f"Your last output {line!r} dropped qualifier predicate(s) "
                f"{sorted(missing_quals)}. The QUESTION text explicitly mentions "
                f"the corresponding noun phrase(s); the claim FOL MUST include "
                f"each of those predicates as part of the entity's condition "
                f"conjunction. Do NOT translate a CONJUNCTION as a single atom."
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


def _naturalize_predicate(name: str) -> str:
    """Convert ``snake_case_or_PascalCase`` predicate name to a normalized
    space-separated lowercase phrase suitable for substring matching in
    a natural-language question.

    Examples:
        honor_roll              -> "honor roll"
        HonorRoll               -> "honor roll"
        EligibleForScholarship  -> "eligible for scholarship"
        completed_pedagogical_training -> "completed pedagogical training"
    """
    # Split PascalCase: insert spaces before capital letters
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    # snake_case → spaces
    spaced = spaced.replace("_", " ")
    return spaced.lower().strip()


_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with",
    "is", "are", "be", "as", "and", "or", "if", "then",
})


def _question_expected_predicates(question: str, vocab: set[str]) -> set[str]:
    """Iter-12: predicates that the question explicitly mentions and that
    therefore MUST appear in the translated claim's conjunction.

    For each predicate ``P`` in the premise vocabulary, naturalize it
    (``HonorRoll`` -> ``"honor roll"``; ``EligibleForScholarship`` ->
    ``"eligible for scholarship"``). Then require ALL of the
    naturalized phrase's CONTENT words (after stop-word strip) to appear
    in the question text. This is permissive enough to handle inserted
    articles (``"eligible for A scholarship"`` ✓) and possessives
    (``"the honor roll's"`` ✓) without false-positive matching on bare
    single words like "student".

    Catches the 3B-translator pattern of dropping qualifier phrases.
    """
    q_words = set(re.findall(r"[a-z]+", question.lower()))
    expected: set[str] = set()
    for pred in vocab:
        natural = _naturalize_predicate(pred)
        if len(natural) < 5 or " " not in natural:
            # Single-word predicates ("Student", "HasGPA") are too common as
            # substrings; we only infer expectation from multi-word phrases.
            continue
        content_words = [
            w for w in natural.split() if w not in _STOP_WORDS and len(w) >= 3
        ]
        if not content_words:
            continue
        if all(w in q_words for w in content_words):
            expected.add(pred)
    return expected


def _collect_predicate_vocab(premises_fol: list[str]) -> set[str]:
    """Pull predicate names out of the premise FOL strings (rough but useful).

    Iter-7 fix: dataset 2026-05-15 uses snake_case lowercase predicates
    (``completed_pedagogical_training``, ``holds_phd``) — the prior regex
    required PascalCase initial cap, returning an EMPTY vocab. With no
    vocab the LLM had no constraint and invented PascalCase predicates
    that never matched, then Z3 returned Unknown on every translated
    claim. Allowing lowercase-leading identifiers fixes 81 logic rows.
    """
    # Match identifier-then-open-paren, but NOT FOL operators ForAll/Exists/
    # And/Or/Not/Implies (which are syntax, not predicates). Allow any
    # alphanumeric start including underscore.
    pred_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]+)\s*\(")
    reserved = {"ForAll", "Exists", "And", "Or", "Not", "Implies", "Iff"}
    out: set[str] = set()
    for line in premises_fol or []:
        for name in pred_re.findall(line):
            if name not in reserved:
                out.add(name)
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
