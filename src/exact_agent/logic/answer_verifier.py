"""Decide an answer for a logic question using the forward-chained facts.

Day-4 baseline (no LLM):

* **yes_no_unknown**: we look for negation cues in the question, build a
  positive claim, and check whether the claim is entailed by the chain.
  Negation flips the answer; ``Unknown`` is returned when we cannot find
  enough support.
* **true_false**: same machinery — the dataset uses ``Yes`` / ``No``
  labels for both task variants.
* **multiple_choice**: score each option by surface-form overlap with the
  derived facts. The highest-scoring option above a small margin wins;
  otherwise we abstain.
* **open**: we abstain at Day-4 baseline — open answers need the LLM.

The verifier returns the answer label *and* the premise indices that
supported it, so the API response can populate the ``premises`` field
even when we abstain on the actual label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from exact_agent.logic.forward_chainer import ChainResult
from exact_agent.logic.premise_selector import _normalize as tokenize

_NEGATION_CUES = (
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoes ?n[oö]t\b",
    r"\bdoesn't\b",
    r"\bisn't\b",
    r"\baren't\b",
)
_NEGATION_RE = re.compile("|".join(_NEGATION_CUES), flags=re.IGNORECASE)


@dataclass(frozen=True)
class VerifierResult:
    answer: str
    """One of ``Yes``, ``No``, ``Unknown``, ``A``/``B``/``C``/``D``, or empty."""

    supports: tuple[int, ...]
    """1-based premise indices used. Always populated when we have signal."""

    rationale: str
    confidence: float


def _normalize_question(question: str) -> str:
    """Drop interrogative prefix so the verifier sees the claim text."""
    body = question.strip()
    # Strip leading "Does/Is/Can/Will ... ?".
    body = re.sub(
        r"^(does|is|are|can|could|will|would|should|do|did)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    return body.rstrip("?.").strip()


def _has_negation(question: str) -> bool:
    return bool(_NEGATION_RE.search(question))


def verify_yes_no(question: str, chain: ChainResult, *, threshold: float = 0.4) -> VerifierResult:
    """Yes/No/Unknown — and True/False — via forward-chained entailment."""
    claim = _normalize_question(question)
    negated = _has_negation(claim)
    match = chain.matches(claim, threshold=threshold)
    if match is None:
        return VerifierResult(
            answer="Unknown",
            supports=chain.all_supports()[:3],
            rationale="No premise chain matched the question claim.",
            confidence=0.3,
        )
    overlap_score = len(match.tokens & frozenset(tokenize(claim))) / max(
        1, len(match.tokens | frozenset(tokenize(claim)))
    )
    answer = "No" if negated else "Yes"
    return VerifierResult(
        answer=answer,
        supports=match.supports,
        rationale=f"Derived from premises {list(match.supports)}: {match.text}",
        confidence=min(0.9, 0.5 + 0.4 * overlap_score),
    )


_OPTION_RE = re.compile(r"^([A-D])\.\s+(.+)$")


def _parse_options(question: str, choices: dict[str, str] | None) -> list[tuple[str, str]]:
    """Return ``[(label, text), ...]`` parsed from the question or choices dict."""
    if choices:
        return [(k, v) for k, v in choices.items()]
    options: list[tuple[str, str]] = []
    for line in question.splitlines():
        m = _OPTION_RE.match(line.strip())
        if m:
            options.append((m.group(1), m.group(2).strip()))
    return options


def verify_multiple_choice(
    question: str,
    choices: dict[str, str] | None,
    chain: ChainResult,
    *,
    margin: float = 0.0,
    min_top_score: float = 0.0,
    abstain_when_all_zero: bool = True,
) -> VerifierResult:
    """Pick the option whose surface form best matches a derived fact.

    E4 policy (Day-21, post official 2026-05-15 release): **"Unknown" is
    a legitimate MC label.** The release retains 168 MCQ records whose
    correct answer is ``Unknown`` (premises genuinely under-determine —
    CHANGELOG_TYPE1; a separate 103 MCQs that previously said
    ``Unknown`` were *fixed* to letters). The Day-4 "always commit"
    policy scored zero on every retained Unknown row.

    The default change is **conservative**: keep the alphabetical-tie
    commit when there is *any* signal (≥1 overlapping token), but
    abstain to ``"Unknown"`` (not ``""``) when literally no option
    overlapped. The stricter ``min_top_score`` and ``margin`` paths are
    *opt-in via kwargs* — measured on the 2026-05-15 holdout, defaults
    of 0.15 / 0.05 over-abstained MC (40% → 0% on the 15-row slice)
    with no clear YNU gain attributable. Keep them available so future
    tuning can lift them when a per-prefix score distribution is known.
    Callers can also disable the all-zero path with ``abstain_when_all_zero=False``.
    """
    options = _parse_options(question, choices)
    if not options:
        return VerifierResult(
            answer="Unknown",
            supports=chain.all_supports()[:3],
            rationale="No multiple-choice options parsed.",
            confidence=0.2,
        )

    scored: list[tuple[str, float, tuple[int, ...]]] = []
    for label, text in options:
        match = chain.matches(text, threshold=0.0)
        score = (
            0.0
            if match is None
            else len(match.tokens & frozenset(tokenize(text)))
            / max(1, len(match.tokens | frozenset(tokenize(text))))
        )
        supports = match.supports if match else ()
        scored.append((label, score, supports))

    scored.sort(key=lambda t: -t[1])
    top = scored[0]
    runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
    spread = max(0.0, top[1] - runner_up_score)

    if top[1] == 0.0 and abstain_when_all_zero:
        return VerifierResult(
            answer="Unknown",
            supports=chain.all_supports()[:3],
            rationale="No option had any token overlap with the derived facts.",
            confidence=0.2,
        )
    if top[1] < min_top_score:
        return VerifierResult(
            answer="Unknown",
            supports=top[2],
            rationale=(
                f"Best option {top[0]!r} score {top[1]:.2f} below min_top_score "
                f"{min_top_score}; insufficient signal -> Unknown."
            ),
            confidence=0.25,
        )
    if margin > 0.0 and spread < margin:
        return VerifierResult(
            answer="Unknown",
            supports=top[2],
            rationale=(
                f"Top option {top[0]!r} (score {top[1]:.2f}) too close to runner-up "
                f"({runner_up_score:.2f}); insufficient discrimination -> Unknown."
            ),
            confidence=0.25,
        )
    return VerifierResult(
        answer=top[0],
        supports=top[2],
        rationale=(
            f"Option {top[0]!r} best matched the derived facts "
            f"(score {top[1]:.2f}, spread {spread:.2f})."
        ),
        confidence=min(0.85, 0.35 + 0.5 * spread + 0.2 * top[1]),
    )
