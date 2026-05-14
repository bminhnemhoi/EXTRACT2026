"""Translate an NL question into a FOL claim string for the Z3 backend.

The dataset already provides ``premises_FOL_clean``; what's missing is the
question side. Once we have ``claim_FOL`` we can fully exercise the Day-5
``z3_verifier`` fallback — that's the unlock from this module.

We deliberately keep the LLM scope tiny: input = (question, predicate
vocabulary derived from premises_FOL); output = a single FOL line.
The verifier is the ground truth — if Z3 returns Unknown the LLM
translation simply didn't help on that round; nothing breaks.
"""

from __future__ import annotations

import re

from exact_agent.llm.prompt_templates import render
from exact_agent.llm.vllm_client import LLMClient

_FOL_LINE_RE = re.compile(r"[∀∃¬A-Za-z(].+")


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
    vocab = ", ".join(sorted(_collect_predicate_vocab(premises_fol))) or "none"
    contextual = f"# Predicate vocabulary already used in this problem: {vocab}\n{question.strip()}"

    prompt = render("nl_to_fol.j2", premise=contextual)
    raw = client.complete(prompt, max_tokens=max_tokens).strip()
    if not raw:
        raise LLMTranslationError("LLM returned empty completion")

    line = _first_fol_line(raw)
    if line is None:
        raise LLMTranslationError(f"no FOL line found; raw={raw[:200]!r}")
    return line


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
