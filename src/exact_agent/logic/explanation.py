"""Render the logic-pipeline trace into the API ``explanation`` field."""

from __future__ import annotations

from collections.abc import Sequence

from exact_agent.logic.answer_verifier import VerifierResult
from exact_agent.logic.forward_chainer import ChainResult


def render_explanation(
    question: str,
    chain: ChainResult,
    verifier: VerifierResult,
    selected_premise_texts: Sequence[str],
) -> str:
    """Build a paragraph explanation backed by the chain trace."""
    parts: list[str] = []

    if verifier.supports:
        labels = ", ".join(f"P{i}" for i in verifier.supports)
        parts.append(f"Using premises {labels}.")
    else:
        parts.append("No premise chain could be matched to the claim.")

    for pid, text in zip(verifier.supports, selected_premise_texts, strict=False):
        parts.append(f"  - P{pid}: {text}")

    parts.append(
        f"Forward chain produced {len(chain.facts)} fact(s) in {chain.iterations} iterations."
    )
    parts.append(verifier.rationale)
    parts.append(f"Answer: {verifier.answer or 'Unknown'} (confidence {verifier.confidence:.2f}).")
    return "\n".join(parts)
