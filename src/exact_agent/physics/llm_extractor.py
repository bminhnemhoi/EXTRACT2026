"""LLM-driven fallback for the regex quantity extractor.

The regex extractor only fires on explicit ``name = value unit`` syntax.
Real questions in the dataset write things like "the voltage across its
plates is 60 V" or "8 cm apart"; for those, we ask the LLM to produce a
``{symbol → {value, unit}}`` JSON envelope keyed by the symbols the
chosen formula needs.

The LLM is **not** asked to compute anything. It only extracts numbers
and units; the deterministic SymPy solver still does the math, the
deterministic verifier still grades it.

E6 (Day-22) adds :func:`extract_with_llm_self_consistent` — the
organizer-endorsed self-consistency vote (Slide 28 "Practical Tips"):
sample N independent extractions, then per-symbol pick the value that
has majority close-agreement and the most common unit string. Robust
against the 3B model's occasional one-off hallucinated number; falls
back cleanly when no consensus exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from exact_agent.llm.prompt_templates import render
from exact_agent.llm.vllm_client import LLMClient, parse_json_completion
from exact_agent.physics.formula_library import Formula
from exact_agent.physics.quantity_extractor import ExtractedQuantity
from exact_agent.physics.rag_retriever import WorkedExample


class LLMExtractionError(RuntimeError):
    """Raised when the LLM produced no parseable JSON for the formula schema."""


def extract_with_llm(
    question: str,
    formula: Formula,
    client: LLMClient,
    *,
    max_tokens: int = 256,
    examples: Sequence[WorkedExample] = (),
) -> list[ExtractedQuantity]:
    """Ask the LLM for the formula's required inputs as JSON, return them.

    Returns one :class:`ExtractedQuantity` per declared input on success.
    Missing keys raise :class:`LLMExtractionError` so the caller can fall
    back to abstaining instead of silently inventing numbers.

    ``examples`` (E8 RAG): optional list of solved training rows to
    inject as few-shot demonstrations before the actual question — the
    Slide-28 "Practical Tips" pattern. Empty tuple ⇒ legacy zero-shot.
    """
    inputs_dict: dict[str, Any] = {}
    for name, spec in formula.inputs.items():
        inputs_dict[name] = {
            "unit": spec.unit,
            "aliases": list(spec.aliases),
        }

    prompt = render(
        "physics_extract.j2",
        formula_id=formula.id,
        description=formula.description,
        inputs=inputs_dict,
        question=question,
        examples=list(examples),
    )

    raw = client.complete(prompt, max_tokens=max_tokens)
    if not raw.strip():
        raise LLMExtractionError("LLM returned empty completion")

    try:
        payload = parse_json_completion(raw)
    except (ValueError, TypeError) as exc:
        raise LLMExtractionError(f"could not parse LLM JSON: {exc}; raw={raw[:200]!r}") from exc

    if not isinstance(payload, dict):
        raise LLMExtractionError(f"expected a JSON object; got {type(payload).__name__}")

    quantities: list[ExtractedQuantity] = []
    missing: list[str] = []
    for symbol in formula.inputs:
        node = payload.get(symbol)
        if not isinstance(node, dict):
            missing.append(symbol)
            continue
        try:
            value = float(node["value"])
        except (KeyError, TypeError, ValueError):
            missing.append(symbol)
            continue
        unit = str(node.get("unit") or "").strip()
        quantities.append(
            ExtractedQuantity(
                name=symbol,
                value=value,
                unit=unit,
                raw_value=str(node.get("value")),
                raw_unit=unit,
            )
        )

    if missing:
        raise LLMExtractionError(f"LLM JSON missing inputs {missing}; payload={payload!r}")
    return quantities


def extract_with_llm_self_consistent(
    question: str,
    formula: Formula,
    client: LLMClient,
    *,
    n_votes: int = 3,
    max_tokens: int = 256,
    tol: float = 0.01,
    examples: Sequence[WorkedExample] = (),
) -> tuple[list[ExtractedQuantity], dict[str, int]]:
    """Vote-aggregate ``n_votes`` independent :func:`extract_with_llm` calls.

    Returns ``(quantities, vote_counts)``. ``vote_counts[symbol]`` is the
    number of attempts whose value for that symbol agreed (within ``tol``
    relative tolerance) with the elected value — useful for the solver
    trace ("q1 elected by 3/3 votes").

    A symbol is elected when at least ``majority`` (= ceil(n_votes/2))
    successful attempts cluster around the same value. The unit string
    is taken from the *most common* unit among the agreeing votes. If
    any symbol has no majority cluster, the whole extraction is
    rejected (LLMExtractionError) — abstaining is sounder than
    half-voting an inconsistent number into the solver.
    """
    if n_votes < 1:
        raise ValueError("n_votes must be >= 1")
    majority = (n_votes // 2) + 1

    attempts: list[list[ExtractedQuantity]] = []
    last_err: Exception | None = None
    for _ in range(n_votes):
        try:
            attempts.append(extract_with_llm(
                question, formula, client,
                max_tokens=max_tokens, examples=examples,
            ))
        except LLMExtractionError as exc:
            last_err = exc

    if len(attempts) < majority:
        raise LLMExtractionError(
            f"self-consistency: only {len(attempts)}/{n_votes} attempts produced "
            f"a parseable schema (need ≥{majority}); last error: {last_err}"
        )

    elected: list[ExtractedQuantity] = []
    vote_counts: dict[str, int] = {}
    for symbol in formula.inputs:
        candidates: list[ExtractedQuantity] = [
            q for attempt in attempts for q in attempt if q.name == symbol
        ]
        if not candidates:
            raise LLMExtractionError(f"self-consistency: no attempts produced {symbol!r}")

        # Cluster by relative tolerance against each candidate's value.
        best_score = 0
        best_value = candidates[0].value
        best_cluster: list[ExtractedQuantity] = [candidates[0]]
        for pivot in candidates:
            cluster = [c for c in candidates if _values_close(c.value, pivot.value, tol)]
            if len(cluster) > best_score:
                best_score = len(cluster)
                best_value = pivot.value
                best_cluster = cluster

        if best_score < majority:
            raise LLMExtractionError(
                f"self-consistency: {symbol!r} has no majority cluster "
                f"({best_score}/{n_votes} agreed; values seen: "
                f"{[round(c.value, 6) for c in candidates]})"
            )
        unit = Counter(c.unit for c in best_cluster).most_common(1)[0][0]
        raw_value_str = next(c.raw_value for c in best_cluster if c.value == best_value)
        elected.append(
            ExtractedQuantity(
                name=symbol,
                value=best_value,
                unit=unit,
                raw_value=raw_value_str,
                raw_unit=unit,
            )
        )
        vote_counts[symbol] = best_score

    return elected, vote_counts


def _values_close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return abs(a - b) <= tol
    return abs(a - b) / max(abs(a), abs(b)) <= tol
