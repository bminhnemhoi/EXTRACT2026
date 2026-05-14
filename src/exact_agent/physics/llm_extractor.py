"""LLM-driven fallback for the regex quantity extractor.

The regex extractor only fires on explicit ``name = value unit`` syntax.
Real questions in the dataset write things like "the voltage across its
plates is 60 V" or "8 cm apart"; for those, we ask the LLM to produce a
``{symbol → {value, unit}}`` JSON envelope keyed by the symbols the
chosen formula needs.

The LLM is **not** asked to compute anything. It only extracts numbers
and units; the deterministic SymPy solver still does the math, the
deterministic verifier still grades it.
"""

from __future__ import annotations

from typing import Any

from exact_agent.llm.prompt_templates import render
from exact_agent.llm.vllm_client import LLMClient, parse_json_completion
from exact_agent.physics.formula_library import Formula
from exact_agent.physics.quantity_extractor import ExtractedQuantity


class LLMExtractionError(RuntimeError):
    """Raised when the LLM produced no parseable JSON for the formula schema."""


def extract_with_llm(
    question: str,
    formula: Formula,
    client: LLMClient,
    *,
    max_tokens: int = 256,
) -> list[ExtractedQuantity]:
    """Ask the LLM for the formula's required inputs as JSON, return them.

    Returns one :class:`ExtractedQuantity` per declared input on success.
    Missing keys raise :class:`LLMExtractionError` so the caller can fall
    back to abstaining instead of silently inventing numbers.
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
