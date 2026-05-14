"""Pick the most likely formula for a physics question.

We have two layers, applied in order:

1. **Keyword router** — if the question text mentions a high-signal cue
   (e.g. "energy stored", "Coulomb", "resistance in parallel"), the
   corresponding formula id is returned directly. This handles ~80% of
   the dataset without any ML.
2. **Heuristic fallback by extracted symbols** — when no keyword fires we
   check which formula's required symbols best matches the variable names
   the extractor found in the question. A small bias picks the formula
   with the most input symbols (more specific equations win ties).

This is deliberately *not* a learned classifier: at Day-2 baseline the
rule set is short and easy to audit. A TF-IDF backup can be added later
if eval shows recurring misses; for now the bias is correctness over
recall.
"""

from __future__ import annotations

from dataclasses import dataclass

from exact_agent.physics.formula_library import Formula, FormulaLibrary
from exact_agent.physics.quantity_extractor import ExtractedQuantity


@dataclass(frozen=True)
class ClassificationResult:
    formula_id: str
    confidence: float
    reason: str


# Lowercase substring → formula id. Ordered for evaluation; first match wins.
_KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("energy stored", "energy in capacitor", "0.5cu", "0.5 cu"), "capacitor_energy"),
    (("capacitance of",), "capacitance_from_charge"),
    (("coulomb",), "coulomb_force"),
    (("electric field",), "electric_field_point_charge"),
    (("resistors in parallel", "in parallel"), "parallel_resistance_two"),
    (("resistors in series", "in series"), "series_resistance_two"),
    (
        (
            "rlc impedance",
            "impedance of",
        ),
        "rlc_impedance",
    ),
    (("resonance frequency", "resonant frequency"), "resonance_frequency"),
    (("solenoid",), "magnetic_field_solenoid"),
    (("long wire", "straight wire"), "magnetic_field_long_wire"),
    (("power dissipated", "joule heating"), "power_i2r"),
    (("power",), "power_voltage_current"),
    (("voltage across", "ohm", "ohm's law"), "ohm_law_voltage"),
)


def _keyword_match(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    for keywords, formula_id in _KEYWORD_RULES:
        for kw in keywords:
            if kw in lower:
                return formula_id, f"keyword '{kw}'"
    return None


def _symbol_match(
    library: FormulaLibrary, quantities: list[ExtractedQuantity]
) -> tuple[Formula, float, str] | None:
    """Score each formula by symbol-set overlap with the extracted variables."""
    if not quantities:
        return None
    extracted_names = {q.name for q in quantities}

    best: tuple[Formula, float, str] | None = None
    for formula in library.all():
        required = set(formula.required_symbols())
        if not required:
            continue
        overlap = required & extracted_names
        if not overlap:
            continue
        # Score = fraction of required symbols present.
        score = len(overlap) / len(required)
        if score < 1.0:
            continue  # demand full coverage at Day 2 — partial matches are noisy
        # Prefer more specific (longer required) formulas.
        candidate = (formula, score, f"symbol coverage {sorted(overlap)}")
        if best is None or len(formula.required_symbols()) > len(best[0].required_symbols()):
            best = candidate
    return best


def classify(
    question: str,
    quantities: list[ExtractedQuantity],
    library: FormulaLibrary,
) -> ClassificationResult | None:
    """Return the best-matching formula id, or None if we cannot decide."""
    keyword = _keyword_match(question)
    if keyword is not None:
        formula_id, reason = keyword
        if formula_id in library:
            return ClassificationResult(formula_id=formula_id, confidence=0.9, reason=reason)

    fallback = _symbol_match(library, quantities)
    if fallback is not None:
        formula, score, reason = fallback
        return ClassificationResult(formula_id=formula.id, confidence=0.5 * score, reason=reason)

    return None
