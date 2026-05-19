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
# More specific patterns first so they outrank generic single-word triggers
# (e.g. "resonant frequency" before plain "frequency"). Capacitor-energy
# variants beat generic "electric field" because real questions like
# "energy stored in the electric field" must route to capacitor_energy.
_KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    # --- Day-11 RLC-resonance / error / inductor closed forms. Highly
    # specific phrases; placed first so they can't be pre-empted. Order
    # inside this block matters: 'pure resistance' (resistance_at_resonance)
    # is checked before the 'resonant rlc circuit' power phrasing so CH001
    # ("...determine the pure resistance R") doesn't fall into power. ---
    (
        (
            "by what factor",
            "factor must the angular frequency",
            "angular frequency must be set to",
            "angular frequency be multiplied",
            "factor of ω0",
            "factor of w0",
        ),
        "resonance_frequency_factor",
    ),
    (
        (
            "what capacitance must",
            "capacitance must the capacitor have",
            "to achieve resonance",
            "to resonate at",
            "resonate at a frequency",
        ),
        "capacitance_for_resonance",
    ),
    (
        ("pure resistance",),
        "resistance_at_resonance",
    ),
    (
        (
            "resonant rlc circuit",
            "rlc resonant circuit",
            "rlc resonant",
        ),
        "power_at_resonance",
    ),
    (
        (
            "relative error",
            "percentage relative error",
            "percentage error",
            "percent error",
        ),
        "relative_error_percent",
    ),
    (
        # "of magnetic energy" excludes "total electromagnetic energy" (DDT354).
        ("of magnetic energy", "stores magnetic energy"),
        "current_from_inductor_energy",
    ),
    # Capacitor-energy-derived unknowns (must come first — they often mention
    # both 'energy' and 'electric field' in the same question).
    (
        (
            "potential difference u (v) between its plates",
            "potential difference between its plates",
            "voltage u (v) between its plates",
            "voltage between its two plates",
            "voltage across the capacitor",
        ),
        "voltage_from_energy_capacitance",
    ),
    (
        (
            "calculate its capacitance",
            "calculate the capacitance (unit: μf",
            "calculate the capacitance (unit: uf",
        ),
        "capacitance_from_energy_voltage",
    ),
    (
        (
            "energy stored",
            "energy in capacitor",
            "energy of the capacitor",
            "energy (mj) stored",
            "energy in the electric field",
            "energy stored in the electric field",
            "electric field energy",
            "field energy",
            "0.5cu",
            "0.5 cu",
        ),
        "capacitor_energy",
    ),
    # Charge-from-capacitance (Q = C·U) — question gives C and U, asks for Q.
    (
        (
            "calculate the charge stored",
            "calculate the charge",
            "charge stored by the capacitor",
            "charge q stored",
        ),
        "charge_from_capacitance",
    ),
    # Capacitance from Q + U.
    (
        ("capacitance of", "capacitance c of", "find the capacitance", "what is the capacitance"),
        "capacitance_from_charge",
    ),
    # Resultant of two GIVEN forces at an angle (law of cosines). Must beat
    # the coulomb rules below — these problems hand you F1/F2 directly and
    # never ask you to compute a force from charges.
    (
        (
            "act at an angle",
            "at an angle of",
            "angle between the two forces",
            "angle between them",
            "two forces with magnitudes",
            "resultant of the two forces",
        ),
        "resultant_two_forces",
    ),
    # Geometry-specific Coulomb closed forms — MUST precede generic
    # coulomb_force ("force acting on") so a 3-charge midpoint /
    # perpendicular-bisector problem doesn't fall into the single-pair
    # formula (right unit, wrong physics).
    (
        ("at the midpoint", "midpoint of the line segment", "midpoint o"),
        "coulomb_force_at_midpoint",
    ),
    (
        ("perpendicular bisector",),
        "coulomb_force_perp_bisector",
    ),
    # Coulomb / force between charges.
    (("coulomb",), "coulomb_force"),
    (("force between", "force acting on", "force on the charge"), "coulomb_force"),
    # Electric field — note: must NOT fire on "field energy" / "stored in the
    # electric field". Use narrower phrasings that imply asking for the field.
    (
        (
            "electric field intensity",
            "electric field strength",
            "magnitude of the electric field",
            "electric field e at",
            "electric field at point",
            "field intensity",
            "field strength at",
        ),
        "electric_field_point_charge",
    ),
    (("resistors in parallel", "connected in parallel", "in parallel"), "parallel_resistance_two"),
    (("resistors in series", "connected in series", "in series"), "series_resistance_two"),
    (("rlc impedance", "impedance of", "total impedance"), "rlc_impedance"),
    (("resonance frequency", "resonant frequency", "natural frequency"), "resonance_frequency"),
    (("solenoid",), "magnetic_field_solenoid"),
    (("long wire", "straight wire", "current-carrying wire"), "magnetic_field_long_wire"),
    (("power dissipated", "joule heating", "heat dissipated"), "power_i2r"),
    (("power consumed", "power output", "electric power"), "power_voltage_current"),
    (("voltage across", "potential difference", "ohm's law"), "ohm_law_voltage"),
)


# Target words that must appear (case-insensitive) for a symbol-only fallback
# to commit to a given formula. Without this guard, any question containing
# both ``C`` and ``U`` would be misclassified as `capacitor_energy`, even when
# the prompt asks for capacitance or charge.
_TARGET_WORDS: dict[str, frozenset[str]] = {
    "capacitor_energy": frozenset({"energy", "joule", "stored"}),
    "capacitance_from_charge": frozenset({"capacitance", "farad"}),
    "charge_from_capacitance": frozenset({"charge", "coulomb", "stored"}),
    "voltage_from_energy_capacitance": frozenset({"voltage", "potential difference"}),
    "capacitance_from_energy_voltage": frozenset({"capacitance"}),
    "coulomb_force": frozenset({"force", "newton"}),
    "coulomb_force_at_midpoint": frozenset({"midpoint", "force"}),
    "coulomb_force_perp_bisector": frozenset({"bisector", "force"}),
    "resultant_two_forces": frozenset({"resultant", "angle"}),
    "electric_field_point_charge": frozenset({"intensity", "strength", "magnitude"}),
    "ohm_law_voltage": frozenset({"voltage", "volt"}),
    "power_voltage_current": frozenset({"power", "watt"}),
    "power_i2r": frozenset({"power", "watt", "dissipat", "heat"}),
    "series_resistance_two": frozenset({"resistance", "ohm"}),
    "parallel_resistance_two": frozenset({"resistance", "ohm"}),
    "rlc_impedance": frozenset({"impedance"}),
    "resonance_frequency": frozenset({"frequency", "resonan", "hertz"}),
    "capacitance_for_resonance": frozenset({"resonate", "resonance"}),
    "resonance_frequency_factor": frozenset({"factor", "angular frequency"}),
    "resistance_at_resonance": frozenset({"resistance", "resonan"}),
    "power_at_resonance": frozenset({"power", "resonan"}),
    "relative_error_percent": frozenset({"error", "%"}),
    "current_from_inductor_energy": frozenset({"current", "inductor", "magnetic"}),
    "magnetic_field_solenoid": frozenset({"magnetic", "field", "tesla"}),
    "magnetic_field_long_wire": frozenset({"magnetic", "field", "tesla"}),
}


def _keyword_match(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    for keywords, formula_id in _KEYWORD_RULES:
        for kw in keywords:
            if kw in lower:
                return formula_id, f"keyword '{kw}'"
    return None


def _symbol_match(
    library: FormulaLibrary,
    quantities: list[ExtractedQuantity],
    question_lower: str,
) -> tuple[Formula, float, str] | None:
    """Score each formula by symbol-set overlap with the extracted variables.

    Requires a target word from :data:`_TARGET_WORDS` to also appear in the
    question, otherwise multiple formulas sharing input symbols (e.g.
    ``{C, U}``) would always pick the first one seen.
    """
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
        score = len(overlap) / len(required)
        if score < 1.0:
            continue  # demand full coverage — partial matches are noisy
        targets = _TARGET_WORDS.get(formula.id, frozenset())
        if targets and not any(t in question_lower for t in targets):
            continue  # right symbols, but the question asks for something else
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
    question_lower = question.lower()

    keyword = _keyword_match(question)
    if keyword is not None:
        formula_id, reason = keyword
        if formula_id in library:
            return ClassificationResult(formula_id=formula_id, confidence=0.9, reason=reason)

    fallback = _symbol_match(library, quantities, question_lower)
    if fallback is not None:
        formula, score, reason = fallback
        return ClassificationResult(formula_id=formula.id, confidence=0.5 * score, reason=reason)

    return None
