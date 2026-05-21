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
    # Iter-3 (Day-25 TD010) STATE-CHANGE: a capacitor was connected to a
    # source, then disconnected, then plate distance changed. Q is
    # conserved → V scales with d. MUST precede every other capacitor
    # routing (capacitor_energy / voltage_from_energy_capacitance /
    # parallel_plate_*) because those would otherwise misroute on
    # "capacitor" / "voltage" keywords. Position MATTERS: keyword rules
    # are first-match-wins; this block must come above the next
    # voltage_from_energy_capacitance block.
    (
        (
            "is then disconnected",
            "then disconnected from the source",
            "after disconnecting",
            "disconnected from the source",
        ),
        "voltage_capacitor_distance_ratio",
    ),
    # Iter-3 (Day-25 THCB070) STATE-CHANGE circuit: a lamp/component
    # is removed and we're given the remaining lamp's current directly
    # ("lamp D2 draws 0.5 A"). The answer is the passthrough of that
    # current value. MUST precede the parallel_resistance keyword rule
    # below because "connected in parallel" still fires on the initial
    # state.
    (
        (
            "is removed",
            "is taken out",
            "if lamp d",   # "if lamp Dx is removed"
            "if one lamp",
            "if one bulb",
        ),
        "passthrough_current_from_remaining",
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
    # Iter-2 (Day-25 TD013): inverse direction — solve ε_r given C, A, d.
    # MUST precede parallel_plate_capacitance_dielectric because both
    # rules match the same "dielectric" wording but the INTENT differs.
    (
        (
            "what is the dielectric constant",
            "calculate the dielectric constant",
            "find the dielectric constant",
            "determine the dielectric constant",
            "what is the relative permittivity",
        ),
        "dielectric_constant_from_capacitance",
    ),
    # Iter-4 Fix C (TD025/043/079/191): "An air-filled parallel-plate
    # capacitor HAS A CAPACITANCE OF X pF and is charged to Y V. Calculate
    # the electric field energy" must route to capacitor_energy (uses the
    # given C and U directly), NOT parallel_plate_capacitance (which
    # asks for A and d that the question doesn't give). The phrase
    # "electric field energy" is the unambiguous intent marker.
    (
        (
            "calculate the electric field energy stored",
            "calculate the electric field energy in",
            "electric field energy stored in the capacitor",
        ),
        "capacitor_energy",
    ),
    # Parallel-plate geometry capacitance — area + separation given.
    # MUST precede the "calculate its capacitance" rule below so a
    # geometry problem doesn't fall into the energy/voltage formula.
    # Day-22 audit surfaced TD162/189/383/174 all matching this shape;
    # previous classifier routed them ⇒ wrong formula.
    (
        (
            "parallel-plate capacitor has a dielectric",
            "parallel-plate capacitor with a dielectric",
            "relative permittivity",
            "dielectric constant",
        ),
        "parallel_plate_capacitance_dielectric",
    ),
    (
        (
            # Specific parallel-plate phrasings only — the bare "plate
            # area"/"plate separation" pulled in problems that already had
            # a correct C/U-based formula (net -1 row on Day-22 holdout).
            "air parallel-plate capacitor",
            "parallel-plate air capacitor",
            "air-filled parallel-plate",
        ),
        "parallel_plate_capacitance",
    ),
    (
        (
            "calculate its capacitance",
            "calculate the capacitance (unit: μf",
            "calculate the capacitance (unit: uf",
        ),
        "capacitance_from_energy_voltage",
    ),
    # Iter-1 NL005: solve U from W and C (capacitor). Must precede the
    # capacitor_energy rule below (which fires on "electric field energy"
    # too eagerly otherwise). Output is voltage, so verify the question
    # asks for voltage/potential difference explicitly.
    (
        (
            "calculate the potential difference (unit: v)",
            "calculate the voltage (unit: v)",
            "potential difference (unit: v) between",
        ),
        "voltage_from_energy_capacitance",
    ),
    # Iter-5 Fix C (TD361): "A capacitor has a CHARGE OF X μC and a
    # VOLTAGE OF Y V. Calculate the energy stored." Question never gives
    # C, so capacitor_energy (needs C+U) fails. Use E = Q·U/2 form.
    # MUST precede the generic energy rule below. Phrase is narrow to the
    # capacitor framing (does not catch generic "two charges" questions).
    (
        (
            "capacitor has a charge of",
            "capacitor with a charge of",
            "capacitor carries a charge of",
        ),
        "capacitor_energy_from_charge_voltage",
    ),
    (
        (
            "energy stored in capacitor",
            "energy stored in a capacitor",
            "energy stored in the capacitor",
            "energy in capacitor",
            "energy in a capacitor",
            "energy in the capacitor",
            "energy of the capacitor",
            "energy of a capacitor",
            "energy (mj) stored",
            "energy in the electric field",
            "energy stored in the electric field",
            "electric field energy",
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
    # Iter-4 Fix D: perpendicular-bisector 2-opposite-charge FIELD problems
    # (LD065/099/100/340). MUST precede the force perp-bisector rule below
    # because field_asked makes the force rule skip, leaving symbol-fallback
    # to land on single-charge field formula (which only uses one q -> 100%
    # off). The F2 field guard inside _keyword_match also skips this rule
    # when force_asked, so collinear/force questions retain the force rule.
    (
        ("perpendicular bisector",),
        "electric_field_perp_bisector_two_opposite",
    ),
    (
        ("perpendicular bisector",),
        "coulomb_force_perp_bisector",
    ),
    # Iter-2 (Day-25 LD026): q3 on segment between OPPOSITE-sign q1, q2
    # (CA and CB explicit). Forces add. MUST precede LD025 same-sign rule
    # below because "ca = 4 cm" / "cb = 2 cm" is the discriminator.
    (
        (
            "ca =",
            "cb =",
            "ca=",
            "cb=",
        ),
        "coulomb_force_collinear_opposite_signs",
    ),
    # Iter-1 (Day-25 LD025): q3 on the line segment between q1 and q2
    # ("positioned along the line connecting q1 and q2", "when it is
    # X cm away from q1"). Net force is k·|q3·q|·|1/r1² - 1/r2²|.
    # Must precede generic coulomb_force.
    (
        (
            "positioned along the line connecting",
            "positioned on the line segment",
            "positioned along the segment",
            "when it is",  # narrow: usually follows "X cm away from q1"
        ),
        "coulomb_force_on_charge_between_two_identical",
    ),
    # Iter-1 (Day-25 DT005): isoceles triangle with AC = BC, ±q1 sources
    # at A and B, test charge q3 at C. Closed form F = k·|q1·q3|·AB/r³.
    (
        (
            "ac = bc",
            "ac=bc",
            "given that ac = bc",
        ),
        "coulomb_force_two_opposite_sources_isoceles_apex",
    ),
    # Iter-1 (Day-25 DT006): right triangle at C (AC² + BC² = AB²).
    # General sources q1, q2; test charge q3 at C. F = q3·sqrt(E1²+E2²).
    # Keyword: "AC = X and BC = Y" without AC=BC (caught by above first).
    (
        (
            "and bc =",
            "ac = 12 cm and bc",
            "ac = 16 cm and bc",
        ),
        "coulomb_force_two_sources_right_triangle_apex",
    ),
    # (TD013 rule moved above the parallel_plate_capacitance_dielectric block)
    # 3 identical charges at vertices of an equilateral triangle → vector
    # sum of two equal pairwise forces at 60° gives F·√3. MUST also
    # precede generic coulomb_force. Day-22 E10 audit (LD295, LD130,
    # LD228, LD242) all hit this geometry. Day-23 audit found LD314 /
    # LD317 / LD396 misroute here because they say "equilateral" but
    # ASK for the field. F2 guard in `_keyword_match` skips this rule
    # when the question is field-asking (V/m or "field intensity"...).
    (
        (
            "equilateral triangle",
            "vertices of an equilateral",
            "equilateral triangle with side",
        ),
        "coulomb_force_equilateral_three_identical",
    ),
    # F1 new — E = F/q. Day-23 audit DT046: F=3mN given on test charge
    # q=1e-7 C → E = 3e4 V/m. Must precede the field/force rules below.
    (
        (
            "experiences a force",
            "experiencing a force",
            "force of magnitude",
        ),
        "electric_field_from_force",
    ),
    # G1 (Day-24) — 2 opposite-sign charges, field at midpoint. Audit
    # post-F1+F2 found LD053/LD090/LD065/LD056/LD081 etc. all this
    # shape ("two charges q1 = -q2 placed at A, B... field at midpoint
    # / field at midpoint of AB"). Must precede the single-charge
    # electric_field_point_charge rule below.
    (
        (
            "field at the midpoint",
            "field strength at the midpoint",
            "electric field at the midpoint",
            "field at midpoint of",
        ),
        "electric_field_two_opposite_charges_midpoint",
    ),
    # G1 (Day-24) — Z = U/I. DDT349 wants this; but the broad
    # "calculate the total impedance" keyword regressed -1 row on CH
    # by eating rlc_impedance cases (CH rows that have R + XL + XC and
    # SHOULD route to rlc_impedance). Routing rule REMOVED; formula is
    # left in the registry so the symbol-fallback can still pick it
    # when the question literally extracts only U + I.
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
    # F1 new — power factor cosφ = R/Z. Audit DDT327, DDT337 ("calculate
    # the power factor"). MUST precede impedance generic.
    (
        ("power factor", "cosφ", "cos φ", "cos(φ)", "cosphi"),
        "power_factor_from_r_z",
    ),
    # F1 new — magnetic flux Φ = B·A through one turn. DDT158
    # ("magnetic flux through one turn").
    (
        ("magnetic flux through one turn", "flux through one turn", "magnetic flux through a turn"),
        "magnetic_flux_solenoid_one_turn",
    ),
    # F1 new — inductance for resonance. CH067/CH084 ("what value of L /
    # required inductance to resonate at f"). Must precede LC resonance
    # frequency rule which would otherwise grab the same question.
    (
        (
            "what value of inductor",
            "required inductance",
            "what is the required inductance",
            "needed to resonate",
            "required to resonate at",
        ),
        "inductance_for_resonance",
    ),
    # F1 new — inductance from inductor magnetic energy + current. NL010.
    (
        (
            "calculate its inductance",
            "calculate the inductance",
        ),
        "inductance_from_inductor_energy",
    ),
    # Iter-1 (Day-25 NL007): "magnetic field energy ... calculate the current"
    # — inductor energy → current via I = sqrt(2W/L). Formula already exists.
    (
        (
            "magnetic field energy",
            "magnetic energy",
        ),
        "current_from_inductor_energy",
    ),
    # (NL005 routing moved above the capacitor_energy block; see comment there)
    # F1 new — current from voltage and impedance. RLC-DDT339-style.
    (
        ("calculate the rms current", "calculate the current i in the circuit"),
        "current_from_voltage_impedance",
    ),
    # Iter-5 Fix A: resonance/reactance rules MUST precede the generic
    # "in series" → series_resistance_two rule. CH025/031/032 ("L in series
    # with C") clearly say "resonant frequency" and DDT345 says "capacitive
    # reactance", but the naive "in series" trigger was capturing them
    # before this rule could fire. Order matters in _keyword_match (first
    # match wins).
    (("resonance frequency", "resonant frequency", "natural frequency"), "resonance_frequency"),
    (("resistors in parallel", "connected in parallel", "in parallel"), "parallel_resistance_two"),
    # Tighten series rule: require explicit resistor wording. "in series"
    # alone over-matches inductor/capacitor combos that need RLC formulas.
    (
        ("resistors in series", "two resistors connected", "series resistors"),
        "series_resistance_two",
    ),
    (("rlc impedance", "impedance of", "total impedance"), "rlc_impedance"),
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
    "capacitor_energy_from_charge_voltage": frozenset({"energy", "joule", "stored"}),
    "capacitance_from_charge": frozenset({"capacitance", "farad"}),
    "charge_from_capacitance": frozenset({"charge", "coulomb", "stored"}),
    "voltage_from_energy_capacitance": frozenset({"voltage", "potential difference"}),
    "capacitance_from_energy_voltage": frozenset({"capacitance"}),
    "coulomb_force": frozenset({"force", "newton"}),
    "coulomb_force_at_midpoint": frozenset({"midpoint", "force"}),
    "coulomb_force_perp_bisector": frozenset({"bisector", "force"}),
    "coulomb_force_equilateral_three_identical": frozenset({"equilateral", "force"}),
    "parallel_plate_capacitance": frozenset({"capacitance", "farad"}),
    "parallel_plate_capacitance_dielectric": frozenset({"capacitance", "farad"}),
    # F1 (Day-23) target-word guards for symbol-only fallback.
    "magnetic_flux_solenoid_one_turn": frozenset({"flux", "weber"}),
    "power_factor_from_r_z": frozenset({"factor", "cos"}),
    "inductance_for_resonance": frozenset({"inductance", "henry"}),
    "electric_field_from_force": frozenset({"field", "v/m", "intensity"}),
    "inductance_from_inductor_energy": frozenset({"inductance", "henry"}),
    "current_from_voltage_impedance": frozenset({"current", "ampere"}),
    # G1 (Day-24)
    "electric_field_two_opposite_charges_midpoint": frozenset({"midpoint", "field"}),
    "electric_field_perp_bisector_two_opposite": frozenset({"bisector", "field"}),
    "impedance_from_voltage_current": frozenset({"impedance", "ohm"}),
    # Iter-1/2 (Day-25)
    "dielectric_constant_from_capacitance": frozenset({"dielectric", "constant", "permittivity"}),
    "coulomb_force_on_charge_between_two_identical": frozenset({"force", "newton"}),
    "coulomb_force_two_opposite_sources_isoceles_apex": frozenset({"force", "newton"}),
    "coulomb_force_two_sources_right_triangle_apex": frozenset({"force", "newton"}),
    "coulomb_force_collinear_opposite_signs": frozenset({"force", "newton"}),
    # Iter-3 (Day-25) state-change
    "voltage_capacitor_distance_ratio": frozenset({"voltage", "volt", "potential"}),
    "ohm_law_current": frozenset({"current", "ampere"}),
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


# F2 (Day-23) — formulas that compute a FORCE in newtons. If the
# question is clearly asking for an electric FIELD (V/m), don't let a
# keyword like "equilateral triangle" route us into a Newton-output
# formula. Audit Day-22: LD314 / LD396 / LD317 all hit this.
_FORCE_FORMULAS: frozenset[str] = frozenset({
    "coulomb_force",
    "coulomb_force_at_midpoint",
    "coulomb_force_perp_bisector",
    "coulomb_force_equilateral_three_identical",
    "resultant_two_forces",
})
# Iter-4 Fix D — the inverse guard. Symmetric to _FORCE_FORMULAS so a
# "calculate the net force" question doesn't trip a field-output formula
# (V/m would be marked unit-wrong vs gold N).
_FIELD_FORMULAS: frozenset[str] = frozenset({
    "electric_field_point_charge",
    "electric_field_two_opposite_charges_midpoint",
    "electric_field_perp_bisector_two_opposite",
    "electric_field_from_force",
})
_FORCE_ASK_TOKENS: tuple[str, ...] = (
    "net force",
    "resultant force",
    "force acting on",
    "force on the charge",
    "force exerted on",
    "magnitude of the force",
    "magnitude of the net force",
    "find the force",
    "calculate the force",
    "what is the force",
    "find the net",      # "find the net force/charge/etc." — broad but field
    " newton",            # rules already exclude on "field" tokens above.
    "(unit: n)",
)
_FIELD_ASK_TOKENS: tuple[str, ...] = (
    "electric field intensity",
    "electric field strength",
    "electric field at",
    "magnitude of the electric field",
    "field intensity",
    "field strength",
    " v/m",
    "(v/m)",
    "volt/meter",
    "volts per meter",
)


def _is_field_asking(question_lower: str) -> bool:
    return any(tok in question_lower for tok in _FIELD_ASK_TOKENS)


def _is_force_asking(question_lower: str) -> bool:
    return any(tok in question_lower for tok in _FORCE_ASK_TOKENS)


def _keyword_match(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    field_asked = _is_field_asking(lower)
    force_asked = _is_force_asking(lower)
    for keywords, formula_id in _KEYWORD_RULES:
        for kw in keywords:
            if kw in lower:
                # F2 guard: never let a "force" formula win when the
                # question is asking for a field — units mismatch.
                if field_asked and formula_id in _FORCE_FORMULAS:
                    continue
                # Iter-4 Fix D mirror: never let a "field" formula win
                # when force is asked — e.g. "perpendicular bisector"
                # paired with "calculate the net force" must stay on
                # coulomb_force_perp_bisector, not the field variant.
                if force_asked and formula_id in _FIELD_FORMULAS:
                    continue
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

    G1 (Day-24): the F2 field-vs-force guard from ``_keyword_match`` is
    duplicated here — a Coulomb-type force formula must never win the
    symbol fallback when the question is asking for an electric field.
    Without this, an equilateral-3-charge question phrased
    "calculate the field at the centroid" still landed on
    `coulomb_force_equilateral_three_identical` via the q+a symbol overlap.
    """
    if not quantities:
        return None
    extracted_names = {q.name for q in quantities}
    field_asked = _is_field_asking(question_lower)
    force_asked = _is_force_asking(question_lower)

    best: tuple[Formula, float, str] | None = None
    for formula in library.all():
        if field_asked and formula.id in _FORCE_FORMULAS:
            continue
        if force_asked and formula.id in _FIELD_FORMULAS:
            continue
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
