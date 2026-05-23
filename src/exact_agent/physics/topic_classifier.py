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

import re
from dataclasses import dataclass

from exact_agent.physics.formula_library import Formula, FormulaLibrary
from exact_agent.physics.quantity_extractor import ExtractedQuantity
from exact_agent.physics.unit_converter import get_registry


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
    # === Iter-19 (Day-29) — placed at the TOP because the existing rules
    # below match the same keywords (e.g. "voltage across the capacitor",
    # "power factor", "ac = bc") and would steal the route via first-match.
    # Each Iter-19 rule is guarded so it only fires on its narrow target
    # pattern; on miss, the existing rule wins naturally.

    # 19a — CH143/144/145: series RLC AT RESONANCE, U_RC given, asks U_C.
    # Guard requires "at resonance" + R-C or C-L section phrasing AND
    # the question to be asking about the capacitor voltage.
    (
        (
            "rms voltage across the capacitor",
            "voltage across the capacitor c",
            "rms voltage across the c",
        ),
        "voltage_capacitor_from_urc_resonance",
        "rlc_resonance_with_urc",
    ),
    # 19e — CH247/249: "LCω² = 1 ... power factor" -> cos(φ) = 1.
    (
        (
            "lcω2 = 1",
            "lcω² = 1",
            "lc*ω2 = 1",
            "lcw2 = 1",
            "satisfies the condition lc",
            "the condition lc",
        ),
        "power_factor_at_resonance_constant",
        "asks_power_factor",
    ),
    # 19c — THCB094/114/133: per-unit mean formulas.
    (("calculate the average mass",), "mean_three_measurements_mass"),
    (("calculate the average voltage",), "mean_three_measurements_voltage"),
    (("calculate the average temperature",), "mean_three_measurements_temperature"),
    # 19b — LD052: same-sign isoceles general (NOT equilateral, NOT opposite).
    # MUST precede the existing "ac = bc" -> coulomb_force_two_opposite_*
    # rule (Iter-15) and "ac = bc" -> 16a opposite-isoceles rule.
    (
        ("ac = bc", "ac=bc"),
        "electric_field_isoceles_two_same_sign_apex",
        "same_sign_isoceles_not_equilateral",
    ),
    # === end Iter-19 ===

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
    # Iter-18c (Day-29) — LD067: 2 opposite-sign charges at midpoint in
    # a DIELECTRIC medium. MUST precede parallel_plate_capacitance_
    # dielectric (which matches "dielectric constant" generically) and
    # the regular midpoint field rule (which doesn't apply 1/ε scaling).
    # Guard requires both midpoint phrasing AND dielectric language; the
    # final classifier dim guard further requires the question to be
    # field-asking (V/m output, not farad).
    (
        (
            "midpoint of the line segment",
            "midpoint of the segment",
            "midpoint of ab",
            "at the midpoint",
        ),
        "electric_field_two_opposite_charges_midpoint_dielectric",
        "has_dielectric_constant",
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
            # Iter-15b: TD062 writes "parallel plate" without the hyphen —
            # add the unhyphenated variants so the routing catches it
            # before the generic "calculate its capacitance" rule below.
            "air parallel plate capacitor",
            "parallel plate air capacitor",
            "air-filled parallel plate",
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
            # Iter-17c (Day-29) — NL103: "Calculate the voltage across its
            # plates". W and C given. Asks for V → voltage_from_energy_capacitance.
            # Narrow so it doesn't steal energy-asking questions (those would
            # also be dim-rejected since output is volt).
            "calculate the voltage across its plates",
            "what is the voltage across its plates",
            "find the voltage across its plates",
            "voltage across the plates",
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
    # Iter-7 NL346: question gives energy + voltage, asks for charge.
    # MUST precede the generic capacitor_energy rule which fires on
    # "energy stored in a capacitor" and would steal this row.
    (
        (
            "what is the charge",
            "find the charge",
            "calculate the charge (mc)",
            "the charge (mc) on the capacitor",
        ),
        "charge_from_energy_voltage",
    ),
    # Iter-7 NL340: LC partition — given W_total and W_L, find W_C. Distinctive
    # phrase "what is the electric field energy" + LC context.
    (
        (
            "electric field energy (j)",
            "what is the electric field energy",
            "find the electric field energy",
        ),
        "lc_partition_electric_energy",
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
            # Iter-6 (LD126/182): "Two electric forces have magnitudes of
            # 3 N and 8 N, acting at a 90° angle" — earlier triggers missed
            # the inverted word order and the "their resultant force" wording.
            "two electric forces have magnitudes",
            "two electric forces have a magnitude",
            "their resultant force",
            "a 90° angle",
            "a 90 degree angle",
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
    # Iter-18c (Day-29) — LD067: midpoint field of 2 OPPOSITE-sign charges
    # in a DIELECTRIC medium. MUST precede the parallel_plate_capacitance_
    # dielectric rule above and the regular midpoint rule below — neither
    # of those applies the 1/ε scaling. Triggered when the question both
    # describes a midpoint geometry and contains dielectric language.
    (
        (
            "midpoint of the line segment",
            "midpoint of the segment",
            "midpoint of ab",
            "field at the midpoint",
        ),
        "electric_field_two_opposite_charges_midpoint_dielectric",
        "has_dielectric_constant",
    ),
    # Iter-18d (Day-29) — LD081 / LD387: SAME-sign perpendicular bisector
    # field. The existing electric_field_perp_bisector_two_opposite formula
    # is OPPOSITE-sign only; same-sign needs the (|q|±|q|) terms swapped.
    # Routing fires when "perpendicular bisector" + signs NOT opposite
    # (guarded). MUST precede the opposite-sign rule directly below.
    (
        ("perpendicular bisector",),
        "electric_field_perp_bisector_two_same_sign",
        "same_sign_two_charges",
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
    # Iter-16a (Day-29): LD053-style "AC = BC = r, q1 = -q2" field-at-apex
    # geometry. Equivalent to perpendicular-bisector with |q1|=|q2|, but
    # questions never spell out "perpendicular bisector" so the rule above
    # never fired. Parameterised in r so extractor doesn't need to compute
    # the perp distance l from Pythagoras. Guarded with opposite-sign
    # check so a same-sign isoceles question (LD052) does NOT route here
    # — that one is intentionally left to a future Iter-16c general
    # isoceles formula. Field-asked guard is applied by _keyword_match.
    (
        ("ac = bc", "ac=bc"),
        "electric_field_two_opposite_sources_isoceles_apex",
        "opposite_sign_two_charges",
    ),
    # Iter-16a (Day-29): LD090 equilateral with q1=-q2 is the special case
    # r=d of the rule above. Guard rule fires before the same-sign
    # equilateral 16b rule so opposite-sign equilaterals are dispatched
    # to the isoceles-opposite formula, not the same-sign vector-sum one.
    (
        ("equilateral triangle",),
        "electric_field_two_opposite_sources_isoceles_apex",
        "opposite_sign_two_charges",
    ),
    # Iter-16b (Day-29): DT051, LD319, LD392 — equilateral with same-sign
    # charges at two (or three) vertices, field at the remaining/queried
    # vertex. Vectors from the two source charges meet at 60° → √3·k|q|/a².
    # MUST precede the coulomb_force_equilateral_three_identical rule
    # below; F2 guard already skips that one on field-asked, but explicit
    # ordering keeps the trace reason "electric_field_equilateral_*" clean.
    # Opposite-sign equilateral case is captured above and won't reach here.
    (
        ("equilateral triangle",),
        "electric_field_equilateral_three_identical",
    ),
    # Iter-16d (Day-29): LD335 — three identical charges at vertices of
    # an isoceles right triangle, field at the right-angle vertex. The
    # two source vectors are perpendicular → √2·k|q|/r². Field-asked
    # guard applied. Distinct from coulomb_force_two_sources_right_triangle
    # (which fires on force questions and takes general q1, q2).
    (
        (
            "isosceles right triangle",
            "right-angle vertex",
            "right angle vertex",
        ),
        "electric_field_right_angle_two_identical",
    ),
    # Iter-18e (Day-29) — LD243: force-output mirror of Iter-16d. Three
    # identical charges at vertices of an isoceles right triangle; force
    # on the right-angle-vertex charge from the other two. F2 guard
    # ensures field-asked questions still hit the Iter-16d field formula.
    (
        (
            "isosceles right triangle",
            "right-angle vertex",
            "right angle vertex",
        ),
        "coulomb_force_right_angle_two_identical",
    ),
    # Iter-17f (Day-29) — LD362 (perpendicular), LD367 (60°): two source
    # charges of DIFFERENT magnitudes, each equidistant from point M,
    # with their source-field vectors at M forming a given angle theta.
    # Distinct from 16d (which requires identical charges + right angle
    # in a 3-charge triangle configuration). Phrase variants from the
    # dataset: "fields they produce at M are perpendicular" / "form an
    # angle of 60°" / "form a 90° angle". F2 / dim guards inherited.
    (
        (
            "are perpendicular to each other",
            "fields they produce at m are perpendicular",
            "form an angle of",
            "form a 90° angle",
            "form a 90 degree angle",
            "fields they produce at m form",
        ),
        "electric_field_two_sources_at_angle",
    ),
    # Iter-17e (Day-29) — THCB094/114/133 cluster DEFERRED: requires a
    # solver-side change so the output_unit passes the input unit through
    # (gold has g/V/°C; a single dimensionless formula can't satisfy
    # unit_match). Routing rule kept commented as a placeholder; un-comment
    # alongside the input-unit-passthrough plumbing in a later iter.
    # (
    #     ("calculate the average mass and the average absolute error",
    #      "calculate the average voltage and the average absolute error",
    #      "calculate the average temperature and the average absolute error"),
    #     "mean_three_measurements",
    # ),
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
    # Iter-17d (Day-29): broadened to catch LD067 phrasing where the
    # "electric field" and "at the midpoint" tokens are separated by a
    # prepositional clause ("electric field vector produced by ... at
    # the midpoint of the line segment"). Earlier triggers required
    # adjacent "field at the midpoint" substring, which LD067 violates.
    (
        (
            "field at the midpoint",
            "field strength at the midpoint",
            "electric field at the midpoint",
            "field at midpoint of",
            "at the midpoint of the line segment",
            "at the midpoint of the segment",
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
    # Iter-19f (Day-29) — added "what is its inductance" / "find its
    # inductance" so NL334 ("What is its inductance (H)?") stops being
    # mis-extracted by current_from_inductor_energy (which expects L).
    (
        (
            "calculate its inductance",
            "calculate the inductance",
            "what is its inductance",
            "find its inductance",
            "determine its inductance",
            "what is the inductance",
            "find the inductance",
        ),
        "inductance_from_inductor_energy",
    ),
    # Iter-18f (Day-29) — CH360: "At resonance with U = X V, R = Y Ω,
    # what is I?". At resonance Z = R, so I = U/R (Ohm's law). Without
    # this rule the symbol fallback picked power_at_resonance (U + R
    # match its inputs + "resonan" target word) and returned 400 W
    # instead of 4 A. Triggers are narrow ("what is i" + "?" anchor) so
    # power-asking resonance questions still route to power_at_resonance.
    (
        (
            "at resonance, what is i",
            "at resonance, find i",
            "at resonance, calculate i",
            "ω, what is i",
            "ohms, what is i",
            ", what is i?",
            # Iter-19d (Day-29) — CH160: "operating at resonance" +
            # "calculate the maximum effective current Imax". At
            # resonance Z = R, so I = U/R. Dim guard ensures power-asking
            # resonance questions still route to power_at_resonance.
            "calculate the maximum effective current",
            "calculate the maximum current",
            "find the maximum effective current",
            "calculate the rms current at resonance",
            "calculate imax",
            "find imax",
        ),
        "ohm_law_current",
    ),
    # Iter-17a (Day-29) — NL040: "An inductor has L=X H, current I=Y A.
    # Calculate the magnetic field energy." Direct W = 0.5·L·I². Must
    # precede the current_from_inductor_energy rule below which keys on
    # the same "magnetic energy" tokens. Trigger requires asking-for-
    # energy phrasing so the inverse problems (find current / find
    # inductance) still route correctly.
    (
        (
            "calculate the magnetic field energy",
            "calculate the magnetic energy",
            "what is the magnetic field energy",
            "find the magnetic field energy",
            "determine the magnetic field energy",
            "calculate w (",  # "calculate W (J)" style
        ),
        "inductor_magnetic_energy",
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
    # Iter-6: quality factor Q = (1/R)·sqrt(L/C). MUST precede the generic
    # resonance/series rules because the question phrasing ("calculate Q",
    # "quality factor") could otherwise be eaten by them.
    (
        (
            "quality factor",
            "calculate the quality factor",
            "calculate q",
            "what is the value of q",
            "determine q",
        ),
        "quality_factor_q",
    ),
    # Iter-6: capacitive reactance X_C = 1/(2*pi*f*C). DDT345-style.
    (
        (
            "capacitive reactance",
            "calculate z_c",
            "calculate the capacitive reactance",
        ),
        "capacitive_reactance",
    ),
    # Iter-6: natural period T = 2*pi*sqrt(L*C). DDT362-style. Distinct
    # from "natural frequency" (which maps to resonance_frequency).
    (
        (
            "natural period",
            "period of oscillation",
            "natural period of oscillation",
        ),
        "natural_period_lc",
    ),
    # Iter-6: total flux linkage Psi = N*Phi (DDT384). Specific phrase
    # so it doesn't conflict with single-turn magnetic_flux_solenoid_one_turn.
    (
        (
            "total flux linkage",
            "flux linkage",
            "calculate the total flux",
        ),
        "total_flux_linkage",
    ),
    # Iter-6: energy-loss percent in LC oscillation (NL092). Specific.
    (
        (
            "percentage loss",
            "percent loss",
            "percentage of energy loss",
            "% loss",
            "loss (%)",
        ),
        "energy_loss_percent",
    ),
    # CH169: at-resonance MAX power. Must precede generic resonance rules.
    (
        (
            "maximum power",
            "max power",
            "pmax",
            "p_max",
        ),
        "power_at_resonance_max",
    ),
    # CH365: voltage across L at resonance.
    (
        (
            "voltage across l",
            "calculate ul",
            "calculate u_l",
            "voltage across the inductor",
        ),
        "voltage_inductor_at_resonance",
    ),
    # DDT339: when Z is given (impedance) + power consumed asked, use the
    # impedance-based active-power formula instead of P=UI (which needs I).
    # MUST precede the generic "power consumed" rule that lands on power_voltage_current.
    (
        (
            "impedance z =",
            "an impedance z",
            "has an impedance",
        ),
        "power_from_voltage_impedance_r",
    ),
    (("resonance frequency", "resonant frequency", "natural frequency"), "resonance_frequency"),
    (("resistors in parallel", "connected in parallel", "in parallel"), "parallel_resistance_two"),
    # Tighten series rule: require explicit resistor wording. "in series"
    # alone over-matches inductor/capacitor combos that need RLC formulas.
    (
        ("resistors in series", "two resistors connected", "series resistors"),
        "series_resistance_two",
    ),
    # Iter-15c: DDT349-style "voltage U = X V and current I = Y A. Calculate
    # the total impedance Z" — direct Z = U/I, not the RLC form which needs
    # R, X_L, X_C. Trigger BEFORE rlc_impedance so the simpler formula wins
    # when the question only gives U and I.
    (
        (
            "voltage u = ", "current i = ", "rms voltage u =", "rms current i =",
        ),
        "impedance_from_voltage_current",
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
    # Iter-6 new formulas:
    "quality_factor_q": frozenset({"quality", "factor", "q"}),
    "capacitive_reactance": frozenset({"reactance", "z_c", "ohm"}),
    "natural_period_lc": frozenset({"period", "oscillation", "second"}),
    "total_flux_linkage": frozenset({"flux", "linkage", "weber"}),
    "energy_loss_percent": frozenset({"loss", "percent", "percentage"}),
    # Iter-7 Tier-1 new formulas:
    "charge_from_energy_voltage": frozenset({"charge", "coulomb", "mc"}),
    "power_at_resonance_max": frozenset({"power", "max", "watt"}),
    "power_from_voltage_impedance_r": frozenset({"power", "consumed", "watt"}),
    "voltage_inductor_at_resonance": frozenset({"voltage", "inductor", "ul"}),
    "lc_partition_electric_energy": frozenset({"electric", "energy", "joule"}),
    "electric_field_point_charge": frozenset({"intensity", "strength", "magnitude"}),
    # Iter-16 (Day-29) target words: require the question to actually be
    # asking for an electric field before letting the symbol fallback
    # commit. q + d/a/r symbols are otherwise too generic.
    "electric_field_two_opposite_sources_isoceles_apex": frozenset(
        {"electric field", "field intensity", "field strength", "v/m"}
    ),
    "electric_field_equilateral_three_identical": frozenset(
        {"electric field", "field intensity", "field strength", "v/m"}
    ),
    "electric_field_right_angle_two_identical": frozenset(
        {"electric field", "field intensity", "field strength", "v/m"}
    ),
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
    # Iter-9b: 4 force-output formulas that were silently bypassing the
    # field-vs-force guard. Safe to include now because the guard scans
    # only the LAST imperative sentence (via _extract_question_sentence),
    # so compound DT005/006-style questions whose ASK is "calculate the
    # electric force" correctly route to one of these force formulas
    # without being blocked by an earlier "field strength" setup phrase.
    "coulomb_force_two_opposite_sources_isoceles_apex",
    "coulomb_force_two_sources_right_triangle_apex",
    "coulomb_force_on_charge_between_two_identical",
    "coulomb_force_collinear_opposite_signs",
    # Iter-18e (Day-29): force-output mirror of Iter-16d field formula.
    "coulomb_force_right_angle_two_identical",
})
# Iter-4 Fix D — the inverse guard. Symmetric to _FORCE_FORMULAS so a
# "calculate the net force" question doesn't trip a field-output formula
# (V/m would be marked unit-wrong vs gold N).
_FIELD_FORMULAS: frozenset[str] = frozenset({
    "electric_field_point_charge",
    "electric_field_two_opposite_charges_midpoint",
    "electric_field_perp_bisector_two_opposite",
    "electric_field_from_force",
    # Iter-16 (Day-29) — three new vector-composition field formulas.
    # Each outputs V/m; the inverse guard rejects them on force-asked.
    "electric_field_two_opposite_sources_isoceles_apex",
    "electric_field_equilateral_three_identical",
    "electric_field_right_angle_two_identical",
    # Iter-17f (Day-29) — general-angle 2-source field. Must be in
    # _FIELD_FORMULAS so a force-asking "form an angle of 60°" question
    # is rejected at the keyword stage instead of stealing the route.
    "electric_field_two_sources_at_angle",
    # Iter-18c/d (Day-29) — dielectric midpoint + same-sign perp-bisector.
    "electric_field_two_opposite_charges_midpoint_dielectric",
    "electric_field_perp_bisector_two_same_sign",
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
    # Iter-17d (Day-29): LD067 phrases the ask as "electric field
    # vector produced by ..." — neither "field strength" nor "field at"
    # appears, so the F2 guard wasn't rejecting force formulas. Adding
    # "electric field vector" and "electric field produced" (with the
    # variant via "field e produced") closes that loophole.
    "electric field vector",
    "electric field e produced",
    "electric field produced by",
    "resultant electric field",
    "total electric field",
)


# Iter-9b: imperative verbs that mark the QUESTION sentence (where the
# actual ask lives). Used to isolate the ask from data/setup sentences.
_IMPERATIVE_VERBS: tuple[str, ...] = (
    "calculate", "find", "determine", "compute", "evaluate",
    "what is", "what are", "how much", "how many",
)

# Sentence splitter — match ". " / "? " / "! " boundaries.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_question_sentence(question_lower: str) -> str:
    """Return the LAST sentence of ``question_lower`` that starts an
    imperative ask (Calculate/Find/What is/...). Falls back to the full
    prompt when no imperative sentence is found.

    Iter-9b motivation: compound physics questions like DT005 mix SETUP
    sentences ("two charges placed at A and B... field strength caused at
    point C") with the ACTUAL ASK ("Calculate the electric force on
    q3 at C"). Substring matching against the whole prompt picks up the
    setup phrase and infers the wrong intent. Isolating the last
    imperative sentence gives much sharper signal.
    """
    sentences = _SENTENCE_SPLIT_RE.split(question_lower.strip())
    imperatives = [s for s in sentences if any(v in s for v in _IMPERATIVE_VERBS)]
    if imperatives:
        return imperatives[-1]
    return question_lower


# Iter-16a (Day-29): detect when the question describes two charges of
# opposite sign and equal magnitude (q1 = -q2 patterns). Used as a per-rule
# routing guard so an "equilateral triangle" / "AC = BC" question routes to
# the perp-bisector-special-case formula only when the charge signs justify
# it. Detection is text-only (no extractor coupling) so the classifier stays
# pure-function on question text.
_OPPOSITE_SIGN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "q1 = -q2" / "q_1 = -q_2" / "q2 = -q1" — chained-equality form
    re.compile(r"q_?1\s*=\s*[−\-]\s*q_?2"),
    re.compile(r"q_?2\s*=\s*[−\-]\s*q_?1"),
    # "q1 = +X ... q2 = -X" / "q1 = +X mC and q2 = -Y mC" — two-statement form
    re.compile(r"q_?2\s*=\s*[−\-]\s*\d"),
    re.compile(r"and\s+q_?2\s*=\s*[−\-]"),
    re.compile(r",\s*q_?2\s*=\s*[−\-]"),
    # Inverse word order: q1 negative, q2 positive
    re.compile(r"q_?1\s*=\s*[−\-]\s*\d"),
    re.compile(r"and\s+q_?1\s*=\s*[−\-]"),
)


def _has_opposite_sign_two_charges(question_lower: str) -> bool:
    return any(p.search(question_lower) for p in _OPPOSITE_SIGN_PATTERNS)


# Iter-18d (Day-29): same-sign detector — q1 = q2 chained equality, or
# two explicit identical positive charges. Used as a per-rule guard so
# the new same-sign perp-bisector formula fires only when the signs
# justify it (and the opposite-sign formula stays the default).
_SAME_SIGN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"q_?1\s*=\s*q_?2"),
    re.compile(r"q_?a\s*=\s*q_?b"),
    re.compile(r"qa\s+and\s+qb,\s*both\s+equal"),
    # "three identical charges"
    re.compile(r"three\s+identical\s+charges"),
    re.compile(r"3\s+identical\s+charges"),
    re.compile(r"both\s+equal\s+to"),
)


def _has_same_sign_two_charges(question_lower: str) -> bool:
    """Permissive same-sign detector: True when opposite-sign is NOT
    detected AND there is at least a 2-charge problem (q1 + q2 mentioned).
    Used as a per-rule guard alongside ``perpendicular bisector`` so the
    same-sign formula wins when signs aren't explicitly opposite.
    """
    if _has_opposite_sign_two_charges(question_lower):
        return False  # opposite-sign takes precedence
    # Explicit positive markers (3 identical charges, qa=qb, etc.)
    if any(p.search(question_lower) for p in _SAME_SIGN_PATTERNS):
        return True
    # Fallback: any pair of "q1 =" / "q2 =" tokens with NO negative signs
    # in front (caught by the opposite-sign check above).
    has_q1 = re.search(r"q_?1\s*=\s*\d", question_lower) is not None
    has_q2 = re.search(r"q_?2\s*=\s*\d", question_lower) is not None
    has_qa_qb = re.search(r"q_?a\s+and\s+q_?b", question_lower) is not None
    return (has_q1 and has_q2) or has_qa_qb


# Iter-18c (Day-29): dielectric-medium detector for the dielectric-aware
# midpoint field formula.
_DIELECTRIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"dielectric\s+constant"),
    re.compile(r"relative\s+permittivity"),
    re.compile(r"ε\s*=\s*\d"),
    re.compile(r"epsilon\s*=\s*\d"),
    re.compile(r"in\s+alcohol"),
    re.compile(r"in\s+oil"),
    re.compile(r"in\s+a\s+dielectric"),
)


def _has_dielectric_constant(question_lower: str) -> bool:
    return any(p.search(question_lower) for p in _DIELECTRIC_PATTERNS)


# Iter-19a (Day-29): the U_C-from-U_RC formula fires only when the
# question describes a series RLC AT RESONANCE plus a U_RC voltage.
def _has_rlc_resonance_with_urc(question_lower: str) -> bool:
    has_resonance = ("at resonance" in question_lower
                     or "in resonance" in question_lower
                     or "currently in resonance" in question_lower)
    has_urc_phrasing = ("r-c" in question_lower
                        or "rc combination" in question_lower
                        or "rc section" in question_lower
                        or "r and c" in question_lower)
    has_both_sections = "c-l" in question_lower or "cl section" in question_lower
    return has_resonance and (has_urc_phrasing or has_both_sections)


# Iter-19e (Day-29): the constant-cos(φ)=1 formula must only fire when
# the question actually asks for the power factor.
def _asks_power_factor(question_lower: str) -> bool:
    return ("power factor" in question_lower
            or "cos(φ)" in question_lower
            or "cosφ" in question_lower
            or "cos φ" in question_lower)


# Iter-19b (Day-29): same-sign isoceles general — must NOT fire on
# equilateral (Iter-16b handles that case better) and NOT on opposite-
# sign (Iter-16a handles that).
def _is_same_sign_isoceles_not_equilateral(question_lower: str) -> bool:
    if _has_opposite_sign_two_charges(question_lower):
        return False
    if "equilateral" in question_lower:
        return False
    return _has_same_sign_two_charges(question_lower)


# Iter-16a: per-rule guard table. A rule in _KEYWORD_RULES may name a guard
# string; the rule only fires when the named guard returns True. Keeps the
# rule table flat and the guard logic centralised.
_RULE_GUARDS: dict[str, "callable[[str], bool]"] = {
    "opposite_sign_two_charges": _has_opposite_sign_two_charges,
    "same_sign_two_charges": _has_same_sign_two_charges,
    "has_dielectric_constant": _has_dielectric_constant,
    "rlc_resonance_with_urc": _has_rlc_resonance_with_urc,
    "asks_power_factor": _asks_power_factor,
    "same_sign_isoceles_not_equilateral": _is_same_sign_isoceles_not_equilateral,
}


def _is_field_asking(question_lower: str) -> bool:
    # Iter-9b: scan only the LAST imperative sentence so a setup phrase
    # like "the field strength at M" in an earlier sentence doesn't
    # falsely trigger this on a force-asking question.
    ask = _extract_question_sentence(question_lower)
    return any(tok in ask for tok in _FIELD_ASK_TOKENS)


def _is_force_asking(question_lower: str) -> bool:
    ask = _extract_question_sentence(question_lower)
    return any(tok in ask for tok in _FORCE_ASK_TOKENS)


# ---------------------------------------------------------------------------
# Iter-10: Target-Unit Guard v2 (the user's "nhìn đơn vị để biết mình có đi
# sai đường không" check, finally done correctly).
#
# Pre-Iter-8 attempt failed because it scanned the WHOLE prompt with first-
# match. Setup phrases like "voltage across it is 150 V" wrongly triggered
# the volt-intent hint and blocked legitimate capacitance/energy formulas.
# This v2 fixes both problems:
#
#   1. Scan only the LAST imperative sentence via _extract_question_sentence
#      (so data-mentions in setup are out of scope).
#   2. Hints must be VERB-PREFIXED ("calculate the X", "find the X", etc.)
#      — bare noun phrases stay out.
#   3. Longest-match wins (so "calculate the electric field energy" beats
#      its prefix "calculate the electric field" for joule vs V/m).
#
# When the guard infers a target dimension, any formula whose declared
# output_unit doesn't share that dimension is REJECTED at routing time
# (in classify) and at fallback time (in _symbol_match), so the dim-
# wrong formula can't silently win.
# ---------------------------------------------------------------------------

_INTENT_DIMENSION_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("calculate the electric field", "find the electric field",
      "what is the electric field", "determine the electric field",
      "magnitude of the electric field",
      "resultant electric field",
      "electric field at point", "field intensity",
      "field strength at"), "volt/meter"),
    (("calculate the force", "find the force", "what is the force",
      "determine the force",
      "calculate the net force", "find the net force",
      "calculate the electric force", "find the electric force",
      "calculate the electrostatic force",
      "magnitude of the force",
      "force acting on", "force on the", "force exerted on"),
     "newton"),
    (("calculate the voltage", "find the voltage",
      "what is the voltage", "determine the voltage",
      "calculate ul", "calculate u_l",
      "calculate the potential difference",
      "what is the potential difference",
      "find the potential difference",
      "calculate the rms voltage", "find the rms voltage"), "volt"),
    (("calculate the charge", "what is the charge", "find the charge",
      "determine the charge",
      "the charge on the capacitor", "the charge (mc)"), "coulomb"),
    (("calculate the capacitance", "what is the capacitance",
      "find the capacitance", "determine the capacitance",
      "calculate its capacitance"), "farad"),
    (("calculate the energy", "what is the energy",
      "find the energy", "determine the energy",
      "calculate the electric field energy",
      "what is the electric field energy",
      "find the electric field energy",
      "calculate the stored energy",
      "energy stored in",
      # Iter-17a (Day-29): must beat the shorter "calculate the magnetic
      # field" -> tesla hint via longest-match. Without these phrases the
      # dim guard rejected inductor_magnetic_energy for NL040 even though
      # the keyword router selected it correctly.
      "calculate the magnetic field energy",
      "what is the magnetic field energy",
      "find the magnetic field energy",
      "determine the magnetic field energy",
      "calculate the magnetic energy",
      "what is the magnetic energy"), "joule"),
    (("calculate the current", "find the current",
      "what is the current", "determine the current",
      "calculate the rms current"), "ampere"),
    (("calculate the resistance",
      "what is the resistance", "find the resistance",
      "determine the resistance",
      "equivalent resistance"), "ohm"),
    (("calculate the impedance", "what is the impedance",
      "find the impedance", "determine the impedance",
      "capacitive reactance", "calculate z_c",
      "calculate the capacitive reactance"), "ohm"),
    (("resonance frequency", "resonant frequency", "natural frequency",
      "frequency of resonance"), "hertz"),
    (("natural period", "period of oscillation"), "second"),
    (("magnetic flux through", "magnetic flux linkage",
      "total flux linkage", "calculate the total flux"), "weber"),
    (("calculate the magnetic field", "find the magnetic field",
      "magnetic field inside", "magnetic flux density"),
     "tesla"),
    (("calculate the inductance", "what is the inductance",
      "find the inductance"), "henry"),
    (("quality factor", "calculate the quality factor",
      "what is the quality factor", "find the quality factor",
      "calculate the power factor", "what is the power factor",
      "find the power factor", "determine the power factor",
      "power factor", "cos(phi)", "cosφ"), "dimensionless"),
    (("percentage loss", "percent loss", "% loss"), "dimensionless"),
    (("calculate the maximum power", "calculate the power",
      "find the power", "what is the power",
      "power consumed", "power dissipated", "active power"), "watt"),
)


def _infer_expected_output_dim(question_lower: str):
    """Iter-10: infer the pint dimensionality the question is asking for.

    Scans ONLY the last imperative sentence (returned by
    :func:`_extract_question_sentence`) and picks the LONGEST matching
    hint phrase. Returns ``None`` when no hint matches — caller must NOT
    reject any formula in that case (insufficient signal).
    """
    ask = _extract_question_sentence(question_lower)
    ureg = get_registry()
    best_len, best_unit = 0, None
    for phrases, target_unit in _INTENT_DIMENSION_HINTS:
        for phrase in phrases:
            if phrase in ask and len(phrase) > best_len:
                best_len = len(phrase)
                best_unit = target_unit
    if best_unit is None:
        return None
    try:
        return ureg.parse_expression(best_unit).dimensionality
    except Exception:
        return None


def _formula_output_compatible(formula: Formula, expected_dim) -> bool:
    """True if ``formula``'s output unit shares ``expected_dim``."""
    if expected_dim is None:
        return True  # no intent inferred -> don't reject anything
    ureg = get_registry()
    try:
        formula_dim = ureg.parse_expression(formula.output_unit).dimensionality
    except Exception:
        return True  # lenient if formula unit doesn't parse
    return bool(formula_dim == expected_dim)


def _keyword_match(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    field_asked = _is_field_asking(lower)
    force_asked = _is_force_asking(lower)
    for rule in _KEYWORD_RULES:
        # Iter-16a: support optional per-rule guard as a 3rd tuple element.
        if len(rule) == 3:
            keywords, formula_id, guard_name = rule
        else:
            keywords, formula_id = rule
            guard_name = None
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
                # Iter-16a: per-rule guard (e.g. opposite-sign check).
                if guard_name is not None:
                    guard_fn = _RULE_GUARDS.get(guard_name)
                    if guard_fn is None or not guard_fn(lower):
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
    # Iter-10: target-unit guard at the fallback level too.
    expected_dim = _infer_expected_output_dim(question_lower)

    best: tuple[Formula, float, str] | None = None
    for formula in library.all():
        if field_asked and formula.id in _FORCE_FORMULAS:
            continue
        if force_asked and formula.id in _FIELD_FORMULAS:
            continue
        # Iter-10: reject any formula whose declared output unit can't
        # reach the dimension the ask sentence wants.
        if not _formula_output_compatible(formula, expected_dim):
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
    """Return the best-matching formula id, or None if we cannot decide.

    Iter-10: when the target-unit guard fires (a verb-prefixed intent
    phrase was matched in the ask sentence), a keyword hit whose declared
    output unit conflicts with the inferred target dimension is dropped
    and routing falls through to the dimension-filtered symbol fallback.
    """
    question_lower = question.lower()
    expected_dim = _infer_expected_output_dim(question_lower)

    keyword = _keyword_match(question)
    if keyword is not None:
        formula_id, reason = keyword
        if formula_id in library:
            kw_formula = library[formula_id]
            if _formula_output_compatible(kw_formula, expected_dim):
                return ClassificationResult(
                    formula_id=formula_id, confidence=0.9, reason=reason,
                )
            # Keyword matched but output dim incompatible with inferred
            # target — fall through to dim-filtered symbol fallback.

    fallback = _symbol_match(library, quantities, question_lower)
    if fallback is not None:
        formula, score, reason = fallback
        return ClassificationResult(formula_id=formula.id, confidence=0.5 * score, reason=reason)

    return None
