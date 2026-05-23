# ADR 0039 — Iter-17 (Day-29): 4 audit-driven small-cluster fixes

## Status
Accepted — 2026-05-23

## Context

After Iter-16e (Physics 53.4%), the remaining 76 wrong rows no longer
have a single dominant failure cluster. Triage:

```
26 numeric_mismatch   (spread; max 4 per formula)
26 missing_input      (5 capacitor_energy, 4 impedance, ...)
19 no_formula_matched (5 LD, 4 NL, 4 CH, 3 THCB, 2 TD, 1 DDT)
 4 llm_recovery_failed
 1 unit_conversion_failed
```

Four small tractable clusters found among the `no_formula_matched`
and `missing_input` buckets:

| Cluster | Rows | Pattern |
|---|---|---|
| 17a NL040 / NL387 | 2 | "inductor has L, I — calculate magnetic field energy" — W = 0.5·L·I² with no existing formula |
| 17c NL103 | 1 | "energy and capacitance given — find voltage across plates" — existing voltage_from_energy_capacitance, missing trigger |
| 17d LD067 | 1 | "electric field vector ... at the midpoint of the line segment" — existing midpoint formula, but trigger required adjacent "field at midpoint" substring |
| 17f LD362 / LD367 | 2 | Two source charges of DIFFERENT magnitudes, equidistant from M, source fields at given angle θ — no formula |

Two clusters that surfaced during triage were intentionally **deferred**:

* **17b NL361** (capacitor_energy on "what is the electric field energy"
  with U = 100√2 V): routing fix is one line, but extractor cannot
  parse `100√2 V` to 141.42, so the formula would fail at compute
  time. Two-step dependency; defer until extractor handles √-prefixed
  literals.
* **17e THCB094/114/133** (3 rows, mean of three measurements): single
  formula `(x1+x2+x3)/3` is trivial, but the gold rows carry **per-row
  units** (g, V, °C) that a dimensionless formula cannot satisfy in
  `unit_match`. Requires a solver-side input-unit-passthrough
  mechanism (the formula's output unit becomes one of its inputs'
  units at runtime) — out of scope for a single-iter batch.

## Decision

Ship the 4 in-scope fixes as a single Iter-17 batch:

### 17a — `inductor_magnetic_energy` (new formula)
`W = 0.5·L·I²`. Routing on "calculate the magnetic field energy" /
"what is the magnetic field energy" placed ABOVE the existing
`current_from_inductor_energy` rule (which keys on the same "magnetic
energy" tokens) so inverse problems remain correctly routed.

**Dim-hint bug fix discovered during implementation**: the existing
`_INTENT_DIMENSION_HINTS` mapped "calculate the magnetic field"
(longest-match=27) to `tesla`, beating any inferred energy intent.
Added "calculate the magnetic field energy" (longest-match=34) to the
joule bucket so the dim guard no longer rejects 17a's keyword match.
This bug had silently affected any future inductor-energy work — caught
only because Iter-17 was the first to add a formula in that ask space.

### 17c — `voltage_from_energy_capacitance` (existing formula, routing fix)
Added narrow triggers "voltage across its plates" / "voltage across the
plates" / "calculate the voltage across its plates" to the existing
rule. Output unit (volt) keeps the dim guard's energy/volt separation
clean — an energy-asking question still skips this rule.

### 17d — `electric_field_two_opposite_charges_midpoint` (existing formula, routing fix)
Added triggers "at the midpoint of the line segment" / "at the midpoint
of the segment" so LD067's "electric field vector produced by ... at the
midpoint of the line segment" matches. Numeric path for LD067 itself is
still blocked: the question specifies a dielectric ε=2.2 and the formula
doesn't apply the 1/ε scaling. A dielectric-aware variant
(`electric_field_two_opposite_charges_midpoint_dielectric`) is the next
step; not shipped in this batch to keep attribution clean.

Side effect: `_FIELD_ASK_TOKENS` extended with "electric field vector",
"electric field produced by", "resultant electric field", "total
electric field" so the F2 guard correctly identifies LD067-style asks
as field questions. This closes a latent loophole where the existing
`coulomb_force_at_midpoint` rule (newton output) was stealing routes
that the dim guard then rejected to None.

### 17f — `electric_field_two_sources_at_angle` (new formula)
Law-of-cosines vector sum for two equidistant source charges with the
angle between their source-fields given:

```
E = sqrt((kq1/r²)² + (kq2/r²)² + 2·(kq1/r²)·(kq2/r²)·cos(θ·π/180))
```

Distinct from the Iter-16d `right_angle_two_identical` formula because
it (a) requires equal magnitudes and (b) is restricted to the
3-charges-on-triangle configuration. Iter-17f handles arbitrary
magnitudes at arbitrary angles for the 2-source equidistant case.

Triggers cover both LD362 ("perpendicular to each other") and LD367
("form an angle of 60°"). Added to `_FIELD_FORMULAS` so the F2 inverse
guard rejects it on force-asked questions (e.g., "two forces F1, F2
at an angle of 60°" stays on `resultant_two_forces`); regression test
`test_17f_force_asked_does_not_steal_field_formula` locks this.

## Consequences

* **Tests**: +11 (3 routing, 4 numeric, 2 regression, 2 inverse-problem
  protection); total 301 → 312.
* **Eval (163-row SFT-unseen, LLM-enabled)**:
  Physics **53.4% → 56.4% (+3.0pp, +5 rows, 0 regressions)**.
  Gained: NL040, NL103, LD362, LD367 (4 targets) + NL387 (bonus from
  17a inductor formula). Session cumulative (Iter-15 → Iter-17):
  **47.2% → 56.4% (+9.2pp, +15 rows, 4 commits)**.
* **Risk surface**: the new "electric field vector" token in
  `_FIELD_ASK_TOKENS` is broader than previous tokens. Mitigated by the
  ask-sentence-scoped scan (Iter-9b) — only the LAST imperative
  sentence is inspected, so setup-phrase mentions don't trip it.
* **Deferred (with reason)**:
  * 17b (NL361) — needs extractor support for `100√2 V` literal
  * 17e (THCB) — needs solver-side input-unit passthrough
  * LD067 numeric — needs dielectric-aware midpoint formula
  * LD034 / LD054 — 3-charge force at a point with implicit geometry;
    no closed-form fits without geometry extraction
  * 4 CH RLC compound-segment questions — multi-step, beyond
    single-formula scope

## Methodology note

Audit → 4 tractable clusters → ship 4 minimal fixes (3 formulas + 1
routing-only) → dim/F2 guards updated as discovered → eval → row-diff.
Same playbook as Iter-15/16/16e. Per-fix commit structure (YAML →
classifier → tests+ADR) keeps each row's gain attributable to a single
edit and revertable in isolation.
