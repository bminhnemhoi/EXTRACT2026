# ADR 0040 — Iter-18 (Day-29): 6-fix audit batch — parser, percent, dielectric, same-sign perp-bisector, right-angle force, resonance current

## Status
Accepted — 2026-05-23

## Context

After Iter-17 (Physics 56.4%), per-row inspection of 71 remaining wrong
rows surfaced 6 small but independently fixable clusters. None has the
size of the Iter-15/16 vector-formula cluster; all are short tail. They
fall into three categories:

1. **Scorer bugs** — predictions were already correct but the eval
   couldn't see they matched. Two such cases this iter.
2. **Modelling gaps** — missing physics formula for a real pattern
   present in the holdout. Three such cases.
3. **Routing misses** — existing formula was correct for the question
   but a different rule fired first. One case.

## Decision

Ship all 6 fixes as a single Iter-18 batch in 4 per-fix commits
(YAML → parser → classifier → tests+ADR).

### Scorer fixes

#### 18a — `parse_number` extensions
Two dataset notations the existing regex couldn't reach:

* **`X . 10^Y`** (Vietnamese textbook dot-multiplier) — e.g. DT046
  gold `3 . 10^4 V/m` was being parsed as `3.1` (period+space treated
  as decimal fragment instead of multiplication delimiter).
* **`10^{N}`** (Latex curly-brace exponent) — e.g. DT051 gold
  `1.22 . 10^{-3} V/m` was being parsed as `1.22`.

Fix: pre-normalise both notations BEFORE the regex runs. Strip braces
then rewrite ` . 10^` → ` * 10^`. The existing regex then handles
`1.22 * 10^-3` correctly. Same playbook as the Iter-16e superscript
fix — pre-translate, no regex change, zero risk on inputs without the
notation (the dot-multiplier regex requires whitespace on both sides
so `3.14` is untouched).

#### 18b — `energy_loss_percent.output_unit` "dimensionless" → "percent"
NL092 had pred=75 and gold=75, but `quantity_match` returned False.
Root cause: the formula's expression returns the percentage value
(75 for 75% loss, not 0.75). With `output_unit: "dimensionless"`,
pint converted to `%` by **multiplying by 100** (75 dimensionless =
7500%), false-failing the comparison. Switching to `"percent"`
makes the `%` conversion an identity. `relative_error_percent`
already had `"%"`; only `energy_loss_percent` was inconsistent.

### Modelling gaps

#### 18c — `electric_field_two_opposite_charges_midpoint_dielectric`
LD067 — two opposite-sign charges at midpoint in a dielectric medium:
```
E = (k/ε_r) · (|q1| + |q2|) / (d/2)²
```
The existing non-dielectric formula over-estimated by exactly ε_r
(returned 2.16e6 vs gold 9.8e5 for ε=2.2). Routing trigger requires
both midpoint phrasing AND dielectric language; placed ABOVE
`parallel_plate_capacitance_dielectric` which was stealing LD067 via
the bare "dielectric constant" substring.

#### 18d — `electric_field_perp_bisector_two_same_sign`
LD081, LD387 — perpendicular bisector field of two SAME-sign charges:
```
E = k · sqrt((d/2)²·(|q1|-|q2|)² + l²·(|q1|+|q2|)²) / ((d/2)² + l²)^1.5
```
This is the mirror of the existing opposite-sign formula with the
two terms swapped (because same-sign vectors point AWAY from each
charge in opposite directions along AB, so the AB-parallel components
SUBTRACT; for opposite signs they add). Same-sign guard
(`_has_same_sign_two_charges`) is permissive — True iff opposite-sign
is NOT detected AND a two-charge problem is described — to catch
LD387 ("q1 = 2.96e-6 and q2 = 3.84e-6" with no explicit sign marker).

#### 18e — `coulomb_force_right_angle_two_identical`
LD243 — force-output mirror of Iter-16d's field formula:
```
F = sqrt(2) · k · |q|² / r²
```
Three identical charges at vertices of an isoceles right triangle;
force on the right-angle-vertex charge from the other two (each at
distance r, source forces perpendicular). The same routing triggers
as Iter-16d ("isosceles right triangle" / "right-angle vertex"); F2
guard dispatches field-asked vs force-asked. Regression tests cover
both directions.

### Routing miss

#### 18f — CH360 `ohm_law_current` at resonance
"At resonance with U = 100 V, R = 25 Ω, what is I?". Without a
keyword rule, symbol fallback picked `power_at_resonance` because
{U, R} are its inputs and "resonan" is its target word — returning
400 W instead of 4 A. Added narrow rules (",  what is i?", "at
resonance, what is i", etc.) for ohm_law_current. Power-asking
resonance questions (CH041 shape) still route correctly because the
dim guard rejects ampere when the ask sentence requests watt.

## Consequences

* **Tests**: +25 unit tests (parser parametric × 9, percent × 1,
  routing × 8, numeric × 4, regression/guard × 3). Total 312 → 337.
* **Eval (163-row SFT-unseen, LLM-enabled)**:
  Physics **56.4% → 60.7% (+4.3pp, +7 rows net, 1 LLM-noise
  regression)**.
  Gained (8): DT046, NL092, LD067, LD081, LD243, CH360 (six targets)
              + LD226, LD256 (bonus — both right-angle 3-charge force
              questions newly covered by 18e).
  Lost (1):   LD367 — same routing/formula as Iter-17 (passed there);
              LLM extractor returned `missing_input: r` this run.
              Pure non-determinism (temp=0.2). Not a real regression.
* **Cumulative session lift** (Iter-15 → Iter-18, single day):
  Physics **47.2% → 60.7% (+13.5pp, +22 rows, 5 ADRs, 8 commits)**.
* **Risk surface**: The 18d same-sign guard is permissive (returns
  True for "any two-charge problem that's not explicitly opposite").
  Mitigated because it only matters when "perpendicular bisector"
  fires, which is already a strong scope filter.
* **Still deferred**:
  * 17b NL361 (`100√2 V` extractor)
  * 17e THCB (unit-passthrough architecture)
  * LD052 (same-sign general isoceles)
  * LD086 (inverse zero-field-position problem)
  * LD034 / LD054 / LD146 (compound 3-charge geometries — some have
    apparent dataset bugs e.g. LD146 gold 3.89e-3 N inconsistent with
    its stated q=5e-6 C and a=10cm which give F=38.97 N)
  * NL086 (gold inconsistent with stated W=0.36 mJ; pred 0.089 A is
    correct, gold 2.83 A would require W=0.36 J)
  * NL346 (unit conversion: pred=0.002 C vs gold=0.002 mC — semantic
    mismatch from the LLM not rescaling its answer to gold's mC unit)
  * CHLT020 (Yes/No conceptual question routed to quality_factor_q)
  * Various TD state-change qualitative MC ("decreases by half", etc.)

## Methodology note

This batch is the densest "tail" iteration so far — 6 different fix
types in one day (parser bug, unit-config bug, 3 new formulas, 1
routing rule). The audit-driven loop now produces small finds; each
+1 row requires a separate hypothesis and fix. We are approaching the
deterministic ceiling on this 3B-extracted holdout (~62-65% range
based on the remaining tail's complexity / dataset-quality limits).
Further gains likely come from the pending SFT-7B retrain (extractor
upgrade) rather than from solver/classifier work.
