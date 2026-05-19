# 0010 — Day-10 resultant-of-two-forces formula

Date: 2026-05-15
Status: Accepted

## Context

ADR 0009's corrected scorer made the remaining physics gaps cleanly
attributable. Categorizing the 46 LD+DT eval rows by sub-type:

| sub-type | count | tractable? |
|---|---:|---|
| `coulomb_resultant_geom` (force/field on a 3rd charge, general geometry) | 18 | needs geometry extraction — hard |
| `coulomb_symmetric` (3rd charge at midpoint / perpendicular bisector) | 12 | needs config + sign logic — medium-hard |
| `other/single` (single pair / collinear attraction) | 11 | mixed |
| `two_forces_angle` (resultant of two GIVEN forces at an angle) | 5 | **closed form — easy** |

The `two_forces_angle` group is the one clean, zero-ambiguity win: the
problem hands you F1, F2, and the angle directly; no charges, no
geometry to infer. `R = √(F1² + F2² + 2·F1·F2·cosθ)`.

## Decision

Add exactly one formula — `resultant_two_forces` — and stop there for
Day 10. Deliberately **do not** hand-code the 30 geometry-dependent
coulomb cases:

* They require extracting positions/triangle geometry and a
  configuration (collinear vs triangle vs perpendicular bisector) plus
  per-case sign handling.
* A local 3B model extracting that reliably is not realistic.
* ADR 0003 already classed vector composition as Phase-2. The honest
  architectural call is to let the **SFT'd model do the multi-step CoT**
  and have the solver verify, rather than grow a brittle geometry
  sub-engine. Revisit after the rented-GPU SFT.

## Implementation

* `configs/physics_formulas.yaml`: `resultant_two_forces` with inputs
  `F1`/`F2` (newton), `theta` (degree); expression
  `sqrt(F1**2 + F2**2 + 2*F1*F2*cos(theta*pi/180))`. Degrees handled in
  the expression (SymPy `cos` is radians); the unit converter passes the
  degree value through unchanged (verified `convert(60,'degree','degree')`).
* `topic_classifier.py`: a keyword rule placed **before** the coulomb
  rules ("act at an angle", "at an angle of", "two forces with
  magnitudes", …) so resultant problems out-rank `force acting on`.
  Target-word guard `{resultant, angle}` for the symbol fallback.
* Tests: integration `test_resultant_two_forces` (LD015 → 15.13 N) plus
  the existing formula-library coverage.

## Effect on the measured numbers

Same holdout, same Ollama `qwen2.5:3b-instruct`, same Day-9 scorer.

| | rule | Day-9 LLM | Day-10 LLM |
|---|---:|---:|---:|
| **Physics Full✓ overall** | 0.8% | 8.3% | **12.0%** |
| `resultant_two_forces` | — | — | **5/5 = 100%** |
| LD prefix numeric | 0% | 0% | **12.5%** |
| LD solved | 0% | 67.5% | 77.5% |
| `no_formula_matched` | — | 45 | 40 |

No regressions: TD 23.1%, NL 18.5%, `charge_from_capacitance` 60%,
`rlc_impedance` 100% all held. All 5 resultant cases verified exactly
correct across angles 30°/60°/135° (LD015/132/168/189/199) — not a
scorer artefact.

Physics Full-correct trajectory: **0.8% → 3.8% → 8.3% → 12.0%** (15×
the rule baseline; solver + scorer only, no training yet).

## What's still open (unchanged priority, now sharper)

* `coulomb_force` 13 + `electric_field_point_charge` 21 — 100%/76%
  solved, 0% numeric. Multi-charge vector composition with geometry.
  **Deferred to post-SFT CoT** per the decision above.
* `no_formula_matched` 40 — THCB (6, measurement error, no formula),
  some DDT magnetic-field variants, classifier misses.
* `magnetic_field_solenoid` 7 — 0% solved (extraction/formula gap).

## Reproduction

```powershell
uv run pytest -q tests/integration/test_physics_solver.py   # incl. LD015
uv run python scripts/run_eval.py --task physics --with-llm --out outputs/eval/day10_llm
```
