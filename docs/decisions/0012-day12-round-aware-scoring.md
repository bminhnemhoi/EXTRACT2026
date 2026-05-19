# 0012 — Day-12 round-aware (stated-precision) scoring

Date: 2026-05-15
Status: Accepted
Refines: ADR 0009 (unit-aware), ADR 0011 (dimensionless)

## Context

ADR 0011 flagged a recurring miss: the dataset frequently says
"round the result to two decimal places" and ships the *rounded* value
as gold. The solver returns the exact value, so e.g.
`√(2·0.54e-3/0.12) = 0.094868 A` is judged wrong against gold `"0.09"`
(5.4 % off → fails the 1 % relative tolerance) even though it is the
same answer at the gold's stated precision.

## Decision

`quantity_match` now accepts on **either**:

1. relative tolerance (`math.isclose`, rel 1 %) — unchanged; or
2. **stated-precision**: if the gold is a clean fixed-point decimal
   with ≥1 decimal place, accept when
   `round(pred, d) == round(gold, d)` where `d` = the gold's decimal
   count.

Guards that keep it conservative:

* Round-aware only fires for `^[-+]?\d+\.\d+$` golds. Integers
  (`"40"`, `"3"`) and scientific/messy golds (`"4 × 10^-6"`,
  `"0.3; 1.5 cm"`) use relative tolerance only — no loosening there.
* It runs *after* unit conversion, so it operates in the gold's own
  unit/scale (the ADR 0009 / 0011 machinery is unchanged; this only
  adds a second acceptance path on the final compare).

Rationale: a gold quoted to 2 dp encodes the true value only to
±0.005. A prediction that rounds to the same 2-dp value is
indistinguishable from gold at the precision the dataset itself chose
— that is exactly how a human grader marks these.

## Effect on the measured numbers

Same 133-row holdout, qwen2.5:3b via Ollama; only the scorer changed
vs Day-11.

| | Day-11 | **Day-12** |
|---|---:|---:|
| **Physics Full✓** | 22.6% | **24.8%** |
| NL prefix Full✓ | 18.5% | 29.6% |
| full-correct rows | 30 | 33 |

Sample audit (day11_final → day12): **0 regressions**, 3 gains
(NL013, NL015, NL018 — all explicit "round to 2 dp" golds the solver
had right to full precision). Numeric✓ == Full✓ (24.8 %).

Physics Full trajectory: **0.8 → 3.8 → 8.3 → 12.0 → 22.6 → 24.8 %**
(≈31× the rule baseline; solver + scorer only, still no training).

## Why this is safe (not metric-gaming)

Verified the guard rejects genuine errors: `quantity_match(0.08,
"coulomb", "26.55", "nC")` → False (TD397 — wrong extraction stays
wrong); `quantity_match(41.0, "ohm", "40", "ohm")` → False (integer
gold not loosened); `quantity_match(4.4e-6, "newton", "4 × 10^-6",
"N")` → False (scientific gold, rel-tol only). 5 new unit tests pin
these.

## Scorer change log (one concern per commit)

* ADR 0009 — unit-aware (`quantity_match`, SI vs prefixed).
* ADR 0011 — dimensionless unit-string equivalence (`unit_match`).
* ADR 0012 — stated-precision round-aware acceptance (this).

The scorer is now stable; further movement should come from the
solver, classifier, or the (pending) SFT model — not the metric.

## Test coverage

`tests/unit/test_metrics.py::TestQuantityMatch` +5 cases (NL015
recovery, raw-path 2dp, genuinely-wrong rejected, integer-gold not
loosened, scientific-gold rel-tol-only). Suite: 216 passing (was 211),
ruff + mypy clean. Reference checkpoint: `outputs/eval/day12_llm/`.
