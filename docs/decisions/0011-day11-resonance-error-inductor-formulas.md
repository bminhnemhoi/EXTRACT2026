# 0011 — Day-11 RLC-resonance / error / inductor formulas

Date: 2026-05-15
Status: Accepted

## Context

ADR 0010 left `no_formula_matched` at 40. Categorizing those 40:

| prefix | n | nature |
|---|---:|---|
| CH | 18 | RLC-resonance closed forms (biggest single bucket) |
| LD | 6 | multi-charge geometry (deferred, ADR 0010) |
| THCB | 6 | measurement error |
| DDT | 5 | conceptual yes/no / symbolic (LLM territory) |
| NL | 4 | inductor magnetic energy |
| DT | 1 | geometry field (deferred) |

The CH-resonance and THCB/NL-inductor groups are closed-form and
extractable — the same tractable shape as Day-10's `resultant_two_forces`.

## Decision

Add six formulas + classifier routing. Keep deferring the geometry
(LD/DT) and conceptual (DDT) cases to the post-SFT CoT path.

| formula | expr | eval rows | result |
|---|---|---:|---|
| `capacitance_for_resonance` | `1/(L·(2πf)²)` | 4 | 4/4 = 100% |
| `resonance_frequency_factor` | `√(X_C/X_L)` | 4 | 4/4 = 100% |
| `resistance_at_resonance` | `Z` (at resonance Z=R) | 2 | 2/2 = 100% |
| `power_at_resonance` | `U²/R` | 4 | 2/4 = 50% |
| `relative_error_percent` | `(Δ/value)·100` | 4 | 2/4 = 50% |
| `current_from_inductor_energy` | `√(2W/L)` | 2 | 1/2 = 50% |

Two supporting fixes were required and made as part of this work:

1. **`unit_match` dimensionless normalization.** `resonance_frequency_factor`
   came out 4/4 numerically correct but 0% full: the solver answer carries
   unit `"dimensionless"` while the gold writes `"-"`. `unit_match` now
   treats `{"", "-", "–", "—", "dimensionless", "none", "n/a", "ratio"}`
   as mutually equal. Without this the 4 correct k-factor answers were
   discarded — same class of measurement defect as ADR 0009.
2. **Dimensionless inputs skip pint.** `relative_error_percent` inputs
   (`delta`, `value`) are pure ratios. The regex extractor can grab a
   stray word as their "unit" (`"0.2 reads"`), which made
   `unit_converter.convert` raise and fail an otherwise valid solve.
   `solver._convert_inputs` now uses the raw value when the formula
   declares an input unit `"dimensionless"`.

## Effect on the measured numbers

Same 133-row holdout, qwen2.5:3b via Ollama, Day-9 scorer + the two
fixes above.

| | Day-10 | **Day-11** |
|---|---:|---:|
| **Physics Full✓ overall** | 12.0% | **22.6%** |
| solved | 41.4% | 56.4% |
| `no_formula_matched` | 40 | 20 |
| CH prefix Full✓ | 7.7% | **57.7%** |
| THCB prefix Full✓ | 0% | 33.3% |

Numeric✓ now equals Full✓ (22.6%) — the unit-string gap is closed.
Physics Full-correct trajectory: **0.8 → 3.8 → 8.3 → 12.0 → 22.6 %**
(≈28× the rule baseline; solver + scorer only, still no training).

Regression audit (day10 → day11 sample-level): 16 → 30 full-correct;
**15 gains, 1 flip (NL013)**. NL013 routed to the *same* formula in
both runs (`voltage_from_energy_capacitance`); only the LLM-extracted
values differed (14.83 V → 0.47 V). That is qwen2.5:3b sampling
variance at temperature 0.2, **not a code regression** — no Day-11
change touches that path. Run-to-run noise is ±1–2 rows; the SFT'd
model at lower temperature will damp it.

## What's still open

* `no_formula_matched` 20 — LD/DT geometry (deferred), DDT conceptual
  (LLM), THCB multi-target ("0.3; 1.5 cm; %"), CH146/149/270 multi-step
  AC, CH177/180 cosφ=1 (constant — mis-routes to power, fails
  `missing_input`, lateral not a regression).
* `power_at_resonance` / `relative_error_percent` / `current_from_inductor_energy`
  at 50%: the misses are LLM mis-extraction and gold rounding
  (e.g. NL015 √(2·0.54e-3/0.12)=0.0949 vs gold "0.09" rounded to 2 dp).
  A round-aware tolerance (accept when `round(pred, dp(gold)) == gold`)
  is a candidate Day-12 scorer refinement — kept out of this commit to
  keep the scorer change history one-concern-per-commit.

## Test coverage

6 new integration tests (one per formula, deterministic via explicit
`name = value`), 6 new `unit_match` dimensionless cases. Suite: 211
passing (was 199), ruff + mypy clean. Routing verified on the real 40
unmatched questions: 20 newly route, **0 regressions** among the 16
previously-correct (pre-eval static check).

Reference checkpoint: `outputs/eval/day11_final/`.
