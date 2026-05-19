# 0009 — Day-9 unit-aware numeric scoring

Date: 2026-05-15
Status: Accepted
Supersedes: the numeric-comparison clause of ADR 0003

## Context

ADR 0008 documented, with evidence, that `eval/metrics.py::numeric_match`
compared the solver's **SI float** against the gold's **prefixed
string** without unit normalization. The physics solver canonicalizes
everything to SI before computing, so its answers come out as e.g.
`5.987e-10` coulomb while the gold is `0.6` nC — physically identical,
scored wrong. ADR 0008 deferred the fix as its own commit because
ADR 0003 had frozen the harness as the measurement source of truth.

This is that commit.

## Decision

Add `quantity_match(pred_value, pred_unit, exp_value, exp_unit, …)` to
`eval/metrics.py` and switch the physics eval call site
(`eval_physics._score_sample`) to use it.

Semantics:

* If both sides carry a unit and the units are dimensionally
  convertible (via the existing `physics.unit_converter.convert`), the
  predicted magnitude is converted into the **gold's** unit, then
  compared with `math.isclose(rel_tol=0.01)`.
* If a unit is missing or the conversion fails, fall back to the legacy
  raw comparison (`numeric_match` semantics). The metric therefore
  never regresses on unit-free answers.
* `numeric_match` is kept unchanged for backward compatibility and any
  unit-free numeric check.

Side benefit: comparing in the gold's natural unit (magnitude ~1)
instead of raw SI (~1e-10) also eliminates the `abs_tol`-swamps-tiny-
values failure mode of the old path (any two sub-1e-9 values used to
compare "equal").

## Effect on the measured numbers

Same holdout (133 rows, seed 42), same solver, same Ollama
`qwen2.5:3b-instruct` — **only the scorer changed**.

| Slice | rule baseline | Day-8 LLM (old scorer) | Day-9 LLM (unit-aware) |
|---|---:|---:|---:|
| **Overall Full✓** | 0.8% | 3.8% | **8.3%** |
| `charge_from_capacitance` | 0% | 0% | **60%** |
| `capacitor_energy` | 0% | 11.8% | 17.6% |
| `TD` prefix | 0% | 0% | 23.1% |
| `NL` prefix | 0% | 11.1% | 18.5% |
| `power_voltage_current` | 0% | 0% | 33.3% |

Rule-only stays 0.8% (it rarely solves, so there's nothing to recover).
The fix more than doubled the reported LLM full-correct with **zero
solver change** — it just stopped discarding answers that were already
right.

Verified not over-counting (spot-check of `charge_from_capacitance`):

| id | predicted | gold | scored | correct call? |
|---|---|---|---|---|
| TD039 | 5.987e-10 C | 0.6 nC | True | ✅ (0.599 ≈ 0.6 nC) |
| TD060 | 7.27e-10 C | 0.73 nC | True | ✅ |
| TD181 | 1.454e-9 C | 1.46 nC | True | ✅ |
| TD006 | 1.2e-9 C | 1 pF | False | ✅ misclassified (gold is capacitance) |
| TD397 | 0.08 C | 26.55 nC | False | ✅ genuinely wrong extraction |

## What this leaves on the table (NOT scorer issues)

The remaining 0% slices are now cleanly attributable to **physics
modelling gaps**, not measurement:

* `coulomb_force` (13): 100% solved, 0% numeric — LD problems need
  *vector composition* of 2–3 pairwise forces; the single-pair formula
  computes a number that's the wrong physics. Phase-2 (ADR 0003).
* `electric_field_point_charge` (21): same — multi-charge field
  superposition.
* `no_formula_matched` 45 — classifier/coverage gaps (THCB measurement
  error has no formula; some DDT magnetic-field variants missing).

These are the highest-value remaining physics work, and they're now
measurable: adding a vector-composition formula will show up directly
in the LD/DT numbers.

## Consequences

1. `outputs/eval/day9_llm/` is the new physics reference checkpoint.
2. Logic eval is unaffected (it uses `label_match`, not numeric).
3. Next physics work order: (a) LD/DT vector composition, (b) THCB +
   missing DDT formulas, (c) classifier `no_formula_matched` reduction.
4. The rented-GPU SFT (still pending) will be measured with this
   corrected scorer from the start.

## Test coverage

`tests/unit/test_metrics.py::TestQuantityMatch` — 8 cases incl. the
real TD039/TD060/TD181 SI-vs-nC cases, the genuinely-wrong TD397 case,
µF gold, missing-unit fallback, inconvertible-unit fallback, and None
inputs. Suite: 198 passing (was 190), ruff + mypy clean.
