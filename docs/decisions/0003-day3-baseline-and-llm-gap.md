# 0003 — Day-3 physics baseline & the LLM extraction gap

Date: 2026-05-15
Status: Accepted

## Context

Day 3 of the plan was to (a) land the eval harness end-to-end and
(b) measure the rule-only physics solver against a deterministic 10%
holdout (`data/eval_split/physics_eval.jsonl`, 133 samples, seed=42).
The number is the input that decides where Day-4+ LLM work should focus.

## Decision

Adopt the following as the Phase-1 baseline (rule + regex only, no LLM):

| Slice | N | Solved | Numeric✓ | Unit✓ | Full✓ |
|---|---:|---:|---:|---:|---:|
| **Overall** | 133 | 3.0% | **0.8%** | 46.6% | **0.8%** |
| CH (mạch) | 26 | 7.7% | 3.8% | 19.2% | 3.8% |
| CHLT (mạch LT) | 1 | 100% | 0% | 0% | 0% |
| DDT (từ trường) | 14 | 0% | 0% | 21.4% | 0% |
| DT (điện thế) | 6 | 0% | 0% | 66.7% | 0% |
| LD (Coulomb) | 40 | 0% | 0% | 72.5% | 0% |
| NL (năng lượng) | 27 | 0% | 0% | 40.7% | 0% |
| TD (tụ điện) | 13 | 7.7% | 0% | 76.9% | 0% |
| THCB (sai số) | 6 | 0% | 0% | 0% | 0% |

Failure mix: 83 `missing_input`, 45 `no_formula_matched`, 1 `unit_conversion_failed`.

The harness, metrics, and report writer in `src/exact_agent/eval/` are
accepted as the source of truth for all subsequent measurements.

## Diagnosis — what the numbers say

* **Quantity extraction is the dominant bottleneck.** 83/133 samples hit
  `missing_input`: the classifier picked the right formula but the regex
  extractor couldn't see one of the required variables. The cleaned
  dataset writes "the voltage across its plates is 60 V", "charged to
  50.1 V", "8 cm apart", "perpendicular bisector of AB" — never
  `U = 60 V` style. Our extractor only fires on explicit `name = value`.
* **Unit dimensionality is encouraging (47%).** When we attempt an
  answer, the unit is usually right (Coulomb 100%, electric field 95%,
  Ohm 100%, power 100%). The gap is numeric, not dimensional.
* **Coulomb / E-field rows need vector composition.** Most LD problems
  combine two or three pairwise interactions; the single-pair
  `coulomb_force` covers ~30% of LD by topology. Refusing rather than
  guessing was the right Day-3 hardening (target-word guard in
  `topic_classifier`).
* **THCB (measurement error) has zero coverage.** That's a formula gap,
  not an extraction bug — Phase-1 backlog.

## Consequences for Day 4+

1. **LLM-assisted variable extraction is the highest-leverage win.** The
   Day-6 SFT prompts (Phase 4) will ask the model to emit a JSON
   envelope of `{symbol → {value, unit}}` given the formula's required
   schema. The deterministic solver still does the math and the
   explanation; the LLM only fills the extraction slot.
2. **Orchestrator policy**: regex extractor first; if `< n_required`
   symbols are matched, fall back to the LLM extractor. Determinism
   wherever possible.
3. **Logic baseline (Day 4) before more physics rule polish** — coverage
   on the logic task is still 0 and the eval harness extends naturally.
4. **Hold off on Coulomb vector composition until Phase 2** — flag those
   LD cases for self-correction feedback (Phase 5) instead of inventing
   ad-hoc rules.
5. The eval harness is fast (~1 ms/sample average) — rerun it on every
   commit during Day 4–6 development.

## Reproduction

```powershell
uv run python scripts\build_eval_split.py   # one-shot, seed=42
uv run python scripts\run_eval.py --task physics
# outputs/eval/physics_report.md + .json
```

## Numbers we won't change

- Eval split: 10% holdout, seed 42, written by `scripts/build_eval_split.py`.
- Numeric tolerance: 1% relative, 1e-9 absolute.
- Unit equality: pint-dimensional, case-insensitive symbol match first.
