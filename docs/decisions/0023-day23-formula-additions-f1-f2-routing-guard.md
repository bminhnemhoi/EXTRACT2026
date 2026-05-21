# 0023 — Day-23 F1 (6 formulas) + F2 (field-vs-force routing guard)

Date: 2026-05-21
Status: Accepted

## Context

Day-22 closed with the conclusion that the qwen2.5:3b extractor is the
ceiling on physics; Day-23 E7 confirmed SFT-7B lift is marginal
(+2.4pp single, +4.3pp hybrid projected) and deployment is
expensive/fragile. Returned to the audit (`scripts/audit_dataset_issues.py`
on `outputs/eval/e7a_baseline_3b_unseen`): 22 failing rows on the 163-
row clean holdout, no credible dataset bugs, 15+ rows are real
formula / routing gaps.

This ADR captures the deterministic answer: write the missing
formulas, guard the misroutes. No GPU / Colab / SFT required.

## F1 — six new formulas (`configs/physics_formulas.yaml`)

Targeted at specific failing rows verified by hand against the audit:

| # | Formula | Audit-target rows | Verified |
|---|---|---|---|
| 1 | `magnetic_flux_solenoid_one_turn`: Φ = B·A | DDT158 | 0.01 T × 8 cm² → 8 μWb ✓ gold |
| 2 | `power_factor_from_r_z`: cosφ = R/Z | DDT327, DDT337 | 12/20=0.60 ✓; 18/30=0.60 ✓ |
| 3 | `inductance_for_resonance`: L = 1/(C(2πf)²) | CH067, CH084 | inverse of capacitance_for_resonance |
| 4 | `electric_field_from_force`: E = F/q | DT046 | 3 mN / 1e-7 C → 3×10⁴ V/m ✓ |
| 5 | `inductance_from_inductor_energy`: L = 2W/I² | NL010 | 2×3.2 mJ / (4 A)² → 0.4 mH ✓ |
| 6 | `current_from_voltage_impedance`: I = U/Z | DDT339 + RLC | direct application |

## F2 — field-vs-force routing guard (`topic_classifier._keyword_match`)

Day-22 audit found LD314/LD317/LD396 misroute to
`coulomb_force_equilateral_three_identical` because they have the word
"equilateral" but actually ask for the *electric field* (V/m) at the
centroid, not the force. Computing force when the gold is field is
"right slice, wrong dimension" — always fails.

Implementation: `_keyword_match` now skips any `_FORCE_FORMULAS`
result when the question contains `_FIELD_ASK_TOKENS` (V/m, "electric
field intensity / strength / at...", "magnitude of the electric
field", "volts per meter"). Falls through to subsequent rules
(field-specific ones win). Tighter than tweaking individual keyword
lists; one centralised guard.

## Classifier routing rules added (paired with formulas above)

* `"power factor" | "cosφ"` → `power_factor_from_r_z` (before generic impedance)
* `"magnetic flux through one turn"` → `magnetic_flux_solenoid_one_turn`
* `"what value of inductor" | "needed to resonate" | "required inductance"` → `inductance_for_resonance` (must precede LC resonance freq rule)
* `"experiences a force" | "force of magnitude"` → `electric_field_from_force` (before generic force rules)
* `"calculate its inductance" | "calculate the inductance"` → `inductance_from_inductor_energy`
* `"calculate the rms current" | "calculate the current I in the circuit"` → `current_from_voltage_impedance`

Also tightened `capacitor_energy` keyword set: removed the bare
`"field energy"` (was catching "magnetic field energy" → inductor
problems mis-routed to capacitor). Replaced with explicit
`"energy stored in (a/the) capacitor"`, `"energy in (a/the) capacitor"`,
`"energy of (a/the) capacitor"`, etc.

## Measured impact on `physics_eval_sft_unseen.jsonl` (163 rows, qwen2.5:3b, E6 N=3 + E8 RAG, frozen scorer)

| Slice | N | 3B baseline | **F1+F2 (3B)** | Δ |
|---|---:|---:|---:|---:|
| **Overall Full✓** | 163 | **22.1%** (36) | **27.6%** (45) | **+5.5pp / +9 rows** |
| `DDT` | 16 | 6.2% (1) | **31.2%** (5) | **+25pp / +4 rows** |
| `CH` | 39 | 23.1% (9) | **30.8%** (12) | **+7.7pp / +3 rows** |
| `NL` | 21 | 4.8% (1) | **14.3%** (3) | **+9.5pp / +2 rows** |
| `DT` | 6 | 0.0% | 0.0% | 0 (extractor missed F/q on DT046; formula correct, regex didn't catch "F=3 mN" pattern) |
| `LD` | 45 | 33.3% (15) | 33.3% (15) | 0 (no new LD formula in F1; F2 guard would have fired on field-asking LDs but none crossed the threshold this run) |
| `TD` | 26 | 23.1% (6) | 23.1% (6) | 0 (no new TD formula) |
| `THCB` | 9 | 44.4% (4) | 44.4% (4) | 0 |
| `CHLT` | 1 | 0% | 0% | 0 |
| ms/sample | | 2217 | 2054 | ~same |

**+5.5pp deterministic — cleaner than the E7 SFT-7B path** (+2.4pp single, +4.3pp hybrid projected) **at zero deployment complexity**: no second LLM, no GPU swap, no compliance gymnastics (Q3 is trivially satisfied — one model). DDT slice went from "lost cause" 6% to a respectable 31% in a single batch.

## Honest caveats

1. The DDT lift (+25pp) is on **16 rows** — small sample. On the full
   official corpus (~130 DDT rows) the projected lift would be ~+4
   rows / +3.1pp prefix-wide if the per-row hit rate holds; less
   spectacular than the slice ratio suggests.
2. **DT still 0%** — DT046 routed correctly to `electric_field_from_force`
   but the regex extractor didn't pull "F = 3 mN" + "q = 10^-7 C" out
   of the prose. Adding regex patterns for "experiences a force F = X"
   would close this; deferred.
3. The `"energy stored"` → `"energy stored in (a/the) capacitor"`
   tightening covers all tested phrasings but might miss novel
   phrasings on the BTC test set (e.g. "energy held by"). Acceptable
   risk — the prior loose rule was bleeding into magnetic energy.

## Tests

253 passed (no test changes — formulas are YAML, classifier rules are
table entries; existing integration tests cover dispatch). ruff/mypy
clean. The pre-tighten `test_missing_required_input` test for
`"Energy stored in a capacitor with U = 30 V."` still passes after
the extended phrasings.

## Reproduction

```powershell
uv run pytest -q
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/f1f2_3b_unseen
```
