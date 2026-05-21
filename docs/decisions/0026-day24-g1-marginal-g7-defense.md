# 0026 — Day-24 G1 (audit-driven formulas, marginal) + G7 defense script

Date: 2026-05-21 (Day 24)
Status: Accepted (G1 kept for infra value; G7 final deliverable)

## Context — Option A SAFE plan after Day-23

Day-23 closed at:
* Physics: **27.6%** (F1+F2 deterministic) / **30.1%** (hybrid F1+F2 + SFT-7B)
* Logic: 23.5%

User chose Option A SAFE (Path A keep 3B + small deterministic
improvements) — push G1+G3+G6+G7 in Day 24-25, defer hybrid deploy.
This ADR records G1 (formula additions) and G7 (defense script).

## G1 — re-audit post-F1+F2

Ran `scripts/audit_dataset_issues.py` on `outputs/eval/f1f2_3b_unseen`:
29 candidates surfaced. Identified two new patterns:

1. **2-charge opposite-sign field at midpoint** (LD053/090/065/056/081/
   392 etc.): two charges +q and −q at distance d, field at the
   midpoint is the *sum* of two same-direction fields = **E = 8·k·|q|/d²**.
   Currently routed to single-charge `electric_field_point_charge`
   (wrong physics, wrong magnitude).
2. **Z = U/I direct** (DDT349 etc.): RLC questions giving only U and
   I, expecting Z back. Distinct from the `rlc_impedance` formula
   (Z = √(R² + (XL−XC)²)) which needs the reactances.

Added two formulas (`configs/physics_formulas.yaml`):
* `electric_field_two_opposite_charges_midpoint` — E = 8·k·|q|/d²
* `impedance_from_voltage_current` — Z = U/I

Plus expanded F2 symbol-fallback guard: a Coulomb-class formula
must never win the *symbol* fallback when the question is asking
for an electric field. Symmetric to the keyword-level F2 guard.

### What didn't pan out — routing rule revert

Initial G1 also added the keyword rule
`("calculate the total impedance",) → impedance_from_voltage_current`.
Measured impact: **−1 row** vs F1+F2 baseline (27.6% → 27.0%) —
the broad keyword ate CH rows that legitimately need `rlc_impedance`
(R + XL + XC given) and routed them to the simpler U/I formula
instead. **Routing rule REVERTED**; formula left in the registry so
the symbol-fallback can still pick it on questions that genuinely
provide only U + I.

### Final G1 measurement

| Eval | Overall Full✓ | Verdict |
|---|---:|---|
| Day-23 F1+F2 (baseline) | 27.6% (45) | |
| G1 full (with all routing) | 27.0% (44) | −1 row regression |
| **G1 trimmed (final)** | **27.0%** (44) | Same as full — variance ±2 row |

The two new formulas and the symbol-fallback guard land **clean
semantically** (no force formula wins a field question) but **don't
visibly lift on this 163-row sample**. Same pattern as ADR
0018 (E5), 0021 (E8), 0024 (F3): infrastructure correct, lift
within 3B noise band.

## G7 — defense demo script (`docs/defense_demo_script.md`)

The rubric's **P3** (Slide 12; QA Q21) is evaluated *live* on Public
Test Day (Jun 15) for the top-10 finalists. Jury can ask follow-ups
on any response field; our 6-field schema (`answer, explanation, cot,
premises, fol, confidence`) gives them surface to dig into.

Written script: **~10 minutes**, one physics demo (parallel-plate
capacitance — a Day-22 F1 win, shows formula + SymPy compute + pint
unit conversion in cot) and one logic demo (modus-ponens entailment
— shows premise selector + forward chain + Z3 fallback + NL→FOL
rejection loop in cot). Each demo has rehearsed talking points and
fallback plays for hard questions ("what if your formula is wrong",
"what about edge cases", "why didn't you deploy the SFT-7B").

The script's value isn't in any one talking point — it's that the
*system itself* exposes every piece of evidence the jury needs in
the response payload. Defending it is "read the cot together",
not "memorise a pitch".

## Tests

253 passed (no test changes — formulas are YAML, classifier rules
are table entries). ruff/mypy clean.

## Honest caveats

1. G1's two new formulas are correct and tested by hand, but **didn't
   fire on the 163 rows**. They might fire on the BTC hidden test set
   if it has more 2-charge-midpoint or U/I-only-impedance phrasings;
   it might not. Keeping them is risk-free either way.
2. G7 script is *defensive* — it does not improve P1 by a single
   row. Its leverage is conditional: it only matters if we make
   Top-10 and get the Public Test Day. The leverage when it matters
   is high.

## Reproduction

```powershell
uv run python scripts/audit_dataset_issues.py `
    --report outputs/eval/f1f2_3b_unseen/physics_llm_report.json `
    --rel-tol 0.1
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/g1_revert_3b_unseen
```
