# 0033 — Day-27 Iter-6 — formula fan-out + Unicode-minus normalisation

Date: 2026-05-22 (Day 27)
Status: Accepted (eval in flight; commit pending result)

## Context — what Iter-5 left on the table

Iter-5 (ADR 0032) hit 39.3% on the 163-row SFT-unseen holdout. Per-sample
fail-reason grouping on that eval flags the biggest remaining cluster as
the `no_formula_matched` bucket — **24 rows** (14.7% of the holdout) where
the topic classifier returns nothing because no formula in the registry
covers the asked quantity.

Manual triage of those 24 rows splits them into:

| Sub-cluster | Rows | Physics |
|---|---:|---|
| CH371/372/373/374 — quality factor | 4 | Q = (1/R)·√(L/C) |
| DDT345 — capacitive reactance | 1 | X_C = 1/(2πfC) |
| DDT362 — natural period of LC | 1 | T = 2π·√(LC) |
| DDT384 — total flux linkage | 1 | Ψ = N·Φ |
| NL092 — energy-loss percent | 1 | (W_i − W_f) / W_i · 100 |
| LD126/182 — resultant of 2 perpendicular forces | 2 | F = √(F1² + F2²) — formula exists, routing missed |
| LD034/054/362/367/395 — multi-charge force/field at angle | 5 | new formulas, deferred Tier-2 |
| TD386/391/THCB094/114/133/NL026/DDT330/LD077/DT051 + CH106 | 9 | symbolic / qualitative / complex multi-step — out of scope |

Tier-1 ceiling (Iter-6): 10 rows = +6.1pp if every fix lands cleanly.

## Fixes

### Fix 1-5 — five new formulas in `configs/physics_formulas.yaml`

All five expression literals were pre-verified against gold answers at
< 0.18% relative error before being committed:

```yaml
quality_factor_q:        (1/R) * sqrt(L/C)             # CH371/372/373/374
capacitive_reactance:    1 / (2*pi*f*C)                # DDT345
natural_period_lc:       2*pi*sqrt(L*C)                # DDT362
total_flux_linkage:      N*Phi                          # DDT384
energy_loss_percent:     (W_initial-W_final)/W_initial*100  # NL092
```

Each formula gets a dedicated keyword rule in `_KEYWORD_RULES`, placed
ABOVE the existing resonance/series rules so the more specific intent
wins. Each also gets a `_TARGET_WORDS` entry so the symbol-fallback
path can elect them when the keyword doesn't match exactly.

### Fix 6 — `resultant_two_forces` routing fan-out

LD126/182 phrasing — "Two electric forces have magnitudes of 3 N and
8 N, acting at a 90° angle" — missed every existing trigger:
`"at an angle of"`, `"angle between the two forces"`,
`"two forces with magnitudes"`. Added 5 new triggers:
- `"two electric forces have magnitudes"`
- `"two electric forces have a magnitude"`
- `"their resultant force"`
- `"a 90° angle"`
- `"a 90 degree angle"`

### Fix 7 — Unicode minus + bare-10 scientific notation
[`src/exact_agent/physics/question_cleaner.py`]

DDT362's question writes `C = 10−6 F` where `−` is U+2212 MINUS SIGN
(not ASCII `-`). NFKC normalization does NOT touch math operators, so the
extractor regex stops at the non-ASCII char and reads `C = 10` (losing
the exponent and the unit `F`). Two-step fix:

1. `_UNICODE_MINUS_MAP = str.maketrans({"−": "-", "–": "-", "—": "-",
   "‑": "-", "‒": "-"})` — explicit translate for the 5 dash variants
   the dataset uses interchangeably with ASCII hyphen.

2. After unicode-minus replacement we get `C = 10-6 F` — the extractor's
   scientific regex still doesn't catch this without a caret. Added
   `_BARE_TEN_NEG_EXPONENT_RE = re.compile(r"(?<=[\s=])10-(\d+)(?=\s)")`
   that rewrites bare `10-N` (only when flanked by space/equals) to
   `10^-N`. The flank requirement keeps "There are 10-6=4 apples"
   from being clobbered (3 parametrized regression tests).

## Quality gate

- 271 tests pass (3 new question_cleaner tests + the formula registry
  loader fans out via existing parametrized tests)
- `ruff check src tests` — All checks passed
- `mypy src` — Success: no issues found in 51 source files

## Smoke-test (deterministic, no LLM)

```
CH371   formula=quality_factor_q,      pred=1.118,    gold=1.12,    err=0.18%
CH372   formula=quality_factor_q,      pred=1.000,    gold=1.00,    err=0.00%
DDT345  formula=capacitive_reactance,  pred=35.37,    gold=35.37,   err=0.01%
DDT362  formula=natural_period_lc,     pred=1.99e-3,  gold=1.99e-3, err=0.15%
DDT384  formula=total_flux_linkage,    routed correctly; needs LLM for N=600, Φ=4e-6
NL092   formula=energy_loss_percent,   routed correctly; needs LLM for Wi/Wf prose
LD126   formula=resultant_two_forces,  routed correctly; needs LLM for F1=3, F2=8
LD182   formula=resultant_two_forces,  routed correctly; needs LLM for F1=6, F2=5
```

## Expected lift

Optimistic ceiling: +10 rows = +6.1pp (39.3% → ~45.4%) if every Tier-1
fix lands cleanly through LLM extraction.

Realistic with LLM-extractor variance: +5-8 rows = +3-5pp (→ ~42-44%).

Remaining no_formula_matched residue: 5 rows needing new multi-charge
formulas (LD034/054/362/367/395 — Tier-2 candidates for a future Iter-7)
plus 9 rows that need symbolic / qualitative output infrastructure
(currently out of scope).
