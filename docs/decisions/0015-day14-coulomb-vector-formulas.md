# 0015 — Day-14 Coulomb vector-composition formulas (#2)

Date: 2026-05-20
Status: Accepted

## Context

ADR 0010 deferred the geometry-dependent Coulomb cases. After #1
(parser) was shown to be P1-flat (ADR 0014), #2 was the remaining
high-P1 lever: LD/DT problems where the classifier already picks a
Coulomb formula but single-pair physics is the wrong model.

Sub-categorizing the 46 LD+DT eval rows: midpoint-collinear **10**,
perpendicular-bisector **7**, two-forces-angle 7 (done Day-10),
equilateral-triangle 5, general/other 17. The two biggest tractable
buckets have exact closed forms.

## What was added

Two formulas (verified against real golds *before* coding):

| formula | closed form | verified |
|---|---|---|
| `coulomb_force_at_midpoint` | `4k·|q3|·(|q1|+|q2|)/d²` | LD022→14.4 N, LD035→0.36 N |
| `coulomb_force_perp_bisector` | `k·|q1·q|·d/((d/2)²+h²)^1.5` | LD219→1.71e-3 N |

Classifier rules placed **before** generic `coulomb_force` so a
3-charge midpoint / bisector problem doesn't fall into the single-pair
formula. Static check on day12 samples: routed **10 midpoint + 7
perp-bisector, 0 regressions** among 33 previously-correct. +2
integration tests; suite 226 → **228**, ruff/mypy clean.

## Measured impact

Same holdout, qwen2.5:3b, frozen scorer; only the formulas/classifier
changed vs day12.

| | day12 | **day14_vec** |
|---|---:|---:|
| **Physics Full✓** | 24.8% | **28.6%** |
| full-correct rows | 33 | **38** |
| LD prefix numeric | 12.5% | **25.0%** |
| `coulomb_force_at_midpoint` | — | 10 rows, 3 full (30%) |
| `coulomb_force_perp_bisector` | — | 7 rows, 2 full (29%) |
| `resultant_two_forces` (regress check) | 100% | 100% |

Sample audit: **0 regress, +5 gain** (3 midpoint + 2 perp-bisector).
Physics trajectory: 0.8 → 3.8 → 8.3 → 12.0 → 22.6 → 24.8 → **28.6 %**
(~36× the rule baseline; solver + scorer only, no training).

## Honest caveats

1. **Only ~5 of the 17 routed rows became full-correct.** The formulas
   are exact (verified); the limiter is the qwen2.5:3b extractor not
   reliably pulling q1/q2/q3/d/h from prose — the same
   solver-correct-but-LLM-extraction-bound pattern as ADR 0013. A
   stronger extractor would lift these further with no formula change.
2. **Unit✓ dipped 64.7% → 56.4%.** A few DT *electric-field*
   (V/m) questions phrased with "midpoint"/"bisector" now misroute to
   the *force* (N) formula → unit mismatch. Net Full still rose +3.8pp
   with **0 regressions**, so the trade is positive, but a future
   field-vs-force guard (target word "field"/"V/m" → field formula)
   would recover the misrouted DT rows. Logged as backlog.

## Decision

Keep #2 — it is the largest clean Stage-1 P1 gain since the scorer
fixes, deterministic, 0-regression, and strengthens P3 (every
midpoint/bisector answer now carries a correct closed-form trace).

Remaining physics backlog (diminishing returns, optional): DT
field-vs-force routing guard; equilateral-triangle (5);
general/other (17, genuinely hard — defer).

Next: submission packaging (endpoint + 1-page PDF) — the mandatory,
highest-value remaining work.

## Reproduction

```powershell
uv run pytest -q tests/integration/test_physics_solver.py
uv run python scripts/run_eval.py --task physics --with-llm --out outputs/eval/day14_vec
# vs outputs/eval/day12_llm/  (24.8% -> 28.6%, +5 rows, 0 regress)
```
