# 0028 — Day-25 Tier 1 batch (G3 regex + J audit + K inference abstention)

Date: 2026-05-22 (Day 25)
Status: Accepted

## Context — MAX PUSH plan for Top-1 goal

User confirmed max-push direction Day-24-end. Day-25 executes Tier 1
(solo, no GPU/Colab cost) — three small improvements aimed at
squeezing a few more rows from the current 3B backbone, ahead of
Day-26 SFT-7B retrain (user) + Day-27 hybrid compose (me).

## G3 — three regex enhancements in `quantity_extractor.py`

1. **Standalone "10^N" → "1eN"** plus **"mantissa * 10^N" → "mantissaeN"**.
   Day-23 audit failures included `q = 10^-7 C` (regex matched only
   `10` and dropped the exponent) and `q1 = q2 = q3 = 2.6 × 10^-6 C`
   (chain regex saw value=2.6 and unit=garbage). The new
   `_denormalize_exponents` collapses both forms to single scientific
   tokens so a single number regex catches every magnitude.
2. **Chained-equality broadcast**: `q1 = q2 = q3 = <value> <unit>` now
   yields three entries with the shared value. Equilateral-triangle
   questions extract correctly.
3. **"<noun> of <value> <unit>" prose pattern**: a 30-noun vocabulary
   maps "plate area", "capacitance", "potential difference", "magnetic
   flux density", "side length", etc. to canonical formula symbols
   (A, C, U, B, a, ...). A second scan fills in symbols the equation
   passes didn't catch. "of magnitude X" also handled.

### Measured (163-row SFT-unseen holdout)

| | F1+F2 baseline | **+ G3** |
|---|---:|---:|
| Overall Full✓ | 27.6% (one run) / ~27.0% mean | **27.0%** |
| TD slice | 23.1% | **26.9%** (+3.8pp / +1 row) |
| Explanation F1 | 0.286 | **0.297** (richer trace because more vars extracted) |
| ms/sample | 2054 | **1789** (-13%; fewer LLM fallback calls) |

Headline unchanged within ±2 row variance band; TD slice + Expl-F1 + latency all moved the right way. Keep ON — clear infrastructure win even if the noise band hides the P1 delta.

## J — output number format audit (no code change)

Hypothesis: BTC scorer might be strict on number format (decimal vs
scientific vs unit-attached). Audited the predicted vs expected
strings on 60 sample rows:

| (pred_fmt, exp_fmt) | count | full_correct? |
|---|---|---|
| dec / dec | 17 | mostly True |
| sci / sci-* | 6 | True when value matches |
| dec / sci-* | 7 | True when round-aware match fires |
| other (empty pred) / * | ≥10 | False — extraction missed entirely |

The scorer's `quantity_match` (ADR 0009 unit-aware + 0011/0012 round-
aware) correctly handles **every** format combination in the sample.
QA Q20 explicitly says BTC's scorer normalises "common mathematical
notations (Unicode √2 vs LaTeX \\sqrt{2}, scientific vs decimal)".

**No code change needed.** The output format is not a bottleneck on
either side.

## K — inference-phrased MC abstention (`answer_verifier.py`)

Audit found **12/81 logic rows tagged `question_type='yes_no_unknown'`
in the dataset but routed to `verify_multiple_choice` by our pipeline**
because the question structure has A/B/C/D options. Per CHANGELOG_TYPE1
the dataset retains 168 MCQs whose gold is `Unknown` (premises don't
determine a unique option) — exactly these questions.

Added `_INFERENCE_PHRASES` ("can be inferred", "which conclusion is
correct", "which statement is correct", "which statement can be",
"which of the following can be inferred", ...) and a small helper
`_is_inference_phrased`. When the MC verifier sees such a question,
it auto-raises `min_top_score` to 0.25 so that without confident
chain overlap the verifier returns "Unknown" instead of committing.

### Measured

| | Day-23 / Day-24 baseline | **+ K** |
|---|---:|---:|
| Logic overall correct | 23.5% (19/81) | **23.5%** (19/81) |
| MC slice | 33.3% (5/15) | 33.3% (5/15) |
| YNU slice | 21.2% (14/66) | 21.2% (14/66) |
| Abstained | 39.5% | 40.7% (+1 row abstain) |

P1 unchanged. K fired on inference-phrased questions but the abstentions
landed roughly evenly on Unknown-gold and letter-gold rows (the 12
wrongly-classified set isn't homogeneous). **Kept ON** — same
"infrastructure correct, lift within 3B noise band" pattern as E5 /
E8 / F3 / G1. Semantic: when a question literally says "which can be
inferred" and the chain has no strong overlap, abstaining to "Unknown"
is sound.

## Cumulative Day-21 → Day-25 trajectory

| Phase | Physics Full | Logic correct |
|---|---:|---:|
| Day-1 rule-only | 0.8% | 35.9% |
| Day-21 honest baseline | 22.1% | 22.2% |
| Day-22 (E5/E6/E8/F3, mean of multi-run) | ~22-25% (within noise) | 23.5% |
| Day-23 F1+F2 (DETERMINISTIC) | **27.6%** | -- |
| Day-23 hybrid ablation | 30.1% (not deployed) | -- |
| **Day-25 (G3 + K, current)** | **27.0%** (TD +3.8pp slice gain) | **23.5%** |

Headline Physics around 27% mean stable across Day-23/24/25 (variance
band ±2 row already characterised). Logic 23.5% rock-solid stable.

## Tests

253 passed (no test changes — G3 adds regex, K adds helper; existing
fixtures cover both paths). ruff/mypy clean (51 src files).

## Reproduction

```powershell
uv run pytest -q
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/g3_3b_unseen
uv run python scripts/run_eval.py --task logic --with-llm `
    --split data/official_v20260515/eval_split/logic_eval.jsonl `
    --out outputs/eval/k_logic
```
