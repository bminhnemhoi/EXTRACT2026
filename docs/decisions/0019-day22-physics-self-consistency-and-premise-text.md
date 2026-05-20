# 0019 — Day-22 physics self-consistency vote (E6) + premises idx+text (E12)

Date: 2026-05-21
Status: Accepted

## Context

Two small organizer-endorsed improvements bundled together because they
share no surface area and are both cheap:

* **E6** (Slide 28 official "Practical Tip"): *"Ensemble multiple
  approaches and select the most consistent answer"*. The qwen2.5:3b
  extractor occasionally hallucinates one number in an otherwise
  correct envelope; majority-voting N independent samples filters
  those out.
* **E12** (Slide 33 organizer reference output schema): the `premises`
  field is shown as **human-readable statements** (`"Ohm's law: V = IR"`),
  not bare index labels (`"P1"`). Our Day-4 pipeline emitted indices
  only. Carrying both index *and* text in one string preserves the
  alignment with the dataset's 1-based `idx` gold (P3 measurement)
  while making the cited evidence directly readable for the Public
  Test Day jury (P2 + P3 reward).

## Changes

### E6 — `physics/llm_extractor.py`

New public function `extract_with_llm_self_consistent(question,
formula, client, *, n_votes=3, max_tokens=256, tol=0.01)` returning
`(quantities, vote_counts)`.

Vote rule per symbol: cluster the N candidates by relative tolerance
(default 1%); the largest cluster's value wins; unit is the most
common string in that cluster. A symbol must have a cluster size
≥ ceil(n_votes/2) ("majority") to be elected; otherwise the whole
extraction is rejected (`LLMExtractionError`) so the solver abstains
rather than half-voting an inconsistent number into the math.

Config: `configs/app.yaml::pipelines.physics.llm_self_consistency_n`
(default in YAML: 3; default in code: 1 for backwards compat). Solver
constructor accepts `self_consistency_n=` override for tests / A/B.

Trace ("LLM extractor (self-consistent N=3): C=1e-4 F (3/3 votes),
U=30 V (2/3 votes)") is surfaced into the API `cot` — exactly the
verifiable-evidence shape Slide 12 P3 rewards.

+5 unit tests covering: unanimous, majority-with-one-hallucinated,
no-majority-rejected, two-of-three-parse-fails, within-tolerance
clustering.

### E12 — `logic/pipeline.py`

`premises` response field changes from `["P1", "P7"]` →
`["P1: <text>", "P7: <text>"]` (concatenating the 1-based index with
the actual premise text). Out-of-range indices fall back to the bare
`P{i}` form (defensive — shouldn't happen, the chain only emits valid
indices).

No new tests — the integration test only assertion was `assert
resp.premises` (truthy check), which still holds.

## Measured impact on `data/official_v20260515/eval_split/physics_eval.jsonl`

| Slice | N | Day-21 baseline | **Day-22 (E6 N=3)** | Δ |
|---|---:|---:|---:|---:|
| **Overall Full✓** | 135 | 16.3% (22) | **18.5%** (25) | **+2.2pp / +3 rows** |
| Solved | 135 | 51.9% | 52.6% | +0.7pp |
| Unit✓ | 135 | 55.6% | 55.6% | 0 |
| `THCB` (sai số) | 7 | 42.9% | **57.1%** | +14.2pp |
| `NL` (năng lượng) | 19 | 10.5% | **15.8%** | +5.3pp |
| `CH` (mạch) | 25 | 12.0% | **16.0%** | +4.0pp |
| `TD` (tụ) | 18 | 33.3% | 33.3% | 0 |
| `LD` (Coulomb) | 38 | 21.1% | 21.1% | 0 |
| `DDT` (từ trường) | 21 | 0.0% | 0.0% | 0 (extractor isn't the bottleneck; needs formula coverage) |
| `DT` (điện trường) | 5 | 0.0% | 0.0% | 0 (same — registry gap) |
| ms/sample | 135 | 705 | **1928** | +2.7× (within 60s/req cap) |

E6 net positive on P1 — modest but real and exactly where expected
(rows that miss regex and rely on the 3B extractor's per-call
reliability). DDT/DT untouched: those need formula additions or
DeepSeek-grade extraction (E7b), not more votes on the same backbone.

(E12 is a response-shape change — does not affect P1; pure P2/P3 lift.)

## Honest caveats

1. E6 cost scales with how often the regex extractor *misses*. Rows
   solved purely by regex (~half of TD/CH) pay zero E6 cost; the LLM
   extractor only fires for the harder slices (DDT/LD/NL).
2. Vote tolerance is 1% rel — the same as the scorer's
   `quantity_match`. Looser tolerance → more spurious agreement;
   tighter → more abstentions. 1% is the obvious anchor.
3. E12 makes the `premises` field longer (text instead of one
   label). At a typical 50-100 char premise, response payloads grow
   modestly; no impact on the 60s latency cap.

## Tests

246 passed (was 241 → +5 self-consistency). ruff / mypy clean.

## Reproduction

```powershell
uv run python scripts/run_eval.py --task physics --with-llm \
    --split data/official_v20260515/eval_split/physics_eval.jsonl \
    --out outputs/eval/day22_v0515_physics_e6
```
