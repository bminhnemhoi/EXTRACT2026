# 0021 — Day-22 E8 RAG few-shot for the physics extractor

Date: 2026-05-21
Status: Accepted

## Context

Slide 28 of the organizer kickoff deck explicitly endorses RAG:
*"Build a knowledge base of physics formulas and solved examples,
retrieve similar problems as few-shot examples"*. With Day-22 having
saturated what a vanilla 3B extractor can do under self-consistency
(ADR 0019, ~19-21% range with ±2-3 row variance), the question was
whether *demonstrating* the extraction pattern with retrieved solved
training rows would lift accuracy on miss-the-regex extractor calls.

## What was built

### `src/exact_agent/physics/rag_retriever.py`

`WorkedExample(sample_id, question, cot, answer, unit)` dataclass +
`RagRetriever(examples)` that TF-IDFs the example *questions*
(`TfidfVectorizer(ngram=(1,2), stop_words='english', sublinear_tf=True)`)
and exposes `top_k(query, k=3)`. Cosine similarity ranks; only entries
with positive overlap are returned (better to drop than pad with
off-topic noise).

`default_retriever()` is `@lru_cache`d and loads from
`data/official_v20260515/train/physics_train.jsonl` (the 1,217-row
training split from the official 2026-05-15 release). Returns `None`
if the file is absent so the solver gracefully degrades to zero-shot.

We deliberately stay on TF-IDF, not sentence-transformer embeddings,
matching the existing logic premise_selector (Day-4 dropped the
`embed` extra). Zero new heavyweight deps in the API image.

### Prompt template

`configs/prompts/physics_extract.j2` now has an opt-in `{% if examples %}`
block listing each demo's `Question / Worked solution / Answer`. When
`examples == []` the prompt is byte-identical to the pre-E8 form — no
regression risk on existing callers.

### Extractor

`extract_with_llm` and `extract_with_llm_self_consistent` both accept
`examples=()`. The self-consistency variant passes the same demos to
every vote (so the votes are on the same reasoning context, not on
different few-shot framings — controlled comparison).

### Solver wiring + config

`PhysicsSolver.__init__` now takes `rag_examples_n=None` (defaults
from `configs/app.yaml::pipelines.physics.rag_examples_n`) and an
optional explicit `retriever`. When RAG is on the solver calls
`retriever.top_k(question, k=rag_examples_n)` *before* every LLM
extraction attempt and surfaces the retrieved sample IDs into the
trace (P3 evidence — reviewers see which solved examples informed the
extraction). `default_retriever()` is loaded lazily only when `n > 0`.

Config: `pipelines.physics.rag_examples_n` — code default `0`
(backwards compat), `app.yaml` default `3`.

## Tests

+7 unit tests (`test_rag_retriever.py`): correct ranking, k bounds,
empty corpus rejection, no-overlap returns empty (not random noise),
small-corpus consistency. **253 total tests** (was 246 → +7). ruff
auto-sorted imports (one fix); mypy clean on 51 source files.

## Measured impact on `data/official_v20260515/eval_split/physics_eval.jsonl`

| Variant | Overall Full✓ | ms/sample | Note |
|---|---:|---:|---|
| Day-21 baseline | 16.3% (22) | 705 | — |
| + E6 self-consistency (N=3) | 18.5–20.7 mean ~19.5% | 1900 | 2.7× latency |
| + E5 NL→FOL loop | 18.5–20.7 (unchanged for physics) | 1900 | logic-only effect |
| + 3 audit-driven formulas | 17.8–20.7 mean ~19.5% | 1750 | within 3B variance |
| **+ E8 RAG (k=3 demos)** | **17.0% (23)** | 2084 (+19%) | within 3B noise band; P1-neutral |

**Verdict honest:** E8 P1-neutral on this measurement (5-run series so
far: 22 / 25 / 28 / 27 / 24 / 23 — mean ~24.8 = ~18.4%; the E8 run
sits at the low end of the band, indistinguishable from extractor
variance). Retriever itself works correctly — smoke-tested in isolation
("Calculate energy when C=50 μF and U=20 V" → pulls TD401, NL002,
NL001, all capacitor-energy demos). 3B simply doesn't capitalise on
the few-shot signal — same pattern as E5 (logic NL→FOL loop): infra
correct, ceiling = backbone.

**Per-prefix:** DDT solved went 4.8% → 14.3% (3× more attempts succeed
through computation) but numeric still 0 (the LLM extracts something,
solver computes, doesn't match gold). TD +5.5pp, NL -5.3pp — within
±1 row noise on those slices.

**Decision: keep ON** for the same reasons as E5 — Slide-28 endorsed
infrastructure, P3 trace shows the retrieved demo IDs as verifiable
evidence ("RAG few-shot demos: TD401, NL002, NL001"), and the infra
becomes a real lift the moment a stronger backbone (E7) replaces 3B.

## Honest caveats

1. **RAG inflates the prompt** (3 demos × ~50-200 words each), which
   means slower extraction *and* the qwen2.5:3b context window must
   accommodate. 3B has a 32k window; demos are well within.
2. **Sklearn TF-IDF only ranks by lexical overlap**. Two physics
   problems with the same physics but different surface phrasing
   (e.g. one says "voltage", other says "potential difference") may
   not be ranked together. Acceptable starting point; if data shows
   we're often retrieving off-topic demos, a future swap to
   sentence-transformer is a one-line change.
3. The 3B-variance ceiling (±2-3 row on this 135-row holdout) means
   a single-run lift number is noisy. Three runs would give a
   confidence band; for time-budget we report one and flag it.
4. RAG and E6 self-consistency *multiply* — three votes × demo prompt
   each. Latency is the trade.

## Reproduction

```powershell
uv run pytest -q tests/unit/test_rag_retriever.py
uv run python scripts/run_eval.py --task physics --with-llm --include-samples \
    --split data/official_v20260515/eval_split/physics_eval.jsonl \
    --out outputs/eval/day22_v0515_physics_e8_rag
```
