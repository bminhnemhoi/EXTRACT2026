# 0017 — Day-21 official dataset migration + Z3 arity fix + MC Unknown abstention

Date: 2026-05-20
Status: Accepted

## Context

Organizer released the official authoritative dataset on 2026-05-15 (zip
under `tailieu_moi/Datasets/`), with two binding companion documents:
`QA.pdf` (28 questions of official rules) and `EXACT_Slides.pdf` (35
slides). Reading them changed three things at once: (1) our internal
"cleaned" dataset is a now-known-stale derivative of the 2026-05-09 drop
with bugs the organizer has since fixed (37 FOL parse fails, 8 UNSAT
records, 103 mislabelled MCQs, 401 QA-prefix annotation-pipe leak in
physics, Vietnamese unit labels, rounding errors); (2) the QA reveals
the scoring rubric (P1 auto exact-match w/ unit-tolerance for Type 2;
P2 committee-reviewed NL explanation; **P3 evaluated LIVE on Public Test
Day for Top 10**) plus the mandatory **Data Disclosure Document**; (3)
several "free" levers we were not yet using (Unknown is a legitimate MC
label per 168 retained MCQ records; sequential multi-model is permitted;
RAG is encouraged).

This ADR captures the Day-21 work: migrate, fix the bugs the migration
exposes, take one of the free levers.

## Decisions

### 1. Adopt the official 2026-05-15 release as the single source of truth

`scripts/import_official_v20260515.py` produces, from the unpacked
release, the JSONL shapes our existing eval harness already consumes,
under a new tree `data/official_v20260515/` (we do **not** overwrite
the prior `data/cleaned/` — kept for diff/audit). Outputs:

- `physics_safe.jsonl` — **1,352** rows (1,755 raw - 401 QA-prefix - 2 empty cot)
- `logic_safe.jsonl` — **808** questions (1 row per question, flattened from 411 records)
- `eval_split/{physics,logic}_eval.jsonl` — 10% holdout (seed 42)
- `train/{physics,logic}_train.jsonl` — 90% remainder

Carries the official `idx` field per question as `used_premise_idx`
(1-based; per CHANGELOG_TYPE1) — this is the gold premise reference the
P3 rubric explicitly rewards citing.

`LD348` and `LD350` (which existed in our prior cleaned data but the
organizer's official cleanup excluded) are dropped — the importer
treats them as no-op since the official CSV no longer contains them.

### 2. Recalibrated baselines on the official split (qwen2.5:3b, frozen scorer)

| Task | Prior (internal data) | Official 2026-05-15 holdout |
|---|---:|---:|
| Physics Full-correct | 28.6% | **16.3%** (22/135) |
| Logic correct | 37.5% | **22.2%** (18/81) |

The drops are honest: prior numbers were on a non-representative slice
of our self-cleaned 2026-05-09 derivative. The official split exposes
DDT (21/135 rows; 0% full-correct on current registry) and includes
the hard MCQ-Unknown rows our old "safe" filter had excluded. From now
on this is the ruler for every subsequent change.

### 3. Z3 mixed-arity fix (regression caught by the new clean FOL)

`_build_environment` previously inferred predicate arity from the first
occurrence and built a single `z3.Function(name, …, BoolSort)` per name.
The 2026-05-15 release fixes rec-31's missing quantifiers (and 37 other
formula bugs) — which exposed that the upstream NL→FOL translator
emits the same predicate name with two arities across premises (e.g.
`Has(Alice)` and `Has(Alice, BA)`). The single funcdecl was then called
with mismatched arity at apply time → `z3.z3types.Z3Exception: index out
of bounds`, killing the entire logic eval.

Fix: key `predicates` and `cmp_funcs` by **`(name, arity)`**. `P/1` and
`P/2` are distinct funcdecls (sound — FOL permits polymorphic names);
display labels suffix the arity (`P` and `P__a2`) so debug prints
distinguish. +2 regression tests (`TestMixedArity`).

### 4. E4 — MC abstain to "Unknown" (not empty string, not a guess)

The release retains 168 MCQ records whose gold is `Unknown` because the
premises genuinely under-determine the answer (CHANGELOG_TYPE1: 103
other MCQs that previously said `Unknown` were *fixed* to letters
because the explanation named one — those 168 are the residual where
the answer truly is Unknown). The Day-4 "always commit alphabetically"
policy was scoring zero on every such row.

`verify_multiple_choice` defaults changed: abstain to `"Unknown"` when
(a) every option scored zero, (b) the best option's token overlap is
below `min_top_score=0.15`, or (c) the spread to the runner-up is below
`margin=0.05`. Callers can disable each via kwargs for legacy
behaviour. Three failing tests updated (the old policy was *required*
by the test, not incidental); +1 new test for the threshold path.

### 5. Compliance artefacts

- `docs/data_disclosure.md` → `docs/data_disclosure.pdf` (2 pages, pandoc+pdflatex). Declares: only EXACT2026 official 2026-05-15; no external/synthetic/crawled data; one LLM at inference (qwen2.5:3b via Ollama OpenAI-compatible at `/v1/models`, verified); SFT'd 7B adapter retained as ablation, **not** deployed. Required by QA Q11/Q23 — submissions without this are grounds for disqualification.
- E2a (Ollama `/v1/models` compat): tested live — returns the OpenAI-standard `{object:"list", data:[{id,object,owned_by,created}]}` shape; `id` reveals the model+size (`qwen2.5:3b-instruct`). Stays on Ollama for the deploy; documented in Disclosure §5.

## Honest caveats

1. **Today's numbers are LOWER than the headline in `solution_description.pdf`** (16.3 vs 28.6, 22.2 vs 37.5). The PDF must be updated before submission (E13). The new numbers are the honest measurement against a representative split of the authoritative dataset; the old numbers were on noise.
2. The E4 thresholds (`min_top_score=0.15`, `margin=0.05`) are conservative defaults; tuning against the new holdout is the next data point.
3. Z3 mixed-arity fix loses cross-arity entailment (a premise stating `Loves(x)` no longer interacts with a claim about `Loves(x, y)`). This was never sound to begin with; the fix surfaces what was implicitly broken.

## Tests

234 passed (was 231; +2 Z3 mixed-arity regression, +1 E4 threshold test, -1 old test repurposed). ruff/mypy clean.

## Reproduction

```powershell
uv run python scripts/import_official_v20260515.py
uv run python scripts/run_eval.py --task physics --with-llm \
    --split data/official_v20260515/eval_split/physics_eval.jsonl \
    --out outputs/eval/day21_v0515_physics
uv run python scripts/run_eval.py --task logic --with-llm \
    --split data/official_v20260515/eval_split/logic_eval.jsonl \
    --out outputs/eval/day21_v0515_logic_e4
pandoc docs/data_disclosure.md -o docs/data_disclosure.pdf --pdf-engine=pdflatex
```
