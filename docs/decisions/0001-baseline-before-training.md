# 0001 — Baseline solver before LLM training

Date: 2026-05-15
Status: Accepted

## Context

EXACT 2026 scores on three axes: P1 (correctness), P2 (explanation quality),
P3 (reasoning depth, especially symbolic / verifiable trace). We have 15 days
to first submission. There are two obvious extreme strategies:

1. **Train an 8B LLM end-to-end** on `sft_train_mixed_solver_clean.jsonl`
   and let it generate answers + explanations directly.
2. **Build deterministic solvers** (SymPy + pint for physics; rule + Z3 for
   logic), use an LLM only to draft and reformat.

## Decision

We commit to strategy (2): land a baseline solver + FastAPI + local eval
**before** any training run. The LLM is added on Day 6 strictly as a draft
generator / formatter, never as the final numeric computation source.

## Rationale

- The cleaning report shows only 2/1329 physics rows carry a verified
  `solver_formula`; the rest are answer-only. An LLM trained on those rows
  has no anchor to ground numerics — failures will be silent.
- P3 rewards verifiable reasoning. A deterministic solver naturally emits
  the trace the rubric asks for; an LLM must be coaxed into it and is easy
  to grade lower on "depth of reasoning."
- Without an eval harness, training iterations are blind. Solver first
  forces us to build the metric pipeline early.
- 15 days is tight. Starting on training without a fall-back means a
  collapsed deadline gives us nothing to submit.

## Consequences

- Day 1–5 = no training. Day 6 onward is the first SFT run.
- We accept that the baseline cap on physics is ≈70% (formula library
  coverage). We grow the library as eval reveals gaps.
- The orchestrator is wired to call solvers first, LLM second; this is
  reflected in `configs/app.yaml` (`pipelines.physics.use_solver_first: true`).
