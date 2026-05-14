# 0007 — Day-7 self-correction loop

Date: 2026-05-15
Status: Accepted

## Context

Days 1-6 stack up to a deterministic API: pipeline → response. Two
quality concerns remain that the pipeline cannot fix on its own:

1. **Cold abstentions.** Logic returns `Unknown` with no premise IDs
   (47% of the YNU split); physics returns an empty answer when the
   solver can't identify a formula.
2. **Low-confidence wins.** The solver got *something* but flagged
   verifier warnings (range out of bounds, etc.) — confidence ≤ 0.4.

Both are recoverable with one or two LLM revision passes if we hand the
model the **specific** verifier complaint. That's exactly what the
plan's Phase 5 calls for.

## Decision

Land a small, opt-in self-correction loop wrapping the orchestrator
output. The loop:

1. Scores the draft via :func:`exact_agent.agent.confidence.score_response`
   — checks empty fields, missing premises on logic, short physics CoT,
   and the solver's own confidence value. Emits a numeric score plus a
   list of human-readable defect reasons.
2. If `score < threshold` (default 0.6), renders
   ``configs/prompts/self_correction.j2`` with the question, draft JSON,
   and concatenated reasons. Asks the LLM for a revised JSON.
3. Validates the revision against :class:`PredictResponse`. If valid
   AND its score is strictly higher than the current draft, keeps it
   and continues. Otherwise breaks out — we never make things worse.
4. Caps at `max_rounds` (default 2). The CoT trace gets a final
   "Self-correction loop applied (revisions kept: N)" line so the
   reasoning-depth (P3) signal reflects the work.

The loop is **off by default** in `configs/app.yaml::self_correction.enabled`
so the Day-1..6 deterministic behavior is preserved without an LLM. The
orchestrator constructs a `SelfCorrector` only when (a) the config
toggle is on AND (b) an `LLMClient` was passed in.

## What ships

| Module | Role |
| --- | --- |
| `agent/confidence.py` | `score_response(response, task_type, threshold)` → `ConfidenceAssessment` |
| `agent/self_correction.py` | `SelfCorrector(llm, max_rounds, threshold).maybe_revise(...)` |
| `agent/orchestrator.py` | Now accepts `llm_client` / `self_corrector`; runs the loop after the pipeline |
| `tests/unit/test_confidence.py` | 7 unit tests covering each defect rule |
| `tests/integration/test_self_correction.py` | 6 tests: no-LLM passthrough, high-confidence skip, accept-when-better, reject invalid JSON, reject no-progress, max-rounds cap |

## Why we wrap the pipeline (instead of re-running it)

The pipelines have already done deterministic work — extraction,
classification, SymPy compute, Z3 entailment. Re-running them on the
same input would produce the same draft. The interesting variable is
the *natural-language phrasing* and the *answer/explanation/premise
fields*; that's where the LLM revision adds value. SymPy and Z3 don't
need a second turn at the wheel.

## What we explicitly didn't do

- **Fire the loop in tests by default.** Self-correction stays an
  explicit opt-in via the orchestrator constructor or app config; the
  existing 174 tests still cover the deterministic path.
- **Rerun the entire pipeline post-revision.** Score the revision in
  isolation and accept-or-keep — simpler and bounded.
- **Cross-check the revision against the solver.** A future refinement:
  if the LLM revises a numeric answer, verify it against `formula.compute`
  again. Out of scope for Day 7.

## Reproduction

```python
from exact_agent.agent.orchestrator import Orchestrator
from exact_agent.llm.vllm_client import VLLMClient

llm = VLLMClient()  # configs/model.yaml
# Toggle configs/app.yaml::self_correction.enabled = true
orch = Orchestrator(llm_client=llm)
response = orch.predict(payload)  # may include 'Self-correction loop applied (...)' in cot
```
