# 0018 — Day-22 NL->FOL solver-rejection loop (E5)

Date: 2026-05-20
Status: Accepted (kept — P3-positive even though P1-flat on holdout)

## Context

Slide 27 of the organizer's official kickoff deck explicitly endorses
the "Neurosymbolic Hybrid" pattern: **LLM generates candidate FOL ->
solver verifies -> if reject, regenerate**. Our Day-6/Day-8 wiring was
single-shot — one LLM call, no feedback to the model on parser /
vocabulary failures. Logic eval on the official 2026-05-15 holdout
showed ~35-40% of rows abstaining; the hypothesis was that some chunk
of those were salvageable with a retry loop.

## Change

`logic/llm_translator.py`: new public function
`translate_question_to_fol_with_verify(question, premises_fol, client,
*, max_rounds=2)` returning `(accepted_fol, trace)`. Rejection signals
(only — Z3 verdict is **not** a reject signal because Unknown is often
the legitimate gold):

1. Empty completion or no FOL-shaped line.
2. `parse_fol(claim) is None` (the parser already covers ∀, ∧, →,
   comparisons, ground atoms; ∃ and ∨ are deliberately unsupported).
3. Every predicate in the proposed claim is novel relative to the
   premise vocabulary (hallucinated translation).

Each rejection appends a *specific* reason to a `feedback` list which is
prepended to the next prompt as a "do not repeat" header. The per-attempt
trace is surfaced into the response `cot` so reviewers (P3) see the
iterative refinement.

`logic/pipeline.py::_try_z3_fallback` now calls the verify variant.
The old single-shot `translate_question_to_fol` is retained for
backwards compat.

+6 unit tests with `MockLLMClient` covering: first-shot accept,
parse-failure retry with feedback, vocab-mismatch retry with feedback,
exhaustion, partial-novel-vocab acceptance, max_rounds=0 path.

## Measured impact on `data/official_v20260515/eval_split/logic_eval.jsonl` (qwen2.5:3b, frozen scorer)

| | Day-21 (E4 v2) | **Day-22 (E5)** |
|---|---:|---:|
| Overall correct | 23.5% | **23.5%** (no change) |
| MC (15 rows) | 33.3% | 33.3% |
| YNU (66 rows) | 21.2% | 21.2% |
| Abstained | 40.7% | 39.5% |
| ms/sample | 744 | **1909 (2.6×)** |

P1 flat. The mechanism is sound but the rejection signals fire rarely:
3B's outputs mostly parse and partially reuse vocabulary, so the loop
short-circuits on the first attempt. The latency rise (still well
within the 60-second cap) is the cost of per-attempt verification.

## Decision — keep it ON anyway

1. **P3 explicitly rewards verifiable reasoning evidence** (QA Q21;
   slide 12). The loop trace in `cot` ("attempt 1 rejected: ∃ not
   supported; attempt 2 accepted") is exactly the kind of evidence
   the Public Test Day jury will be looking at for Top-10 — it shows
   the model is being held to a *verifiable* refinement protocol, not
   just emitting plausible-sounding FOL once.
2. With a **stronger** translator (E7 — DeepSeek-R1-Distill-Qwen-8B
   distilled-reasoning, or the SFT-7B as sequential specialist) the
   first-attempt vocab-novelty rate may shift such that the loop
   actually corrects more rows. The infrastructure is now in place to
   benefit when the backbone changes.
3. 2.6× of ~750ms = ~2s — order-of-magnitude under the 60s/req cap.

## Honest caveats

1. On this holdout (81 rows, qwen2.5:3b) the P1 lift is **zero**.
   Don't sell E5 as a P1 mover in the writeup; sell it as P3
   infrastructure + headroom-unlock for a stronger backbone.
2. The "vocab mismatch" check is heuristic. We accept any claim that
   shares ≥1 predicate with the premises; only a *fully* novel claim
   triggers retry. Conservative on purpose (false-positive retries
   are expensive); could be tuned per-prefix later.

## Tests

241 passed (was 234 → +6 translator + 1 redundant unused; net +6). ruff / mypy clean.

## Reproduction

```powershell
uv run pytest -q tests/unit/test_llm_translator.py
uv run python scripts/run_eval.py --task logic --with-llm \
    --split data/official_v20260515/eval_split/logic_eval.jsonl \
    --out outputs/eval/day22_v0515_logic_e5
```
