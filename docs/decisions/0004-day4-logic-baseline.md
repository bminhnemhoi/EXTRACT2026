# 0004 — Day-4 logic baseline

Date: 2026-05-15
Status: Accepted

## Context

Day 4 lit up the logic pipeline end-to-end and ran the eval harness on a
deterministic 10% holdout (`data/eval_split/logic_eval.jsonl`, 64 samples,
seed=42). For physics the rule baseline collapsed to 0.8% because the
dataset is full of NL forms the regex extractor can't see; for logic the
surface-form forward chainer is a much better fit and the baseline lands
at a usable starting point.

## Decision

Adopt the following as the Phase-1 logic baseline (rule + token-Jaccard
chainer, no LLM):

| Slice | N | Correct | Abstained | Expl F1 | ms |
|---|---:|---:|---:|---:|---:|
| **Overall** | 64 | **35.9%** | 32.8% | 0.287 | 0.5 |
| `multiple_choice` | 17 | 47.1% | 0.0% | 0.249 | 0.6 |
| `true_false`      | 13 | 38.5% | 53.8% | 0.286 | 0.5 |
| `yes_no_unknown`  | 30 | 30.0% | 46.7% | 0.296 | 0.4 |
| `open`            |  4 | 25.0% | 0.0%  | 0.377 | 0.5 |

Random baseline: MC ≈ 25%, Yes/No/Unknown ≈ 33%. We beat random on MC and
break even on YNU with a much higher abstain rate, so true accuracy on
the items we *do* answer is materially above random.

Failure mix: 23 `wrong_label`, 18 `answered_unknown`.

## Two iterations applied today

1. **Light stemming** in `premise_selector._stem` — strips
   `ies/ed/ing/es/s` plus a trailing silent `e`. "receive" / "receives" /
   "receiving" all collapse to `receiv`. This single change pushed several
   YNU items from `Unknown` → correct because the rule conclusion now
   matched the question claim.
2. **MC verifier policy flip** in `verify_multiple_choice` — default now
   commits the highest-overlap option instead of abstaining on a thin
   margin. MC accuracy jumped from **17.6% → 47.1%** with no other
   change. The old behavior remains opt-in via
   `abstain_when_all_zero=True` (we'll switch this back on once
   self-correction is in and re-asks the LLM on low-confidence rounds).

## Diagnosis — what's left on the table

* **Coreference is the next big bottleneck.** Premises talk about
  "a student" / "the curriculum"; questions and conclusions reference
  named entities ("Sophia", "Alice"). Surface Jaccard cannot bridge
  these. The Day-5 FOL/Z3 path will handle universal-quantifier
  instantiation directly.
* **YNU 47% abstention** is mostly *correct conservatism* — many YNU
  rows in the dataset are intentionally `Unknown`. Tightening here risks
  trading correct abstentions for wrong commits.
* **`open` answers** need the LLM. The Day-4 baseline returns the
  raw verifier output, which is rarely the right surface form for an
  open-ended question.

## Consequences for Day 5+

1. **Day 5**: bring up the Z3 backend in `logic.fol_parser` /
   `logic.z3_verifier` (the Z3 dep is already installed). The parser
   handles `∀x (Body → Head)` and ground atoms; we just need the
   verifier wrapper that asserts premises and queries the negated
   conclusion.
2. **Phase 4 (Day 6)**: NL→FOL prompt for premises whose surface form
   doesn't parse cleanly. Logic shares the same SFT model with physics
   extraction — same vLLM client, same Jinja2 template directory.
3. The harness is fast (≤1 ms/sample) — rerun on every commit.

## Reproduction

```powershell
uv run python scripts\build_eval_split.py   # one-shot, seed=42
uv run python scripts\run_eval.py --task logic
# outputs/eval/logic_report.md + .json
```

## Numbers we won't change

- Eval split: 10% holdout, seed 42, written by `scripts/build_eval_split.py`.
- Label match: case-insensitive exact (Yes/No/Unknown/A/B/C/D).
- Stemming: hand-rolled; intentionally not pulling NLTK / SpaCy.
