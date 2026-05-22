# 0034 — Day-28 Iter-8 — Unit-Sanity Guard experiment (net-zero; reverted)

Date: 2026-05-22 (Day 28 calendar; Iter-7 baseline = 46.0% / 24.7%)
Status: Accepted (experiment closed; code reverted to Iter-7)

## Context — the user's architectural feedback

After Iter-7 landed at Physics 46.0% / Logic 24.7% I reported the ceiling
analysis claiming Physics 70% needs SFT-7B + multi-step planner + symbolic
infrastructure. The user pushed back with a sharper framing:

> Hệ thống của bạn hiện đang collapse "Understanding" + "Solving" thành
> 1 bước, không có lớp validate. Cách giải quyết đúng là: ưu tiên đọc đề
> + xác định vai trò dữ kiện + kiểm tra đơn vị, không phải chỉ đưa LLM
> mạnh hơn. Hãy thêm Problem Understanding + Role-aware Extraction +
> Unit Sanity Check + State-change Detector trước khi chốt cần SFT.

This is the **Mixture-of-Refinement-Agents** pattern (MoRA, arXiv
2412.00821) and the NSVIF formulation-checking split (arXiv 2511.09008).

I agreed and ran a focused experiment on **just the Unit Sanity Check**
piece — operationalized as: "infer the dimensionality the question is
asking for, reject any formula whose declared output unit can't reach
that dimensionality". This is the user's "nhìn đơn vị để biết mình có
đi sai đường không".

## Hypothesis (from `scripts/analyze_iter7_failures.py`)

The Iter-7 fail dump on the 163-row holdout shows **20 wrong rows with
predicted-unit-vs-gold-unit DIMENSION MISMATCH** (23% of wrong rows).
Top pattern: force-output formula (newton) won routing on a question
whose gold is electric field (V/m). The named-set guard `_FORCE_FORMULAS`
was missing 4 of the 9 force-output formulas, silently bypassing the
field-vs-force check.

Predicted ceiling: +20pp if we cleanly redirect each mismatched row to
a dimension-compatible formula.

## What was implemented (5 progressive variants)

| Variant | Mechanism | Result |
|---|---|---:|
| 8a | Expand `_FORCE_FORMULAS` + naive intent dim guard (first-match) | 38.7% (**−7.3pp**) |
| 8b | + longest-match instead of first-match | 41.1% (−4.9pp) |
| 8c | + tighten dim hints to verb-prefixed forms only, add `power factor` | 44.8% (−1.2pp) |
| 8d | + rightmost-occurrence anchor for compound questions | 44.8% (−1.2pp) |
| 8e | + intent dim guard supersedes legacy `_is_field_asking` | 43.6% (−2.4pp) |
| 8-final | Revert dim guard, keep only `_FORCE_FORMULAS` expansion | 44.8% (−1.2pp) |
| **8-revert** | **Full revert to Iter-7 baseline** | **46.0% (≡ baseline)** |

Best variant of 5 attempts produced **−1.2pp net regression**. The trade-off
that appeared in every configuration:

- Expanding the force-formula set **fixes** LD052/053/056/394 (field questions
  that wrongly route to a force formula → +4 rows on the LD slice)
- Expanding it **breaks** DT005/006 (compound questions whose SETUP mentions
  "field strength" but whose ACTUAL ask in the last sentence is "calculate
  the electric force" → −2 rows on the DT slice)
- Plus extractor-LLM variance shifts ~2-3 other rows in unpredictable ways

The dim guard then introduces second-order regressions:
- Intent inference via substring matching has many false positives
  (`"voltage across it is 150 V"` looks like a voltage-intent phrase but
  is actually data)
- Longest-match and rightmost-anchor each fix one class while breaking
  another
- The guard ALWAYS rejects valid formulas when intent inference is wrong

## Root cause of the experiment failure

**Naive substring matching cannot reliably infer question intent on
compound multi-sentence physics prose.** The Vietnamese physics dataset
routinely intermixes setup statements ("there is a field strength at
point M") with the actual ask ("calculate the force on q3 at M"), and
the noun phrases for both appear in the same prompt.

A correct implementation would need:
1. Sentence-level segmentation (split on `.`, `?`, imperative verbs)
2. Identify the LAST sentence containing an imperative verb (calculate,
   find, what is, determine)
3. Run intent inference ONLY on that sentence

This is doable but requires either a small NLP layer or a dedicated LLM
call per row — neither feasible in the time budget for this sprint.

## Decision

- **Keep**: `scripts/analyze_iter7_failures.py` — the per-sample root-cause
  classifier that surfaced the 20-row dimension-mismatch hypothesis is
  reusable for future audits.
- **Revert**: every line of `topic_classifier.py` / `pyproject.toml` /
  `test_topic_classifier.py` touched by this experiment. Tree returns
  exactly to Iter-7 (271 tests, ruff/mypy clean, 46.0% / 24.7%).
- **Document for future**: the right next attempt is a sentence-segmented
  intent inferrer (not a naive substring search). Park as ADR 0034 NOTE.

## Honest reflection

The user's architectural framing is correct — collapsing
Understanding + Solving is the root design flaw. But the FIRST attempt
at decoupling them via this specific Unit-Sanity Guard didn't pay off
because substring-based intent inference is too noisy on this dataset.
The user's broader recommendation (semantic frame, role-aware extraction,
state-change as first-class) remains the right north star; this ADR
documents one tactical detour that didn't move the needle.

Net session result: **Iter-7 ceiling 46.0% / 24.7% confirmed reproducible**;
no improvement from Iter-8. Best move forward: USER-side Day-26 SFT-7B
retrain (script ready, +6.2pp projected), then ship submit-ready at the
Iter-7 numbers.
