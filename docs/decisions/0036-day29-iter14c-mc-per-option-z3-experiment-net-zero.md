# 0036 — Day-29 Iter-14c — MC Per-Option Z3 experiment (net-zero, reverted)

Date: 2026-05-23
Status: Accepted (experiment closed; reverted to Iter-14b baseline)

## Context — user-directed Hướng-2 checklist item #3

The user's analysis listed three logic improvements for Iter-14:
1. ✓ Witness-aware skeptical existential (Iter-14a, +3.7pp)
2. ✓ Post-Z3 qualifier hard guard (Iter-14b, soundness-only)
3. **MC option support guard** — translate each option to FOL, Z3-verify
   independently, pick the uniquely entailed option

Pattern (per user):
```
Mỗi option cần có:
{ option, claim, supporting_premises, status:
  entailed | contradicted | unsupported, missing_conditions }

Nếu option được chọn là `entailed` nhưng không có supporting premise
rõ ràng → hạ confidence hoặc không chọn.
```

## Implementation (`_try_mc_per_option_z3` in pipeline.py)

For each MC option:
1. Translate the option text to FOL via the existing rejection-loop
   translator (`translate_question_to_fol_with_verify`).
2. Z3-verify the option's FOL against the premises (`verify_with_z3`).
3. Collect (label, fol, verdict) triples.

Decision logic:
- Exactly one entailed → pick that option (confidence 0.85).
- Multiple entailed → pick most specific (longest FOL string, confidence 0.55).
- All-but-one refuted → remaining wins (confidence 0.70).
- Otherwise return `None` → fall back to surface MC verifier.

Wired in `LogicPipeline.run` to fire BEFORE the generic
`_try_z3_fallback` for MC questions.

## Result on 81-row holdout

| | Iter-14b (no per-option) | Iter-14c (per-option) |
|---|---:|---:|
| Overall Logic | 28.4% | **28.4%** (=) |
| MC slice | 33.3% | **33.3%** (=) |
| MC latency | 2.5s/row | **7.6s/row** (3× slower) |
| MC predictions changed | — | **0** |

The function returned `None` on **ALL 15 MC rows** — fell back to
surface verifier every time, identical predictions.

## Root cause (trace on 12_q0)

```
Q: Based on the requirements, which statement about Sarah is correct?
A. She can take advanced classes because she has approval
B. She cannot take advanced classes due to insufficient completed courses
C. She is eligible but lacks approval
D. Her active status alone qualifies her
```

Per-option translation attempts:
- **A**: `has_approval(She) → can_take_advanced_classes(She)` →
  parser REJECTS (rule with arrow not in ground-atom form).
- **B**: `¬(∀c (completed_courses(c) → has_approval(c)) ∧ ∃c (¬completed_courses(c)))` →
  parser REJECTS (negation of complex formula).
- **C**: `∃x (eligible_advanced(x) ∧ ¬has_approval(x))` → parses,
  Z3 returns Unknown (predicates not in premise vocab in this form).
- **D**: `active_status(Her)` → parses, Z3 returns Unknown.

**Structural mismatch:** MC option texts are *compound reasoning
statements* ("X because Y", "X due to Z", "X but lacks Y") that
1. The 3B translator emits as universal rules with arrows, not ground
   atoms, OR
2. As negations of complex formulas the parser doesn't support, OR
3. As ground atoms with novel predicates that don't match premise vocab.

The per-option path is sound IN PRINCIPLE but the
3B-translator × simple-FOL-parser combo can't handle these compound
options on this dataset. Result: function always returns `None` →
identical to baseline.

## Decision

- **Revert** Iter-14c. Adds 5s/MC-row latency for **0** lift; not worth
  the code-surface cost (~115 lines).
- **Document the failure mode** so future iterations don't re-attempt
  the same shape without addressing one of:
  - Bigger/better LLM translator (7B/8B) that can keep option texts
    as ground atoms.
  - Parser extension for `body → head` ground rules + negation of
    conjunctions.
  - Few-shot prompt with MC-style option examples (closer to dataset
    distribution).
- **Architecture lessons learned** retained for future configs where
  the underlying components support compound option translation.

## State after Iter-14c revert

Tree returns exactly to Iter-14b: **Physics 46.0% / Logic 28.4% / 277
tests pass**. The MC slice stays at the 33.3% surface ceiling (5/15);
breaking past that requires LLM or parser upgrades, not pipeline
restructuring.

## Honest takeaway

The user's MC option support guard is conceptually right — each option
SHOULD be verified independently. But this dataset's MC options are
compound enough that they require a stronger translator + richer
parser to express them. With our current 3B + simple-FOL-grammar combo,
the implementation collapses to fall-back behavior every time.

This is the second "architecturally sound, empirically net-zero" iter
this session (after Iter-9's `_FORCE_FORMULAS` expansion). The pattern
is consistent: when a fix's net effect depends on downstream LLM
quality, and current LLM is weak, the fix doesn't show through. Future
SFT-7B or 8B-Deepseek upgrades will likely unlock both Iter-9b's
expanded force-formula guard AND Iter-14c's per-option MC path.
