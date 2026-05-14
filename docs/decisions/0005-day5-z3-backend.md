# 0005 — Day-5 Z3 entailment backend

Date: 2026-05-15
Status: Accepted

## Context

The Day-4 logic baseline closed at 35.9% with two clear failure modes:

* **Coreference & quantifier instantiation**. Premises talk about
  "a student"; questions talk about "Sophia". Surface-form Jaccard cannot
  prove the bridge; the conclusion lands as `Unknown`.
* **53% of FOL strings don't parse** with the current single-variable
  `∀x` regex (the dataset uses nested `∀x (ForAll(d, …))` for
  multi-arity rules).

The cleaned dataset already ships `premises_FOL_clean` for every logic
row, so Z3 doesn't have to wait for the LLM to translate premises — only
to translate the question into a `claim_FOL`. That makes Z3 an obvious
Day-5 win.

## Decision

Land an SMT entailment backend in `logic.z3_verifier`:

* One Z3 sort `Entity`. Constants from premises become `Const`s.
* Each predicate becomes a `Function(Entity, …, Bool)`, arity inferred
  from the first occurrence.
* Universal rules → `ForAll([x], Implies(body, head))`.
* Ground facts asserted directly.
* Verdict policy:
  * `Yes` ⇔ premises ∧ ¬claim is **unsat**
  * `No` ⇔ premises ∧ claim is **unsat**
  * `Unknown` everywhere else (including timeouts, parse failure)

Wire it into `LogicPipeline` as a *fallback* — only consulted when the
surface verifier returned `Unknown` (or a low-confidence Yes/No), and
the request carries both `premises-FOL` and `claim-FOL`. This keeps the
pipeline deterministic for the easy cases and only pays the SMT cost
when it might pay off.

## What ships today

* `src/exact_agent/logic/z3_verifier.py` — `verify_with_z3(premises, claim) → Yes/No/Unknown`
* `LogicPipeline._try_z3_fallback` — gating policy (no `claim_FOL` →
  no Z3 call; confident surface answer → no Z3 call).
* New schema field `claim_FOL` (alias `claim-FOL`) on `PredictRequest`.
* Tests:
  * 6 unit tests on `verify_with_z3` (modus ponens, two-step chain,
    contradiction → No, unrelated → Unknown, unparseable premises
    skipped, unparseable claim → Unknown).
  * 5 unit tests on `parse_atom` / `parse_fol`.
  * 1 integration test proving the pipeline actually invokes the
    fallback and recovers a `Yes` where the surface chain can't.

## What we measured (and what we didn't)

The eval split (`logic_eval.jsonl`) contains `premises_FOL_clean` but
no `claim_FOL`. Without a question→FOL translator, the fallback
gating drops every row, so the **eval number is unchanged: 35.9%**.

That's expected. The Z3 path is a *capability*, not a measurement, until
Phase 4 lands an LLM translator that can produce `claim_FOL` from the
NL question. Once it does, every Yes/No/True-False row that currently
abstains is a candidate.

## Consequences for Phase 4

1. The vLLM client + Jinja2 prompt for NL→FOL question translation
   becomes the next high-leverage piece. Reuse `configs/prompts/nl_to_fol.j2`
   (already in the repo) with a small adaptation for the question side.
2. Eval harness extension: pass `claim_FOL = llm.translate(question)`
   into the request and re-measure. The plumbing is in place.
3. **Multi-variable parsing**: the parser still rejects nested
   `∀x (ForAll(d, …))` constructs. Either extend the regex or have
   the LLM normalize during translation. The Z3 verifier is agnostic
   to which.

## Reproduction

```powershell
# unit tests
uv run pytest tests/unit/test_z3_verifier.py -v
# integration: surface chain → Unknown, Z3 → Yes
uv run pytest tests/integration/test_logic_pipeline.py::TestLogicPipelineEndToEnd::test_z3_fallback_resolves_coreference -v
```
