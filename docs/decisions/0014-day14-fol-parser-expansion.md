# 0014 — Day-14 FOL parser/Z3 expansion: P3 capability, P1 unchanged

Date: 2026-05-20
Status: Accepted

## Context

ADR 0005 flagged that ~52% of dataset FOL strings don't parse (the
single-`∀x` regex). The hypothesis: expanding the parser → more Z3
proofs → higher logic P1. We measured *before* over-investing.

## What was built

- `logic/types.py`: `Comparison` literal (`f(args) OP value`); `Rule`
  is now multi-variable (`quantified_vars: tuple[str,...]`, with a
  back-compat `quantified_var` property).
- `logic/fol_parser.py`: rewritten — peels nested/multi-var universals
  in **both** syntaxes (`∀x (∀y …)`, `∀x ∀y …`,
  `ForAll(a, ForAll(b, ForAll(c, …)))`), parses comparison literals
  (`≥ ≤ ≠ > < =`, ascii too) as body conjuncts, heads, or bare facts.
- `logic/z3_verifier.py`: multi-variable `ForAll([…])`; comparison
  literals modelled as **Real-valued** Z3 functions with arithmetic
  constraints; Bool predicates and Real cmp-funcs in separate
  namespaces.
- +10 unit tests (multi-var transitivity, arithmetic thresholds,
  ascii ops, back-compat). Suite 216 → **226**, ruff/mypy clean.

Parser coverage on the eval FOL set: 48% → **51%** (Rule 283→311,
+Comparison). The rest is ∃/Exists (96, deliberately out of scope —
existentials need different handling) and disjunction.

## The measured result — stated plainly

Logic eval, expanded parser, **same** qwen2.5:3b translator as the
37.5% baseline:

| | baseline | expanded parser |
|---|---:|---:|
| Logic correct | 37.5% (24/64) | **37.5% (24/64)** |

Sample-level: **0 regress, 0 gain**. The parser expansion produced
**zero measured logic-P1 change.**

Root cause (confirmed, not speculated): the Z3 fallback only fires with
a `claim_FOL`, whose only source is the weak qwen2.5:3b NL→FOL
translator; and of the 21 Z3-eligible rows only ~4 have a fully
parseable premise theory (the other ~15 mix ∃ premises → incomplete
theory → Z3 cannot soundly decide). Parser quality was never the
binding constraint — `claim_FOL` is.

## Decision

**Keep the expansion** (committed) but for its real value, not P1:

1. **P3 (the finals-decisive axis).** The rubric explicitly rewards
   FOL inferences / structured evidence, and the top-10 finals are
   judges evaluating reasoning depth live. The system can now emit
   *verifiable* multi-step Z3 proofs — transitivity chains,
   arithmetic-threshold entailments — instead of only single-rule
   modus ponens. That is exactly the neuro-symbolic story the
   organizers (URA, neuro-symbolic XAI workshop) reward, and it is
   correct, tested, and reproducible.
2. **Future-proofing.** When a stronger translator lands (better SFT,
   or a deterministic claim builder), Z3 must *already* handle these
   shapes or the better claims still won't decide. The capability is a
   prerequisite, not dead code.
3. **Do not invest further in logic P1 this cycle.** The binding
   constraint is `claim_FOL` generation, which SFT (ADR 0013) did not
   improve and which a reliable deterministic builder is itself a
   research problem. Logic stays ~37.5%; that is an honest, measured
   ceiling for the available translator.

Pivot to **#2 (physics vector composition)** — it has verified,
countable Stage-1 P1 rows (LD/DT midpoint + perpendicular-bisector,
closed forms validated against real golds: LD022→14.4 N, LD035→0.36 N,
LD219→1.71e-3 N).

## Why this is the right call (not a failure)

Baseline-first (ADR 0001) exists precisely to surface this: we spent
~half a day proving, with data, that the logic lever is blocked
upstream of the parser — instead of spending two days on a parser +
claim-builder gamble that the measurement now shows would not have
moved Stage-1. The capability gained is genuinely valuable for P3 /
finals; the P1 expectation was falsified early and cheaply.

## Reproduction

```powershell
uv run pytest -q tests/unit/test_fol_parser.py tests/unit/test_z3_verifier.py
uv run python scripts/run_eval.py --task logic --with-llm --out outputs/eval/day14_fol
# vs outputs/eval/day8_llm/ (baseline) — identical 24/64
```
