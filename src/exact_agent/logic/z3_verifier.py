"""Z3 backend for the logic pipeline.

When the surface-form forward chainer can't prove a claim — typically
because it requires universal-quantifier instantiation across an entity
boundary the chainer doesn't see — we try the FOL-translated premises
through Z3.

Approach:

* Single ``Entity`` sort. Every constant that appears in any premise
  becomes a Z3 ``Const`` of that sort. Variables (lowercase, single-char)
  are quantifier-bound.
* Every distinct predicate becomes a Z3 ``Function(Entity, ..., Bool)``,
  arity inferred from its first occurrence.
* Universal rules ``∀x (body → head)`` are asserted with ``ForAll([x], …)``.
* Ground facts (``P(John)``, ``¬P(John)``) are asserted directly.
* The query is the parsed FOL form of the claim. We check
  ``solver.add(Not(claim))`` and ask for ``unsat`` — that's classical
  entailment.

The verifier returns one of ``"Yes"`` / ``"No"`` / ``"Unknown"`` plus the
premise IDs that participated in the assertion stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import z3

from exact_agent.logic.fol_parser import parse_fol
from exact_agent.logic.types import Atom, Comparison, Existential, Rule

_Item = "Atom | Comparison | Rule | Existential"
_Lit = "Atom | Comparison"


@dataclass(frozen=True)
class Z3VerificationResult:
    """Outcome of a Z3 entailment check."""

    verdict: str
    """Always one of ``"Yes"``, ``"No"``, ``"Unknown"``."""

    supports: tuple[int, ...]
    """1-based premise indices used to assert the theory."""

    rationale: str

    parsed_premises: int
    skipped_premises: int


def _bound_vars(item: Atom | Comparison | Rule | Existential) -> set[str]:
    if isinstance(item, (Rule, Existential)):
        return set(item.quantified_vars)
    return set()


def _is_variable(token: str) -> bool:
    """Heuristic for a free/bound logic variable: single lowercase letter."""
    return len(token) == 1 and token.islower()


def _lits(item: Atom | Comparison | Rule | Existential) -> tuple[Atom | Comparison, ...]:
    if isinstance(item, Rule):
        return (*item.body, item.head)
    if isinstance(item, Existential):
        return tuple(item.body)
    return (item,)


def _collect_constants(items: list[Atom | Comparison | Rule | Existential]) -> set[str]:
    constants: set[str] = set()
    for item in items:
        bound = _bound_vars(item)
        for lit in _lits(item):
            for arg in lit.args:
                if arg in bound or _is_variable(arg):
                    continue
                constants.add(arg)
    return constants


def _build_environment(
    parsed_items: list[Atom | Comparison | Rule | Existential],
    extras: list[Atom | Comparison | Rule | Existential] | None = None,
) -> tuple[
    z3.SortRef,
    dict[tuple[str, int], z3.FuncDeclRef],
    dict[tuple[str, int], z3.FuncDeclRef],
    dict[str, z3.ExprRef],
]:
    """Shared Entity sort + Bool predicates + Real comparison-functions +
    ground constants. ``extras`` (the parsed claim) is registered too so
    its symbols exist before asserting.

    Predicates / comparison funcs are keyed by ``(name, arity)``: a
    translator that emits ``Student(x)`` in one premise and ``Student(x,
    year)`` in another would otherwise crash Z3 ("index out of bounds") at
    apply time. Treating ``P/1`` and ``P/2`` as distinct funcdecls is
    sound (FOL permits this), keeps the solver running, and is the only
    practical option given imperfect upstream NL->FOL translation.
    """
    Entity = z3.DeclareSort("Entity")
    everything: list[Atom | Comparison | Rule | Existential] = list(parsed_items)
    if extras:
        everything.extend(extras)

    pred_arities: set[tuple[str, int]] = set()
    cmp_arities: set[tuple[str, int]] = set()
    for item in everything:
        for lit in _lits(item):
            if isinstance(lit, Comparison):
                cmp_arities.add((lit.func, len(lit.args)))
            else:
                pred_arities.add((lit.predicate, len(lit.args)))

    def _decl_name(name: str, arity: int, seen: dict[str, int]) -> str:
        # Z3 doesn't actually care if two FuncDecls share a Python label;
        # only the FuncDeclRef identity matters. We still suffix the
        # second+ arity so error messages and debug prints distinguish.
        seen[name] = seen.get(name, 0) + 1
        return name if seen[name] == 1 else f"{name}__a{arity}"

    seen_pred: dict[str, int] = {}
    predicates: dict[tuple[str, int], z3.FuncDeclRef] = {}
    for name, arity in sorted(pred_arities):
        label = _decl_name(name, arity, seen_pred)
        predicates[name, arity] = z3.Function(
            label, *([Entity] * max(arity, 0)), z3.BoolSort()
        )

    seen_cmp: dict[str, int] = {}
    cmp_funcs: dict[tuple[str, int], z3.FuncDeclRef] = {}
    for name, arity in sorted(cmp_arities):
        label = _decl_name(name, arity, seen_cmp)
        cmp_funcs[name, arity] = z3.Function(
            label, *([Entity] * max(arity, 0)), z3.RealSort()
        )

    constants_map: dict[str, z3.ExprRef] = {
        name: z3.Const(name, Entity) for name in sorted(_collect_constants(everything))
    }
    return Entity, predicates, cmp_funcs, constants_map


_OP_FN = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def _args_to_z3(
    raw_args: tuple[str, ...],
    *,
    Entity: z3.SortRef,
    constants_map: dict[str, z3.ExprRef],
    binders: dict[str, z3.ExprRef] | None,
) -> list[z3.ExprRef]:
    out: list[z3.ExprRef] = []
    for raw in raw_args:
        if binders is not None and raw in binders:
            out.append(binders[raw])
        elif _is_variable(raw):
            out.append(z3.Const(f"_free_{raw}", Entity))
        else:
            if raw not in constants_map:
                constants_map[raw] = z3.Const(raw, Entity)
            out.append(constants_map[raw])
    return out


def _lit_to_z3(
    lit: Atom | Comparison,
    *,
    Entity: z3.SortRef,
    predicates: dict[tuple[str, int], z3.FuncDeclRef],
    cmp_funcs: dict[tuple[str, int], z3.FuncDeclRef],
    constants_map: dict[str, z3.ExprRef],
    binders: dict[str, z3.ExprRef] | None = None,
) -> z3.BoolRef:
    arity = len(lit.args)
    if isinstance(lit, Comparison):
        fn = cmp_funcs[lit.func, arity]
        args = _args_to_z3(lit.args, Entity=Entity, constants_map=constants_map, binders=binders)
        lhs = fn(*args) if args else fn()
        return _OP_FN[lit.op](lhs, z3.RealVal(lit.value))

    func = predicates[lit.predicate, arity]
    args = _args_to_z3(lit.args, Entity=Entity, constants_map=constants_map, binders=binders)
    expr = func(*args) if args else func()
    return z3.Not(expr) if lit.polarity == "neg" else expr


def _rule_to_z3(
    rule: Rule,
    *,
    Entity: z3.SortRef,
    predicates: dict[tuple[str, int], z3.FuncDeclRef],
    cmp_funcs: dict[tuple[str, int], z3.FuncDeclRef],
    constants_map: dict[str, z3.ExprRef],
) -> z3.BoolRef:
    binders = {v: z3.Const(v, Entity) for v in rule.quantified_vars}
    body_terms = [
        _lit_to_z3(
            a,
            Entity=Entity,
            predicates=predicates,
            cmp_funcs=cmp_funcs,
            constants_map=constants_map,
            binders=binders,
        )
        for a in rule.body
    ]
    head_term = _lit_to_z3(
        rule.head,
        Entity=Entity,
        predicates=predicates,
        cmp_funcs=cmp_funcs,
        constants_map=constants_map,
        binders=binders,
    )
    body_conj = body_terms[0] if len(body_terms) == 1 else z3.And(*body_terms)
    return z3.ForAll(list(binders.values()), z3.Implies(body_conj, head_term))


def _existential_to_z3(
    ex: Existential,
    *,
    Entity: z3.SortRef,
    predicates: dict[tuple[str, int], z3.FuncDeclRef],
    cmp_funcs: dict[tuple[str, int], z3.FuncDeclRef],
    constants_map: dict[str, z3.ExprRef],
) -> z3.BoolRef:
    """Iter-9: encode ``∃x_1 … x_n (a(...) ∧ b(...) ∧ …)`` as a z3.Exists.

    Soundness note: Z3 handles ∃-claims and ∃-premises in the standard
    way. The crucial behaviour for 42_q1-style problems is that having
    two SEPARATE ∃ premises ``∃x(P(x))`` and ``∃x(Q(x))`` does NOT
    entail ``∃x(P(x) ∧ Q(x))`` — different witnesses can satisfy each.
    Z3 will correctly return ``sat`` for the ``premises ∧ ¬claim`` check
    (claim is not entailed) AND ``sat`` for ``premises ∧ claim`` (claim
    is not refuted either) → verdict = Unknown for ambiguous existentials.
    For the case where premises explicitly provide a witness for the
    conjunction, Z3 returns the right entailment.
    """
    binders = {v: z3.Const(v, Entity) for v in ex.quantified_vars}
    body_terms = [
        _lit_to_z3(
            a,
            Entity=Entity,
            predicates=predicates,
            cmp_funcs=cmp_funcs,
            constants_map=constants_map,
            binders=binders,
        )
        for a in ex.body
    ]
    body_conj = body_terms[0] if len(body_terms) == 1 else z3.And(*body_terms)
    return z3.Exists(list(binders.values()), body_conj)


def _parse_premises(
    premises_fol: list[str],
) -> tuple[list[Atom | Comparison | Rule | Existential], list[int], list[int]]:
    """Parse every FOL premise; return (parsed, parsed_idx, skipped_idx)."""
    parsed: list[Atom | Comparison | Rule | Existential] = []
    parsed_idx: list[int] = []
    skipped_idx: list[int] = []
    for i, raw in enumerate(premises_fol, start=1):
        item = parse_fol(raw, premise_id=f"P{i}")
        if item is None:
            skipped_idx.append(i)
            continue
        parsed.append(item)
        parsed_idx.append(i)
    return parsed, parsed_idx, skipped_idx


def verify_with_z3(
    premises_fol: list[str],
    claim_fol: str,
    *,
    timeout_ms: int = 4000,
) -> Z3VerificationResult:
    """Check whether ``claim_fol`` is entailed by ``premises_fol``.

    Strategy:

    * ``Yes`` ⇔ ``premises ∧ ¬claim`` is unsat.
    * ``No``  ⇔ ``premises ∧ claim`` is unsat.
    * ``Unknown`` for everything else (including timeout, parse failure).
    """
    parsed_claim = parse_fol(claim_fol, premise_id="QUERY")
    if parsed_claim is None:
        return Z3VerificationResult(
            verdict="Unknown",
            supports=(),
            rationale=f"Claim FOL did not parse: {claim_fol!r}",
            parsed_premises=0,
            skipped_premises=len(premises_fol),
        )

    parsed_items, parsed_idx, skipped_idx = _parse_premises(premises_fol)
    Entity, predicates, cmp_funcs, constants_map = _build_environment(
        parsed_items, extras=[parsed_claim]
    )

    def _encode(item: Atom | Comparison | Rule | Existential) -> z3.BoolRef:
        if isinstance(item, Rule):
            return _rule_to_z3(
                item,
                Entity=Entity,
                predicates=predicates,
                cmp_funcs=cmp_funcs,
                constants_map=constants_map,
            )
        if isinstance(item, Existential):
            return _existential_to_z3(
                item,
                Entity=Entity,
                predicates=predicates,
                cmp_funcs=cmp_funcs,
                constants_map=constants_map,
            )
        return _lit_to_z3(
            item,
            Entity=Entity,
            predicates=predicates,
            cmp_funcs=cmp_funcs,
            constants_map=constants_map,
        )

    def assert_theory(solver: z3.Solver) -> None:
        for item in parsed_items:
            solver.add(_encode(item))

    def encode_claim() -> z3.BoolRef:
        return _encode(parsed_claim)

    claim_expr = encode_claim()
    supports = tuple(parsed_idx)

    # Test entailment: theory ∪ ¬claim → unsat?
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    assert_theory(solver)
    solver.add(z3.Not(claim_expr))
    entails = solver.check()

    # Test refutation: theory ∪ claim → unsat?
    refute = z3.Solver()
    refute.set("timeout", timeout_ms)
    assert_theory(refute)
    refute.add(claim_expr)
    refutes = refute.check()

    if entails == z3.unsat and refutes != z3.unsat:
        verdict = "Yes"
        rationale = "premises ⊨ claim (¬claim is unsat)"
    elif refutes == z3.unsat and entails != z3.unsat:
        verdict = "No"
        rationale = "premises ⊨ ¬claim (claim is unsat)"
    elif entails == z3.unsat and refutes == z3.unsat:
        verdict = "Unknown"
        rationale = "premises are inconsistent — both claim and ¬claim are unsat"
    else:
        verdict = "Unknown"
        rationale = (
            f"Z3 could not decide entailment "
            f"(entails={entails}, refutes={refutes}, parsed={len(parsed_items)}, "
            f"skipped={len(skipped_idx)})"
        )

    return Z3VerificationResult(
        verdict=verdict,
        supports=supports,
        rationale=rationale,
        parsed_premises=len(parsed_items),
        skipped_premises=len(skipped_idx),
    )
