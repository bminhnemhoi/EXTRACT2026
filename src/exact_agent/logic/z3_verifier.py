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
from exact_agent.logic.types import Atom, Rule


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


def _is_variable(token: str) -> bool:
    """Heuristic: lowercase, length 1 → bound variable. Anything else → constant."""
    return len(token) == 1 and token.islower()


def _collect_constants(items: list[Atom | Rule]) -> set[str]:
    constants: set[str] = set()

    def add_atom(atom: Atom) -> None:
        for arg in atom.args:
            if not _is_variable(arg):
                constants.add(arg)

    for item in items:
        if isinstance(item, Atom):
            add_atom(item)
        else:  # Rule
            for body_atom in item.body:
                add_atom(body_atom)
            add_atom(item.head)
    return constants


def _arity_for(predicate: str, items: list[Atom | Rule]) -> int:
    """Infer predicate arity from the first occurrence in any premise."""
    for item in items:
        atoms = (item,) if isinstance(item, Atom) else (*item.body, item.head)
        for atom in atoms:
            if atom.predicate == predicate:
                return len(atom.args)
    return 1


def _build_environment(
    parsed_items: list[Atom | Rule],
    extras: list[Atom | Rule] | None = None,
) -> tuple[z3.SortRef, dict[str, z3.FuncDeclRef], dict[str, z3.ExprRef]]:
    """Create the shared sort + predicate functions + ground constants.

    ``extras`` is the (parsed) claim plus any auxiliaries; we want to make
    sure their predicates and constants are also registered before
    asserting.
    """
    Entity = z3.DeclareSort("Entity")
    everything: list[Atom | Rule] = list(parsed_items)
    if extras:
        everything.extend(extras)

    # Predicates → Z3 functions.
    predicates: dict[str, z3.FuncDeclRef] = {}
    seen_preds: set[str] = set()
    for item in everything:
        atoms = (item,) if isinstance(item, Atom) else (*item.body, item.head)
        for atom in atoms:
            seen_preds.add(atom.predicate)
    for predicate in sorted(seen_preds):
        arity = _arity_for(predicate, everything)
        predicates[predicate] = z3.Function(predicate, *([Entity] * arity), z3.BoolSort())

    # Constants → Z3 consts.
    constants_map: dict[str, z3.ExprRef] = {}
    for name in sorted(_collect_constants(everything)):
        constants_map[name] = z3.Const(name, Entity)

    return Entity, predicates, constants_map


def _atom_to_z3(
    atom: Atom,
    *,
    Entity: z3.SortRef,
    predicates: dict[str, z3.FuncDeclRef],
    constants_map: dict[str, z3.ExprRef],
    binders: dict[str, z3.ExprRef] | None = None,
) -> z3.BoolRef:
    func = predicates[atom.predicate]
    args: list[z3.ExprRef] = []
    for raw in atom.args:
        if _is_variable(raw) and binders is not None and raw in binders:
            args.append(binders[raw])
        elif _is_variable(raw):
            # Free variable — promote to a fresh constant (rare; normally
            # bound by the rule). This keeps the solver from crashing on
            # malformed inputs.
            args.append(z3.Const(f"_free_{raw}", Entity))
        else:
            if raw not in constants_map:
                constants_map[raw] = z3.Const(raw, Entity)
            args.append(constants_map[raw])
    expr = func(*args) if args else func()
    return z3.Not(expr) if atom.polarity == "neg" else expr


def _rule_to_z3(
    rule: Rule,
    *,
    Entity: z3.SortRef,
    predicates: dict[str, z3.FuncDeclRef],
    constants_map: dict[str, z3.ExprRef],
) -> z3.BoolRef:
    binder = z3.Const(rule.quantified_var, Entity)
    binders = {rule.quantified_var: binder}
    body_terms = [
        _atom_to_z3(
            a,
            Entity=Entity,
            predicates=predicates,
            constants_map=constants_map,
            binders=binders,
        )
        for a in rule.body
    ]
    head_term = _atom_to_z3(
        rule.head,
        Entity=Entity,
        predicates=predicates,
        constants_map=constants_map,
        binders=binders,
    )
    body_conj = body_terms[0] if len(body_terms) == 1 else z3.And(*body_terms)
    return z3.ForAll([binder], z3.Implies(body_conj, head_term))


def _parse_premises(premises_fol: list[str]) -> tuple[list[Atom | Rule], list[int], list[int]]:
    """Parse every FOL premise; return (parsed, parsed_idx, skipped_idx)."""
    parsed: list[Atom | Rule] = []
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
    Entity, predicates, constants_map = _build_environment(parsed_items, extras=[parsed_claim])

    def assert_theory(solver: z3.Solver) -> None:
        for item in parsed_items:
            if isinstance(item, Atom):
                solver.add(
                    _atom_to_z3(
                        item,
                        Entity=Entity,
                        predicates=predicates,
                        constants_map=constants_map,
                    )
                )
            else:
                solver.add(
                    _rule_to_z3(
                        item,
                        Entity=Entity,
                        predicates=predicates,
                        constants_map=constants_map,
                    )
                )

    def encode_claim() -> z3.BoolRef:
        if isinstance(parsed_claim, Atom):
            return _atom_to_z3(
                parsed_claim,
                Entity=Entity,
                predicates=predicates,
                constants_map=constants_map,
            )
        return _rule_to_z3(
            parsed_claim,
            Entity=Entity,
            predicates=predicates,
            constants_map=constants_map,
        )

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
