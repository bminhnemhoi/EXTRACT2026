"""Surface-level forward chainer over parsed premises.

Each iteration scans rules whose ``conditions`` already match the working
set of facts (approximate Jaccard match). Newly-derived conclusions join
the facts, and the next iteration fires any rule that was unlocked. The
loop stops at a fixpoint or a configurable iteration cap.

The trace records which premise IDs supported each derivation, so the
explanation can populate the ``premises`` field of the API response.

Day-4 caveat: surface-form chaining cannot resolve coreference
("they"/"he"/"she" → entity) or universal-quantifier instantiation. That
work belongs to the FOL/Z3 path (Day 5 / Phase 4). The Jaccard threshold
is deliberately permissive — we'd rather over-chain than refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exact_agent.logic.premise_selector import _normalize as tokenize
from exact_agent.logic.rule_parser import Fact, Rule


def _token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class DerivedFact:
    """A fact in the working set, with the premise IDs that supported it."""

    text: str
    tokens: frozenset[str]
    supports: tuple[int, ...]
    """1-based ``P{i}`` premise indices."""


@dataclass
class ChainResult:
    facts: list[DerivedFact] = field(default_factory=list)
    iterations: int = 0
    converged: bool = True

    def matches(self, query_text: str, *, threshold: float = 0.5) -> DerivedFact | None:
        """Return the highest-overlap fact matching ``query_text``, or None."""
        q = _token_set(query_text)
        if not q:
            return None
        best: DerivedFact | None = None
        best_score = threshold
        for fact in self.facts:
            score = _jaccard(q, fact.tokens)
            if score >= best_score:
                best = fact
                best_score = score
        return best

    def entails(self, query_text: str, *, threshold: float = 0.5) -> bool:
        return self.matches(query_text, threshold=threshold) is not None

    def all_supports(self) -> tuple[int, ...]:
        seen: list[int] = []
        for fact in self.facts:
            for p in fact.supports:
                if p not in seen:
                    seen.append(p)
        return tuple(seen)


def _to_one_based(zero_based: int) -> int:
    """Premise indices in the dataset are 0-based; explanations use ``P{i+1}``."""
    return zero_based + 1


def _initial_facts(facts: list[Fact]) -> list[DerivedFact]:
    derived: list[DerivedFact] = []
    for fact in facts:
        tokens = _token_set(fact.text)
        if not tokens:
            continue
        # Deduplicate by token-set equality.
        if any(d.tokens == tokens for d in derived):
            continue
        derived.append(
            DerivedFact(
                text=fact.text,
                tokens=tokens,
                supports=(_to_one_based(fact.premise_index),),
            )
        )
    return derived


def _conditions_satisfied(
    rule: Rule, facts: list[DerivedFact], threshold: float
) -> tuple[bool, tuple[int, ...]]:
    supports: list[int] = []
    for cond in rule.conditions:
        cond_tokens = _token_set(cond)
        if not cond_tokens:
            return False, ()
        best_score = threshold
        best_fact: DerivedFact | None = None
        for fact in facts:
            score = _jaccard(cond_tokens, fact.tokens)
            if score >= best_score:
                best_score = score
                best_fact = fact
        if best_fact is None:
            return False, ()
        for sid in best_fact.supports:
            if sid not in supports:
                supports.append(sid)
    return True, tuple(supports)


def forward_chain(
    rules: list[Rule],
    facts: list[Fact],
    *,
    max_iterations: int = 50,
    threshold: float = 0.5,
) -> ChainResult:
    """Run forward chaining over ``rules`` and seed ``facts`` to fixpoint."""
    derived = _initial_facts(facts)

    for iteration in range(1, max_iterations + 1):
        new_facts: list[DerivedFact] = []
        for rule in rules:
            ok, supports = _conditions_satisfied(rule, derived, threshold)
            if not ok:
                continue
            concl_tokens = _token_set(rule.conclusion)
            if not concl_tokens:
                continue
            if any(_jaccard(concl_tokens, f.tokens) >= 0.9 for f in derived):
                continue
            full_supports = tuple(dict.fromkeys((*supports, _to_one_based(rule.premise_index))))
            new_facts.append(
                DerivedFact(
                    text=rule.conclusion,
                    tokens=concl_tokens,
                    supports=full_supports,
                )
            )
        if not new_facts:
            return ChainResult(facts=derived, iterations=iteration, converged=True)
        derived.extend(new_facts)

    return ChainResult(facts=derived, iterations=max_iterations, converged=False)
