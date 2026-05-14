"""Data model shared by the logic pipeline.

We deliberately keep this small and Pydantic-free: forward chaining hot
loops dozens of times per question, and we want plain dataclasses with no
validation overhead. The Pydantic schemas in :mod:`exact_agent.schemas`
live at the API boundary, not in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Polarity = Literal["pos", "neg"]


@dataclass(frozen=True)
class Atom:
    """A (possibly negated) predicate applied to constants or variables.

    Example: ``can_propose_courses(John)`` → ``Atom("can_propose_courses",
    ("John",), polarity="pos")``. The single-variable case from
    ``∀x P(x)`` is represented with ``args = ("x",)``.
    """

    predicate: str
    args: tuple[str, ...]
    polarity: Polarity = "pos"

    def negate(self) -> Atom:
        return Atom(self.predicate, self.args, "neg" if self.polarity == "pos" else "pos")

    def is_ground(self) -> bool:
        # Constants are capitalized in the dataset (John, Sophia, Alice); a
        # variable is a single lowercase letter (x, y, c).
        return all(not arg.islower() or len(arg) > 1 for arg in self.args)

    def substitute(self, var: str, value: str) -> Atom:
        new_args = tuple(value if a == var else a for a in self.args)
        return Atom(self.predicate, new_args, self.polarity)

    def __str__(self) -> str:
        sign = "" if self.polarity == "pos" else "¬"
        return f"{sign}{self.predicate}({', '.join(self.args)})"


@dataclass(frozen=True)
class Rule:
    """A universally quantified Horn-ish rule.

    Bodies are conjunctions of atoms; head is a single atom (possibly
    negated). Disjunctive heads are uncommon in the dataset and ignored
    by the Day-4 baseline.
    """

    quantified_var: str  # 'x', 'c', 'y' ...
    body: tuple[Atom, ...]
    head: Atom
    source_premise_id: str  # e.g. "P1"

    def __str__(self) -> str:
        body_str = " ∧ ".join(str(a) for a in self.body)
        return f"∀{self.quantified_var} ({body_str} → {self.head})"


@dataclass(frozen=True)
class Fact:
    """A grounded atom, with a pointer back to the premise that introduced it."""

    atom: Atom
    source_premise_id: str  # e.g. "P5" for a ground fact, or "derived"


@dataclass
class ProofStep:
    """One forward-chaining derivation step (used for the explanation)."""

    derived: Fact
    rule_premise_id: str  # which rule fired
    binding: dict[str, str]  # variable → constant
    supporting_premise_ids: tuple[str, ...]  # facts that satisfied the body


@dataclass
class KnowledgeBase:
    """Aggregated rules + facts + derivation trace.

    The forward chainer mutates this in-place during fixpoint iteration.
    Atoms are deduplicated via a frozenset key, so re-deriving an existing
    fact is a cheap noop.
    """

    rules: list[Rule] = field(default_factory=list)
    facts: dict[tuple[str, tuple[str, ...], Polarity], Fact] = field(default_factory=dict)
    proof: list[ProofStep] = field(default_factory=list)

    def add_fact(self, fact: Fact) -> bool:
        """Insert a fact. Returns True if it was new."""
        key = (fact.atom.predicate, fact.atom.args, fact.atom.polarity)
        if key in self.facts:
            return False
        self.facts[key] = fact
        return True

    def has_fact(self, atom: Atom) -> bool:
        return (atom.predicate, atom.args, atom.polarity) in self.facts

    def lookup(self, atom: Atom) -> Fact | None:
        return self.facts.get((atom.predicate, atom.args, atom.polarity))


# ---------------------------------------------------------------------------
# Answer typing
# ---------------------------------------------------------------------------

YNUVerdict = Literal["Yes", "No", "Unknown"]
