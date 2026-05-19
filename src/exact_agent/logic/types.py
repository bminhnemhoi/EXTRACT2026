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
class Comparison:
    """An arithmetic/comparison literal: ``f(args) OP value``.

    The dataset uses these heavily for threshold reasoning, e.g.
    ``∀x ((membership_duration(x) ≥ 6) → eligible_trainer(x))`` or the
    ground fact ``membership_duration(Alex) = 8``. The surface chainer
    cannot evaluate ``8 ≥ 6``; only the Z3 backend (Real-valued
    functions) can, which is exactly why these were dead weight before.
    """

    func: str
    args: tuple[str, ...]
    op: str  # one of: = != >= <= > <
    value: float

    def __str__(self) -> str:
        return f"{self.func}({', '.join(self.args)}) {self.op} {self.value:g}"


# A body conjunct / head can be a boolean predicate atom or a comparison.
Literal_ = "Atom | Comparison"


@dataclass(frozen=True)
class Rule:
    """A (possibly multi-variable) universally quantified Horn-ish rule.

    Bodies are conjunctions of atoms/comparisons; head is a single
    atom/comparison (possibly negated). Disjunctive heads and ∃ are
    ignored by the parser (returns ``None``).
    """

    quantified_vars: tuple[str, ...]  # ('x',) or ('x', 'd') or ('a','b','c')
    body: tuple[Atom | Comparison, ...]
    head: Atom | Comparison
    source_premise_id: str  # e.g. "P1"

    @property
    def quantified_var(self) -> str:
        """Back-compat: the first bound variable."""
        return self.quantified_vars[0] if self.quantified_vars else "x"

    def __str__(self) -> str:
        body_str = " ∧ ".join(str(a) for a in self.body)
        qs = "".join(f"∀{v}" for v in self.quantified_vars)
        return f"{qs} ({body_str} → {self.head})"


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
