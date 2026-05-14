"""Parse the Unicode first-order-logic strings shipped with the dataset.

The cleaned dataset uses a stable subset of FOL:

* Universal rules: ``∀x (body → head)``
* Conjunctive body: ``(A(x) ∧ B(x) ∧ C(x))``
* Negation: ``¬P(x)``
* Ground facts: ``P(John)`` or ``¬P(John)``

The parser is intentionally regex-driven — robust enough for the 90%+ of
premises that conform to this shape, and graceful (returns ``None``) on
the rest. Day-4 baseline only consumes premises that parse cleanly; the
LLM fallback in Phase 4 will handle the odd ones.
"""

from __future__ import annotations

import re

from exact_agent.logic.types import Atom, Rule

# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

# Predicate followed by parenthesised, comma-separated args.
_ATOM_RE = re.compile(
    r"""
    (?P<neg>¬\s*)?                         # optional negation
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)       # predicate name
    \s*\(\s*
    (?P<args>[^()]*?)                       # args (no nested parens)
    \s*\)
    """,
    re.VERBOSE,
)


def parse_atom(text: str) -> Atom | None:
    """Parse a single atom string (possibly negated)."""
    text = text.strip()
    if not text:
        return None
    match = _ATOM_RE.fullmatch(text)
    if not match:
        return None
    args_raw = match.group("args").strip()
    args: tuple[str, ...] = ()
    if args_raw:
        args = tuple(arg.strip() for arg in args_raw.split(","))
    return Atom(
        predicate=match.group("name"),
        args=args,
        polarity="neg" if match.group("neg") else "pos",
    )


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


def _split_conjuncts(body_text: str) -> list[str]:
    """Split a conjunction string at top-level ``∧``.

    Respects parenthesis nesting. Handles the leading/trailing parens that
    the dataset typically wraps bodies in.
    """
    text = body_text.strip()
    if text.startswith("(") and text.endswith(")"):
        # Strip outer parens only if balanced.
        depth = 0
        balanced_outer = True
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    balanced_outer = False
                    break
        if balanced_outer:
            text = text[1:-1].strip()

    conjuncts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch == "∧":
            conjuncts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        conjuncts.append(tail)
    return conjuncts


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(
    r"""
    ∀\s*(?P<var>[A-Za-z])    # quantified variable
    \s*\(?\s*
    (?P<body>.+?)
    \s*→\s*
    (?P<head>.+?)
    \s*\)?\s*$
    """,
    re.VERBOSE | re.DOTALL,
)


def parse_fol(text: str, premise_id: str) -> Atom | Rule | None:  # noqa: PLR0911
    """Parse a single FOL line into either a ground :class:`Atom` (fact)
    or a :class:`Rule` (universal Horn-like rule).

    Returns ``None`` for anything we can't handle (∃ quantifiers,
    disjunctive heads, malformed syntax).
    """
    raw = text.strip()
    if not raw:
        return None

    # Universal rule?
    if raw.startswith("∀"):
        match = _RULE_RE.fullmatch(raw)
        if match is None:
            return None
        var = match.group("var")
        body_raw = match.group("body").strip()
        head_raw = match.group("head").strip()
        # Body may still be wrapped in parens. _split_conjuncts strips them.
        body_atoms = []
        for conj in _split_conjuncts(body_raw):
            atom = parse_atom(conj)
            if atom is None:
                return None
            body_atoms.append(atom)
        head_atom = parse_atom(head_raw)
        if head_atom is None:
            return None
        return Rule(
            quantified_var=var,
            body=tuple(body_atoms),
            head=head_atom,
            source_premise_id=premise_id,
        )

    if raw.startswith("∃"):
        return None  # not supported in Day-4 baseline

    # Otherwise treat as a ground (possibly negated) atom.
    return parse_atom(raw)
