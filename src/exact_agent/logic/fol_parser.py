"""Parse the Unicode/ASCII first-order-logic strings shipped with the dataset.

Covers (Day-14 expansion):

* Universal rules, single OR nested / multi-variable, both syntaxes:
  ``∀x (body → head)``, ``∀x ∀y (...)``, ``∀x (∀y ...)``,
  ``∀x (ForAll(d, ...))``, ``ForAll(a, ForAll(b, ForAll(c, ...)))``.
* Conjunctive bodies ``(A(x) ∧ B(x))``; negation ``¬P(x)``.
* **Comparison / arithmetic literals** ``f(args) OP n`` with
  ``OP ∈ {=, ≠, ≥, ≤, >, <}`` (and ascii ``== != >= <= > <``), as a
  body conjunct, a head, or a bare ground fact (``dur(Alex) = 8``).
* Ground (possibly negated) atoms.

Still out of scope (returns ``None``): ∃ / Exists, disjunction ∨,
biconditional ↔. The Z3 backend consumes whatever parses; unparsed
premises are simply skipped (theory stays sound, just weaker).
"""

from __future__ import annotations

import re

from exact_agent.logic.types import Atom, Comparison, Existential, Rule

# ---------------------------------------------------------------------------
# Atoms & comparisons
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(
    r"""
    (?P<neg>¬\s*)?
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*\(\s*
    (?P<args>[^()]*?)
    \s*\)
    """,
    re.VERBOSE,
)

# f(args) OP number   — unicode or ascii operators.
_CMP_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*\(\s*(?P<args>[^()]*?)\s*\)\s*
    (?P<op>≥|≤|≠|>=|<=|==|!=|=|>|<)\s*
    (?P<num>[-+]?\d+(?:\.\d+)?(?:\s*[×x*]\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?)
    \s*$
    """,
    re.VERBOSE,
)

_OP_CANON = {
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "==": "=",
    "=": "=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "!=": "!=",
}


def _parse_num(raw: str) -> float:
    s = raw.strip().replace(" ", "")
    s = re.sub(r"[×x*]10\^?", "e", s)
    return float(s)


def parse_atom(text: str) -> Atom | Comparison | None:
    """Parse one literal: a predicate atom or a comparison ``f(..) OP n``."""
    text = text.strip()
    if not text:
        return None

    cmp_m = _CMP_RE.match(text)
    if cmp_m:
        args_raw = cmp_m.group("args").strip()
        args = tuple(a.strip() for a in args_raw.split(",")) if args_raw else ()
        try:
            value = _parse_num(cmp_m.group("num"))
        except ValueError:
            return None
        return Comparison(
            func=cmp_m.group("name"),
            args=args,
            op=_OP_CANON[cmp_m.group("op")],
            value=value,
        )

    m = _ATOM_RE.fullmatch(text)
    if not m:
        return None
    args_raw = m.group("args").strip()
    args = tuple(a.strip() for a in args_raw.split(",")) if args_raw else ()
    return Atom(
        predicate=m.group("name"),
        args=args,
        polarity="neg" if m.group("neg") else "pos",
    )


# ---------------------------------------------------------------------------
# Paren-aware splitting
# ---------------------------------------------------------------------------


def _strip_balanced_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i != len(text) - 1:
                ok = False
                break
        if not ok:
            break
        text = text[1:-1].strip()
    return text


def _split_top(text: str, sep: str) -> list[str]:
    """Split ``text`` at top-level (depth-0) occurrences of ``sep``."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch == sep:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _split_conjuncts(body_text: str) -> list[str]:
    return _split_top(_strip_balanced_parens(body_text), "∧")


# ---------------------------------------------------------------------------
# Universal-quantifier peeling (handles both syntaxes + nesting)
# ---------------------------------------------------------------------------

_UNI_UNICODE = re.compile(r"^∀\s*(?P<var>[A-Za-z]\w*)\s*")
_UNI_FORALL = re.compile(r"^ForAll\s*\(\s*(?P<var>[A-Za-z]\w*)\s*,\s*", re.IGNORECASE)
_EXI_UNICODE = re.compile(r"^∃\s*(?P<var>[A-Za-z]\w*)\s*")
_EXI_EXISTS = re.compile(r"^Exists\s*\(\s*(?P<var>[A-Za-z]\w*)\s*,\s*", re.IGNORECASE)


def _peel_universals(text: str) -> tuple[list[str], str] | None:
    """Strip leading ∀/ForAll layers. Returns (vars, matrix) or None if an
    unsupported quantifier (∃/Exists) is encountered."""
    vars_: list[str] = []
    text = text.strip()
    while True:
        text = _strip_balanced_parens(text).strip()
        if text.startswith("∃") or re.match(r"^Exists\s*\(", text, re.IGNORECASE):
            return None
        m = _UNI_UNICODE.match(text)
        if m:
            vars_.append(m.group("var"))
            text = text[m.end() :].strip()
            continue
        m = _UNI_FORALL.match(text)
        if m:
            vars_.append(m.group("var"))
            rest = text[m.end() :].strip()
            # Drop the single trailing ')' that closes this ForAll(...).
            if rest.endswith(")"):
                rest = rest[:-1].strip()
            text = rest
            continue
        break
    return vars_, text


def _peel_existentials(text: str) -> tuple[list[str], str] | None:
    """Iter-9: strip leading ∃/Exists layers. Returns (vars, matrix) or
    None if a ∀ leaks into a position we don't support (mixed ∃∀ quantifier
    alternation is rare in this dataset; we punt to None to stay sound).
    """
    vars_: list[str] = []
    text = text.strip()
    while True:
        text = _strip_balanced_parens(text).strip()
        # ∀ inside an ∃ scope is an unsupported alternation for our model.
        if text.startswith("∀") or re.match(r"^ForAll\s*\(", text, re.IGNORECASE):
            return None
        m = _EXI_UNICODE.match(text)
        if m:
            vars_.append(m.group("var"))
            text = text[m.end() :].strip()
            continue
        m = _EXI_EXISTS.match(text)
        if m:
            vars_.append(m.group("var"))
            rest = text[m.end() :].strip()
            if rest.endswith(")"):
                rest = rest[:-1].strip()
            text = rest
            continue
        break
    if not vars_:
        return None
    return vars_, text


def parse_fol(  # noqa: PLR0911, PLR0912
    text: str, premise_id: str
) -> Atom | Comparison | Rule | Existential | None:
    """Parse one FOL line into a ground Atom/Comparison/Rule/Existential.

    Iter-9: ``∃x(...)`` / ``Exists(x, ...)`` patterns previously returned
    ``None``, silently dropping ~138 dataset premises (68% of rows). Now
    parsed into an :class:`Existential` node so the Z3 backend can emit
    ``z3.Exists`` and let entailment queries see the witness.
    """
    raw = text.strip()
    if not raw:
        return None
    if "∨" in raw or "↔" in raw:
        return None  # disjunction / biconditional not supported

    # Iter-9: existential body — strip the leading ∃ layers then parse
    # the conjunction inside.
    is_existential = raw.startswith("∃") or re.match(r"^Exists\s*\(", raw, re.IGNORECASE)
    if is_existential:
        peeled = _peel_existentials(raw)
        if peeled is None:
            return None
        bound_vars, matrix = peeled
        matrix = _strip_balanced_parens(matrix)
        if not matrix:
            return None
        # An existential matrix is a conjunction of literals (no implication).
        # We deliberately do NOT support ∃ over implications since those are
        # semantically unusual and rare in this dataset.
        if "→" in matrix:
            return None
        body_atoms: list[Atom | Comparison] = []
        for conj in _split_conjuncts(matrix):
            a = parse_atom(conj)
            if a is None:
                return None
            body_atoms.append(a)
        if not body_atoms:
            return None
        return Existential(
            quantified_vars=tuple(bound_vars),
            body=tuple(body_atoms),
            source_premise_id=premise_id,
        )

    is_universal = raw.startswith("∀") or re.match(r"^ForAll\s*\(", raw, re.IGNORECASE)
    if is_universal:
        peeled = _peel_universals(raw)
        if peeled is None:
            return None
        bound_vars, matrix = peeled
        if not bound_vars:
            return None
        matrix = _strip_balanced_parens(matrix)
        sides = _split_top(matrix, "→")
        if len(sides) != 2:
            return None  # need exactly body → head
        body_atoms = []
        for conj in _split_conjuncts(sides[0]):
            a = parse_atom(conj)
            if a is None:
                return None
            body_atoms.append(a)
        head = parse_atom(_strip_balanced_parens(sides[1]))
        if head is None or not body_atoms:
            return None
        return Rule(
            quantified_vars=tuple(bound_vars),
            body=tuple(body_atoms),
            head=head,
            source_premise_id=premise_id,
        )

    # Ground fact: a (possibly negated) atom or a comparison.
    return parse_atom(raw)
