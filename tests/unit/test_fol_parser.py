"""Tests for the Unicode FOL parser."""

from __future__ import annotations

from exact_agent.logic.fol_parser import parse_atom, parse_fol
from exact_agent.logic.types import Atom, Comparison, Rule


class TestParseAtom:
    def test_no_args(self) -> None:
        atom = parse_atom("Sunny()")
        assert atom == Atom(predicate="Sunny", args=(), polarity="pos")

    def test_one_arg_constant(self) -> None:
        atom = parse_atom("CompletedCapstone(Sophia)")
        assert atom is not None
        assert atom.predicate == "CompletedCapstone"
        assert atom.args == ("Sophia",)

    def test_negation(self) -> None:
        atom = parse_atom("¬Eligible(Bob)")
        assert atom is not None
        assert atom.polarity == "neg"

    def test_two_args(self) -> None:
        atom = parse_atom("HasDegree(Alice, BA)")
        assert atom is not None
        assert atom.args == ("Alice", "BA")

    def test_rejects_bare_text(self) -> None:
        assert parse_atom("just plain text") is None


class TestParseFol:
    def test_universal_horn_rule(self) -> None:
        text = "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))"
        parsed = parse_fol(text, premise_id="P1")
        assert isinstance(parsed, Rule)
        assert parsed.quantified_var == "x"
        assert len(parsed.body) == 2
        assert parsed.head.predicate == "ReceivesCredit"

    def test_ground_fact(self) -> None:
        parsed = parse_fol("Student(Alice)", premise_id="P2")
        assert isinstance(parsed, Atom)
        assert parsed.args == ("Alice",)

    def test_negated_ground_fact(self) -> None:
        parsed = parse_fol("¬Failed(Alice)", premise_id="P3")
        assert isinstance(parsed, Atom)
        assert parsed.polarity == "neg"

    def test_existential_skipped(self) -> None:
        # Day-4/5 baseline doesn't handle ∃; should return None.
        parsed = parse_fol("∃x P(x)", premise_id="P4")
        assert parsed is None

    def test_unbalanced_returns_none(self) -> None:
        parsed = parse_fol("∀x (P(x) → ", premise_id="P5")
        assert parsed is None


class TestMultiVarAndComparison:
    def test_nested_forall_two_vars(self) -> None:
        s = "∀x (ForAll(d, (faculty_member(x) ∧ has_degree(x, d)) → teach(x)))"
        p = parse_fol(s, premise_id="P1")
        assert isinstance(p, Rule)
        assert p.quantified_vars == ("x", "d")
        assert len(p.body) == 2
        assert p.head.predicate == "teach"

    def test_deeply_nested_three_vars(self) -> None:
        s = "∀a (ForAll(b, ForAll(c, (higher(a, b) ∧ higher(b, c)) → higher(a, c))))"
        p = parse_fol(s, premise_id="P2")
        assert isinstance(p, Rule)
        assert p.quantified_vars == ("a", "b", "c")

    def test_bare_comparison_fact(self) -> None:
        p = parse_fol("membership_duration(Alex) = 8", premise_id="P3")
        assert isinstance(p, Comparison)
        assert p.func == "membership_duration"
        assert p.args == ("Alex",)
        assert p.op == "=" and p.value == 8.0

    def test_comparison_in_rule_body(self) -> None:
        p = parse_fol("∀x ((membership_duration(x) ≥ 6) → eligible_trainer(x))", premise_id="P4")
        assert isinstance(p, Rule)
        assert isinstance(p.body[0], Comparison)
        assert p.body[0].op == ">="
        assert p.head.predicate == "eligible_trainer"

    def test_ascii_operators(self) -> None:
        assert parse_fol("score(Bob) >= 5", "P").op == ">="  # type: ignore[union-attr]
        assert parse_fol("count(X) != 3", "P").op == "!="  # type: ignore[union-attr]

    def test_back_compat_quantified_var_property(self) -> None:
        p = parse_fol("∀x ∀y (R(x, y) → S(x))", premise_id="P")
        assert isinstance(p, Rule)
        assert p.quantified_var == "x"  # first bound var
        assert p.quantified_vars == ("x", "y")
