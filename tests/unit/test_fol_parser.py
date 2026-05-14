"""Tests for the Unicode FOL parser."""

from __future__ import annotations

from exact_agent.logic.fol_parser import parse_atom, parse_fol
from exact_agent.logic.types import Atom, Rule


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
