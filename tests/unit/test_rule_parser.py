"""Tests for surface-level premise parsing."""

from __future__ import annotations

from exact_agent.logic.rule_parser import parse_premise, parse_premises


class TestParsePremise:
    def test_if_then_simple(self) -> None:
        parsed = parse_premise(
            "If a student completes all required courses, then they are eligible for graduation.",
            premise_index=0,
        )
        assert parsed.rule is not None
        assert parsed.rule.conditions == ("a student completes all required courses",)
        assert "eligible for graduation" in parsed.rule.conclusion

    def test_if_and_then(self) -> None:
        parsed = parse_premise(
            "If a curriculum has exercises and provides advanced resources, "
            "then it enhances critical thinking.",
            premise_index=2,
        )
        assert parsed.rule is not None
        assert len(parsed.rule.conditions) == 2

    def test_implicit_conditional_no_then(self) -> None:
        # Dataset frequently drops "then": "If X, it is Y."
        parsed = parse_premise(
            "If a Python project is well-structured, it is optimized.", premise_index=4
        )
        assert parsed.rule is not None
        assert "well-structured" in parsed.rule.conditions[0]

    def test_grounded_fact(self) -> None:
        parsed = parse_premise("Sophia has completed her capstone project.", premise_index=10)
        assert parsed.fact is not None
        assert parsed.rule is None

    def test_unrecognized_keeps_as_fact(self) -> None:
        parsed = parse_premise("There is something unusual going on.", premise_index=0)
        # The implicit conditional pattern doesn't match (no "If ..."); falls through
        # to the free-form Fact branch.
        assert parsed.fact is not None
        assert parsed.fact.text  # not empty


class TestParsePremises:
    def test_splits_into_rules_and_facts(self) -> None:
        rules, facts = parse_premises(
            [
                "If a student passes the exam, then they receive credit.",
                "Alice has passed the exam.",
            ]
        )
        assert len(rules) == 1
        assert len(facts) == 1
        assert rules[0].premise_index == 0
        assert facts[0].premise_index == 1

    def test_handles_empty_list(self) -> None:
        rules, facts = parse_premises([])
        assert rules == []
        assert facts == []
