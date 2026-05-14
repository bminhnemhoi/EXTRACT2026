"""Tests for the Jaccard forward chainer."""

from __future__ import annotations

from exact_agent.logic.forward_chainer import forward_chain
from exact_agent.logic.rule_parser import parse_premises


class TestForwardChain:
    def test_modus_ponens_style(self) -> None:
        rules, facts = parse_premises(
            [
                "If a student passes the exam, then they receive credit.",
                "A student passes the exam.",
            ]
        )
        chain = forward_chain(rules, facts)
        # The conclusion text "they receive credit" should be derivable.
        assert chain.entails("receive credit", threshold=0.3)

    def test_records_supporting_premises(self) -> None:
        rules, facts = parse_premises(
            [
                "If a student passes the exam, then they receive credit.",
                "A student passes the exam.",
            ]
        )
        chain = forward_chain(rules, facts)
        match = chain.matches("receive credit", threshold=0.3)
        assert match is not None
        # Both the rule (P1) and the fact (P2) should appear in supports.
        assert 1 in match.supports
        assert 2 in match.supports

    def test_no_match_returns_none(self) -> None:
        _, facts = parse_premises(["Alice has passed the exam."])
        chain = forward_chain([], facts)
        # 'unrelated quantum' is not in any fact tokens.
        assert chain.matches("unrelated quantum gravitation") is None

    def test_converges(self) -> None:
        rules, facts = parse_premises(
            [
                "If condition A, then conclusion B.",
                "Condition A holds.",  # falls through as a fact
            ]
        )
        chain = forward_chain(rules, facts, max_iterations=5)
        assert chain.converged is True
        assert chain.iterations <= 5

    def test_caps_iterations(self) -> None:
        # With no rules to fire, chain converges on iteration 1 — exercise the cap.
        chain = forward_chain([], [], max_iterations=10)
        assert chain.iterations <= 10
