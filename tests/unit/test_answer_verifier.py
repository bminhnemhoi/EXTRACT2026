"""Tests for the answer-verifier (MC and Yes/No/Unknown branches)."""

from __future__ import annotations

from exact_agent.logic.answer_verifier import verify_multiple_choice, verify_yes_no
from exact_agent.logic.forward_chainer import forward_chain
from exact_agent.logic.rule_parser import parse_premises


def _build_chain(premises: list[str]):
    rules, facts = parse_premises(premises)
    return forward_chain(rules, facts)


class TestVerifyYesNo:
    def test_simple_yes(self) -> None:
        chain = _build_chain(
            [
                "If a student passes the exam, then they receive credit.",
                "A student passes the exam.",
            ]
        )
        result = verify_yes_no("Does the student receive credit?", chain)
        assert result.answer == "Yes"
        assert result.supports

    def test_negated_question_inverts(self) -> None:
        chain = _build_chain(
            [
                "If a student passes the exam, then they receive credit.",
                "A student passes the exam.",
            ]
        )
        result = verify_yes_no("Does the student not receive credit?", chain)
        assert result.answer == "No"

    def test_unknown_when_no_chain(self) -> None:
        chain = _build_chain(["Some unrelated fact."])
        result = verify_yes_no("Is the moon made of cheese?", chain)
        assert result.answer == "Unknown"


class TestVerifyMultipleChoice:
    def test_picks_overlapping_option(self) -> None:
        chain = _build_chain(
            [
                "If a student passes the exam, then they receive credit.",
                "A student passes the exam.",
            ]
        )
        choices = {
            "A": "The student fails the exam",
            "B": "The student receives credit",
            "C": "The teacher fails the exam",
            "D": "Nobody passes",
        }
        result = verify_multiple_choice("Which is true?", choices, chain)
        assert result.answer == "B"

    def test_commits_guess_when_options_tie(self) -> None:
        chain = _build_chain(["Unrelated background fact about kittens."])
        choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
        result = verify_multiple_choice("Pick one.", choices, chain)
        # Day-4 policy: don't abstain — commit to the top alphabetical option
        # (label A wins ties from sort stability) with low confidence.
        assert result.answer == "A"
        assert result.confidence < 0.5

    def test_abstains_when_all_zero_with_explicit_flag(self) -> None:
        chain = _build_chain(["Unrelated background fact about kittens."])
        choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
        result = verify_multiple_choice("Pick one.", choices, chain, abstain_when_all_zero=True)
        assert result.answer == ""
