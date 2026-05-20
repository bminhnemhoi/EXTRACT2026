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

    def test_abstains_unknown_when_no_signal(self) -> None:
        # E4 policy (2026-05-15+): when no option overlaps the chain at
        # all, the answer is the legitimate label "Unknown" — not empty,
        # not an alphabetical guess. The official release retains 168
        # MCQs whose gold IS "Unknown".
        chain = _build_chain(["Unrelated background fact about kittens."])
        choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
        result = verify_multiple_choice("Pick one.", choices, chain)
        assert result.answer == "Unknown"
        assert result.confidence < 0.5

    def test_commits_letter_when_any_signal_exists(self) -> None:
        # Conservative default (post-retune): one overlapping token is
        # enough to commit a letter rather than abstain — measured the
        # stricter threshold over-abstained MC on the holdout.
        chain = _build_chain(["The student receives credit."])
        choices = {
            "A": "credit",
            "B": "completely unrelated thing about kittens",
            "C": "another unrelated thing",
            "D": "yet another",
        }
        result = verify_multiple_choice("Pick one.", choices, chain)
        assert result.answer == "A"

    def test_caller_can_opt_in_to_strict_threshold(self) -> None:
        # The strict path is reachable for callers that want it.
        chain = _build_chain(["The student receives credit."])
        choices = {
            "A": "credit",
            "B": "completely unrelated thing about kittens",
            "C": "another unrelated thing",
            "D": "yet another",
        }
        result = verify_multiple_choice("Pick one.", choices, chain, min_top_score=0.9)
        assert result.answer == "Unknown"

    def test_caller_can_disable_unknown_abstention(self) -> None:
        chain = _build_chain(["Unrelated background fact about kittens."])
        choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
        result = verify_multiple_choice(
            "Pick one.", choices, chain, abstain_when_all_zero=False
        )
        assert result.answer == "A"
