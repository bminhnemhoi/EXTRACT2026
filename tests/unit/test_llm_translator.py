"""Tests for the E5 solver-rejection NL->FOL translator loop."""

from __future__ import annotations

import pytest

from exact_agent.llm.vllm_client import MockLLMClient
from exact_agent.logic.llm_translator import (
    LLMTranslationError,
    translate_question_to_fol_with_verify,
)


class TestSolverRejectionLoop:
    def test_accepts_first_valid_attempt(self) -> None:
        client = MockLLMClient(["FOL: Student(Alice)"])
        line, trace = translate_question_to_fol_with_verify(
            "Is Alice a student?", ["Student(Bob)"], client,
        )
        assert line == "Student(Alice)"
        assert len(trace) == 1 and "accepted" in trace[0]
        assert len(client.calls) == 1   # no retry needed

    def test_retries_on_parse_failure_with_feedback(self) -> None:
        # First: malformed (∨ still unsupported post-Iter-9; ∃ now parses);
        # second: valid universal rule.
        client = MockLLMClient([
            "FOL: ∀x (Student(x) ∨ Faculty(x))",  # disjunction never parses
            "FOL: ∀x (Student(x) → Eligible(x))",
        ])
        line, _trace = translate_question_to_fol_with_verify(
            "Are all students eligible?", ["Student(Alice)"], client,
        )
        assert line == "∀x (Student(x) → Eligible(x))"
        assert len(client.calls) == 2
        # The second prompt must carry the rejection reason from the first.
        retry_prompt = client.calls[1]["prompt"]
        assert "did NOT parse" in retry_prompt or "∨" in retry_prompt

    def test_retries_on_vocabulary_mismatch(self) -> None:
        # First claim uses brand-new predicates; second reuses premise vocab.
        client = MockLLMClient([
            "FOL: Brandnew(Alice)",
            "FOL: Student(Alice)",
        ])
        line, _trace = translate_question_to_fol_with_verify(
            "Is Alice a student?", ["Student(Bob)", "Eligible(Bob)"], client,
        )
        assert line == "Student(Alice)"
        assert len(client.calls) == 2
        retry_prompt = client.calls[1]["prompt"]
        # The retry must include the actual offending claim AND the
        # allowed vocab so the model can self-correct.
        assert "Brandnew" in retry_prompt
        assert "Student" in retry_prompt

    def test_raises_when_all_attempts_fail(self) -> None:
        # 3 bad attempts (1 initial + 2 retries) — all ∨ unsupported.
        # (Post-Iter-9 ∃ now parses, so we use disjunction for the
        # "stays unparseable forever" test instead.)
        client = MockLLMClient([
            "FOL: P(x) ∨ Q(x)",
            "FOL: A(x) ∨ B(x)",
            "FOL: C(x) ∨ D(x)",
        ])
        with pytest.raises(LLMTranslationError):
            translate_question_to_fol_with_verify(
                "Anything?", ["Student(A)"], client, max_rounds=2,
            )
        # All 3 attempts consumed.
        assert len(client.calls) == 3

    def test_does_not_reject_partial_novel_vocab(self) -> None:
        # Claim reuses ONE premise predicate and introduces ONE new term —
        # legitimate (the question often introduces a related concept).
        client = MockLLMClient([
            "FOL: ∀x (Student(x) → Eligible(x))",   # Student known, Eligible new
        ])
        line, _ = translate_question_to_fol_with_verify(
            "Are students eligible?", ["Student(Alice)", "PassedExam(Alice)"], client,
        )
        assert line == "∀x (Student(x) → Eligible(x))"

    def test_max_rounds_zero_means_single_shot(self) -> None:
        client = MockLLMClient(["FOL: Student(Alice)"])
        line, _ = translate_question_to_fol_with_verify(
            "Is Alice a student?", ["Student(Bob)"], client, max_rounds=0,
        )
        assert line == "Student(Alice)"
        assert len(client.calls) == 1
