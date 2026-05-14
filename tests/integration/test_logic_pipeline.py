"""End-to-end logic pipeline integration test (no LLM)."""

from __future__ import annotations

import pytest

from exact_agent.logic.pipeline import LogicPipeline
from exact_agent.schemas import PredictRequest


@pytest.fixture(scope="module")
def pipeline() -> LogicPipeline:
    return LogicPipeline()


class TestLogicPipelineEndToEnd:
    def test_yes_no_modus_ponens(self, pipeline: LogicPipeline) -> None:
        payload = PredictRequest.model_validate(
            {
                "premises-NL": [
                    "If a student passes the exam, then they receive credit.",
                    "A student passes the exam.",
                ],
                "question": "Does the student receive credit?",
            }
        )
        resp = pipeline.run(payload)
        assert resp.task_type == "logic"
        assert resp.answer == "Yes"
        assert resp.premises  # should reference at least one premise
        assert resp.confidence and resp.confidence > 0.4

    def test_multiple_choice_question_in_question_body(self, pipeline: LogicPipeline) -> None:
        payload = PredictRequest.model_validate(
            {
                "premises-NL": [
                    "If a student passes the exam, then they receive credit.",
                    "A student passes the exam.",
                ],
                "question": (
                    "Which conclusion follows?\n"
                    "A. The student fails the exam\n"
                    "B. The student receives credit\n"
                    "C. The teacher fails the exam\n"
                    "D. Nobody passes"
                ),
            }
        )
        resp = pipeline.run(payload)
        assert resp.task_type == "logic"
        assert resp.answer == "B"

    def test_unknown_when_no_signal(self, pipeline: LogicPipeline) -> None:
        payload = PredictRequest.model_validate(
            {
                "premises-NL": ["Some unrelated background fact about kittens."],
                "question": "Is the moon made of cheese?",
            }
        )
        resp = pipeline.run(payload)
        # Pipeline must not crash and must not over-claim.
        assert resp.answer in ("Unknown", "")

    def test_z3_fallback_resolves_coreference(self, pipeline: LogicPipeline) -> None:
        """Surface chain can't bridge ``a student``↔``Sophia``; Z3 should.

        With ``premises-FOL`` + ``claim-FOL`` supplied, the pipeline
        delegates the entailment check to Z3. The expected answer is
        ``Yes`` even though the surface chainer alone returns ``Unknown``.
        """
        payload = PredictRequest.model_validate(
            {
                "premises-NL": [
                    "All students who pass the exam receive credit.",
                    "Sophia has passed the exam.",
                ],
                "premises-FOL": [
                    "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))",
                    "Student(Sophia)",
                    "PassedExam(Sophia)",
                ],
                "claim-FOL": "ReceivesCredit(Sophia)",
                "question": "Does Sophia receive credit?",
            }
        )
        resp = pipeline.run(payload)
        assert resp.answer == "Yes"
        # The Z3 fallback marker should be in the CoT trace.
        assert resp.cot is not None
        assert any("Z3" in step for step in resp.cot)
