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
