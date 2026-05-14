"""Integration tests for the self-correction loop."""

from __future__ import annotations

from exact_agent.agent.self_correction import SelfCorrector
from exact_agent.llm.vllm_client import MockLLMClient
from exact_agent.schemas import PredictRequest, PredictResponse


def _draft(**fields) -> PredictResponse:
    base = {"answer": "Unknown", "explanation": "no signal", "task_type": "logic"}
    return PredictResponse.model_validate({**base, **fields})


def _payload() -> PredictRequest:
    return PredictRequest.model_validate(
        {
            "premises-NL": ["If a student passes the exam, then they receive credit."],
            "question": "Does the student receive credit?",
        }
    )


class TestSelfCorrector:
    def test_no_llm_means_no_revision(self) -> None:
        corrector = SelfCorrector(llm=None, max_rounds=2)
        draft = _draft(confidence=0.2)
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.rounds_used == 0
        assert outcome.accepted is False
        assert outcome.response is draft

    def test_high_confidence_skips_revision(self) -> None:
        llm = MockLLMClient([])  # would crash if invoked
        corrector = SelfCorrector(llm=llm, max_rounds=2)
        draft = _draft(answer="Yes", confidence=0.9, premises=["P1"])
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.rounds_used == 0
        assert outcome.accepted is False

    def test_llm_revises_into_higher_confidence(self) -> None:
        revised_json = (
            '{"answer": "Yes", "explanation": "Premise P1 entails it.", '
            '"premises": ["P1"], "confidence": 0.85}'
        )
        llm = MockLLMClient([revised_json])
        corrector = SelfCorrector(llm=llm, max_rounds=2)
        draft = _draft(confidence=0.2)
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.accepted is True
        assert outcome.rounds_used == 1
        assert outcome.response.answer == "Yes"
        assert outcome.response.premises == ["P1"]
        assert outcome.final_score > outcome.initial_score
        # Trace annotation should mention the revision.
        assert outcome.response.cot is not None
        assert any("Self-correction" in step for step in outcome.response.cot)

    def test_invalid_llm_json_keeps_draft(self) -> None:
        llm = MockLLMClient(["this is not valid JSON at all"])
        corrector = SelfCorrector(llm=llm, max_rounds=2)
        draft = _draft(confidence=0.2)
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.accepted is False
        assert outcome.response is draft

    def test_llm_revision_no_better_keeps_draft(self) -> None:
        # Returns identical-confidence revision → no progress, no acceptance.
        same_json = '{"answer": "Unknown", "explanation": "still no signal", "confidence": 0.2}'
        llm = MockLLMClient([same_json])
        corrector = SelfCorrector(llm=llm, max_rounds=2)
        draft = _draft(confidence=0.2)
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.accepted is False

    def test_max_rounds_caps_attempts(self) -> None:
        # First revision marginally better, second much better — both kept.
        first = (
            '{"answer": "Unknown", "explanation": "narrowed it down", '
            '"premises": ["P1"], "confidence": 0.4}'
        )
        second = (
            '{"answer": "Yes", "explanation": "full proof", "premises": ["P1"], "confidence": 0.9}'
        )
        llm = MockLLMClient([first, second])
        corrector = SelfCorrector(llm=llm, max_rounds=2)
        draft = _draft(confidence=0.2)
        outcome = corrector.maybe_revise(draft, _payload(), task_type="logic")
        assert outcome.accepted is True
        assert outcome.rounds_used == 2
        assert outcome.response.answer == "Yes"
