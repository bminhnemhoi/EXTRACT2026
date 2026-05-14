"""Tests for the response confidence scorer."""

from __future__ import annotations

from exact_agent.agent.confidence import score_response
from exact_agent.schemas import PredictResponse


def _make(answer: str = "Yes", **kw) -> PredictResponse:
    return PredictResponse.model_validate(
        {
            "answer": answer,
            "explanation": kw.pop("explanation", "because reasons"),
            **kw,
        }
    )


class TestScoreResponse:
    def test_high_confidence_passes(self) -> None:
        resp = _make(confidence=0.85, premises=["P1"], cot=["s1", "s2"])
        result = score_response(resp, task_type="logic")
        assert result.score >= 0.6
        assert not result.needs_revision

    def test_empty_answer_demands_revision(self) -> None:
        resp = _make(answer=" ", confidence=0.9)
        result = score_response(resp, task_type="logic")
        assert result.score == 0.0
        assert result.needs_revision
        assert "answer field is empty" in result.reasons

    def test_logic_unknown_without_premises_flagged(self) -> None:
        resp = _make(answer="Unknown", confidence=0.5)
        result = score_response(resp, task_type="logic")
        assert result.needs_revision
        assert any("Unknown" in r for r in result.reasons)

    def test_logic_unknown_with_premises_not_flagged_for_that_reason(self) -> None:
        resp = _make(answer="Unknown", premises=["P1"], confidence=0.65)
        result = score_response(resp, task_type="logic")
        assert not any("Unknown" in r for r in result.reasons)

    def test_mc_without_premises_flagged(self) -> None:
        resp = _make(answer="B", confidence=0.5)
        result = score_response(resp, task_type="logic")
        assert any("multiple-choice" in r for r in result.reasons)

    def test_physics_low_solver_confidence_flagged(self) -> None:
        resp = _make(answer="0.045 J", confidence=0.3, cot=["s1", "s2", "s3"])
        result = score_response(resp, task_type="physics")
        assert result.needs_revision
        assert any("solver confidence" in r for r in result.reasons)

    def test_physics_short_cot_flagged(self) -> None:
        resp = _make(answer="0.045 J", confidence=0.9, cot=["only one step"])
        result = score_response(resp, task_type="physics")
        assert any("CoT" in r for r in result.reasons)
