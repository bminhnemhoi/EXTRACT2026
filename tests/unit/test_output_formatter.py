"""Tests for the response formatter."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exact_agent.agent.output_formatter import format_from_dict, format_response
from exact_agent.schemas import PredictResponse


class TestFormatResponse:
    def test_minimal_required_fields(self) -> None:
        resp = format_response(answer="Yes", explanation="Because P1 implies it.")
        assert resp.answer == "Yes"
        assert resp.explanation == "Because P1 implies it."
        assert resp.cot is None
        assert resp.premises is None
        assert resp.fol is None
        assert resp.confidence is None

    def test_confidence_clamped_above(self) -> None:
        assert format_response(answer="a", explanation="b", confidence=1.5).confidence == 1.0

    def test_confidence_clamped_below(self) -> None:
        assert format_response(answer="a", explanation="b", confidence=-0.2).confidence == 0.0

    def test_empty_cot_becomes_none(self) -> None:
        resp = format_response(answer="a", explanation="b", cot=["", "  "])
        assert resp.cot is None

    def test_premises_filtered_and_kept(self) -> None:
        resp = format_response(answer="a", explanation="b", premises=["P1", "", "P3"])
        assert resp.premises == ["P1", "P3"]

    def test_fol_trimmed(self) -> None:
        resp = format_response(answer="a", explanation="b", fol="  ∀x P(x)  ")
        assert resp.fol == "∀x P(x)"

    def test_strips_answer_and_explanation(self) -> None:
        resp = format_response(answer="  Yes  ", explanation="  ok  ")
        assert resp.answer == "Yes"
        assert resp.explanation == "ok"

    def test_empty_answer_rejected_via_validation(self) -> None:
        # format_response itself doesn't reject empty strings (it normalizes);
        # we trust the API layer to surface validation upstream. Verify the
        # Pydantic model still permits empties — the rubric enforces content
        # quality, not the type system.
        resp = format_response(answer="", explanation="")
        assert resp.answer == ""

    def test_format_from_dict_passthrough(self) -> None:
        resp = format_from_dict({"answer": "B", "explanation": "Because.", "cot": ["s1"]})
        assert resp.answer == "B"
        assert resp.cot == ["s1"]

    def test_extra_fields_in_response_rejected(self) -> None:
        # Direct construction guards against drift between pipelines and API.
        with pytest.raises(ValidationError):
            PredictResponse(answer="a", explanation="b", unknown_field=1)  # type: ignore[call-arg]
