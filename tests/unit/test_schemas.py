"""Tests for request/response Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exact_agent.schemas import PredictRequest


class TestPredictRequest:
    def test_accepts_hyphenated_premises_key(self) -> None:
        req = PredictRequest.model_validate({"premises-NL": ["p1", "p2"], "question": "q"})
        assert req.premises_NL == ["p1", "p2"]

    def test_accepts_underscored_premises_key(self) -> None:
        req = PredictRequest(premises_NL=["p"], question="q")
        assert req.premises_NL == ["p"]

    def test_question_required(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest.model_validate({"premises-NL": ["p"]})

    def test_empty_premises_list_normalized_to_none(self) -> None:
        req = PredictRequest.model_validate({"question": "q", "premises-NL": []})
        assert req.premises_NL is None

    def test_question_min_length(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(question="")
