"""Tests for the E6 self-consistency vote over physics LLM extraction."""

from __future__ import annotations

import json

import pytest

from exact_agent.llm.vllm_client import MockLLMClient
from exact_agent.physics.formula_library import default_library
from exact_agent.physics.llm_extractor import (
    LLMExtractionError,
    extract_with_llm_self_consistent,
)

LIB = default_library()
CAP = LIB["capacitor_energy"]   # inputs: C (farad), U (volt)


def _envelope(c_val: float, u_val: float, *, c_unit: str = "F", u_unit: str = "V") -> str:
    return json.dumps({
        "C": {"value": c_val, "unit": c_unit},
        "U": {"value": u_val, "unit": u_unit},
    })


class TestSelfConsistencyVote:
    def test_unanimous_three_attempts(self) -> None:
        client = MockLLMClient([
            _envelope(1e-4, 30.0), _envelope(1e-4, 30.0), _envelope(1e-4, 30.0),
        ])
        out, votes = extract_with_llm_self_consistent(
            "Calculate energy when C=100 μF and U=30 V.", CAP, client, n_votes=3,
        )
        names = {q.name: q for q in out}
        assert names["C"].value == 1e-4
        assert names["U"].value == 30.0
        assert votes == {"C": 3, "U": 3}

    def test_majority_with_one_hallucinated_value(self) -> None:
        # Two attempts agree on the right number; the third hallucinates U.
        client = MockLLMClient([
            _envelope(1e-4, 30.0),
            _envelope(1e-4, 30.0),
            _envelope(1e-4, 9999.0),
        ])
        out, votes = extract_with_llm_self_consistent(
            "...", CAP, client, n_votes=3,
        )
        names = {q.name: q for q in out}
        assert names["U"].value == 30.0   # the hallucinated 9999 lost the vote
        assert votes["U"] == 2

    def test_no_majority_raises(self) -> None:
        # All three give different U values: vote can't elect.
        client = MockLLMClient([
            _envelope(1e-4, 10.0),
            _envelope(1e-4, 30.0),
            _envelope(1e-4, 100.0),
        ])
        with pytest.raises(LLMExtractionError) as excinfo:
            extract_with_llm_self_consistent("...", CAP, client, n_votes=3)
        assert "no majority cluster" in str(excinfo.value)

    def test_two_of_three_attempts_fail_to_parse_raises(self) -> None:
        client = MockLLMClient([
            "not even json",
            "{not json either",
            _envelope(1e-4, 30.0),
        ])
        with pytest.raises(LLMExtractionError) as excinfo:
            extract_with_llm_self_consistent("...", CAP, client, n_votes=3)
        assert "only 1/3 attempts" in str(excinfo.value)

    def test_values_within_relative_tolerance_cluster(self) -> None:
        # Two attempts agree to within 1% rel — should still count as one cluster.
        client = MockLLMClient([
            _envelope(1e-4, 30.0),
            _envelope(1e-4, 30.2),   # 0.67% off -> within tol
            _envelope(1e-4, 50.0),
        ])
        out, votes = extract_with_llm_self_consistent("...", CAP, client, n_votes=3)
        names = {q.name: q for q in out}
        assert abs(names["U"].value - 30.0) < 0.3
        assert votes["U"] == 2
