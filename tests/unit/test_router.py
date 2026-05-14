"""Routing tests — both PredictRequest and raw-dict variants."""

from __future__ import annotations

from exact_agent.router import route_task
from exact_agent.schemas import PredictRequest


class TestRouteTask:
    def test_premises_nl_routes_to_logic(self) -> None:
        req = PredictRequest.model_validate(
            {
                "premises-NL": ["All students are humans."],
                "question": "Are students humans?",
            }
        )
        assert route_task(req) == "logic"

    def test_no_premises_routes_to_physics(self) -> None:
        req = PredictRequest(question="Calculate the energy stored when C=100uF and U=30V.")
        assert route_task(req) == "physics"

    def test_explicit_task_type_wins(self) -> None:
        req = PredictRequest.model_validate(
            {
                "premises-NL": ["foo"],
                "question": "bar",
                "task_type": "physics",
            }
        )
        assert route_task(req) == "physics"

    def test_dict_payload_with_hyphen_key(self) -> None:
        assert route_task({"premises-NL": ["x"], "question": "y"}) == "logic"

    def test_dict_payload_with_underscore_key(self) -> None:
        assert route_task({"premises_NL": ["x"], "question": "y"}) == "logic"

    def test_dict_payload_empty_premises_falls_back_to_physics(self) -> None:
        assert route_task({"premises-NL": [], "question": "y"}) == "physics"

    def test_dict_payload_explicit_task_type(self) -> None:
        assert route_task({"task_type": "logic", "question": "y"}) == "logic"
