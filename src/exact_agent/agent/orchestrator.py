"""Top-level orchestrator: route → run pipeline → format response.

This is the only entry point the API layer needs to know about. Pipelines
themselves never touch HTTP concerns.
"""

from __future__ import annotations

from exact_agent.logic.pipeline import LogicPipeline
from exact_agent.physics.pipeline import PhysicsPipeline
from exact_agent.router import route_task
from exact_agent.schemas import PredictRequest, PredictResponse


class Orchestrator:
    def __init__(
        self,
        logic_pipeline: LogicPipeline | None = None,
        physics_pipeline: PhysicsPipeline | None = None,
    ) -> None:
        self._logic = logic_pipeline or LogicPipeline()
        self._physics = physics_pipeline or PhysicsPipeline()

    def predict(self, payload: PredictRequest) -> PredictResponse:
        task = route_task(payload)
        if task == "logic":
            return self._logic.run(payload)
        return self._physics.run(payload)
