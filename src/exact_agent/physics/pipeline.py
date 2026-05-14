"""Physics pipeline — facade exposed to the API.

Day 2 wires the real solver chain: extractor → classifier → SymPy compute.
A `confidence == 0.0` response is the documented signal to the agent
self-correction loop (Phase 5) that the LLM should retry.
"""

from __future__ import annotations

from exact_agent.agent.output_formatter import format_response
from exact_agent.physics.explanation import render_explanation
from exact_agent.physics.solver import PhysicsSolver, SolverResult
from exact_agent.schemas import PredictRequest, PredictResponse


class PhysicsPipeline:
    """Orchestrates: classifier → extractor → unit_converter → solver → verifier."""

    def __init__(self, solver: PhysicsSolver | None = None) -> None:
        self._solver = solver or PhysicsSolver()

    def run(self, payload: PredictRequest) -> PredictResponse:
        result: SolverResult = self._solver.solve(payload.question)

        if not result.success:
            return format_response(
                answer="",
                explanation=render_explanation(result),
                cot=result.trace,
                confidence=0.0,
                task_type="physics",
            )

        answer_str = result.answer_str
        if result.answer_unit:
            answer_str = f"{answer_str} {result.answer_unit}"

        return format_response(
            answer=answer_str,
            explanation=render_explanation(result),
            cot=result.trace,
            confidence=result.confidence,
            task_type="physics",
        )
