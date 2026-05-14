"""Physics pipeline — facade exposed to the API.

Mock implementation for Day 1; the real solver chain lands Day 2-3.
"""

from __future__ import annotations

from exact_agent.agent.output_formatter import format_response
from exact_agent.schemas import PredictRequest, PredictResponse


class PhysicsPipeline:
    """Orchestrates: classifier → extractor → unit_converter → solver → verifier."""

    def run(self, payload: PredictRequest) -> PredictResponse:
        # TODO(Day 2-3): wire topic_classifier + quantity_extractor + solver.
        return format_response(
            answer="0",
            explanation=(
                "[stub] Physics pipeline not yet implemented. The real SymPy "
                "solver with pint unit handling will be wired in Phase 1 / Day 2-3."
            ),
            cot=["[stub] formula extraction + numeric solve not yet implemented"],
            confidence=0.0,
            task_type="physics",
        )
