"""Logic pipeline — facade exposed to the API.

Day 1 deliberately returns a *mock* but schema-valid response so the
end-to-end API can be smoke-tested before the real solver lands on Day 4.
Subsequent phases will replace the body of :meth:`LogicPipeline.run`
without touching the public signature.
"""

from __future__ import annotations

from exact_agent.agent.output_formatter import format_response
from exact_agent.schemas import PredictRequest, PredictResponse


class LogicPipeline:
    """Orchestrates: premise_selector → rule_parser → forward_chainer → explanation."""

    def run(self, payload: PredictRequest) -> PredictResponse:
        # TODO(Day 4): wire premise_selector + forward_chainer + answer_verifier.
        premise_ids = (
            [f"P{i + 1}" for i in range(len(payload.premises_NL))] if payload.premises_NL else None
        )
        return format_response(
            answer="Unknown",
            explanation=(
                "[stub] Logic pipeline not yet implemented. The real solver "
                "will parse the premises, run forward chaining, and produce "
                "a verifier-grounded answer in Phase 1 / Day 4."
            ),
            cot=["[stub] forward chaining not yet implemented"],
            premises=premise_ids,
            confidence=0.0,
            task_type="logic",
        )
