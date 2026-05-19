"""Top-level orchestrator: route → run pipeline → optional self-correction.

This is the only entry point the API layer needs to know about. Pipelines
themselves never touch HTTP concerns.

LLM resolution policy:

* An explicit ``llm_client`` always wins.
* Otherwise, if ``configs/app.yaml::self_correction.enabled`` is true, the
  orchestrator auto-constructs a :class:`VLLMClient` from
  ``configs/model.yaml`` so the live API exercises the LLM fallback paths
  (physics extractor, logic NL→FOL) *and* the self-correction loop. The
  client is lazy — if the endpoint is unreachable the fallbacks degrade to
  the deterministic answer rather than erroring.
"""

from __future__ import annotations

from exact_agent.agent.self_correction import SelfCorrector
from exact_agent.config import get_settings
from exact_agent.llm.vllm_client import LLMClient, VLLMClient
from exact_agent.logic.pipeline import LogicPipeline
from exact_agent.physics.pipeline import PhysicsPipeline
from exact_agent.physics.solver import PhysicsSolver
from exact_agent.router import route_task
from exact_agent.schemas import PredictRequest, PredictResponse


class Orchestrator:
    def __init__(
        self,
        logic_pipeline: LogicPipeline | None = None,
        physics_pipeline: PhysicsPipeline | None = None,
        llm_client: LLMClient | None = None,
        self_corrector: SelfCorrector | None = None,
    ) -> None:
        client = self._resolve_llm(llm_client)
        self._logic = logic_pipeline or LogicPipeline(llm_client=client)
        self._physics = physics_pipeline or PhysicsPipeline(solver=PhysicsSolver(llm_client=client))
        self._corrector = self._build_corrector(client, self_corrector)

    def predict(self, payload: PredictRequest) -> PredictResponse:
        task = route_task(payload)
        draft = self._logic.run(payload) if task == "logic" else self._physics.run(payload)
        if self._corrector is None:
            return draft
        outcome = self._corrector.maybe_revise(draft, payload, task)
        return outcome.response

    @staticmethod
    def _resolve_llm(explicit: LLMClient | None) -> LLMClient | None:
        if explicit is not None:
            return explicit
        if get_settings().self_correction.enabled:
            # Construct from configs/model.yaml. httpx.Client creation does
            # not open a socket; an unreachable endpoint only bites on the
            # first call, where the fallbacks already handle failure.
            return VLLMClient()
        return None

    @staticmethod
    def _build_corrector(
        llm_client: LLMClient | None,
        explicit: SelfCorrector | None,
    ) -> SelfCorrector | None:
        if explicit is not None:
            return explicit
        cfg = get_settings().self_correction
        if not cfg.enabled or llm_client is None:
            return None
        return SelfCorrector(llm=llm_client, max_rounds=cfg.max_rounds)
