"""HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from exact_agent.__version__ import __version__
from exact_agent.agent.orchestrator import Orchestrator
from exact_agent.schemas import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


def get_orchestrator() -> Orchestrator:
    # The Orchestrator is stateless on Day 1; later phases can swap in a
    # process-wide singleton with LLM clients / model weights pre-loaded.
    return Orchestrator()


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]


@router.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest, orchestrator: OrchestratorDep) -> PredictResponse:
    return orchestrator.predict(payload)
