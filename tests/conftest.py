"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from exact_agent.api.app import create_app


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture()
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_logic_payload() -> dict[str, object]:
    return {
        "premises-NL": [
            "If a student completes all required courses, they are eligible for graduation.",
            "Alice has completed all required courses.",
        ],
        "question": "Is Alice eligible for graduation?",
    }


@pytest.fixture()
def sample_physics_payload() -> dict[str, object]:
    return {
        "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.",
    }
