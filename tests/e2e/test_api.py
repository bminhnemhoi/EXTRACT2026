"""End-to-end API smoke tests against the in-process FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_predict_logic_routes_and_returns_schema(
    client: TestClient, sample_logic_payload: dict[str, object]
) -> None:
    response = client.post("/predict", json=sample_logic_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    # Required fields must always be present.
    assert "answer" in body and isinstance(body["answer"], str)
    assert "explanation" in body and isinstance(body["explanation"], str)
    # Day-1 stub routes to the logic pipeline because premises-NL is set.
    assert body["task_type"] == "logic"
    # Optional fields appear and obey schema.
    if body.get("premises") is not None:
        assert isinstance(body["premises"], list)
    if body.get("confidence") is not None:
        assert 0.0 <= body["confidence"] <= 1.0


def test_predict_physics_routes_and_returns_schema(
    client: TestClient, sample_physics_payload: dict[str, object]
) -> None:
    response = client.post("/predict", json=sample_physics_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_type"] == "physics"
    assert "answer" in body
    assert "explanation" in body


def test_predict_physics_capacitor_energy_returns_correct_value(client: TestClient) -> None:
    """The Day-2 solver should compute E = 0.045 J for the textbook problem."""
    payload = {
        "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "physics"
    # Answer is rendered as "0.045 joule".
    assert "0.045" in body["answer"]
    assert "joule" in body["answer"].lower()
    assert body["confidence"] >= 0.8
    # Reasoning trace should reference the formula and conversion.
    assert any("0.5" in step for step in body["cot"])


def test_predict_missing_question_rejected(client: TestClient) -> None:
    response = client.post("/predict", json={"premises-NL": ["foo"]})
    assert response.status_code == 422


def test_predict_request_id_header_echoed(
    client: TestClient, sample_physics_payload: dict[str, object]
) -> None:
    response = client.post(
        "/predict",
        json=sample_physics_payload,
        headers={"X-Request-ID": "test-req-42"},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-req-42"
    assert "X-Response-Time-ms" in response.headers
