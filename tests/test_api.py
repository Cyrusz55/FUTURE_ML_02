"""API and priority-rule tests using only the local model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.main import app
from machine_learning.machine_learning import assign_priority


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_response(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={"text": "My laptop is broken and crashes urgently."},
    )
    body = response.json()
    assert response.status_code == 200
    assert {"category", "category_confidence", "top_categories", "priority"}.issubset(body)
    assert 0 < body["category_confidence"] <= 1
    assert body["priority"] in {"High", "Medium", "Low"}


def test_invalid_ticket_payloads(client: TestClient) -> None:
    invalid = [{}, {"text": ""}, {"text": " four"}, {"text": "x" * 5001}, {"text": 42}]
    for payload in invalid:
        assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_batch_limits_and_consistency(client: TestClient) -> None:
    text = "My laptop is broken and crashes urgently."
    batch = client.post("/api/v1/predict/batch", json={"texts": [text] * 100})
    assert batch.status_code == 200
    assert batch.json()["results"][0] == client.post("/api/v1/predict", json={"text": text}).json()
    assert client.post("/api/v1/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/api/v1/predict/batch", json={"texts": [text] * 101}).status_code == 422


def test_response_model_filters_extra_keys(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"text": "Please help with a planned upgrade."})
    assert "priority_terms_matched" not in response.json()


def test_model_info_and_index(client: TestClient) -> None:
    info = client.get("/api/v1/model-info")
    assert info.status_code == 200
    assert {
        "model_name",
        "hf_repo_id",
        "hf_revision",
        "class_names",
        "temperature",
        "rules_version",
        "high_at",
        "medium_at",
        "library_versions",
    } == set(info.json())
    assert client.get("/").status_code == 200
    assert "IT Ticket Classifier" in client.get("/").text


def test_priority_explicit_urgency() -> None:
    priority, points, rules, _ = assign_priority("urgent access issue", "Access")
    assert (priority, points) == ("High", 4)
    assert "explicit urgency" in rules


def test_priority_service_failure() -> None:
    priority, points, rules, _ = assign_priority("the service is down", "Miscellaneous")
    assert (priority, points) == ("Medium", 2)
    assert "service failure" in rules


def test_priority_routine_request() -> None:
    priority, points, rules, _ = assign_priority("how to request an upgrade", "Purchase")
    assert priority == "Low"
    assert points == -3
    assert "routine request" in rules


def test_priority_maintenance_notice() -> None:
    priority, points, rules, _ = assign_priority("planned maintenance", "Storage")
    assert priority == "Low"
    assert points == -6
    assert {"scheduled or planned", "maintenance notice"} <= set(rules)


def test_priority_wide_impact_and_blocked() -> None:
    priority, points, rules, _ = assign_priority("everyone is unable to log in", "Access")
    assert priority == "High"
    assert points == 5
    assert {"wide impact", "blocked from working"} <= set(rules)
