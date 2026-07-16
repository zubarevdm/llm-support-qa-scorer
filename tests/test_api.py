"""API smoke tests using the offline HeuristicJudge (no LLM_API_KEY set)."""

import os

import pytest
from fastapi.testclient import TestClient

from support_qa.api import app


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_rubric_endpoint(client):
    resp = client.get("/rubric")
    assert resp.status_code == 200
    assert resp.json()["criteria"]


def test_score_endpoint(client):
    payload = {
        "ticket": {
            "id": "T-1",
            "messages": [
                {"role": "customer", "text": "I was double charged."},
                {"role": "agent", "text": "Sorry about that, I've refunded the duplicate charge already."},
            ],
        }
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_id"] == "T-1"
    assert 0 <= body["overall_score"] <= 100
    assert body["verdict"] in {"excellent", "good", "needs_improvement", "poor"}
