"""API tests with FastAPI's TestClient: no server needed, and they run in CI (the model file is committed)."""
import pytest
from fastapi.testclient import TestClient

import api
from src.config import COLS
from src.scorer import Scorer


@pytest.fixture
def client():
    api.scorer = Scorer(api.scorer.model)   # fresh, empty state for every test; same loaded model
    return TestClient(api.app)


def flight(cycle):
    return {"cycle": cycle, **{c: 1.0 for c in COLS[2:]}}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"].startswith("mlp_seed0.joblib@")


def test_normal_sequence_of_flights(client):
    for cycle in [1, 2, 3]:
        r = client.post("/engines/7/cycles", json=flight(cycle))
        assert r.status_code == 200
    body = r.json()
    assert body["unit"] == 7 and body["cycle"] == 3 and body["cycles_seen"] == 3
    assert isinstance(body["predicted_rul"], float)
    assert body["alert"] == (body["predicted_rul"] <= 20)
    assert client.get("/engines/7").json() == body   # GET returns the latest answer


def test_missing_field_is_422(client):
    body = flight(1)
    del body["s4"]
    r = client.post("/engines/7/cycles", json=body)
    assert r.status_code == 422
    assert "s4" in r.text


def test_non_numeric_or_non_finite_value_is_422(client):
    for bad in ["abc", None]:
        assert client.post("/engines/7/cycles", json={**flight(1), "s2": bad}).status_code == 422
    nan_body = '{"cycle": 1, ' + ", ".join(f'"{c}": 1.0' for c in COLS[2:]).replace('"s2": 1.0', '"s2": NaN') + "}"
    r = client.post("/engines/7/cycles", content=nan_body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_repeated_cycle_is_409(client):
    assert client.post("/engines/7/cycles", json=flight(5)).status_code == 200
    r = client.post("/engines/7/cycles", json=flight(5))
    assert r.status_code == 409
    assert "not after its last cycle" in r.json()["detail"]
    assert client.get("/engines/7").json()["cycles_seen"] == 1   # the rejected flight was not stored


def test_unknown_engine_is_404(client):
    r = client.get("/engines/999")
    assert r.status_code == 404
    assert "999" in r.json()["detail"]
