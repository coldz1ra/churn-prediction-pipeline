import pathlib

import pytest
from fastapi.testclient import TestClient

ARTIFACTS_OK = (
    pathlib.Path("artifacts/model.joblib").exists()
    and pathlib.Path("artifacts/columns.json").exists()
)
pytestmark = pytest.mark.skipif(not ARTIFACTS_OK, reason="artifacts not found, skip API smoke")


def test_health():
    from src.app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_one():
    from src.app import app

    client = TestClient(app)
    payload = {
        "data": [
            {
                "revenue": 120,
                "active_days": 12,
                "tenure_days": 180,
                "contract_type": "Month-to-month",
            }
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert "churn_proba" in js
    assert isinstance(js["churn_proba"], list)
