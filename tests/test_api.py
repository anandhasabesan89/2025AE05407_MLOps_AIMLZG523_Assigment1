"""Integration tests for the FastAPI serving and observability endpoints (Task 6 & Task 8)."""

from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import app


def test_api_health_and_predict_endpoints(
    trained_dummy_model_dir: Path,
    sample_patient_payload,
    monkeypatch,
):
    """Test /health, /predict, /metrics, and /dashboard endpoints."""
    model_path = str(trained_dummy_model_dir / "heart_disease_pipeline.joblib")
    monkeypatch.setenv("MODEL_PATH", model_path)

    with TestClient(app) as client:
        # 1. Health endpoint
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["model_loaded"] is True

        # 2. Single prediction endpoint
        pred_res = client.post("/predict", json=sample_patient_payload)
        assert pred_res.status_code == 200
        body = pred_res.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["risk_probability"] <= 1.0
        assert 0.5 <= body["confidence"] <= 1.0
        assert "request_id" in body

        # 3. Validation error on invalid patient age
        bad_payload = dict(sample_patient_payload, age=5)
        bad_res = client.post("/predict", json=bad_payload)
        assert bad_res.status_code == 422

        # 4. Prometheus metrics endpoint
        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200
        assert "heart_api_predictions_total" in metrics_res.text
        assert "heart_api_http_requests_total" in metrics_res.text

        # 5. Live HTML monitoring dashboard
        dash_res = client.get("/dashboard")
        assert dash_res.status_code == 200
        assert "Cardio-Risk MLOps" in dash_res.text
