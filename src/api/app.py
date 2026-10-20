"""FastAPI Microservice for Heart Disease Risk Prediction & Observability.

Implements Tasks 6 & 8:
- Exposes `/predict`, `/predict/batch`, `/health`, `/metrics`, and `/dashboard`.
- Structured JSON request/prediction logging to stdout and `logs/api_requests.log`.
- Prometheus telemetry counters, histograms, and gauges for API latency and
  clinical prediction drift monitoring.
"""

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Deque, Dict, Any
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.schemas import (
    BatchPatientRequest,
    HealthResponse,
    PatientHeartRecord,
    PredictionResponse,
)
from src.models.predict import HeartDiseasePredictor

# Configure structured logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api_requests.log"

logger = logging.getLogger("cardio_risk_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def _get_or_create_metric(metric_cls, name: str, documentation: str, labelnames=(), **kwargs):
    """Safely register or retrieve an existing Prometheus metric across reloads."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_cls(name, documentation, labelnames=labelnames, **kwargs)


HTTP_REQUESTS_TOTAL = _get_or_create_metric(
    Counter,
    "heart_api_http_requests_total",
    "Total HTTP requests processed by the Heart Disease API",
    labelnames=("method", "endpoint", "http_status"),
)
HTTP_REQUEST_LATENCY = _get_or_create_metric(
    Histogram,
    "heart_api_request_latency_seconds",
    "Request latency in seconds by endpoint",
    labelnames=("endpoint",),
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTIONS_TOTAL = _get_or_create_metric(
    Counter,
    "heart_api_predictions_total",
    "Total clinical predictions categorized by diagnosis label and risk tier",
    labelnames=("prediction_label", "risk_tier"),
)
PREDICTION_CONFIDENCE = _get_or_create_metric(
    Histogram,
    "heart_api_prediction_confidence",
    "Distribution of model prediction confidence scores",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
MODEL_LOADED_GAUGE = _get_or_create_metric(
    Gauge,
    "heart_api_model_loaded",
    "Indicates whether the ML pipeline is loaded and ready (1=loaded, 0=unloaded)",
)

START_TIME = time.time()
RECENT_LOGS: Deque[Dict[str, Any]] = deque(maxlen=50)
STATE: Dict[str, Any] = {
    "predictor": None,
    "total_predictions": 0,
    "class_counts": {"No Heart Disease": 0, "Heart Disease Detected": 0},
    "risk_tiers": {"LOW": 0, "MODERATE": 0, "HIGH": 0},
}


def load_predictor_if_available() -> None:
    """Load the packaged model pipeline into memory."""
    model_path = os.environ.get("MODEL_PATH", "models/heart_disease_pipeline.joblib")
    metadata_path = os.environ.get("METADATA_PATH", "models/model_metadata.json")
    try:
        STATE["predictor"] = HeartDiseasePredictor(
            model_path=model_path,
            metadata_path=metadata_path,
        )
        MODEL_LOADED_GAUGE.set(1)
        logger.info(
            "Loaded HeartDiseasePredictor from %s (model=%s)",
            model_path,
            STATE["predictor"].metadata.get("model_name", "Unknown"),
        )
    except FileNotFoundError as exc:
        STATE["predictor"] = None
        MODEL_LOADED_GAUGE.set(0)
        logger.warning("Model file not yet available during startup: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI startup and shutdown lifecycle."""
    load_predictor_if_available()
    yield


app = FastAPI(
    title="Cardio-Risk MLOps Inference API",
    description=(
        "Production-grade REST API for UCI Heart Disease Risk Classification "
        "with integrated Prometheus telemetry and structured audit logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Record HTTP request counts and latency histograms for Prometheus."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    path = request.url.path
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=path,
        http_status=str(response.status_code),
    ).inc()
    HTTP_REQUEST_LATENCY.labels(endpoint=path).observe(elapsed)
    return response


@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Root service discovery endpoint."""
    return {
        "service": "Cardio-Risk MLOps Inference API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
def health_check() -> HealthResponse:
    """Kubernetes liveness and readiness probe endpoint."""
    if STATE["predictor"] is None:
        load_predictor_if_available()
    predictor = STATE["predictor"]
    is_loaded = predictor is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=predictor.metadata.get("model_name", "unloaded") if is_loaded else "unloaded",
        model_version=predictor.metadata.get("version", "0.0.0") if is_loaded else "0.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
        total_predictions_served=STATE["total_predictions"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_heart_disease(patient: PatientHeartRecord, request: Request) -> PredictionResponse:
    """Predict heart disease risk and confidence for a single patient."""
    if STATE["predictor"] is None:
        load_predictor_if_available()
    predictor: HeartDiseasePredictor = STATE["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:12]
    payload = patient.model_dump()
    result = predictor.predict_single(payload)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # Update Prometheus & internal monitoring counters
    PREDICTIONS_TOTAL.labels(
        prediction_label=result["prediction_label"],
        risk_tier=result["risk_tier"],
    ).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])

    STATE["total_predictions"] += 1
    STATE["class_counts"][result["prediction_label"]] = (
        STATE["class_counts"].get(result["prediction_label"], 0) + 1
    )
    STATE["risk_tiers"][result["risk_tier"]] = (
        STATE["risk_tiers"].get(result["risk_tier"], 0) + 1
    )

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "client_ip": request.client.host if request.client else "local",
        "patient_summary": {
            "age": payload["age"],
            "sex": payload["sex"],
            "cp": payload["cp"],
            "trestbps": payload["trestbps"],
            "chol": payload["chol"],
            "thalach": payload["thalach"],
        },
        "prediction": result["prediction"],
        "prediction_label": result["prediction_label"],
        "risk_probability": result["risk_probability"],
        "confidence": result["confidence"],
        "risk_tier": result["risk_tier"],
        "latency_ms": latency_ms,
    }
    RECENT_LOGS.appendleft(log_entry)
    logger.info("PREDICTION_EVENT | %s", json.dumps(log_entry))

    return PredictionResponse(
        **result,
        request_id=request_id,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(batch_req: BatchPatientRequest, request: Request) -> Dict[str, Any]:
    """Run batch inference across multiple patient records."""
    responses = [predict_heart_disease(p, request) for p in batch_req.patients]
    return {
        "batch_size": len(responses),
        "predictions": [r.model_dump() for r in responses],
    }


@app.get("/metrics", tags=["Observability"])
def metrics() -> Response:
    """Prometheus scrape endpoint exposing HTTP and ML model metrics."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Observability"])
def monitoring_dashboard() -> HTMLResponse:
    """Built-in visual monitoring and request log dashboard."""
    predictor = STATE["predictor"]
    model_name = predictor.metadata.get("model_name", "Not Loaded") if predictor else "Not Loaded"
    uptime = round(time.time() - START_TIME, 1)
    rows_html = ""
    for item in list(RECENT_LOGS)[:15]:
        badge_color = "#d95f02" if item["prediction"] == 1 else "#2ca02c"
        rows_html += (
            f"<tr>"
            f"<td><code>{item['request_id']}</code></td>"
            f"<td>{item['timestamp'][:19]}Z</td>"
            f"<td>Age {int(item['patient_summary']['age'])}, CP={item['patient_summary']['cp']}, "
            f"BP={int(item['patient_summary']['trestbps'])}, HR={int(item['patient_summary']['thalach'])}</td>"
            f"<td><span style='color:white;background:{badge_color};padding:2px 8px;border-radius:4px;'>"
            f"{item['prediction_label']}</span></td>"
            f"<td><b>{item['risk_probability']:.3f}</b> ({item['risk_tier']})</td>"
            f"<td>{item['confidence']:.3f}</td>"
            f"<td>{item['latency_ms']:.2f} ms</td>"
            f"</tr>"
        )
    if not rows_html:
        rows_html = (
            "<tr><td colspan='7' style='text-align:center;'>"
            "No prediction requests logged yet.</td></tr>"
        )

    pos_count = STATE["class_counts"]["Heart Disease Detected"]
    neg_count = STATE["class_counts"]["No Heart Disease"]
    tiers = STATE["risk_tiers"]
    tier_str = f"{tiers['LOW']} / {tiers['MODERATE']} / {tiers['HIGH']}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cardio-Risk MLOps — Live Observability Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; background: #f4f6f9; color: #222; }}
            .header {{ background: #1b365d; color: #fff; padding: 18px; border-radius: 8px; margin-bottom: 18px; }}
            .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
            .card {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
            .card h3 {{ margin: 0 0 8px 0; font-size: 13px; color: #555; text-transform: uppercase; }}
            .card .val {{ font-size: 26px; font-weight: bold; color: #1b365d; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 11px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 13.5px; }}
            th {{ background: #2b5c8f; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2 style="margin:0;">Cardio-Risk MLOps — Real-Time API Monitoring & Audit Log Dashboard</h2>
            <p style="margin:6px 0 0 0; opacity:0.9;">
                Active Model: <b>{model_name}</b> | Uptime: <b>{uptime}s</b> |
                Prometheus Endpoint: <a href="/metrics" style="color:#ffd166;">/metrics</a>
            </p>
        </div>
        <div class="cards">
            <div class="card"><h3>Total Predictions</h3><div class="val">{STATE['total_predictions']}</div></div>
            <div class="card"><h3>Heart Disease</h3><div class="val" style="color:#d95f02;">{pos_count}</div></div>
            <div class="card"><h3>No Disease</h3><div class="val" style="color:#2ca02c;">{neg_count}</div></div>
            <div class="card"><h3>Risk Tier (L / M / H)</h3><div class="val">{tier_str}</div></div>
        </div>
        <h3>Recent Inference Audit Logs (Last 15 Requests)</h3>
        <table>
            <thead>
                <tr>
                    <th>Request ID</th><th>Timestamp (UTC)</th><th>Patient Vitals Summary</th>
                    <th>Prediction</th><th>Risk Prob (Tier)</th><th>Confidence</th><th>Latency</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
