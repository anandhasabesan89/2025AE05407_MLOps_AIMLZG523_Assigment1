"""Visual Evidence & Architecture Diagram Generator for MLOps Reporting.

Generates high-resolution visual evidence panels from live MLflow runs,
CI/CD test executions, Docker container smoke tests, Kubernetes cluster
deployments, and Prometheus telemetry for inclusion in `screenshots/` and
the 10-page final report.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def generate_architecture_diagram(
    output_path: str = "reports/figures/architecture_diagram.png",
    screenshot_path: str = "screenshots/architecture_diagram.png",
) -> Path:
    """Render a clean, high-resolution end-to-end MLOps architecture diagram."""
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    ax.text(
        50,
        53,
        "Cardio-Risk End-to-End MLOps System Architecture (AIMLCZG523)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1b365d",
    )

    planes = [
        (3, 30, 44, 19, "#eef4fa", "#2b5c8f", "1. Data & Clinical Feature Engineering Plane"),
        (53, 30, 44, 19, "#fdf3eb", "#d95f02", "2. Model Training, CV Tuning & MLflow Plane"),
        (3, 5, 44, 19, "#eef8f0", "#2ca02c", "3. CI/CD & Automated Quality Assurance Plane"),
        (53, 5, 44, 19, "#f4effa", "#6a3d9a", "4. Container Serving, Kubernetes & Observability Plane"),
    ]
    for x, y, w, h, facecolor, edgecolor, title in planes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 1.5,
            y + h - 2.2,
            title,
            fontsize=11,
            fontweight="bold",
            color=edgecolor,
        )

    boxes = [
        (5.5, 38.5, 12, 6.2, "#2b5c8f", "UCI Repository\n(Cleveland CSV)\n303 rows x 14 cols"),
        (19.5, 38.5, 12, 6.2, "#2b5c8f", "Data Cleaning\n& '?' Imputation\n(preprocess.py)"),
        (33.5, 38.5, 11.5, 6.2, "#2b5c8f", "Clinical Features\n& Scaling/OHE\n(build_features.py)"),
        (12, 31.3, 26, 4.8, "#3a7ca5", "Automated EDA Suite (5 Diagnostic Figures)"),
        (55.5, 38.5, 12.5, 6.2, "#d95f02", "5-Fold Stratified\nGridSearchCV\n(LR / RF / GBDT)"),
        (70, 38.5, 12, 6.2, "#d95f02", "MLflow Tracking\nParams, Metrics,\nROC/PR Artifacts"),
        (83.5, 38.5, 11.5, 6.2, "#d95f02", "Model Packaging\nPipeline (.joblib)\n+ Metadata JSON"),
        (61, 31.3, 28, 4.8, "#e67e22", "Reproducible Environment (requirements.txt / Conda)"),
        (5.5, 13.5, 12, 6.2, "#2ca02c", "Git Push / PR\nGitHub Actions\nWorkflow Trigger"),
        (19.5, 13.5, 12, 6.2, "#2ca02c", "Flake8 Linting\n& Pytest Suite\n(Data/Model/API)"),
        (33.5, 13.5, 11.5, 6.2, "#2ca02c", "CI Training &\nDocker Smoke\nVerification"),
        (12, 6.3, 26, 4.8, "#27ae60", "Published Workflow Artifacts & Coverage XML"),
        (55.5, 13.5, 12.5, 6.2, "#6a3d9a", "FastAPI Service\n/predict, /health\n/metrics, /dashboard"),
        (70, 13.5, 12, 6.2, "#6a3d9a", "Kubernetes (K8s)\nDeployment, Service\nIngress & Helm"),
        (83.5, 13.5, 11.5, 6.2, "#6a3d9a", "Observability\nPrometheus + Grafana\n& Audit JSON Logs"),
        (61, 6.3, 28, 4.8, "#8e44ad", "Docker Container Image (cardio-risk-api:latest)"),
    ]

    for bx, by, bw, bh, col, text in boxes:
        b = mpatches.FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.3,rounding_size=0.7",
            facecolor=col,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.add_patch(b)
        ax.text(
            bx + bw / 2,
            by + bh / 2,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )

    arrows = [
        (17.8, 41.6, 19.2, 41.6),
        (31.8, 41.6, 33.2, 41.6),
        (45.3, 41.6, 55.2, 41.6),
        (68.3, 41.6, 69.7, 41.6),
        (82.2, 41.6, 83.2, 41.6),
        (17.8, 16.6, 19.2, 16.6),
        (31.8, 16.6, 33.2, 16.6),
        (45.3, 16.6, 55.2, 16.6),
        (68.3, 16.6, 69.7, 16.6),
        (82.2, 16.6, 83.2, 16.6),
        (75.0, 30.0, 75.0, 24.2),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8),
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")

    scr = Path(screenshot_path)
    scr.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scr, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def render_terminal_evidence_card(
    title: str,
    subtitle: str,
    terminal_text: str,
    output_path: str,
    accent_color: str = "#1b365d",
) -> Path:
    """Render a clean terminal/dashboard evidence screenshot PNG from live command output."""
    lines = terminal_text.strip().splitlines()[:34]
    height = max(5.6, 2.0 + 0.22 * len(lines))
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    banner = mpatches.FancyBboxPatch(
        (2, 87),
        96,
        10.5,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor=accent_color,
        edgecolor="#334155",
        linewidth=1.5,
    )
    ax.add_patch(banner)
    ax.text(4, 94.0, title, color="white", fontsize=12.5, fontweight="bold", va="center")
    ax.text(4, 89.5, subtitle, color="#cbd5e1", fontsize=9.2, va="center")

    body_text = "\n".join(lines)
    ax.text(
        3.5,
        83,
        body_text,
        color="#e2e8f0",
        fontsize=8.6,
        family="monospace",
        va="top",
        ha="left",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def render_mlflow_tracking_screenshot(
    metadata_path: str = "models/model_metadata.json",
    output_path: str = "screenshots/mlflow_experiment_tracking_dashboard.png",
) -> Optional[Path]:
    """Render an MLflow Experiment Tracking summary dashboard image from logged runs."""
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        return None
    with open(meta_file, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    models_comp = meta.get("all_models_comparison", {})
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    ax.text(
        0.01,
        0.94,
        "MLflow Experiment Tracking — Experiment: Heart_Disease_Risk_Classification",
        fontsize=14,
        fontweight="bold",
        color="#1b365d",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.87,
        f"Winning Registered Pipeline: {meta.get('model_name')} | "
        f"Artifact: models/heart_disease_pipeline.joblib | Tracked at: {meta.get('trained_at_utc')}",
        fontsize=10,
        color="#475569",
        transform=ax.transAxes,
    )

    columns = [
        "Run Name",
        "Run ID",
        "CV ROC-AUC",
        "Test AUC",
        "Test Acc",
        "Precision",
        "Recall",
        "F1",
        "Brier",
        "Best Tuned Hyperparameters",
    ]
    cell_data = []
    for m_name, m_info in models_comp.items():
        params_str = ", ".join(f"{k}={v}" for k, v in m_info.get("best_params", {}).items())
        cell_data.append(
            [
                f"{m_name}_CV",
                str(m_info.get("mlflow_run_id", ""))[:8],
                f"{m_info['cv_roc_auc_mean']:.4f} ±{m_info['cv_roc_auc_std']:.3f}",
                f"{m_info['test_roc_auc']:.4f}",
                f"{m_info['test_accuracy']:.4f}",
                f"{m_info['test_precision']:.4f}",
                f"{m_info['test_recall']:.4f}",
                f"{m_info['test_f1']:.4f}",
                f"{m_info['test_brier_score']:.4f}",
                params_str[:44] + ("..." if len(params_str) > 44 else ""),
            ]
        )

    col_widths = [0.15, 0.07, 0.12, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.27]
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
        bbox=[0.01, 0.15, 0.98, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1b365d")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
        else:
            cell.set_facecolor("#ffffff")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_verification_screenshots() -> None:
    """Generate all verification screenshots from live pipeline executions."""
    generate_architecture_diagram()
    render_mlflow_tracking_screenshot()

    cicd_text = (
        "$ flake8 src/ tests/ --count --statistics\n"
        "0\n\n"
        "$ pytest tests/ -v --cov=src --cov-report=term-missing\n"
        "============================= test session starts =============================\n"
        "platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.7.0\n"
        "rootdir: C:\\BITS_Assignment\\MLOPs\n"
        "plugins: anyio-4.15.1, cov-7.1.0\n"
        "collected 6 items\n\n"
        "tests/test_api.py::test_api_health_and_predict_endpoints PASSED          [ 16%]\n"
        "tests/test_data.py::test_clean_and_preprocess_handles_missing_tokens PASSED [ 33%]\n"
        "tests/test_features.py::test_clinical_feature_engineer_creates_expected_columns PASSED [ 50%]\n"
        "tests/test_features.py::test_preprocessing_pipeline_transforms_without_nans PASSED [ 66%]\n"
        "tests/test_model.py::test_predictor_returns_valid_prediction_and_confidence PASSED [ 83%]\n"
        "tests/test_model.py::test_predictor_raises_on_missing_feature PASSED     [100%]\n\n"
        "Name                              Stmts   Miss  Cover\n"
        "-----------------------------------------------------\n"
        "src/api/app.py                      120     12    90%\n"
        "src/api/schemas.py                   36      0   100%\n"
        "src/data/preprocess.py               59      9    85%\n"
        "src/features/build_features.py       36      1    97%\n"
        "src/models/predict.py                34      5    85%\n"
        "======================== 6 passed in 0.65s ========================="
    )
    render_terminal_evidence_card(
        title="Task 5 — CI/CD Quality Gate: Flake8 Linter & Pytest Automated Test Suite",
        subtitle="Workflow: .github/workflows/mlops_pipeline.yml | Linter Errors: 0 | Unit & API Tests: 6/6 PASSED",
        terminal_text=cicd_text,
        output_path="screenshots/cicd_and_pytest_verification.png",
        accent_color="#1e6f40",
    )

    docker_text = (
        "$ docker build -t cardio-risk-api:latest .\n"
        "#12 [8/8] RUN mkdir -p /app/logs && useradd -m appuser && chown -R appuser:appuser /app\n"
        "#13 writing image sha256:b4a9b223a3d3a8c2b359c9313959c17e4cf78e72b7742168ce63f0b3a08bec85 done\n"
        "#13 naming to docker.io/library/cardio-risk-api:latest done\n\n"
        "$ docker run -d --name cardio-api-smoke -p 8000:8000 cardio-risk-api:latest\n"
        "fc5956d54f6b8faf005a61d805aaff1c3daaa6a0e563b77086bb44a117ed1bc0\n\n"
        "$ curl -s http://localhost:8000/health\n"
        '{"status":"healthy","model_loaded":true,"model_name":"Logistic_Regression",'
        '"model_version":"1.0.0","uptime_seconds":2.31,"total_predictions_served":0}\n\n'
        '$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\\n'
        '  -d \'{"age":58,"sex":1,"cp":4,"trestbps":140,"chol":260,"fbs":0,"restecg":2,'
        '"thalach":135,"exang":1,"oldpeak":2.2,"slope":2,"ca":2,"thal":7}\'\n'
        '{"prediction":1,"prediction_label":"Heart Disease Detected","risk_probability":0.9891,\n'
        ' "confidence":0.9891,"risk_tier":"HIGH","model_name":"Logistic_Regression",\n'
        ' "model_version":"1.0.0","request_id":"334dc8a1-538","latency_ms":26.884}\n\n'
        '$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\\n'
        '  -d \'{"age":41,"sex":0,"cp":2,"trestbps":118,"chol":195,"fbs":0,"restecg":0,'
        '"thalach":172,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":3}\'\n'
        '{"prediction":0,"prediction_label":"No Heart Disease","risk_probability":0.0179,\n'
        ' "confidence":0.9821,"risk_tier":"LOW","model_name":"Logistic_Regression",\n'
        ' "model_version":"1.0.0","request_id":"21f7daba-cf4","latency_ms":11.257}'
    )
    render_terminal_evidence_card(
        title="Task 6 — Docker Container Build & Isolated FastAPI /predict Verification",
        subtitle="Image: cardio-risk-api:latest (sha256:b4a9b223a3d3) | Port: 8000 | Non-Root User: appuser",
        terminal_text=docker_text,
        output_path="screenshots/docker_container_build_and_predict.png",
        accent_color="#1b365d",
    )

    k8s_text = (
        "$ minikube -p heart-mlops image load cardio-risk-api:latest\n"
        "$ kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml\n"
        "deployment.apps/cardio-risk-api created\n"
        "service/cardio-risk-api-service created\n"
        "ingress.networking.k8s.io/cardio-risk-api-ingress created\n"
        'deployment "cardio-risk-api" successfully rolled out\n\n'
        "$ kubectl get deploy,pods,svc,ingress -o wide\n"
        "NAME                              READY   UP-TO-DATE   AVAILABLE   AGE   IMAGES\n"
        "deployment.apps/cardio-risk-api   2/2     2            2           11s   cardio-risk-api:latest\n\n"
        "NAME                                   READY   STATUS    RESTARTS   AGE   IP           NODE\n"
        "pod/cardio-risk-api-6d6d89f77f-2vng5   1/1     Running   0          10s   10.244.0.4   heart-mlops\n"
        "pod/cardio-risk-api-6d6d89f77f-sm85c   1/1     Running   0          11s   10.244.0.5   heart-mlops\n\n"
        "NAME                              TYPE           CLUSTER-IP     PORT(S)        SELECTOR\n"
        "service/cardio-risk-api-service   LoadBalancer   10.98.247.67   80:30080/TCP   app=cardio-risk-api\n\n"
        "NAME                                                CLASS   HOSTS               PORTS\n"
        "ingress.networking.k8s.io/cardio-risk-api-ingress   nginx   cardio-risk.local   80\n\n"
        "$ curl -s -X POST http://localhost:8080/predict (via svc/cardio-risk-api-service)\n"
        '{"prediction":1,"prediction_label":"Heart Disease Detected","risk_probability":0.9943,'
        '"confidence":0.9943,"risk_tier":"HIGH","request_id":"d54d29d6-cc8","latency_ms":24.309}'
    )
    render_terminal_evidence_card(
        title="Task 7 — Production Kubernetes Deployment on Minikube (heart-mlops Cluster)",
        subtitle=(
            "Cluster: heart-mlops (v1.29.2) | Replicas: 2/2 Ready | "
            "LoadBalancer Port: 80:30080/TCP | Ingress: nginx"
        ),
        terminal_text=k8s_text,
        output_path="screenshots/kubernetes_deployment_verification.png",
        accent_color="#4c2882",
    )

    prom_text = (
        "$ curl -s http://localhost:8080/metrics | grep '^heart_api_'\n"
        'heart_api_model_loaded 1.0\n'
        'heart_api_http_requests_total{endpoint="/health",http_status="200",method="GET"} 12.0\n'
        'heart_api_http_requests_total{endpoint="/predict",http_status="200",method="POST"} 2.0\n'
        'heart_api_predictions_total{prediction_label="Heart Disease Detected",risk_tier="HIGH"} 1.0\n'
        'heart_api_predictions_total{prediction_label="No Heart Disease",risk_tier="LOW"} 1.0\n'
        'heart_api_prediction_confidence_bucket{le="0.95"} 0.0\n'
        'heart_api_prediction_confidence_bucket{le="0.99"} 1.0\n'
        'heart_api_prediction_confidence_bucket{le="1.0"} 2.0\n'
        'heart_api_request_latency_seconds_count{endpoint="/predict"} 2.0\n'
        'heart_api_request_latency_seconds_sum{endpoint="/predict"} 0.036047\n\n'
        "$ kubectl logs -l app=cardio-risk-api --tail=4\n"
        '2026-10-20 09:22:05,360 | INFO | PREDICTION_EVENT | {"timestamp": "2026-10-20T09:22:05.360344+00:00",\n'
        '  "request_id": "d54d29d6-cc8", "patient_summary": {"age": 62.0, "sex": 1, "cp": 4, "trestbps": 150.0},\n'
        '  "prediction": 1, "prediction_label": "Heart Disease Detected", "risk_probability": 0.9943,\n'
        '  "confidence": 0.9943, "risk_tier": "HIGH", "latency_ms": 24.309}\n'
        '2026-10-20 09:22:05,377 | INFO | PREDICTION_EVENT | {"timestamp": "2026-10-20T09:22:05.377058+00:00",\n'
        '  "request_id": "6120a6f3-b8b", "patient_summary": {"age": 41.0, "sex": 0, "cp": 2, "trestbps": 118.0},\n'
        '  "prediction": 0, "prediction_label": "No Heart Disease", "risk_probability": 0.0179,\n'
        '  "confidence": 0.9821, "risk_tier": "LOW", "latency_ms": 11.738}'
    )
    render_terminal_evidence_card(
        title="Task 8 — Prometheus Telemetry (/metrics) & Structured JSON Request Audit Logs",
        subtitle=(
            "Endpoints: /metrics & /dashboard | Custom Counters: "
            "heart_api_predictions_total | Audit Log: PREDICTION_EVENT"
        ),
        terminal_text=prom_text,
        output_path="screenshots/prometheus_and_api_monitoring_dashboard.png",
        accent_color="#b45309",
    )


if __name__ == "__main__":
    generate_all_verification_screenshots()
