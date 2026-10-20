#!/usr/bin/env bash
# ==============================================================================
# Cardio-Risk End-to-End MLOps Master Demo Script (BITS Prayogshala VM)
# Student ID: 2025AE05407 | Course: AIMLCZG523 (Assignment 01)
# Usage: bash run_demo.sh
# ==============================================================================

set -e

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

banner "STEP 0: Environment Setup & Dependency Installation"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
export PYTHONPATH="$(pwd)"
export MLFLOW_ALLOW_FILE_STORE="true"
export MLFLOW_DISABLE_AGENT_HINT="1"
export GIT_PYTHON_REFRESH="quiet"
export MPLBACKEND="Agg"
echo -e "${GREEN}[OK] Python virtual environment ready.${NC}"

banner "TASK 1: Data Acquisition, Cleaning & Exploratory Data Analysis (EDA)"
python scripts/01_eda_and_cleaning.py

banner "TASKS 2, 3 & 4: Feature Engineering, 5-Fold CV Tuning, MLflow & Packaging"
python scripts/02_feature_engineering_and_modeling.py
python scripts/03_inference_and_verification.py

banner "TASK 5: Code Quality Linting (Flake8) & Automated Unit Test Suite (Pytest)"
python -m flake8 src/ scripts/ tests/ --count --statistics
python -m pytest tests/ -v --cov=src --cov-report=term-missing

banner "TASK 6: Docker Container Build & Isolated FastAPI Smoke Test"
if command -v docker >/dev/null 2>&1; then
    docker rm -f cardio-api-demo >/dev/null 2>&1 || true
    docker build -t cardio-risk-api:latest .
    docker run -d --name cardio-api-demo -p 8000:8000 cardio-risk-api:latest
    echo "Waiting for container health check..."
    for i in $(seq 1 15); do
        if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    echo -e "${YELLOW}--> GET /health:${NC}"
    curl -s http://localhost:8000/health && echo ""
    echo -e "${YELLOW}--> POST /predict (High-Risk Patient):${NC}"
    curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":58,"sex":1,"cp":4,"trestbps":140,"chol":260,"fbs":0,"restecg":2,"thalach":135,"exang":1,"oldpeak":2.2,"slope":2,"ca":2,"thal":7}' && echo ""
    echo -e "${YELLOW}--> POST /predict (Low-Risk Patient):${NC}"
    curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":41,"sex":0,"cp":2,"trestbps":118,"chol":195,"fbs":0,"restecg":0,"thalach":172,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":3}' && echo ""
    docker rm -f cardio-api-demo >/dev/null 2>&1 || true
else
    echo "Docker CLI not found in PATH; skipping container build."
fi

banner "TASKS 7 & 8: Kubernetes Deployment (Minikube) & Prometheus Monitoring"
if command -v kubectl >/dev/null 2>&1 && command -v minikube >/dev/null 2>&1; then
    if ! minikube status >/dev/null 2>&1; then
        echo "Starting Minikube cluster..."
        minikube start
    fi
    echo "Loading cardio-risk-api:latest into Minikube..."
    minikube image load cardio-risk-api:latest
    kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
    kubectl rollout status deployment/cardio-risk-api --timeout=90s
    kubectl get deploy,pods,svc,ingress -o wide

    pkill -f "kubectl port-forward" >/dev/null 2>&1 || true
    nohup kubectl port-forward svc/cardio-risk-api-service 8080:80 >/tmp/k8s_pf.log 2>&1 &
    sleep 3

    echo -e "${YELLOW}--> Live Kubernetes Service POST /predict:${NC}"
    curl -s -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{"age":62,"sex":1,"cp":4,"trestbps":150,"chol":270,"fbs":0,"restecg":2,"thalach":120,"exang":1,"oldpeak":2.8,"slope":2,"ca":2,"thal":7}' && echo ""

    echo -e "${YELLOW}--> Live Prometheus Telemetry (/metrics):${NC}"
    curl -s http://localhost:8080/metrics | grep "^heart_api_" | head -n 15

    echo -e "${YELLOW}--> Kubernetes Pod Structured Request Audit Logs:${NC}"
    kubectl logs -l app=cardio-risk-api --tail=8
else
    echo "Starting local FastAPI server on port 8080..."
    pkill -f "uvicorn src.api.app:app" >/dev/null 2>&1 || true
    nohup python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8080 >/tmp/uvicorn.log 2>&1 &
    sleep 3
fi

banner "STARTING MLFLOW UI SERVER (PORT 5000)"
pkill -f "mlflow ui" >/dev/null 2>&1 || true
nohup mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000 >/tmp/mlflow_ui.log 2>&1 &
sleep 2

banner "ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!"
echo -e "Open these URLs in your VM browser for the video demonstration:"
echo -e "  1. MLflow Tracking UI        : ${GREEN}http://localhost:5000${NC}"
echo -e "  2. FastAPI Swagger Docs      : ${GREEN}http://localhost:8080/docs${NC}"
echo -e "  3. Live Monitoring Dashboard : ${GREEN}http://localhost:8080/dashboard${NC}"
echo -e "  4. Prometheus Metrics        : ${GREEN}http://localhost:8080/metrics${NC}"
echo ""
