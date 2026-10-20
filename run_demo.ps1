# ==============================================================================
# Cardio-Risk End-to-End MLOps Master Demo Script (Windows PowerShell)
# Student ID: 2025AE05407 | Course: AIMLCZG523 (Assignment 01)
# Usage: .\run_demo.ps1
# ==============================================================================

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = (Get-Location).Path
$env:MLFLOW_ALLOW_FILE_STORE = "true"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:GIT_PYTHON_REFRESH = "quiet"
$env:MPLBACKEND = "Agg"

function Write-Banner($Title) {
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Green
    Write-Host "==============================================================================" -ForegroundColor Cyan
}

Write-Banner "STEP 0: Environment Setup"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt -q

Write-Banner "TASK 1: Data Acquisition, Cleaning & Exploratory Data Analysis (EDA)"
& $py scripts/01_eda_and_cleaning.py

Write-Banner "TASKS 2, 3 & 4: Feature Engineering, 5-Fold CV Tuning, MLflow & Packaging"
& $py scripts/02_feature_engineering_and_modeling.py
& $py scripts/03_inference_and_verification.py

Write-Banner "TASK 5: Code Quality Linting (Flake8) & Automated Unit Test Suite (Pytest)"
& $py -m flake8 src/ scripts/ tests/ --count --statistics
& $py -m pytest tests/ -v --cov=src --cov-report=term-missing

Write-Banner "TASK 6: Docker Container Build & Isolated FastAPI Smoke Test"
docker rm -f cardio-api-demo 2>$null
docker build -t cardio-risk-api:latest .
docker run -d --name cardio-api-demo -p 8000:8000 cardio-risk-api:latest
Start-Sleep -Seconds 4
Write-Host "--> GET /health:" -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json -Compress
Write-Host "--> POST /predict (High-Risk Patient):" -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"age":58,"sex":1,"cp":4,"trestbps":140,"chol":260,"fbs":0,"restecg":2,"thalach":135,"exang":1,"oldpeak":2.2,"slope":2,"ca":2,"thal":7}' | ConvertTo-Json -Compress
docker rm -f cardio-api-demo 2>$null

Write-Banner "TASKS 7 & 8: Kubernetes Deployment & Prometheus Telemetry"
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
kubectl get deploy,pods,svc,ingress -o wide
Start-Process -FilePath "kubectl" -ArgumentList "port-forward svc/cardio-risk-api-service 8080:80" -WindowStyle Hidden
Start-Process -FilePath ".\.venv\Scripts\mlflow.exe" -ArgumentList "ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000" -WindowStyle Hidden
Start-Sleep -Seconds 3

Write-Banner "ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!"
Write-Host "Open these URLs in your browser for the video demonstration:"
Write-Host "  1. MLflow Tracking UI        : http://localhost:5000" -ForegroundColor Green
Write-Host "  2. FastAPI Swagger Docs      : http://localhost:8080/docs" -ForegroundColor Green
Write-Host "  3. Live Monitoring Dashboard : http://localhost:8080/dashboard" -ForegroundColor Green
Write-Host "  4. Prometheus Metrics        : http://localhost:8080/metrics" -ForegroundColor Green
