# Production Dockerfile for Cardio-Risk MLOps FastAPI Service (Task 6)
FROM python:3.11-slim

LABEL maintainer="BITS Pilani MLOps Assignment 01" \
      service="cardio-risk-inference-api" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/heart_disease_pipeline.joblib \
    METADATA_PATH=/app/models/model_metadata.json \
    PORT=8000

WORKDIR /app

# Install runtime dependencies only for a minimal, fast container image
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy>=1.26.0,<2.0.0" \
        "pandas>=2.1.0" \
        "scikit-learn>=1.4.0" \
        "scipy>=1.11.0" \
        "joblib>=1.3.2" \
        "pyyaml>=6.0.1" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.28.0" \
        "pydantic>=2.6.0" \
        "prometheus-client>=0.20.0" \
        "prometheus-fastapi-instrumentator>=7.0.0"

# Copy application source, configuration, and packaged model artifacts
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY models/ /app/models/

RUN mkdir -p /app/logs && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
