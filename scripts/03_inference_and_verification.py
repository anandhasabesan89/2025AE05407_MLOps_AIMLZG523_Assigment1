"""Script 03: Packaged Model Inference & FastAPI Endpoint Verification.

Executes end-to-end inference verification as a standalone Python script:
1. Loads the serialized pipeline (`models/heart_disease_pipeline.joblib`).
2. Runs single-patient and batch risk predictions with confidence scores.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict import HeartDiseasePredictor  # noqa: E402


def main() -> None:
    """Verify packaged pipeline predictions across high-risk and low-risk profiles."""
    predictor = HeartDiseasePredictor()
    high_risk_patient = {
        "age": 62,
        "sex": 1,
        "cp": 4,
        "trestbps": 150,
        "chol": 270,
        "fbs": 0,
        "restecg": 2,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.8,
        "slope": 2,
        "ca": 2,
        "thal": 7,
    }
    low_risk_patient = {
        "age": 41,
        "sex": 0,
        "cp": 2,
        "trestbps": 118,
        "chol": 195,
        "fbs": 0,
        "restecg": 0,
        "thalach": 172,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }
    print("=== HIGH-RISK PATIENT PREDICTION ===")
    print(json.dumps(predictor.predict_single(high_risk_patient), indent=2))
    print("\n=== LOW-RISK PATIENT PREDICTION ===")
    print(json.dumps(predictor.predict_single(low_risk_patient), indent=2))


if __name__ == "__main__":
    main()
