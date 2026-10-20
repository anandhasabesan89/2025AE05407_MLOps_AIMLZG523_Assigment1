"""Reusable Inference Module for the Packaged Heart Disease Pipeline.

Loads the serialized scikit-learn pipeline (`models/heart_disease_pipeline.joblib`)
and accompanying metadata to perform deterministic single-patient or batch
predictions with calibrated confidence scores.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd

from src.features.build_features import REQUIRED_INPUT_FEATURES

DEFAULT_MODEL_PATH = Path("models/heart_disease_pipeline.joblib")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")


class HeartDiseasePredictor:
    """Production wrapper around the serialized scikit-learn Pipeline."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Packaged model pipeline not found at {self.model_path}. "
                "Run `python -m src.models.train` first."
            )
        self.pipeline = joblib.load(self.model_path)
        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict_single(self, patient_data: Dict[str, Union[int, float]]) -> Dict[str, Any]:
        """Predict heart disease risk and confidence for a single patient record.

        Args:
            patient_data: Dictionary containing the 13 clinical input features.

        Returns:
            Dictionary with prediction (0/1), label, probability of heart disease,
            confidence in predicted class, risk tier, and model version metadata.
        """
        missing = [col for col in REQUIRED_INPUT_FEATURES if col not in patient_data]
        if missing:
            raise ValueError(f"Missing required clinical features: {missing}")

        row_df = pd.DataFrame([{col: patient_data[col] for col in REQUIRED_INPUT_FEATURES}])
        pred_class = int(self.pipeline.predict(row_df)[0])
        probs = self.pipeline.predict_proba(row_df)[0]
        disease_prob = float(probs[1])
        confidence = float(probs[pred_class])

        if disease_prob >= 0.70:
            risk_tier = "HIGH"
        elif disease_prob >= 0.35:
            risk_tier = "MODERATE"
        else:
            risk_tier = "LOW"

        return {
            "prediction": pred_class,
            "prediction_label": (
                "Heart Disease Detected" if pred_class == 1 else "No Heart Disease"
            ),
            "risk_probability": round(disease_prob, 4),
            "confidence": round(confidence, 4),
            "risk_tier": risk_tier,
            "model_name": self.metadata.get("model_name", "PipelineClassifier"),
            "model_version": self.metadata.get("version", "1.0.0"),
        }


if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    high_risk_sample = {
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
    low_risk_sample = {
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
    print("High-Risk Patient Inference:", json.dumps(predictor.predict_single(high_risk_sample), indent=2))
    print("Low-Risk Patient Inference:", json.dumps(predictor.predict_single(low_risk_sample), indent=2))
