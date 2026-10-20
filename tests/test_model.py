"""Unit tests for packaged model inference and error handling (Task 4 & Task 5)."""

from pathlib import Path
import pytest

from src.models.predict import HeartDiseasePredictor


def test_predictor_returns_valid_prediction_and_confidence(
    trained_dummy_model_dir: Path,
    sample_patient_payload,
):
    """Verify HeartDiseasePredictor returns valid probability, confidence, and risk tier."""
    model_file = trained_dummy_model_dir / "heart_disease_pipeline.joblib"
    predictor = HeartDiseasePredictor(model_path=model_file)

    result = predictor.predict_single(sample_patient_payload)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["risk_tier"] in ("LOW", "MODERATE", "HIGH")


def test_predictor_raises_on_missing_feature(
    trained_dummy_model_dir: Path,
    sample_patient_payload,
):
    """Verify missing input features raise a clear ValueError."""
    model_file = trained_dummy_model_dir / "heart_disease_pipeline.joblib"
    predictor = HeartDiseasePredictor(model_path=model_file)

    incomplete = dict(sample_patient_payload)
    incomplete.pop("thalach")

    with pytest.raises(ValueError, match="Missing required clinical features"):
        predictor.predict_single(incomplete)
