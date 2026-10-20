"""Unit tests for clinical feature engineering and ColumnTransformer (Task 2 & Task 5)."""

import numpy as np
import pandas as pd

from src.features.build_features import (
    ENGINEERED_NUMERIC_FEATURES,
    REQUIRED_INPUT_FEATURES,
    ClinicalFeatureEngineer,
    build_preprocessing_pipeline,
)


def test_clinical_feature_engineer_creates_expected_columns(sample_patient_payload):
    """Verify custom clinical features are accurately computed."""
    df_in = pd.DataFrame([sample_patient_payload])
    engineer = ClinicalFeatureEngineer()
    df_out = engineer.fit_transform(df_in)

    for col in ENGINEERED_NUMERIC_FEATURES:
        assert col in df_out.columns

    expected_rpp = (sample_patient_payload["trestbps"] * sample_patient_payload["thalach"]) / 1000.0
    assert np.isclose(df_out["rate_pressure_product"].iloc[0], expected_rpp)
    expected_hr_ratio = sample_patient_payload["thalach"] / (220.0 - sample_patient_payload["age"])
    assert np.isclose(df_out["hr_reserve_ratio"].iloc[0], expected_hr_ratio)


def test_preprocessing_pipeline_transforms_without_nans(sample_patient_payload):
    """Verify full preprocessing pipeline produces finite numeric vectors."""
    row2 = dict(sample_patient_payload, age=44, sex=0, cp=2, chol=210, thalach=168, ca=0, thal=3)
    df = pd.DataFrame([sample_patient_payload, row2])[REQUIRED_INPUT_FEATURES]

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(REQUIRED_INPUT_FEATURES)
    assert not np.isnan(transformed).any()
