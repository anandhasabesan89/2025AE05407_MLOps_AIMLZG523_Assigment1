"""Shared pytest fixtures for data, feature engineering, model, and API tests."""

from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.build_features import REQUIRED_INPUT_FEATURES, build_preprocessing_pipeline


@pytest.fixture
def sample_raw_csv(tmp_path: Path) -> Path:
    """Create a synthetic raw UCI Heart Disease CSV containing '?' missing tokens."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 0],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, "?", 3, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, "?", 0],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 3],
        [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0, 3, 0],
        [63, 1, 4, 130, 254, 0, 2, 147, 0, 1.4, 2, 1, 7, 2],
        [53, 1, 4, 140, 203, 1, 2, 155, 1, 3.1, 3, 0, 7, 1],
        [57, 1, 4, 140, 192, 0, 0, 148, 0, 0.4, 2, 0, 6, 0],
        [56, 0, 2, 140, 294, 0, 2, 153, 0, 1.3, 2, 0, 3, 0],
    ]
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    df = pd.DataFrame(rows, columns=cols)
    csv_file = tmp_path / "heart_raw.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_patient_payload() -> Dict[str, Union[int, float]]:
    """Return a valid single-patient clinical feature dictionary."""
    return {
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 260,
        "fbs": 0,
        "restecg": 2,
        "thalach": 135,
        "exang": 1,
        "oldpeak": 2.2,
        "slope": 2,
        "ca": 2,
        "thal": 7,
    }


@pytest.fixture
def trained_dummy_model_dir(tmp_path: Path) -> Path:
    """Create a lightweight fitted pipeline artifact for fast unit/API tests."""
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "age": rng.integers(35, 75, size=n),
            "sex": rng.integers(0, 2, size=n),
            "cp": rng.integers(1, 5, size=n),
            "trestbps": rng.integers(100, 180, size=n),
            "chol": rng.integers(150, 320, size=n),
            "fbs": rng.integers(0, 2, size=n),
            "restecg": rng.integers(0, 3, size=n),
            "thalach": rng.integers(100, 190, size=n),
            "exang": rng.integers(0, 2, size=n),
            "oldpeak": np.round(rng.uniform(0.0, 4.0, size=n), 1),
            "slope": rng.integers(1, 4, size=n),
            "ca": rng.integers(0, 4, size=n),
            "thal": rng.choice([3, 6, 7], size=n),
        }
    )
    y = rng.integers(0, 2, size=n)

    pipe = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipe.fit(df[REQUIRED_INPUT_FEATURES], y)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "heart_disease_pipeline.joblib")
    return model_dir
