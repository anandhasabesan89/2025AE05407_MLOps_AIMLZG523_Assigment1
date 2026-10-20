"""Feature Engineering and Preprocessing Pipeline Module.

Defines a custom scikit-learn compatible `ClinicalFeatureEngineer` transformer
and constructs the full `ColumnTransformer` preprocessing pipeline used across
cross-validation, model training, and production FastAPI inference.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERIC_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "hr_reserve_ratio",
    "rate_pressure_product",
    "chol_age_ratio",
    "st_depression_per_hr",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

REQUIRED_INPUT_FEATURES: List[str] = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific cardiovascular feature engineering transformer.

    Computes four clinically grounded interaction ratios from raw patient vitals:
    1. `hr_reserve_ratio`: Achieved max heart rate relative to age-predicted ceiling (220 - age).
    2. `rate_pressure_product`: Myocardial oxygen consumption proxy (trestbps * thalach / 1000).
    3. `chol_age_ratio`: Serum cholesterol normalized by patient age.
    4. `st_depression_per_hr`: Exercise ST depression normalized by heart rate response.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Stateless fit method returning self for sklearn pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append engineered cardiovascular features to input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=REQUIRED_INPUT_FEATURES)
        df = X.copy()

        age_safe = np.maximum(df["age"].astype(float), 18.0)
        max_hr_pred = np.maximum(220.0 - age_safe, 60.0)
        thalach_float = df["thalach"].astype(float)
        trestbps_float = df["trestbps"].astype(float)
        chol_float = df["chol"].astype(float)
        oldpeak_float = df["oldpeak"].astype(float)

        df["hr_reserve_ratio"] = thalach_float / max_hr_pred
        df["rate_pressure_product"] = (trestbps_float * thalach_float) / 1000.0
        df["chol_age_ratio"] = chol_float / age_safe
        df["st_depression_per_hr"] = oldpeak_float / np.maximum(thalach_float / 100.0, 0.5)

        return df


def build_preprocessing_pipeline() -> Pipeline:
    """Construct full feature engineering and ColumnTransformer preprocessing pipeline.

    Returns:
        scikit-learn Pipeline that transforms 13 raw clinical features into
        scaled numeric + one-hot encoded categorical feature vectors.
    """
    all_numeric = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("clinical_features", ClinicalFeatureEngineer()),
            ("column_preprocessor", column_preprocessor),
        ]
    )
