"""Data Cleaning and Preprocessing Module for the UCI Heart Disease Dataset.

Handles missing value token ('?') replacement, numeric type coercion,
deterministic imputation, duplicate removal, and binary target encoding.
Outputs a clean CSV dataset alongside a JSON data quality audit report.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.download_data import download_heart_disease_data, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_preprocess")

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def clean_and_preprocess_data(
    raw_path: Optional[str] = None,
    processed_path: Optional[str] = None,
    quality_report_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean raw UCI Heart Disease data and persist processed CSV & quality report.

    Args:
        raw_path: Path to raw CSV file. Downloaded automatically if missing.
        processed_path: Path to save cleaned CSV file.
        quality_report_path: Path to save JSON data quality summary.

    Returns:
        Tuple of (cleaned DataFrame, quality summary dictionary).
    """
    config = load_config()
    data_cfg = config.get("data", {})

    raw_file = Path(raw_path or data_cfg.get("raw_path", "data/raw/heart_disease_raw.csv"))
    proc_file = Path(
        processed_path or data_cfg.get("processed_path", "data/processed/heart_disease_cleaned.csv")
    )
    report_file = Path(
        quality_report_path or data_cfg.get("quality_report_path", "reports/data_quality_summary.json")
    )

    if not raw_file.exists():
        logger.info("Raw dataset not found at %s; triggering download...", raw_file)
        download_heart_disease_data(output_path=str(raw_file))

    df_raw = pd.read_csv(raw_file, dtype=str)
    initial_rows = len(df_raw)

    # Replace '?' or whitespace tokens with NaN and coerce to float
    df = df_raw.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

    missing_before = {col: int(df[col].isna().sum()) for col in df.columns}
    total_missing_before = int(df.isna().sum().sum())
    logger.info("Missing values detected prior to imputation: %s", missing_before)

    # Impute discrete/categorical columns with mode, continuous columns with median
    imputation_values: Dict[str, float] = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = float(df[col].mode(dropna=True).iloc[0])
            df[col] = df[col].fillna(mode_val)
            imputation_values[col] = mode_val

    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = float(df[col].median(skipna=True))
            df[col] = df[col].fillna(median_val)
            imputation_values[col] = median_val

    # Remove exact duplicates if any
    duplicates_removed = int(df.duplicated().sum())
    if duplicates_removed > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Binarize target: num == 0 -> 0 (Absence), num in {1,2,3,4} -> 1 (Presence)
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])
    elif "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    # Enforce clean integer types on discrete categorical columns
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    proc_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_file, index=False)

    class_counts = df["target"].value_counts().to_dict()
    quality_summary = {
        "initial_row_count": initial_rows,
        "final_row_count": int(len(df)),
        "feature_count": int(df.shape[1] - 1),
        "duplicates_removed": duplicates_removed,
        "total_missing_values_handled": total_missing_before,
        "missing_by_column_before": missing_before,
        "imputation_values_applied": imputation_values,
        "missing_after_cleaning": int(df.isna().sum().sum()),
        "target_distribution": {
            "0_no_disease": int(class_counts.get(0, 0)),
            "1_heart_disease": int(class_counts.get(1, 0)),
            "positive_rate": round(float(df["target"].mean()), 4),
        },
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)

    logger.info(
        "Cleaned dataset saved to %s (rows=%d, cols=%d, positive_rate=%.2f%%)",
        proc_file,
        len(df),
        df.shape[1],
        quality_summary["target_distribution"]["positive_rate"] * 100,
    )
    return df, quality_summary


if __name__ == "__main__":
    clean_and_preprocess_data()
