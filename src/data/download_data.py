"""Automated Data Acquisition Script for the UCI Heart Disease Dataset.

Downloads the processed Cleveland Heart Disease dataset from the UCI Machine
Learning Repository, verifies schema integrity, computes a SHA-256 checksum
for reproducibility, and persists the raw CSV to disk.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_acquisition")

DEFAULT_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file for auditability."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_heart_disease_data(
    url: Optional[str] = None,
    output_path: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Download the UCI Heart Disease dataset and save as structured CSV.

    Args:
        url: Source URL for the raw dataset.
        output_path: Destination file path for the raw CSV.
        columns: Column names to assign to the dataset.

    Returns:
        DataFrame containing the raw dataset (including uncleaned '?' tokens).
    """
    config = load_config()
    data_cfg = config.get("data", {})

    target_url = url or data_cfg.get("uci_url", UCI_CLEVELAND_URL)
    target_path = Path(output_path or data_cfg.get("raw_path", "data/raw/heart_disease_raw.csv"))
    col_names = columns or data_cfg.get("columns", DEFAULT_COLUMNS)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching UCI Heart Disease dataset from: %s", target_url)
    try:
        response = requests.get(target_url, timeout=30)
        response.raise_for_status()
        raw_lines = [
            line.strip()
            for line in response.text.strip().splitlines()
            if line.strip()
        ]
        records = [line.split(",") for line in raw_lines]
        df = pd.DataFrame(records, columns=col_names)
        df.to_csv(target_path, index=False)
    except requests.RequestException as exc:
        if target_path.exists():
            logger.warning(
                "Network request to UCI failed (%s); using cached raw dataset at %s",
                exc,
                target_path,
            )
            df = pd.read_csv(target_path)
        else:
            raise

    checksum = compute_sha256(target_path)
    logger.info(
        "Saved raw dataset to %s (rows=%d, cols=%d, sha256=%s)",
        target_path,
        df.shape[0],
        df.shape[1],
        checksum[:16],
    )
    return df


if __name__ == "__main__":
    download_heart_disease_data()
