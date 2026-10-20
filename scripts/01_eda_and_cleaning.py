"""Script 01: Data Acquisition, Cleaning & Exploratory Data Analysis (EDA).

Executes Task 1 end-to-end as a standalone Python script:
1. Downloads the raw UCI Heart Disease dataset (`data/raw/heart_disease_raw.csv`).
2. Cleans and imputes missing '?' tokens (`data/processed/heart_disease_cleaned.csv`).
3. Generates 5 multi-panel clinical EDA figures (`reports/figures/`).
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download_data import download_heart_disease_data  # noqa: E402
from src.data.preprocess import clean_and_preprocess_data  # noqa: E402
from src.visualization.eda_plots import generate_all_eda_plots  # noqa: E402


def main() -> None:
    """Run data download, preprocessing, and EDA visualization pipeline."""
    download_heart_disease_data()
    df_clean, quality_summary = clean_and_preprocess_data()
    print("=== DATA QUALITY AUDIT SUMMARY ===")
    print(json.dumps(quality_summary, indent=2))
    print("\n=== SUMMARY STATISTICS ===")
    print(df_clean.describe().T[["count", "mean", "std", "min", "50%", "max"]])
    plots = generate_all_eda_plots()
    print("\nGenerated EDA Plots:", [str(p) for p in plots])


if __name__ == "__main__":
    main()
