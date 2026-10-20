"""Script 02: Feature Engineering, Model Training, CV Tuning & MLflow Tracking.

Executes Tasks 2, 3, and 4 end-to-end as a standalone Python script:
1. Applies custom cardiovascular feature engineering (`ClinicalFeatureEngineer`).
2. Tunes Logistic Regression, Random Forest, and Gradient Boosting via 5-fold Stratified CV.
3. Logs all parameters, metrics, diagnostic plots, and models to MLflow.
4. Packages the winning pipeline into `models/heart_disease_pipeline.joblib`.
"""

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import train_and_evaluate_models  # noqa: E402


def main() -> None:
    """Run multi-model training, MLflow logging, and pipeline packaging."""
    metadata = train_and_evaluate_models()
    print(f"\nWinning Model Selected: {metadata['model_name']}")
    comparison_df = pd.DataFrame(metadata["all_models_comparison"]).T[
        [
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "test_roc_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_brier_score",
        ]
    ]
    print("\n=== 5-FOLD CV & HOLD-OUT TEST BENCHMARK ===")
    print(comparison_df.to_string())


if __name__ == "__main__":
    main()
