"""Unit tests for data acquisition and preprocessing (Task 1 & Task 5)."""

from pathlib import Path

from src.data.preprocess import clean_and_preprocess_data


def test_clean_and_preprocess_handles_missing_tokens(sample_raw_csv: Path, tmp_path: Path):
    """Verify '?' tokens are imputed, 'num' is binarized to 'target', and no NaNs remain."""
    processed_path = tmp_path / "heart_cleaned.csv"
    report_path = tmp_path / "quality_report.json"

    df_clean, summary = clean_and_preprocess_data(
        raw_path=str(sample_raw_csv),
        processed_path=str(processed_path),
        quality_report_path=str(report_path),
    )

    assert processed_path.exists()
    assert report_path.exists()
    assert df_clean.isna().sum().sum() == 0
    assert "target" in df_clean.columns
    assert "num" not in df_clean.columns
    assert set(df_clean["target"].unique()).issubset({0, 1})
    assert summary["total_missing_values_handled"] == 2
    assert summary["missing_after_cleaning"] == 0
