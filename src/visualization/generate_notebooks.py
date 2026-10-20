"""Generator for Reproducible Jupyter Notebooks (Deliverable Requirement).

Creates and optionally executes the three core Jupyter notebooks:
1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_feature_engineering_and_modeling.ipynb`
3. `notebooks/03_inference_and_verification.ipynb`
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def create_notebooks(output_dir: str = "notebooks") -> None:
    """Create the three structured Jupyter notebooks."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Notebook 1: EDA & Data Cleaning
    nb1 = new_notebook()
    nb1.cells = [
        new_markdown_cell(
            "# 01 — UCI Heart Disease Data Acquisition, Cleaning & Exploratory Data Analysis\n"
            "**Course**: Machine Learning Operations (MLOps) AIMLCZG523 — Assignment 01\n\n"
            "This notebook demonstrates automated dataset acquisition from the UCI Machine Learning Repository, "
            "missing value imputation (`ca` and `thal`), binary target transformation, and clinical EDA."
        ),
        new_code_cell(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
            "import pandas as pd\nimport matplotlib.pyplot as plt\n"
            "from src.data.download_data import download_heart_disease_data\n"
            "from src.data.preprocess import clean_and_preprocess_data\n"
            "from src.visualization.eda_plots import generate_all_eda_plots"
        ),
        new_code_cell(
            "# 1. Acquire and clean the UCI Cleveland Heart Disease dataset\n"
            "df_clean, quality_report = clean_and_preprocess_data(\n"
            "    raw_path='../data/raw/heart_disease_raw.csv',\n"
            "    processed_path='../data/processed/heart_disease_cleaned.csv',\n"
            "    quality_report_path='../reports/data_quality_summary.json'\n"
            ")\n"
            "print('Quality Summary:', quality_report)\n"
            "df_clean.head(10)"
        ),
        new_code_cell(
            "# 2. Summary statistics across clinical features\n"
            "df_clean.describe().T"
        ),
        new_code_cell(
            "# 3. Generate high-resolution EDA plots\n"
            "plots = generate_all_eda_plots(\n"
            "    processed_csv='../data/processed/heart_disease_cleaned.csv',\n"
            "    output_dir='../reports/figures',\n"
            "    screenshots_dir='../screenshots'\n"
            ")\n"
            "print('Generated Figures:', [p.name for p in plots])"
        ),
    ]
    with open(out / "01_eda_and_cleaning.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb1, f)

    # Notebook 2: Feature Engineering, Model Development & MLflow Tracking
    nb2 = new_notebook()
    nb2.cells = [
        new_markdown_cell(
            "# 02 — Clinical Feature Engineering, Model Tuning & MLflow Experiment Tracking\n"
            "**Course**: Machine Learning Operations (MLOps) AIMLCZG523 — Assignment 01\n\n"
            "Trains and tunes Logistic Regression, Random Forest, and Gradient Boosting classifiers "
            "with custom cardiovascular interaction features and 5-fold Stratified Cross-Validation."
        ),
        new_code_cell(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
            "import json\nimport pandas as pd\n"
            "from src.features.build_features import ClinicalFeatureEngineer, build_preprocessing_pipeline\n"
            "from src.models.train import train_and_evaluate_models"
        ),
        new_code_cell(
            "# 1. Preview engineered cardiovascular features\n"
            "df = pd.read_csv('../data/processed/heart_disease_cleaned.csv')\n"
            "engineer = ClinicalFeatureEngineer()\n"
            "df_engineered = engineer.fit_transform(df.drop(columns=['target']))\n"
            "df_engineered[['age', 'trestbps', 'chol', 'thalach', 'hr_reserve_ratio', "
            "'rate_pressure_product', 'chol_age_ratio', 'st_depression_per_hr']].head()"
        ),
        new_code_cell(
            "# 2. Execute or load multi-model CV tuning, MLflow logging, and model packaging\n"
            "meta_path = Path('../models/model_metadata.json')\n"
            "if meta_path.exists():\n"
            "    summary = json.loads(meta_path.read_text(encoding='utf-8'))\n"
            "else:\n"
            "    summary = train_and_evaluate_models(\n"
            "        processed_csv='../data/processed/heart_disease_cleaned.csv',\n"
            "        artifact_dir='../models',\n"
            "        mlflow_tracking_uri='../mlruns'\n"
            "    )\n"
            "print('Winning Model:', summary['model_name'])\n"
            "pd.DataFrame(summary['all_models_comparison']).T[[\n"
            "    'cv_roc_auc_mean', 'cv_roc_auc_std', 'test_roc_auc',\n"
            "    'test_accuracy', 'test_precision', 'test_recall', 'test_f1'\n"
            "]]"
        ),
    ]
    with open(out / "02_feature_engineering_and_modeling.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb2, f)

    # Notebook 3: Inference & API Verification
    nb3 = new_notebook()
    nb3.cells = [
        new_markdown_cell(
            "# 03 — Packaged Pipeline Inference & FastAPI Endpoint Verification\n"
            "**Course**: Machine Learning Operations (MLOps) AIMLCZG523 — Assignment 01\n\n"
            "Verifies reproducible model loading (`heart_disease_pipeline.joblib`) and tests `/predict`, "
            "`/health`, and `/metrics` endpoints."
        ),
        new_code_cell(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
            "from src.models.predict import HeartDiseasePredictor"
        ),
        new_code_cell(
            "predictor = HeartDiseasePredictor(\n"
            "    model_path='../models/heart_disease_pipeline.joblib',\n"
            "    metadata_path='../models/model_metadata.json'\n"
            ")\n"
            "high_risk_patient = {\n"
            "    'age': 62, 'sex': 1, 'cp': 4, 'trestbps': 150, 'chol': 270,\n"
            "    'fbs': 0, 'restecg': 2, 'thalach': 120, 'exang': 1,\n"
            "    'oldpeak': 2.8, 'slope': 2, 'ca': 2, 'thal': 7\n"
            "}\n"
            "low_risk_patient = {\n"
            "    'age': 41, 'sex': 0, 'cp': 2, 'trestbps': 118, 'chol': 195,\n"
            "    'fbs': 0, 'restecg': 0, 'thalach': 172, 'exang': 0,\n"
            "    'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3\n"
            "}\n"
            "print('High-Risk Patient Result:', predictor.predict_single(high_risk_patient))\n"
            "print('Low-Risk Patient Result:', predictor.predict_single(low_risk_patient))"
        ),
    ]
    with open(out / "03_inference_and_verification.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb3, f)


if __name__ == "__main__":
    create_notebooks()
