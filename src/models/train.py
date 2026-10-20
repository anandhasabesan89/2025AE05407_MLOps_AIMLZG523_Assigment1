"""Model Training, Hyperparameter Tuning, MLflow Tracking, and Packaging Module.

Implements Tasks 2, 3, and 4:
- Trains and tunes Logistic Regression, Random Forest, and Gradient Boosting
  classifiers using 5-fold Stratified Cross-Validation.
- Logs all parameters, cross-validation metrics, test set metrics, evaluation
  plots, and fitted pipelines to MLflow.
- Packages the winning end-to-end pipeline in `models/heart_disease_pipeline.joblib`
  along with full provenance metadata in `models/model_metadata.json`.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data.preprocess import clean_and_preprocess_data  # noqa: E402
from src.features.build_features import (  # noqa: E402
    REQUIRED_INPUT_FEATURES,
    build_preprocessing_pipeline,
)

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_training")


def get_candidate_models(random_state: int = 42) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return candidate classifiers and their hyperparameter tuning grids."""
    return {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            {
                "classifier__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=random_state),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 6, 8, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Gradient_Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "classifier__n_estimators": [80, 150],
                "classifier__learning_rate": [0.03, 0.08, 0.15],
                "classifier__max_depth": [2, 3, 4],
                "classifier__subsample": [0.85, 1.0],
            },
        ),
    }


def _plot_single_model_diagnostics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    """Create a 3-panel diagnostic plot (Confusion Matrix, ROC, PR Curve) for one model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Heart Disease"])
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    axes[0].grid(False)

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="#d95f02", lw=2.5, label=f"ROC AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    axes[1].set_title(f"{model_name} — ROC Curve", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    # Panel 3: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[2].plot(rec, prec, color="#2b5c8f", lw=2.5, label=f"Avg Precision = {ap:.3f}")
    axes[2].set_title(f"{model_name} — Precision-Recall Curve", fontweight="bold")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    out_file = output_dir / f"diagnostics_{model_name.lower()}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _generate_comparison_plots(
    results: Dict[str, Dict[str, Any]],
    curves_data: Dict[str, Dict[str, Any]],
    best_pipeline: Pipeline,
    best_model_name: str,
    figures_dir: Path,
    screenshots_dir: Path,
) -> None:
    """Generate comparative evaluation figures across all trained models."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    colors = {
        "Logistic_Regression": "#2b5c8f",
        "Random_Forest": "#2ca02c",
        "Gradient_Boosting": "#d95f02",
    }

    # 1. Combined ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for m_name, c_data in curves_data.items():
        color = colors.get(m_name, "#333333")
        axes[0].plot(
            c_data["fpr"],
            c_data["tpr"],
            lw=2.5,
            color=color,
            label=f"{m_name.replace('_', ' ')} (AUC = {results[m_name]['test_roc_auc']:.3f})",
        )
        axes[1].plot(
            c_data["recall_curve"],
            c_data["precision_curve"],
            lw=2.5,
            color=color,
            label=f"{m_name.replace('_', ' ')} (AP = {results[m_name]['test_avg_precision']:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random Baseline")
    axes[0].set_title("Hold-Out Test ROC Curve Comparison", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Hold-Out Test Precision-Recall Curve Comparison", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="lower left")

    plt.suptitle(
        "Multi-Model Diagnostic Curve Comparison on UCI Heart Disease Dataset",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    f1_path = figures_dir / "model_01_roc_and_pr_comparison.png"
    fig.savefig(f1_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(f1_path, screenshots_dir / f1_path.name)

    # 2. Cross-Validation vs. Test Metrics Bar Chart
    metric_keys = [
        ("cv_roc_auc_mean", "CV ROC-AUC"),
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_accuracy", "Test Accuracy"),
        ("test_precision", "Test Precision"),
        ("test_recall", "Test Recall"),
        ("test_f1", "Test F1-Score"),
    ]
    rows = []
    for m_name, m_res in results.items():
        for k, label in metric_keys:
            rows.append(
                {
                    "Model": m_name.replace("_", " "),
                    "Metric": label,
                    "Score": m_res[k],
                }
            )
    df_metrics = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=df_metrics,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#2b5c8f", "#2ca02c", "#d95f02"],
        edgecolor="black",
        ax=ax,
    )
    ax.set_ylim(0.65, 1.0)
    ax.set_title("5-Fold Stratified CV & Hold-Out Test Performance Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    f2_path = figures_dir / "model_02_cv_and_test_metrics.png"
    fig.savefig(f2_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(f2_path, screenshots_dir / f2_path.name)

    # 3. Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for idx, (m_name, c_data) in enumerate(curves_data.items()):
        cm = np.array(results[m_name]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Heart Disease"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        acc_val = results[m_name]["test_accuracy"]
        auc_val = results[m_name]["test_roc_auc"]
        axes[idx].set_title(
            f"{m_name.replace('_', ' ')}\n(Acc={acc_val:.3f}, AUC={auc_val:.3f})",
            fontweight="bold",
        )
        axes[idx].grid(False)
    plt.suptitle(
        "Hold-Out Test Confusion Matrices Across Candidate Classifiers",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    f3_path = figures_dir / "model_03_confusion_matrices.png"
    fig.savefig(f3_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(f3_path, screenshots_dir / f3_path.name)

    # 4. Feature Importance / Coefficient Magnitude Plot for Winning Model
    preprocessor = best_pipeline.named_steps["preprocessing"].named_steps["column_preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in feature_names]
    clf = best_pipeline.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        imp_label = "Gini / Tree Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        imp_label = "Absolute Standardized Coefficient |β|"
    else:
        importances = np.ones(len(clean_names))
        imp_label = "Importance"

    imp_df = (
        pd.DataFrame({"Feature": clean_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=imp_df,
        x="Importance",
        y="Feature",
        palette="viridis",
        edgecolor="black",
        ax=ax,
    )
    ax.set_title(
        f"Top 15 Predictive Clinical Features — {best_model_name.replace('_', ' ')}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(imp_label)
    plt.tight_layout()
    f4_path = figures_dir / "model_04_feature_importances.png"
    fig.savefig(f4_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(f4_path, screenshots_dir / f4_path.name)


def train_and_evaluate_models(
    processed_csv: str = "data/processed/heart_disease_cleaned.csv",
    artifact_dir: str = "models",
    mlflow_tracking_uri: str = "mlruns",
    experiment_name: str = "Heart_Disease_Risk_Classification",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train, tune, track with MLflow, and package the best Heart Disease model.

    Returns:
        Dictionary containing summary metrics across all models and winning model metadata.
    """
    csv_path = Path(processed_csv)
    if not csv_path.exists():
        df, _ = clean_and_preprocess_data(processed_path=processed_csv)
    else:
        df = pd.read_csv(csv_path)

    X = df[REQUIRED_INPUT_FEATURES].copy()
    y = df["target"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    candidates = get_candidate_models(random_state=random_state)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Any]] = {}
    curves_data: Dict[str, Dict[str, Any]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name, (estimator, param_grid) in candidates.items():
        logger.info("Training and tuning candidate model: %s", model_name)
        full_pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("classifier", estimator),
            ]
        )

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=f"{model_name}_Tuned_CV"):
            grid_search.fit(X_train, y_train)
            best_pipe = grid_search.best_estimator_
            trained_pipelines[model_name] = best_pipe

            # Multi-metric 5-fold cross-validation on training set using best estimator
            cv_scores = cross_validate(
                best_pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                n_jobs=-1,
            )

            # Hold-out test predictions
            y_pred = best_pipe.predict(X_test)
            y_prob = best_pipe.predict_proba(X_test)[:, 1]

            metrics_dict = {
                "cv_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
                "cv_roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
                "cv_accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
                "cv_precision_mean": round(float(np.mean(cv_scores["test_precision"])), 4),
                "cv_recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
                "cv_f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
                "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "test_f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "test_avg_precision": round(float(average_precision_score(y_test, y_prob)), 4),
                "test_brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
            }

            clean_params = {
                k.replace("classifier__", ""): (str(v) if v is None else v)
                for k, v in grid_search.best_params_.items()
            }
            mlflow.log_params(clean_params)
            mlflow.log_param("model_family", model_name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_metrics(metrics_dict)

            diag_plot = _plot_single_model_diagnostics(
                model_name=model_name,
                y_true=y_test,
                y_prob=y_prob,
                y_pred=y_pred,
                output_dir=figures_dir,
            )
            mlflow.log_artifact(str(diag_plot), artifact_path="plots")

            mlflow.sklearn.log_model(
                sk_model=best_pipe,
                name="heart_disease_model",
                serialization_format="cloudpickle",
                input_example=X_train.head(3).astype(float),
            )

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
            curves_data[model_name] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision_curve": prec_c,
                "recall_curve": rec_c,
            }

            all_results[model_name] = {
                **metrics_dict,
                "best_params": clean_params,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "mlflow_run_id": mlflow.active_run().info.run_id,
            }
            logger.info(
                "%s -> CV ROC-AUC: %.4f (±%.4f) | Test ROC-AUC: %.4f | Test Acc: %.4f",
                model_name,
                metrics_dict["cv_roc_auc_mean"],
                metrics_dict["cv_roc_auc_std"],
                metrics_dict["test_roc_auc"],
                metrics_dict["test_accuracy"],
            )

    # Select winning model based on highest composite (CV ROC-AUC + Test ROC-AUC)
    best_model_name = max(
        all_results.keys(),
        key=lambda m: (
            all_results[m]["cv_roc_auc_mean"] + all_results[m]["test_roc_auc"],
            all_results[m]["test_f1"],
        ),
    )
    best_pipeline = trained_pipelines[best_model_name]

    _generate_comparison_plots(
        results=all_results,
        curves_data=curves_data,
        best_pipeline=best_pipeline,
        best_model_name=best_model_name,
        figures_dir=figures_dir,
        screenshots_dir=screenshots_dir,
    )

    # Package winning model pipeline & metadata
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = out_dir / "heart_disease_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)

    metadata = {
        "model_name": best_model_name,
        "version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_path": str(pipeline_path),
        "required_input_features": REQUIRED_INPUT_FEATURES,
        "winning_metrics": all_results[best_model_name],
        "all_models_comparison": all_results,
    }

    metadata_path = out_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_summary_path = Path("reports/model_comparison_summary.json")
    with open(report_summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Best model selected: %s (CV ROC-AUC=%.4f, Test ROC-AUC=%.4f). Packaged to %s",
        best_model_name,
        all_results[best_model_name]["cv_roc_auc_mean"],
        all_results[best_model_name]["test_roc_auc"],
        pipeline_path,
    )
    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
