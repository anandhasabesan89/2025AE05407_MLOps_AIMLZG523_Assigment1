"""Exploratory Data Analysis (EDA) Visualization Generator.

Produces publication-grade clinical visualizations for Task 1 of the MLOps
assignment and saves high-resolution PNG figures to `reports/figures/` and
`screenshots/`.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.preprocess import clean_and_preprocess_data

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eda_plots")

PALETTE = {0: "#2b5c8f", 1: "#d95f02"}
LABEL_MAP = {0: "No Disease (0)", 1: "Heart Disease (1)"}


def generate_all_eda_plots(
    processed_csv: str = "data/processed/heart_disease_cleaned.csv",
    output_dir: str = "reports/figures",
    screenshots_dir: Optional[str] = "screenshots",
) -> List[Path]:
    """Generate all EDA plots from the cleaned dataset.

    Args:
        processed_csv: Path to cleaned CSV dataset.
        output_dir: Directory where figures will be saved.
        screenshots_dir: Optional screenshots directory to mirror key EDA plots.

    Returns:
        List of generated file paths.
    """
    csv_path = Path(processed_csv)
    if not csv_path.exists():
        df, _ = clean_and_preprocess_data(processed_path=processed_csv)
    else:
        df = pd.read_csv(csv_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

    df_plot = df.copy()
    df_plot["Diagnosis"] = df_plot["target"].map(LABEL_MAP)

    generated: List[Path] = []

    # 1. Target Class Balance & Demographic Breakdown
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    counts = df_plot["Diagnosis"].value_counts()
    bars = axes[0].bar(
        counts.index,
        counts.values,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        width=0.55,
    )
    axes[0].set_title("Target Class Balance (Cleveland Cohort)", fontweight="bold")
    axes[0].set_ylabel("Number of Patients")
    for bar in bars:
        height = bar.get_height()
        pct = 100.0 * height / len(df_plot)
        axes[0].annotate(
            f"{int(height)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

    sex_df = df_plot.copy()
    sex_df["Sex"] = sex_df["sex"].map({0: "Female", 1: "Male"})
    sns.countplot(
        data=sex_df,
        x="Sex",
        hue="Diagnosis",
        palette=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("Heart Disease Prevalence by Sex", fontweight="bold")
    axes[1].set_ylabel("Patient Count")

    sns.kdeplot(
        data=df_plot,
        x="age",
        hue="Diagnosis",
        fill=True,
        common_norm=False,
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.45,
        linewidth=2,
        ax=axes[2],
    )
    axes[2].set_title("Age Distribution by Diagnosis Status", fontweight="bold")
    axes[2].set_xlabel("Age (Years)")
    plt.suptitle("UCI Heart Disease Dataset — Class Balance & Demographics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = out_path / "eda_01_class_balance_demographics.png"
    fig.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close(fig)
    generated.append(p1)

    # 2. Continuous Feature Distributions (Histograms + KDE)
    num_cols = [
        ("age", "Age (Years)"),
        ("trestbps", "Resting Blood Pressure (mm Hg)"),
        ("chol", "Serum Cholesterol (mg/dL)"),
        ("thalach", "Max Heart Rate Achieved (bpm)"),
        ("oldpeak", "ST Depression Induced by Exercise"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes_flat = axes.flatten()
    for idx, (col, title) in enumerate(num_cols):
        ax = axes_flat[idx]
        sns.histplot(
            data=df_plot,
            x=col,
            hue="Diagnosis",
            kde=True,
            palette=["#2b5c8f", "#d95f02"],
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col)

    # 6th panel: Boxplot of standardized continuous features
    ax_box = axes_flat[5]
    norm_df = df[[c for c, _ in num_cols]].apply(lambda x: (x - x.mean()) / x.std())
    norm_df["Diagnosis"] = df_plot["Diagnosis"]
    melted = norm_df.melt(id_vars="Diagnosis", var_name="Feature", value_name="Z-Score")
    sns.boxplot(
        data=melted,
        x="Feature",
        y="Z-Score",
        hue="Diagnosis",
        palette=["#2b5c8f", "#d95f02"],
        ax=ax_box,
        fliersize=2,
    )
    ax_box.set_title("Standardized Feature Spread (Z-Score)", fontweight="bold")
    plt.suptitle(
        "Continuous Clinical Biomarker Distributions by Heart Disease Status",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    p2 = out_path / "eda_02_continuous_distributions.png"
    fig.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    generated.append(p2)

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 9.5))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.65,
        vmax=0.65,
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title(
        "Feature Correlation Matrix — UCI Heart Disease Clinical Attributes",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    p3 = out_path / "eda_03_correlation_heatmap.png"
    fig.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)
    generated.append(p3)

    # 4. Categorical Risk Factor Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cat_panels = [
        ("cp", "Chest Pain Type (1=Typ, 2=Atyp, 3=Non-Ang, 4=Asymp)", axes[0, 0]),
        ("ca", "Major Vessels Colored by Fluoroscopy (0-3)", axes[0, 1]),
        ("thal", "Thalassemia (3=Normal, 6=Fixed, 7=Reversible)", axes[1, 0]),
        ("exang", "Exercise-Induced Angina (0=No, 1=Yes)", axes[1, 1]),
    ]
    for col, title, ax in cat_panels:
        ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
        ct.plot(
            kind="bar",
            stacked=True,
            color=["#2b5c8f", "#d95f02"],
            edgecolor="black",
            ax=ax,
            rot=0,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Percentage of Patients (%)")
        ax.legend(["No Disease (0)", "Heart Disease (1)"], loc="upper right")
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.0f%%",
                label_type="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )
    plt.suptitle(
        "Heart Disease Risk Stratification Across Categorical Clinical Indicators",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    p4 = out_path / "eda_04_categorical_risk_factors.png"
    fig.savefig(p4, dpi=220, bbox_inches="tight")
    plt.close(fig)
    generated.append(p4)

    # 5. Multivariate Physiological Interaction Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.scatterplot(
        data=df_plot,
        x="age",
        y="thalach",
        hue="Diagnosis",
        style="Diagnosis",
        palette=["#2b5c8f", "#d95f02"],
        s=75,
        alpha=0.8,
        ax=axes[0],
    )
    age_grid = np.linspace(df["age"].min(), df["age"].max(), 100)
    axes[0].plot(age_grid, 220 - age_grid, "--", color="gray", label="220 - Age Ceiling")
    axes[0].set_title("Chronotropic Response: Max Heart Rate vs. Age", fontweight="bold")
    axes[0].set_xlabel("Age (Years)")
    axes[0].set_ylabel("Maximum Heart Rate Achieved (thalach)")
    axes[0].legend()

    df_plot["rate_pressure_product"] = (df_plot["trestbps"] * df_plot["thalach"]) / 1000.0
    sns.boxplot(
        data=df_plot,
        x="cp",
        y="oldpeak",
        hue="Diagnosis",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[1],
    )
    axes[1].set_title("ST Depression (oldpeak) Across Chest Pain Types & Diagnosis", fontweight="bold")
    axes[1].set_xlabel("Chest Pain Type (cp)")
    axes[1].set_ylabel("ST Depression (oldpeak)")
    plt.tight_layout()
    p5 = out_path / "eda_05_multivariate_physiology.png"
    fig.savefig(p5, dpi=220, bbox_inches="tight")
    plt.close(fig)
    generated.append(p5)

    if screenshots_dir:
        s_dir = Path(screenshots_dir)
        s_dir.mkdir(parents=True, exist_ok=True)
        for p in generated:
            shutil.copy2(p, s_dir / p.name)

    logger.info("Generated %d EDA figures in %s", len(generated), out_path)
    return generated


if __name__ == "__main__":
    generate_all_eda_plots()
