"""
Reusable plotting helpers for the Nutriscore predictor pipeline.

Every function:
  - accepts an optional ``output_path`` argument to save the figure to disk.
  - closes the figure after saving/showing to prevent memory leaks.
  - returns nothing — side-effect only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

matplotlib.rcParams.update({"figure.autolayout": True})


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------


def plot_missing_values(missing_pct: pd.Series, output_path: Path | None = None) -> None:
    """Bar chart of missing-value percentage per column."""
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title("Missing Values – OpenFoodFacts Dataset", fontsize=22, color="darkblue")
    ax.set_xlabel("Column", fontsize=15, color="darkblue")
    ax.set_ylabel("% Missing", fontsize=15, color="darkblue")
    missing_pct.plot(kind="bar", ax=ax, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    _save_or_show(fig, output_path)


def plot_pnns_distribution(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Pie chart of product counts per PNNS category."""
    effectif = df.groupby("pnns_groups_1")["code"].count().sort_values()
    fig, ax = plt.subplots(figsize=(12, 12))
    effectif.plot.pie(labels=effectif.index, autopct="%.1f%%", fontsize=12, ax=ax)
    ax.set_title("Product Distribution by PNNS Category", fontsize=20)
    ax.set_ylabel("")
    _save_or_show(fig, output_path)


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Annotated heatmap of pairwise Pearson correlations (numeric columns only)."""
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(22, 14))
    sns.heatmap(corr, fmt=".2f", annot=True, cbar=False, ax=ax, cmap="coolwarm")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.xticks(rotation=30, ha="right")
    _save_or_show(fig, output_path)


def plot_nutriscore_by_pnns(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Box-plot of Nutriscore distribution per PNNS food category."""
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(
        y="nutrition-score-fr_100g",
        x="pnns_groups_1",
        data=df,
        whis=[0, 100],
        width=0.6,
        palette="vlag",
        ax=ax,
    )
    ax.set_title("Nutriscore Distribution by PNNS Food Category", fontsize=20)
    ax.set_xlabel("PNNS Group")
    ax.set_ylabel("Nutriscore")
    plt.xticks(rotation=30, ha="right")
    _save_or_show(fig, output_path)


def plot_nutriscore_by_nova(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Box-plot of Nutriscore distribution per Nova processing group."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        y="nutrition-score-fr_100g",
        x="nova_group",
        data=df,
        whis=[0, 100],
        width=0.6,
        palette="vlag",
        ax=ax,
    )
    ax.set_title("Nutriscore Distribution by Nova Group", fontsize=20)
    ax.set_xlabel("Nova Group (1 = unprocessed → 4 = ultra-processed)")
    ax.set_ylabel("Nutriscore")
    _save_or_show(fig, output_path)


def plot_feature_vs_nutriscore(
    df: pd.DataFrame,
    features: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """Line plots showing the relationship between key nutrients and Nutriscore."""
    features = features or ["fat_100g", "carbohydrates_100g", "sugars_100g"]
    fig, axes = plt.subplots(1, len(features), figsize=(7 * len(features), 7), sharey=False)
    if len(features) == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        ax.set_title(col)
        subset = df[df[col] <= 100]
        sns.lineplot(ax=ax, data=subset, y=col, x="nutrition-score-fr_100g")
    fig.suptitle("Nutritional Components vs. Nutriscore", fontsize=18)
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------


def plot_knn_tuning(k_results: dict, output_path: Path | None = None) -> None:
    """Dual-axis plot of MAE and R² as a function of the number of neighbours."""
    ks = list(k_results.keys())
    maes = [k_results[k]["mae"] for k in ks]
    r2s = [k_results[k]["r2"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(ks, maes, "o-", color="steelblue", label="MAE")
    ax1.set_ylabel("Mean Absolute Error", color="steelblue")
    ax1.set_xlabel("K (Number of Neighbours)")
    ax1.set_title("KNN Hyperparameter Tuning – MAE & R² by K", fontsize=18)
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, r2s, "o-", color="firebrick", label="R²")
    ax2.set_ylabel("Coefficient of Determination (R²)", color="firebrick")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    _save_or_show(fig, output_path)


def plot_regression_line(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual Nutriscore",
    output_path: Path | None = None,
) -> None:
    """Regression scatter-plot of true vs. predicted Nutriscore values."""
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.regplot(x=y_true.ravel(), y=y_pred.ravel(), x_estimator=np.mean, ax=ax)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Actual Nutriscore")
    ax.set_ylabel("Predicted Nutriscore")
    ticks = list(range(-15, 41, 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.yaxis.grid(True, alpha=0.4)
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Dimensionality reduction plots
# ---------------------------------------------------------------------------


def plot_pca_variance(pca: PCA, output_path: Path | None = None) -> None:
    """Bar chart of explained variance per principal component."""
    n = pca.n_components_
    cols = [f"PC{i + 1}" for i in range(n)]
    variance_pct = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(cols, variance_pct, color="steelblue", edgecolor="white")
    ax.set_title("Explained Variance by Principal Component", fontsize=16)
    ax.set_ylabel("Variance (%)")
    for bar, v in zip(ax.patches, variance_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{v:.1f}%", ha="center")
    _save_or_show(fig, output_path)


def plot_pca_scatter(
    df_pca: pd.DataFrame,
    y: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Scatter plot of the first two PCs, coloured by Nutriscore."""
    df = df_pca.copy()
    df["nutriscore"] = y
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.scatterplot(
        data=df[(df["PC1"] <= 5) & (df["PC2"].between(-6, 5))],
        x="PC1",
        y="PC2",
        hue="nutriscore",
        palette="Spectral",
        legend="auto",
        ax=ax,
        alpha=0.7,
        s=30,
    )
    ax.set_title("PCA Projection – Products Coloured by Nutriscore", fontsize=18)
    _save_or_show(fig, output_path)


def plot_pca_correlation_circle(
    pca: PCA,
    feature_names: list[str],
    output_path: Path | None = None,
) -> None:
    """Correlation circle for the first two principal components."""
    pcs = pca.components_
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="steelblue", linewidth=1.5)
    ax.add_patch(circle)

    for i, (x, y) in enumerate(pcs[[0, 1]].T):
        ax.annotate(
            feature_names[i],
            xy=(x, y),
            fontsize=10,
            ha="center",
            color="darkred",
            alpha=0.8,
        )
        ax.arrow(0, 0, x * 0.95, y * 0.95, head_width=0.02, color="grey", alpha=0.6)

    ax.plot([-1, 1], [0, 0], color="grey", ls="--", lw=0.8)
    ax.plot([0, 0], [-1, 1], color="grey", ls="--", lw=0.8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title("PCA Correlation Circle", fontsize=16)
    ax.set_aspect("equal")
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved → %s", output_path)
    else:
        plt.show()
    plt.close(fig)
