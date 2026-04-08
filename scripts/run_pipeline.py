#!/usr/bin/env python3
"""
Nutriscore Predictor – full ML pipeline.

Runs the end-to-end workflow:
  1. Data cleaning
  2. Exploratory visualisations
  3. Feature engineering (scaling + PCA)
  4. KNN hyperparameter tuning
  5. Model training and evaluation

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --raw data/openfoodfacts_data_new.csv
    python scripts/run_pipeline.py --no-figures
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when running from the repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nutriscore.config import (
    DATA_CLEAN,
    DATA_RAW,
    FIGURES_DIR,
    KNN_K_RANGE,
    MODEL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from nutriscore.data import preprocessing
from nutriscore.features import engineering
from nutriscore.models import train as model_train
from nutriscore.visualization import plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nutriscore ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw", type=Path, default=DATA_RAW, help="Raw OpenFoodFacts TSV")
    p.add_argument("--clean", type=Path, default=DATA_CLEAN, help="Output cleaned CSV")
    p.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Output figures directory")
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = None if args.no_figures else args.figures
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Clean raw data
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 1 — Data Cleaning")
    logger.info("=" * 55)
    df = preprocessing.run(raw_path=args.raw, output_path=args.clean)

    # ------------------------------------------------------------------
    # Step 2: EDA visualisations
    # ------------------------------------------------------------------
    if fig_dir:
        logger.info("=" * 55)
        logger.info("STEP 2 — Exploratory Visualisations")
        logger.info("=" * 55)
        plots.plot_pnns_distribution(df, fig_dir / "pnns_distribution.png")
        plots.plot_correlation_matrix(df, fig_dir / "correlation_matrix.png")
        plots.plot_nutriscore_by_pnns(df, fig_dir / "nutriscore_by_pnns.png")
        plots.plot_nutriscore_by_nova(df, fig_dir / "nutriscore_by_nova.png")
        plots.plot_feature_vs_nutriscore(df, output_path=fig_dir / "features_vs_nutriscore.png")

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 3 — Feature Engineering")
    logger.info("=" * 55)
    X, y = engineering.prepare_xy(df)
    X_scaled, scaler = engineering.scale(X)
    df_pca, pca = engineering.apply_pca(X_scaled)

    if fig_dir:
        plots.plot_pca_variance(pca, fig_dir / "pca_variance.png")
        plots.plot_pca_scatter(df_pca, y, fig_dir / "pca_scatter.png")
        plots.plot_pca_correlation_circle(pca, MODEL_FEATURES, fig_dir / "pca_circle.png")

    # ------------------------------------------------------------------
    # Step 4: KNN hyperparameter tuning
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 4 — KNN Hyperparameter Tuning")
    logger.info("=" * 55)
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    knn_tuning = model_train.tune_knn(X_tr, y_tr, X_te, y_te, k_range=KNN_K_RANGE)
    if fig_dir:
        plots.plot_knn_tuning(knn_tuning, fig_dir / "knn_tuning.png")

    # ------------------------------------------------------------------
    # Step 5: Model training and evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 5 — Model Training & Evaluation")
    logger.info("=" * 55)
    results = model_train.run(X_scaled, y)

    print("\n  === Final Results ===")
    for r in results:
        print(r.report())

    best = results[0]
    print(f"\n  Best model: {best.name}  (R² = {best.r2:.3f}, MAE = {best.mae:.3f})")

    if fig_dir:
        y_pred = best.model.predict(X_te)
        plots.plot_regression_line(
            y_te,
            y_pred,
            title=f"Predicted vs. Actual Nutriscore – {best.name}",
            output_path=fig_dir / "regression_line.png",
        )
        logger.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
