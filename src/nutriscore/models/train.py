"""
Model training, hyperparameter search, and evaluation.

Two regression models are compared:
  - KNN Regressor  (best performing, k=2 selected via sweep)
  - Linear Regression  (interpretable baseline)

All public functions return plain objects — no side effects — so they are
trivially testable and composable in any pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from nutriscore.config import KNN_BEST_K, KNN_K_RANGE, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for a single model's evaluation metrics."""

    name: str
    mse: float
    mae: float
    r2: float
    model: object = field(repr=False)

    def report(self) -> str:
        return (
            f"\n{'─' * 40}\n"
            f"  Model : {self.name}\n"
            f"  MSE   : {self.mse:.3f}\n"
            f"  MAE   : {self.mae:.3f}\n"
            f"  R²    : {self.r2:.3f}\n"
            f"{'─' * 40}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split(X: np.ndarray, y: np.ndarray):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> ModelResult:
    y_pred = model.predict(X_test)
    result = ModelResult(
        name=name,
        mse=float(mean_squared_error(y_test, y_pred)),
        mae=float(mean_absolute_error(y_test, y_pred)),
        r2=float(r2_score(y_test, y_pred)),
        model=model,
    )
    logger.info(result.report())
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tune_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_range=KNN_K_RANGE,
) -> dict[int, dict[str, float]]:
    """
    Sweep over k values and record MAE and R² for each.

    Returns
    -------
    dict mapping k → {"mae": float, "r2": float}
    """
    results: dict[int, dict[str, float]] = {}
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        results[k] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        logger.debug("k=%d  MAE=%.3f  R²=%.3f", k, results[k]["mae"], results[k]["r2"])
    return results


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = KNN_BEST_K,
) -> KNeighborsRegressor:
    """Fit a KNN regressor with the given number of neighbours."""
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def train_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """Fit an ordinary least-squares linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def run(X: np.ndarray, y: np.ndarray) -> list[ModelResult]:
    """
    Train and evaluate all models on an 80/20 train-test split.

    Parameters
    ----------
    X: Feature matrix (already scaled).
    y: Target vector.

    Returns
    -------
    List of ModelResult, one per model, ordered best → worst by R².
    """
    X_train, X_test, y_train, y_test = _split(X, y)

    knn = train_knn(X_train, y_train, k=KNN_BEST_K)
    lr = train_linear(X_train, y_train)

    results = [
        _evaluate(f"KNN Regressor (k={KNN_BEST_K})", knn, X_test, y_test),
        _evaluate("Linear Regression", lr, X_test, y_test),
    ]
    return sorted(results, key=lambda r: r.r2, reverse=True)
