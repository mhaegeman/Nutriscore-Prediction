"""
Feature engineering: encoding, scaling, and dimensionality reduction.

Workflow
--------
1. ``encode_pnns``  – convert PNNS category strings to ordered integer codes.
2. ``prepare_xy``   – select feature columns and target; drop incomplete rows.
3. ``scale``        – standardise features with zero mean and unit variance.
4. ``apply_pca``    – project features onto principal components for EDA.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nutriscore.config import MODEL_FEATURES, PCA_COMPONENTS, TARGET

logger = logging.getLogger(__name__)


def encode_pnns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace PNNS group strings with stable integer codes (1-based, sorted).

    The mapping is derived from the unique values present in the dataset so
    it remains consistent across train and predict calls on the same data.
    """
    df = df.copy()
    categories = sorted(df["pnns_groups_1"].dropna().unique())
    mapping = {cat: i + 1 for i, cat in enumerate(categories)}
    df["pnns_groups_1"] = df["pnns_groups_1"].map(mapping)
    logger.debug("PNNS encoding map: %s", mapping)
    return df


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix X and target vector y from a cleaned DataFrame.

    Rows with any NaN in the selected features or the target are dropped.

    Parameters
    ----------
    df:
        Cleaned OpenFoodFacts DataFrame (output of preprocessing.run).
    feature_cols:
        List of column names to use as features. Defaults to MODEL_FEATURES.
    target_col:
        Name of the target column.

    Returns
    -------
    X: np.ndarray of shape (n_samples, n_features)
    y: np.ndarray of shape (n_samples,)
    """
    feature_cols = feature_cols or MODEL_FEATURES
    required = feature_cols + [target_col]
    df = df[required].dropna()
    df = encode_pnns(df)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    logger.info("Dataset ready: %d samples, %d features", *X.shape)
    return X, y


def scale(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardise the feature matrix (zero mean, unit variance).

    Returns
    -------
    X_scaled: np.ndarray
    scaler: fitted StandardScaler (keep to transform new samples at inference)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_pca(
    X_scaled: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> tuple[pd.DataFrame, PCA]:
    """
    Reduce dimensionality with PCA and return a tidy DataFrame of projections.

    Parameters
    ----------
    X_scaled:
        Standardised feature matrix (output of ``scale``).
    n_components:
        Number of principal components to retain.

    Returns
    -------
    df_pca: DataFrame with columns PC1, PC2, …
    pca:    Fitted PCA object (exposes .explained_variance_ratio_, .components_)
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    col_names = [f"PC{i + 1}" for i in range(n_components)]
    df_pca = pd.DataFrame(components, columns=col_names)

    variance_info = {
        col: f"{v * 100:.1f}%" for col, v in zip(col_names, pca.explained_variance_ratio_)
    }
    logger.info("PCA explained variance → %s", variance_info)
    return df_pca, pca
