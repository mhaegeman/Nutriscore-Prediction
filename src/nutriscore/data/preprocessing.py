"""
Data loading and cleaning pipeline for OpenFoodFacts data.

This module converts the raw OpenFoodFacts TSV export into a clean,
analysis-ready CSV by:
  1. Removing duplicate products (same barcode + name).
  2. Keeping only the 20 relevant columns.
  3. Normalising PNNS group labels to human-readable English strings.
  4. Dropping rows where per-100g nutritional components fall outside [0, 100].
"""

import logging
from pathlib import Path

import pandas as pd

from nutriscore.config import (
    DATA_CLEAN,
    DATA_RAW,
    IDENTIFIERS,
    NUMERIC_COMPONENTS,
    PNNS_RENAME_MAP,
    SELECTED_FEATURES,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load the raw OpenFoodFacts TSV export into a DataFrame."""
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows that share the same barcode and product name."""
    before = len(df)
    df = df.drop_duplicates(IDENTIFIERS, keep="first")
    logger.info("Removed %d duplicate rows (%d remaining)", before - len(df), len(df))
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns relevant to the Nutriscore prediction task."""
    available = [c for c in SELECTED_FEATURES if c in df.columns]
    missing = set(SELECTED_FEATURES) - set(available)
    if missing:
        logger.warning("Columns not found in raw data and skipped: %s", missing)
    return df[available].copy()


def normalize_pnns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise PNNS group labels to consistent English strings."""
    df = df.copy()
    df["pnns_groups_1"] = df["pnns_groups_1"].replace(PNNS_RENAME_MAP)
    return df


def remove_aberrant_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where any per-100g component is outside the valid range [0, 100].

    Energy and score columns are intentionally excluded — they use different
    scales (kJ/kcal and -15…40 respectively).
    """
    valid_cols = [c for c in NUMERIC_COMPONENTS if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in valid_cols:
        mask &= df[col].between(0, 100, inclusive="both") | df[col].isna()

    removed = (~mask).sum()
    logger.info("Removed %d rows with out-of-range component values", removed)
    return df.loc[mask].reset_index(drop=True)


def run(raw_path: Path = DATA_RAW, output_path: Path = DATA_CLEAN) -> pd.DataFrame:
    """
    Execute the full cleaning pipeline and persist the result.

    Parameters
    ----------
    raw_path:
        Path to the raw OpenFoodFacts TSV file.
    output_path:
        Destination path for the cleaned CSV.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    df = load_raw(raw_path)
    df = drop_duplicates(df)
    df = select_features(df)
    df = normalize_pnns(df)
    df = remove_aberrant_values(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Clean dataset saved → %s  (%d rows, %d cols)", output_path, *df.shape)
    return df
