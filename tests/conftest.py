"""
Shared pytest fixtures.

The fixtures here generate small synthetic DataFrames that mirror the
schema of the real OpenFoodFacts cleaned dataset, allowing tests to run
without requiring the actual (large) data file.
"""

import numpy as np
import pandas as pd
import pytest

from nutriscore.config import TARGET


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """
    100-row synthetic DataFrame with the same schema as openfoodfacts_clean.csv.

    All numeric columns are drawn from realistic ranges so that the
    preprocessing and model code can run end-to-end without modification.
    """
    rng = np.random.default_rng(42)
    n = 100

    return pd.DataFrame(
        {
            "code": range(n),
            "product_name": [f"product_{i}" for i in range(n)],
            "pnns_groups_1": rng.choice(
                ["Sugary snacks", "Beverages", "Cereals and potatoes", "Fruits and vegetables"],
                n,
            ),
            "nova_group": rng.integers(1, 5, n).astype(float),
            "energy_100g": rng.uniform(100, 2000, n),
            "energy-kcal_100g": rng.uniform(25, 480, n),
            "proteins_100g": rng.uniform(0, 30, n),
            "fat_100g": rng.uniform(0, 50, n),
            "carbohydrates_100g": rng.uniform(0, 80, n),
            "sugars_100g": rng.uniform(0, 60, n),
            "salt_100g": rng.uniform(0, 5, n),
            "sodium_100g": rng.uniform(0, 2, n),
            "saturated-fat_100g": rng.uniform(0, 30, n),
            "fiber_100g": rng.uniform(0, 15, n),
            "calcium_100g": rng.uniform(0, 1, n),
            "cholesterol_100g": rng.uniform(0, 0.5, n),
            "trans-fat_100g": rng.uniform(0, 2, n),
            "iron_100g": rng.uniform(0, 0.05, n),
            "additives_n": rng.integers(0, 10, n).astype(float),
            TARGET: rng.integers(-15, 41, n).astype(float),
        }
    )


@pytest.fixture
def sample_raw_df(sample_clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic DataFrame mimicking the raw (uncleaned) export.

    Adds deliberate issues: duplicates, out-of-range component values,
    and non-normalised PNNS labels — exactly the problems that preprocessing
    should fix.
    """
    df = sample_clean_df.copy()
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Introduce out-of-range values
    df.loc[0, "fat_100g"] = 150.0
    df.loc[1, "sugars_100g"] = -5.0
    # Add non-normalised PNNS labels
    df.loc[2, "pnns_groups_1"] = "fruits-and-vegetables"
    df.loc[3, "pnns_groups_1"] = "sugary-snacks"
    return df
