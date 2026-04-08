"""
Unit tests for nutriscore.data.preprocessing.

Each test class targets a single function and covers:
  - The happy path (correct behaviour on valid input).
  - Edge cases that the function is specifically designed to handle.
"""

import pandas as pd
import pytest

from nutriscore.config import TARGET
from nutriscore.data.preprocessing import (
    drop_duplicates,
    normalize_pnns,
    remove_aberrant_values,
    select_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**overrides) -> dict:
    """Return a minimal valid row dict, with any field overridden."""
    base = {
        "code": 1,
        "product_name": "Test Product",
        "pnns_groups_1": "Beverages",
        "nova_group": 2.0,
        "energy_100g": 800.0,
        "energy-kcal_100g": 191.0,
        "proteins_100g": 4.0,
        "fat_100g": 5.0,
        "carbohydrates_100g": 30.0,
        "sugars_100g": 20.0,
        "salt_100g": 0.3,
        "sodium_100g": 0.12,
        "saturated-fat_100g": 2.0,
        "fiber_100g": 1.5,
        "calcium_100g": 0.12,
        "cholesterol_100g": 0.01,
        "trans-fat_100g": 0.0,
        "iron_100g": 0.002,
        "additives_n": 3.0,
        TARGET: 8.0,
    }
    base.update(overrides)
    return base


def _make_df(*rows) -> pd.DataFrame:
    if not rows:
        rows = ({},)  # default: one row with all defaults
    return pd.DataFrame([_make_row(**r) for r in rows])


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    def test_removes_exact_duplicates(self):
        df = _make_df({}, {"fat_100g": 99.0})  # same code + name
        df.loc[1, "code"] = 1
        df.loc[1, "product_name"] = "Test Product"
        result = drop_duplicates(df)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        df = _make_df({"fat_100g": 10.0}, {"fat_100g": 99.0})
        df.loc[1, "code"] = 1  # same key
        result = drop_duplicates(df)
        assert result.iloc[0]["fat_100g"] == 10.0

    def test_preserves_distinct_products(self):
        df = _make_df({"code": 1}, {"code": 2})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = _make_df({"code": 1}, {"code": 2}, {"code": 3})
        result = drop_duplicates(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# normalize_pnns
# ---------------------------------------------------------------------------


class TestNormalizePnns:
    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("fruits-and-vegetables", "Fruits and vegetables"),
            ("sugary-snacks", "Sugary snacks"),
            ("cereals-and-potatoes", "Cereals and potatoes"),
            ("salty-snacks", "Salty snacks"),
        ],
    )
    def test_renames_slug_to_human_readable(self, slug, expected):
        df = _make_df({"pnns_groups_1": slug})
        result = normalize_pnns(df)
        assert result.loc[0, "pnns_groups_1"] == expected

    def test_leaves_already_normalised_unchanged(self):
        df = _make_df({"pnns_groups_1": "Beverages"})
        result = normalize_pnns(df)
        assert result.loc[0, "pnns_groups_1"] == "Beverages"

    def test_does_not_modify_original(self):
        df = _make_df({"pnns_groups_1": "sugary-snacks"})
        original_value = df.loc[0, "pnns_groups_1"]
        normalize_pnns(df)
        assert df.loc[0, "pnns_groups_1"] == original_value


# ---------------------------------------------------------------------------
# remove_aberrant_values
# ---------------------------------------------------------------------------


class TestRemoveAberrantValues:
    def test_removes_row_with_value_above_100(self):
        df = _make_df({"fat_100g": 150.0}, {})
        result = remove_aberrant_values(df)
        assert len(result) == 1

    def test_removes_row_with_negative_value(self):
        df = _make_df({"sugars_100g": -0.1}, {})
        result = remove_aberrant_values(df)
        assert len(result) == 1

    def test_keeps_boundary_values(self):
        df = _make_df({"fat_100g": 0.0}, {"fat_100g": 100.0})
        result = remove_aberrant_values(df)
        assert len(result) == 2

    def test_nan_values_are_tolerated(self):
        df = _make_df({"fiber_100g": float("nan")})
        result = remove_aberrant_values(df)
        assert len(result) == 1

    def test_valid_dataset_unchanged(self, sample_clean_df):
        n_before = len(sample_clean_df)
        result = remove_aberrant_values(sample_clean_df)
        assert len(result) == n_before


# ---------------------------------------------------------------------------
# select_features
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    def test_drops_unrequested_columns(self):
        df = _make_df()
        df["irrelevant_col"] = "garbage"
        result = select_features(df)
        assert "irrelevant_col" not in result.columns

    def test_keeps_target_column(self):
        df = _make_df()
        result = select_features(df)
        assert TARGET in result.columns

    def test_handles_missing_optional_columns_gracefully(self):
        df = _make_df()
        df.drop(columns=["calcium_100g"], inplace=True)
        result = select_features(df)
        assert "calcium_100g" not in result.columns
