"""
Unit tests for nutriscore.features.engineering and nutriscore.models.train.

Tests are designed to run fully offline using the synthetic fixture from
conftest.py — no real data file is required.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from nutriscore.config import TARGET
from nutriscore.features.engineering import apply_pca, encode_pnns, prepare_xy, scale
from nutriscore.models.train import (
    ModelResult,
    _split,
    train_knn,
    train_linear,
    tune_knn,
)
from nutriscore.models.train import (
    run as run_models,
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestEncodePnns:
    def test_output_is_numeric(self, sample_clean_df):
        result = encode_pnns(sample_clean_df)
        assert pd.api.types.is_numeric_dtype(result["pnns_groups_1"])

    def test_preserves_row_count(self, sample_clean_df):
        result = encode_pnns(sample_clean_df)
        assert len(result) == len(sample_clean_df)

    def test_codes_start_at_one(self, sample_clean_df):
        result = encode_pnns(sample_clean_df)
        assert result["pnns_groups_1"].min() >= 1


class TestPrepareXy:
    def test_returns_2d_x_and_1d_y(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        assert X.ndim == 2
        assert y.ndim == 1

    def test_sample_counts_match(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        assert X.shape[0] == y.shape[0]

    def test_drops_rows_with_missing_target(self, sample_clean_df):
        df = sample_clean_df.copy()
        df.loc[:4, TARGET] = np.nan
        X, y = prepare_xy(df)
        assert len(y) <= len(sample_clean_df) - 5

    def test_custom_feature_cols(self, sample_clean_df):
        cols = ["fat_100g", "sugars_100g", "nova_group", "pnns_groups_1"]
        X, y = prepare_xy(sample_clean_df, feature_cols=cols)
        assert X.shape[1] == len(cols)


class TestScale:
    def test_scaled_mean_near_zero(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        assert np.abs(X_scaled.mean(axis=0)).max() < 0.2

    def test_returns_standard_scaler(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        _, scaler = scale(X)
        assert isinstance(scaler, StandardScaler)

    def test_output_shape_unchanged(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        assert X_scaled.shape == X.shape


class TestApplyPca:
    def test_output_columns_equal_n_components(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        df_pca, _ = apply_pca(X_scaled, n_components=2)
        assert list(df_pca.columns) == ["PC1", "PC2"]

    def test_row_count_preserved(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        df_pca, _ = apply_pca(X_scaled)
        assert len(df_pca) == len(X_scaled)

    def test_explained_variance_sums_to_at_most_one(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        _, pca = apply_pca(X_scaled)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_all_explained_variance_ratios_positive(self, sample_clean_df):
        X, _ = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        _, pca = apply_pca(X_scaled)
        assert all(v > 0 for v in pca.explained_variance_ratio_)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


class TestTrainKnn:
    def test_returns_fitted_knn(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, _ = _split(X_scaled, y)
        knn = train_knn(X_tr, y_tr, k=2)
        assert isinstance(knn, KNeighborsRegressor)

    def test_prediction_length_matches_test_set(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, y_te = _split(X_scaled, y)
        knn = train_knn(X_tr, y_tr, k=2)
        assert len(knn.predict(X_te)) == len(y_te)

    def test_predictions_within_nutriscore_range(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, _ = _split(X_scaled, y)
        knn = train_knn(X_tr, y_tr, k=2)
        preds = knn.predict(X_te)
        assert preds.min() >= -20 and preds.max() <= 50


class TestTrainLinear:
    def test_returns_fitted_linear_model(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, _ = _split(X_scaled, y)
        lr = train_linear(X_tr, y_tr)
        assert isinstance(lr, LinearRegression)

    def test_prediction_length_matches_test_set(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, y_te = _split(X_scaled, y)
        lr = train_linear(X_tr, y_tr)
        assert len(lr.predict(X_te)) == len(y_te)


class TestTuneKnn:
    def test_returns_result_for_each_k(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, y_te = _split(X_scaled, y)
        results = tune_knn(X_tr, y_tr, X_te, y_te, k_range=range(1, 4))
        assert set(results.keys()) == {1, 2, 3}

    def test_each_entry_has_mae_and_r2(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        X_tr, X_te, y_tr, y_te = _split(X_scaled, y)
        results = tune_knn(X_tr, y_tr, X_te, y_te, k_range=range(1, 3))
        assert all("mae" in v and "r2" in v for v in results.values())


class TestRunModels:
    def test_returns_two_model_results(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        results = run_models(X_scaled, y)
        assert len(results) == 2

    def test_all_results_are_model_result_instances(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        results = run_models(X_scaled, y)
        assert all(isinstance(r, ModelResult) for r in results)

    def test_results_sorted_best_r2_first(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        results = run_models(X_scaled, y)
        r2_values = [r.r2 for r in results]
        assert r2_values == sorted(r2_values, reverse=True)

    def test_metrics_are_finite(self, sample_clean_df):
        X, y = prepare_xy(sample_clean_df)
        X_scaled, _ = scale(X)
        results = run_models(X_scaled, y)
        for r in results:
            assert np.isfinite(r.mse)
            assert np.isfinite(r.mae)
            assert np.isfinite(r.r2)


# ---------------------------------------------------------------------------
# ModelResult
# ---------------------------------------------------------------------------


class TestModelResult:
    def test_report_contains_name(self):
        r = ModelResult(name="TestModel", mse=1.0, mae=0.5, r2=0.9, model=None)
        assert "TestModel" in r.report()

    def test_report_contains_metrics(self):
        r = ModelResult(name="X", mse=2.5, mae=1.2, r2=0.88, model=None)
        report = r.report()
        assert "2.500" in report or "2.5" in report
        assert "0.88" in report
