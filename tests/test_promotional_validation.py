"""Tests for promotional_validation module."""

import pandas as pd

from src.analytics.promotional_validation import (
    generate_synthetic_promo_data,
    run_promo_detection_validation,
)


class TestGenerateSyntheticPromoData:
    def test_returns_dataframe_and_dict(self):
        df, weeks = generate_synthetic_promo_data(n_weeks=26, n_products=2, random_seed=42)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(weeks, dict)
        assert len(df) > 0
        assert len(weeks) == 2

    def test_required_columns_present(self):
        df, _ = generate_synthetic_promo_data(n_weeks=10, n_products=2)
        for col in ["date", "stockcode", "price", "quantity"]:
            assert col in df.columns


class TestRunPromoDetectionValidation:
    def test_returns_dataframe(self):
        result = run_promo_detection_validation(n_weeks=26, n_products=2, promo_weeks_fraction=0.2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_expected_columns_present(self):
        result = run_promo_detection_validation(n_weeks=26, n_products=2)
        for col in ["method", "precision", "recall", "f1"]:
            assert col in result.columns

    def test_includes_both_methods(self):
        result = run_promo_detection_validation(
            n_weeks=26, n_products=2, methods=["fixed_threshold", "adaptive_zscore"]
        )
        assert len(result) == 2
        assert set(result["method"].tolist()) == {"fixed_threshold", "adaptive_zscore"}
