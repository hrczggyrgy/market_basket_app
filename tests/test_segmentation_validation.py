"""Tests for segmentation_validation module."""

import pandas as pd

from src.analytics.segmentation_validation import (
    generate_synthetic_customer_segments,
    run_segmentation_validation,
)


class TestGenerateSyntheticCustomerSegments:
    def test_returns_dataframe_and_labels(self):
        df, labels = generate_synthetic_customer_segments(
            n_customers=50, n_true_segments=3, random_seed=42
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(labels, dict)
        assert len(df) > 0
        assert len(labels) == 50

    def test_required_columns_present(self):
        df, _ = generate_synthetic_customer_segments(n_customers=30, n_true_segments=2)
        for col in ["date", "transaction_id", "stockcode", "customer_id", "price", "quantity"]:
            assert col in df.columns

    def test_correct_number_of_segments(self):
        n_seg = 4
        _, labels = generate_synthetic_customer_segments(n_customers=80, n_true_segments=n_seg)
        assert len(set(labels.values())) == n_seg


class TestRunSegmentationValidation:
    def test_returns_dataframe(self):
        result = run_segmentation_validation(n_customers=60, n_true_segments=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_expected_columns_present(self):
        result = run_segmentation_validation(n_customers=60, n_true_segments=3)
        for col in [
            "method",
            "adjusted_rand_index",
            "normalized_mutual_info",
            "n_segments_found",
            "n_true_segments",
        ]:
            assert col in result.columns

    def test_includes_all_methods(self):
        methods = ["rfm_quantile", "rfm_kmeans", "behavioral"]
        result = run_segmentation_validation(n_customers=60, n_true_segments=3, methods=methods)
        assert len(result) == len(methods)
        assert set(result["method"].tolist()) == set(methods)
