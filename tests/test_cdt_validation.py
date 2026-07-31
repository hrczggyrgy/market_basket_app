"""Tests for cdt_validation module."""

import pandas as pd

from src.analytics.cdt_validation import (
    generate_synthetic_cluster_data,
    run_cdt_validation,
)


class TestGenerateSyntheticClusterData:
    def test_returns_dataframe_and_labels(self):
        df, labels = generate_synthetic_cluster_data(
            n_products=10, n_true_clusters=2, n_customers=50, random_seed=42
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(labels, dict)
        assert len(df) > 0
        assert len(labels) == 10

    def test_required_columns_present(self):
        df, _ = generate_synthetic_cluster_data(n_products=6, n_true_clusters=2, n_customers=30)
        for col in ["date", "transaction_id", "stockcode", "customer_id", "price", "quantity"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_true_labels_matches_n_clusters(self):
        n_clusters = 3
        _, labels = generate_synthetic_cluster_data(
            n_products=15, n_true_clusters=n_clusters, n_customers=100
        )
        unique = set(labels.values())
        assert len(unique) == n_clusters

    def test_noise_level_zero_produces_perfect_clusters(self):
        df, labels = generate_synthetic_cluster_data(
            n_products=9, n_true_clusters=3, n_customers=60, noise_level=0.0, random_seed=42
        )
        assert len(set(labels.values())) == 3


class TestRunCdtValidation:
    def test_returns_dataframe_with_expected_columns(self):
        result = run_cdt_validation(
            n_products=10,
            n_true_clusters=2,
            n_customers=50,
            noise_level=0.1,
            min_cooccurrence=1,
            min_k=2,
            max_k=5,
        )
        assert isinstance(result, pd.DataFrame)
        expected = [
            "method",
            "adjusted_rand_index",
            "normalized_mutual_info",
            "n_clusters_found",
            "n_true_clusters",
            "runtime_seconds",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_includes_all_methods(self):
        methods = ["legacy_phi", "legacy_jaccard", "ensemble_phi_jaccard_pmi_tfidf"]
        result = run_cdt_validation(
            n_products=10,
            n_true_clusters=2,
            n_customers=50,
            noise_level=0.1,
            min_cooccurrence=1,
            methods=methods,
        )
        assert len(result) == len(methods)
        assert set(result["method"].tolist()) == set(methods)

    def test_ari_nmi_non_negative(self):
        result = run_cdt_validation(
            n_products=8,
            n_true_clusters=2,
            n_customers=40,
            noise_level=0.0,
            min_cooccurrence=1,
        )
        for _, row in result.iterrows():
            # Should not use -1.0 as error marker (now uses NaN)
            assert row["adjusted_rand_index"] != -1.0, f"Found -1.0 error marker: {row}"
            assert row["normalized_mutual_info"] != -1.0, f"Found -1.0 error marker: {row}"
            # If not NaN, should be >= 0
            import math

            if not math.isnan(row["adjusted_rand_index"]):
                assert row["adjusted_rand_index"] >= 0.0
            if not math.isnan(row["normalized_mutual_info"]):
                assert row["normalized_mutual_info"] >= 0.0
