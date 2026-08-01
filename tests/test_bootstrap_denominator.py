"""Tests for bootstrap reproducibility and denominator handling."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.bootstrap import bootstrap_ci, bootstrap_ci_customer
from src.analytics.basket_metrics import compute_basket_penetration, compute_basket_value_uplift


class TestBootstrapReproducibility:
    """Test bootstrap confidence intervals are reproducible with fixed seed."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "transaction_id": range(n),
            "stockcode": np.random.choice(["A", "B", "C", "D"], n),
            "product": ["Prod"] * n,
            "customer_id": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n),
            "price": np.random.uniform(10, 100, n),
            "quantity": np.random.randint(1, 5, n),
        })

    def test_bootstrap_ci_deterministic(self, sample_data):
        """Same seed produces identical results."""
        def stat_fn(df):
            return df["price"].mean()

        ci1 = bootstrap_ci(sample_data, stat_fn, n_resamples=100, random_seed=42)
        ci2 = bootstrap_ci(sample_data, stat_fn, n_resamples=100, random_seed=42)

        assert ci1["estimate"] == ci2["estimate"]
        assert ci1["lower"] == ci2["lower"]
        assert ci1["upper"] == ci2["upper"]
        assert ci1["std_error"] == ci2["std_error"]

    def test_bootstrap_ci_different_seeds_different(self, sample_data):
        """Different seeds produce different (but similar) results."""
        def stat_fn(df):
            return df["price"].mean()

        ci1 = bootstrap_ci(sample_data, stat_fn, n_resamples=100, random_seed=42)
        ci2 = bootstrap_ci(sample_data, stat_fn, n_resamples=100, random_seed=123)

        # Estimates should be very close (same data)
        assert abs(ci1["estimate"] - ci2["estimate"]) < 0.01
        # But CIs may differ slightly
        assert ci1["lower"] != ci2["lower"] or ci1["upper"] != ci2["upper"]

    def test_bootstrap_ci_customer_level(self, sample_data):
        """Customer-level bootstrap preserves within-customer structure."""
        def stat_fn(df):
            return df.groupby("customer_id")["price"].mean().mean()

        ci = bootstrap_ci_customer(sample_data, stat_fn, n_resamples=50, random_seed=42)
        assert "estimate" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert ci["n_resamples"] > 0


class TestDenominatorHandling:
    """Test that metrics use correct denominators."""

    @pytest.fixture
    def transaction_data(self):
        """Create transaction data with known structure."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "transaction_id": range(100),
            "stockcode": ["A"] * 30 + ["B"] * 30 + ["C"] * 40,
            "product": ["A"] * 30 + ["B"] * 30 + ["C"] * 40,
            "customer_id": [1, 2] * 15 + [3, 4] * 15 + [5, 6, 7, 8] * 10,
            "price": [10.0] * 30 + [20.0] * 30 + [15.0] * 40,
            "quantity": [1] * 100,
        })

    def test_basket_penetration_denominator(self, transaction_data):
        """Basket penetration = baskets with product / total baskets."""
        result = compute_basket_penetration(transaction_data)
        row_a = result[result["stockcode"] == "A"].iloc[0]

        # 30 transactions of A, but unique baskets = 30 (each transaction is unique)
        # Total baskets = 100
        # But A appears in 30 unique baskets
        assert row_a["baskets_with_product"] == 30
        assert row_a["basket_penetration"] == 0.3  # 30/100

    def test_shopper_penetration_denominator(self, transaction_data):
        """Shopper penetration = unique customers buying / total unique customers."""
        result = compute_basket_penetration(transaction_data)
        row_a = result[result["stockcode"] == "A"].iloc[0]

        # A bought by customers 1, 2 (2 unique)
        # Total customers = 8 (1,2,3,4,5,6,7,8)
        assert row_a["unique_customers"] == 2
        assert row_a["unique_shopper_penetration"] == 0.25  # 2/8

    def test_basket_value_uplift_denominator(self, transaction_data):
        """Basket uplift uses mean basket value without product as denominator."""
        result = compute_basket_value_uplift(transaction_data, top_n=3)
        row_a = result[result["stockcode"] == "A"].iloc[0]

        # uplift_pct = (with - without) / without
        assert "avg_basket_value_with" in row_a
        assert "avg_basket_value_without" in row_a
        assert "basket_value_uplift_pct" in row_a

        if row_a["avg_basket_value_without"] > 0:
            expected_pct = (row_a["avg_basket_value_with"] - row_a["avg_basket_value_without"]) / row_a["avg_basket_value_without"] * 100
            assert abs(row_a["basket_value_uplift_pct"] - expected_pct) < 0.01

    def test_repeat_rate_denominator(self, transaction_data):
        """Repeat rate = buyers with 2+ purchases / total buyers."""
        from src.analytics.product_performance import compute_repeat_rate

        result = compute_repeat_rate(transaction_data)
        row_a = result[result["stockcode"] == "A"].iloc[0]

        # A: 30 transactions, customers 1,2 (15 each) -> both have 2+ purchases
        assert row_a["total_buyers"] == 2
        assert row_a["repeat_buyers"] == 2
        assert row_a["repeat_rate"] == 1.0

    def test_basket_penetration_with_fallback_basket_id(self):
        """Basket penetration works with customer+date fallback when no transaction_id."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "stockcode": ["A", "B", "A", "C"],
            "customer_id": [1, 1, 2, 2],
            "price": [10, 20, 10, 30],
            "quantity": [1, 1, 1, 1],
        })
        # No transaction_id column
        result = compute_basket_penetration(df)

        # 2 baskets (cust1-20240101, cust2-20240102)
        # A in both baskets -> penetration = 1.0
        row_a = result[result["stockcode"] == "A"].iloc[0]
        assert row_a["basket_penetration"] == 1.0


class TestLowSupportWarnings:
    """Test warnings for low-support results."""

    def test_basket_uplift_low_support_warning(self):
        """Basket uplift handles low-support products gracefully."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "transaction_id": range(10),
            "stockcode": ["A"] * 2 + ["B"] * 8,
            "customer_id": [1] * 10,
            "price": [10.0] * 10,
            "quantity": [1] * 10,
        })

        result = compute_basket_value_uplift(df, top_n=2)
        # A only in 2 baskets - should still compute but with warning-worthy support
        row_a = result[result["stockcode"] == "A"].iloc[0]
        assert row_a["baskets_with"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])