"""Smoke tests for Bayesian hierarchical elasticity model."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.pricing import estimate_bayesian_hierarchical_elasticity


@pytest.fixture
def bayesian_sample_data():
    """Generate weekly-level synthetic data with known elasticity ≈ -1.5."""
    np.random.seed(42)
    rows = []
    for sku in ["S001", "S002", "S003"]:
        cats = {"S001": "CatA", "S002": "CatA", "S003": "CatB"}
        for w in range(30):
            p = 2.0 + np.random.uniform(-0.5, 0.5)
            q = max(1, int(100 * p**-1.5 + np.random.normal(0, 5)))
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w),
                    "transaction_id": f"T{sku}{w}",
                    "stockcode": sku,
                    "product": sku,
                    "customer_id": "C001",
                    "price": p,
                    "quantity": q,
                    "category": cats[sku],
                }
            )
    return pd.DataFrame(rows)


class TestBayesianElasticity:
    def test_advi_returns_valid_structure(self, bayesian_sample_data):
        """ADVI path returns expected columns and non-empty result."""
        df = bayesian_sample_data
        result = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=200,
            bayesian_mode="fast (ADVI)",
        )
        assert not result.empty
        expected_cols = [
            "stockcode",
            "category",
            "elasticity_mean",
            "elasticity_sd",
            "elasticity_hdi_lower",
            "elasticity_hdi_upper",
            "n_obs",
            "avg_price",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == 3  # 3 SKUs

    def test_advi_elasticities_negative(self, bayesian_sample_data):
        """Estimated elasticities should be negative (downward-sloping demand)."""
        df = bayesian_sample_data
        result = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=200,
            bayesian_mode="fast (ADVI)",
        )
        assert (result["elasticity_mean"] < 0).all(), "All elasticities should be negative"

    def test_nuts_returns_trace(self, bayesian_sample_data):
        """NUTS path returns trace object with posterior."""
        df = bayesian_sample_data
        result, trace = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=50,
            n_tune=50,
            bayesian_mode="full (NUTS)",
            return_trace=True,
        )
        assert not result.empty
        assert trace is not None
        assert hasattr(trace, "posterior")
        assert "beta_sku" in trace.posterior.data_vars
        assert len(result) == 3

    def test_nuts_elasticities_negative(self, bayesian_sample_data):
        """NUTS elasticities should be negative."""
        df = bayesian_sample_data
        result, _ = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=50,
            n_tune=50,
            bayesian_mode="full (NUTS)",
            return_trace=True,
        )
        assert (result["elasticity_mean"] < 0).all()

    def test_hdi_order(self, bayesian_sample_data):
        """HDI lower should be <= HDI upper."""
        df = bayesian_sample_data
        result = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=200,
            bayesian_mode="fast (ADVI)",
        )
        assert (result["elasticity_hdi_lower"] <= result["elasticity_hdi_upper"]).all()

    def test_insufficient_data_returns_empty(self):
        """Very small dataset with poor price variation returns empty DataFrame."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "transaction_id": ["T1", "T2"],
                "stockcode": ["S001", "S001"],
                "product": ["P1", "P1"],
                "customer_id": ["C1", "C1"],
                "price": [10.0, 10.0],
                "quantity": [5, 5],
                "category": ["CatA", "CatA"],
            }
        )
        result = estimate_bayesian_hierarchical_elasticity(
            df, min_periods=5, min_price_variation=0.01, n_samples=100
        )
        assert result.empty
