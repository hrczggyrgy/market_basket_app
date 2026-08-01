"""Tests for data quality summaries and method readiness."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.data_quality import (
    summarize_data_quality,
    validate_price_quantity,
    find_sku_description_conflicts,
    calculate_method_readiness,
    format_readiness_for_ui,
)


class TestSummarizeDataQuality:
    """Test data quality summary function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "transaction_id": range(100),
            "stockcode": np.random.choice(["A", "B", "C", "D"], 100),
            "product": ["Product A"] * 25 + ["Product B"] * 25 + ["Product C"] * 25 + ["Product D"] * 25,
            "customer_id": np.random.choice([1, 2, 3, 4, 5], 100),
            "price": np.random.uniform(10, 100, 100),
            "quantity": np.random.randint(1, 5, 100),
            "category": np.random.choice(["Cat1", "Cat2"], 100),
            "brand": np.random.choice(["BrandA", "BrandB"], 100),
        })

    def test_basic_counts(self, sample_df):
        """Summary includes basic counts."""
        summary = summarize_data_quality(sample_df)
        assert summary["n_rows"] == 100
        assert summary["n_transactions"] == 100
        assert summary["n_customers"] == 5
        assert summary["n_products"] == 4
        assert summary["date_span_days"] == 99

    def test_missing_customer_id(self, sample_df):
        """Detects missing customer IDs."""
        df = sample_df.copy()
        df.loc[0:10, "customer_id"] = np.nan
        summary = summarize_data_quality(df)
        assert summary["missing_customer_id"] == 11
        assert summary["missing_customer_id_pct"] == 11.0

    def test_nonpositive_price_quantity(self, sample_df):
        """Detects non-positive price/quantity."""
        df = sample_df.copy()
        df.loc[0:4, "price"] = 0
        df.loc[5:9, "quantity"] = -1
        summary = summarize_data_quality(df)
        assert summary["nonpositive_price"] == 5
        assert summary["nonpositive_quantity"] == 5

    def test_sparse_sku_count(self, sample_df):
        """Identifies sparse SKUs."""
        summary = summarize_data_quality(sample_df)
        assert "sparse_sku_count" in summary
        assert "sparse_sku_threshold" in summary

    def test_attribute_coverage(self, sample_df):
        """Reports attribute column coverage."""
        summary = summarize_data_quality(sample_df)
        assert "category" in summary["attribute_coverage"]
        assert "brand" in summary["attribute_coverage"]
        assert summary["attribute_coverage"]["category"]["pct"] == 100.0


class TestValidatePriceQuantity:
    """Test price/quantity validation."""

    def test_clean_data(self):
        """No errors/warnings for clean data."""
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "quantity": [1, 2, 3],
        })
        result = validate_price_quantity(df)
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_negative_price_error(self):
        """Negative price raises error."""
        df = pd.DataFrame({"price": [-10.0], "quantity": [1]})
        result = validate_price_quantity(df)
        assert any("negative" in e.lower() for e in result["errors"])

    def test_zero_price_warning(self):
        """Zero price raises warning."""
        df = pd.DataFrame({"price": [0.0], "quantity": [1]})
        result = validate_price_quantity(df)
        assert any("zero" in w.lower() for w in result["warnings"])

    def test_negative_quantity_info(self):
        """Negative quantity is info (returns)."""
        df = pd.DataFrame({"price": [10.0], "quantity": [-1]})
        result = validate_price_quantity(df)
        assert any("return" in i.lower() for i in result["info"])


class TestFindSKUDescriptionConflicts:
    """Test SKU description conflict detection."""

    def test_no_conflicts(self):
        """No conflicts when one description per SKU."""
        df = pd.DataFrame({
            "stockcode": ["A", "A", "B"],
            "product": ["Product A", "Product A", "Product B"],
        })
        conflicts = find_sku_description_conflicts(df)
        assert conflicts.empty

    def test_conflicts_detected(self):
        """Conflicts detected when multiple descriptions per SKU."""
        df = pd.DataFrame({
            "stockcode": ["A", "A", "B"],
            "product": ["Product A", "Product A v2", "Product B"],
        })
        conflicts = find_sku_description_conflicts(df)
        assert len(conflicts) == 1
        assert conflicts.iloc[0]["stockcode"] == "A"
        assert conflicts.iloc[0]["n_descriptions"] == 2


class TestCalculateMethodReadiness:
    """Test method readiness assessment."""

    @pytest.fixture
    def elasticity_ready_df(self):
        """DataFrame with sufficient price variation for elasticity."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=52, freq="W")
        skus = [f"SKU{i}" for i in range(10)]
        rows = []
        for sku in skus:
            for d in dates:
                price = np.random.uniform(5, 20)
                rows.append({
                    "date": d,
                    "stockcode": sku,
                    "price": price,
                    "quantity": np.random.poisson(5) + 1,
                    "customer_id": f"C{np.random.randint(100)}",
                    "transaction_id": f"TX{len(rows)}",
                })
        return pd.DataFrame(rows)

    def test_elasticity_ready(self, elasticity_ready_df):
        """Elasticity readiness with sufficient data."""
        readiness = calculate_method_readiness(elasticity_ready_df, "elasticity")
        assert readiness["status"] in ("ready", "directional")
        assert "eligible_skus" in readiness["details"]

    def test_elasticity_blocked_low_variation(self):
        """Elasticity blocked when no price variation."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20),
            "stockcode": ["SKU1"] * 20,
            "price": [10.0] * 20,  # No variation
            "quantity": [1] * 20,
            "customer_id": ["C1"] * 20,
            "transaction_id": range(20),
        })
        readiness = calculate_method_readiness(df, "elasticity")
        assert readiness["status"] == "blocked"

    def test_segmentation_blocked_few_customers(self):
        """Segmentation blocked with too few customers."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "stockcode": ["SKU1"] * 10,
            "price": [10.0] * 10,
            "quantity": [1] * 10,
            "customer_id": ["C1"] * 10,  # Only 1 customer
            "transaction_id": range(10),
        })
        readiness = calculate_method_readiness(df, "segmentation")
        assert readiness["status"] == "blocked"

    def test_format_readiness_for_ui(self):
        """Formatting produces readable output."""
        readiness = {
            "status": "ready",
            "reason": "All good",
            "details": {"eligible_skus": 10, "total_skus": 10},
            "requirements": ["Req 1", "Req 2"],
        }
        formatted = format_readiness_for_ui(readiness)
        assert "READY" in formatted
        assert "eligible_skus: 10" in formatted
        assert "Req 1" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])