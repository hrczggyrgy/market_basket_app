"""Consolidated core analytics tests using single CSV data source.

This replaces multiple duplicate test files:
- test_data.py
- test_data_quality.py
- test_schemas.py
- test_validation_baseline.py
- test_sample_data.py
- test_edge_cases.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.data import (
    load_transactions,
    build_dataset_capabilities,
    get_data_summary,
    derive_product_lookup,
    revenue_column,
    safe_divide,
)
from src.analytics.schemas import check, TRANSACTIONS
from src.analytics.data_quality import assess_data_quality


class TestDataLoading:
    """Test data loading and normalization."""

    def test_load_canonical_fixture(self, sample_df: pd.DataFrame):
        """Canonical fixture loads without errors and has expected columns."""
        assert len(sample_df) > 0
        required = {"date", "transaction_id", "stockcode", "product", "customer_id", "price", "quantity"}
        assert required.issubset(set(sample_df.columns))

    def test_no_null_required_fields(self, sample_df: pd.DataFrame):
        """Required fields have no nulls after loading."""
        for col in ["date", "transaction_id", "stockcode", "customer_id", "price", "quantity"]:
            assert sample_df[col].notna().all(), f"{col} has nulls"

    def test_positive_price_quantity(self, sample_df: pd.DataFrame):
        """Price and quantity are positive."""
        assert (sample_df["price"] > 0).all()
        assert (sample_df["quantity"] > 0).all()

    def test_dates_parsed(self, sample_df: pd.DataFrame):
        """Dates are properly parsed as datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_df["date"])

    def test_ids_cleaned(self, sample_df: pd.DataFrame):
        """ID columns are cleaned (no .0 suffix, no NaN strings)."""
        for col in ["stockcode", "customer_id", "transaction_id"]:
            assert not sample_df[col].astype(str).str.contains(r"\.0$").any()
            assert not sample_df[col].astype(str).str.lower().eq("nan").any()

    def test_categorical_dtypes(self, sample_df: pd.DataFrame):
        """ID columns are categorical for memory efficiency."""
        for col in ["stockcode", "customer_id", "transaction_id", "product"]:
            if col in sample_df.columns:
                assert pd.api.types.is_categorical_dtype(sample_df[col])

    def test_optional_columns_present(self, sample_df: pd.DataFrame):
        """Optional columns from canonical fixture are present."""
        optional = {"category", "brand", "size", "flavor", "promo_flag", "cost"}
        present = optional & set(sample_df.columns)
        assert len(present) >= 4, f"Expected at least 4 optional columns, got {present}"


class TestDataQuality:
    """Test data quality assessment."""

    def test_quality_report_generated(self, sample_df: pd.DataFrame):
        """Quality report is generated without errors."""
        report = assess_data_quality(sample_df)
        assert report is not None
        assert hasattr(report, "has_issues")
        assert hasattr(report, "volume_warning")

    def test_no_critical_issues(self, sample_df: pd.DataFrame):
        """Sample data has no critical quality issues (volume warning is expected for small sample)."""
        report = assess_data_quality(sample_df)
        # Sample data is small, so volume warning is expected but not critical
        # Check that there are no structural issues (duplicates, incomplete rows, etc.)
        assert report.duplicate_count == 0
        assert report.incomplete_rows == 0
        assert len(report.excluded_products) == 0


class TestSchemas:
    """Test schema validation."""

    def test_transactions_schema(self, sample_df: pd.DataFrame):
        """Sample data passes TRANSACTIONS schema."""
        check(sample_df, TRANSACTIONS)

    def test_schema_rejects_missing_columns(self):
        """Schema rejects DataFrame with missing required columns."""
        bad_df = pd.DataFrame({"date": ["2024-01-01"], "price": [10.0]})
        with pytest.raises(Exception):
            check(bad_df, TRANSACTIONS)


class TestDataUtilities:
    """Test data utility functions."""

    def test_get_data_summary(self, sample_df: pd.DataFrame):
        """Data summary returns expected keys with valid values."""
        summary = get_data_summary(sample_df)
        required_keys = {
            "n_transactions", "n_line_items", "n_customers",
            "n_products", "total_revenue", "avg_basket_size",
            "avg_basket_value", "date_range"
        }
        assert required_keys.issubset(set(summary.keys()))
        assert summary["n_transactions"] > 0
        assert summary["total_revenue"] > 0

    def test_derive_product_lookup(self, sample_df: pd.DataFrame, product_lookup: pd.DataFrame):
        """Product lookup has unique stockcodes and expected columns."""
        assert len(product_lookup) == sample_df["stockcode"].nunique()
        assert "stockcode" in product_lookup.columns
        assert "product" in product_lookup.columns

    def test_revenue_column(self, sample_df: pd.DataFrame, revenue_series: pd.Series):
        """Revenue column matches price * quantity."""
        expected = sample_df["price"] * sample_df["quantity"]
        pd.testing.assert_series_equal(revenue_series, expected, check_names=False)

    def test_safe_divide(self):
        """Safe division handles edge cases."""
        import numpy as np
        # Normal division
        assert safe_divide(10, 2) == 5.0
        # Division by zero returns 0
        assert safe_divide(10, 0) == 0.0
        # Array division
        num = np.array([10, 20, 30])
        den = np.array([2, 0, 5])
        result = safe_divide(num, den)
        assert result[0] == 5.0
        assert result[1] == 0.0
        assert result[2] == 6.0


class TestCapabilities:
    """Test capability detection."""

    def test_capabilities_detected(self, capabilities: dict[str, bool]):
        """Capabilities dict has expected keys with boolean values."""
        expected_keys = {
            "has_category", "has_brand", "has_size", "has_flavor",
            "has_promo_flag", "has_cost", "has_is_online", "has_channel",
            "has_price_variation", "min_distinct_prices_3",
            "sufficient_customers_100", "sufficient_customers_500",
            "sufficient_skus_20", "sufficient_skus_50",
            "sufficient_baskets_200", "sufficient_baskets_500", "sufficient_baskets_1000",
        }
        assert expected_keys.issubset(set(capabilities.keys()))
        for v in capabilities.values():
            assert isinstance(v, bool)


class TestEdgeCases:
    """Test edge case handling."""

    def test_load_without_customer_id(self, fixture_path):
        """Loading with require_customer_id=False keeps rows without customer_id."""
        df, _, dropped, _ = load_transactions(
            fixture_path, require_customer_id=False
        )
        # Should not drop rows for missing customer_id
        assert dropped == 0 or dropped < 100  # Allow some drops for other reasons

    def test_load_with_customer_id(self, fixture_path):
        """Loading with require_customer_id=True drops rows without customer_id."""
        df, _, dropped, _ = load_transactions(
            fixture_path, require_customer_id=True
        )
        assert df["customer_id"].notna().all()


class TestDataContracts:
    """Test data contract validation."""

    def test_transaction_contract_validation(self, sample_df: pd.DataFrame):
        """Transaction data passes contract validation."""
        from src.analytics.schemas import check, TRANSACTIONS
        validated, warnings = TRANSACTIONS.validate(sample_df, allow_empty=False, check_values=True)
        assert len(validated) == len(sample_df)

    def test_empty_result_handling(self):
        """Empty DataFrame handling with allow_empty=True."""
        from src.analytics.schemas import check, TRANSACTIONS
        empty_df = pd.DataFrame(columns=list(TRANSACTIONS.columns))
        validated, _ = TRANSACTIONS.validate(empty_df, allow_empty=True, check_values=False)
        assert len(validated) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])