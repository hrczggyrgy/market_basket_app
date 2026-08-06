"""Tests for transaction loading, capabilities, and summaries."""

import io

import pandas as pd
import pytest

from src.analytics.data import (
    build_dataset_capabilities,
    derive_product_lookup,
    detect_column_mapping,
    get_data_summary,
    load_transactions,
    revenue_column,
    safe_divide,
)
from src.analytics.schemas import TRANSACTIONS


def _csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    return io.BytesIO(df.to_csv(index=False).encode())


def test_load_transactions_canonicalizes(sample_df: pd.DataFrame) -> None:
    TRANSACTIONS.validate(sample_df)
    assert len(sample_df) > 0


def test_load_transactions_auto_mapping(fixture_path) -> None:
    df, _, _, _ = load_transactions(fixture_path)
    TRANSACTIONS.validate(df)


def test_load_transactions_manual_mapping() -> None:
    raw = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-01-01"],
            "order_no": ["A1", "A1"],
            "sku_code": ["X", "Y"],
            "item_desc": ["Item X", "Item Y"],
            "buyer": ["C1", "C1"],
            "unit_price": [2.5, 1.0],
            "units_sold": [2, 1],
        }
    )
    mapping = detect_column_mapping(raw.columns.tolist())
    assert mapping["date"] == "order_date"
    df, _, _, _ = load_transactions(_csv_bytes(raw), mapping)
    TRANSACTIONS.validate(df)
    assert set(df["stockcode"]) == {"X", "Y"}


def test_load_transactions_drops_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "bad-date", "2024-01-02", "2024-01-02"],
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "stockcode": ["A", "B", "C", "D"],
            "product": ["p", "p", "p", "p"],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "price": [1.0, 2.0, -3.0, 4.0],
            "quantity": [1, 1, 1, 1],
        }
    )
    df, warning, dropped, _ = load_transactions(_csv_bytes(raw))
    assert dropped == 2
    assert "Removed" in warning
    TRANSACTIONS.validate(df)


def test_load_transactions_missing_required_column() -> None:
    raw = pd.DataFrame({"date": [], "transaction_id": []})
    with pytest.raises(ValueError, match="Missing required columns"):
        load_transactions(_csv_bytes(raw))


def test_build_dataset_capabilities(sample_df: pd.DataFrame) -> None:
    caps = build_dataset_capabilities(sample_df)
    assert caps["has_category"] and caps["has_brand"] and caps["has_promo_flag"] and caps["has_cost"]
    assert not caps["has_channel"]
    minimal = sample_df[list(TRANSACTIONS.columns)]
    caps2 = build_dataset_capabilities(minimal)
    assert not any(v for v in caps2.values())


def test_get_data_summary(sample_df: pd.DataFrame) -> None:
    summary = get_data_summary(sample_df)
    assert summary["n_transactions"] > 0
    assert summary["n_customers"] > 0
    assert summary["n_products"] > 0
    assert summary["total_revenue"] > 0
    assert summary["avg_basket_size"] >= 1
    assert summary["avg_basket_value"] > 0
    assert "to" in str(summary["date_range"])


def test_derive_product_lookup_unique(sample_df: pd.DataFrame) -> None:
    lookup = derive_product_lookup(sample_df)
    assert lookup["stockcode"].is_unique
    assert lookup["category"].notna().all()


def test_revenue_column(sample_df: pd.DataFrame) -> None:
    rev = revenue_column(sample_df)
    assert (rev > 0).all()
    assert abs(rev.sum() - (sample_df["price"] * sample_df["quantity"]).sum()) < 1e-6


def test_safe_divide() -> None:
    import numpy as np

    result = safe_divide(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
    assert result[0] == 0.5
    assert result[1] == 0.0
    assert safe_divide(1, 0) == 0.0
