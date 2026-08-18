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
from src.analytics.schemas import RFM_FEATURES, TRANSACTIONS


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
    assert (
        caps["has_category"] and caps["has_brand"] and caps["has_promo_flag"] and caps["has_cost"]
    )
    assert not caps["has_channel"]
    minimal = sample_df[list(TRANSACTIONS.columns)]
    caps2 = build_dataset_capabilities(minimal)
    # Column-based capabilities should all be False for minimal data
    column_caps = ["has_category", "has_brand", "has_size", "has_flavor", "has_promo_flag", "has_cost", "has_is_online", "has_channel"]
    assert not any(caps2.get(c, False) for c in column_caps)


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


def test_build_dataset_capabilities_cv_edge_case() -> None:
    """Verify has_price_variation is an aggregate (max CV across SKUs).

    SKU A: prices [10, 10, 10] -> CV = 0
    SKU B: prices [100, 200, 300] -> mean=200, std≈81.65, CV≈0.408
    SKU C: prices [5, 5, 5] -> CV = 0

    has_price_variation should be True because SKU B has CV >= 0.05,
    even though 2/3 SKUs have zero variation.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "stockcode": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "price": [10.0, 10.0, 10.0, 100.0, 200.0, 300.0, 5.0, 5.0, 5.0],
            "date": pd.date_range("2024-01-01", periods=9),
            "transaction_id": [f"T{i}" for i in range(9)],
            "customer_id": ["C1"] * 9,
            "quantity": [1] * 9,
            "product": ["P"] * 9,
        }
    )
    caps = build_dataset_capabilities(df)
    assert caps["has_price_variation"] is True


def test_load_transactions_detects_returns_and_excludes_from_aggregates() -> None:
    """Policy: DQ report only — returns (negative quantity/price) are detected,
    reported in the warning, and dropped. Downstream aggregates (RFM, baskets)
    reflect only the kept positive rows. No netting.
    """
    from src.analytics.segmentation import compute_rfm_features

    raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "stockcode": ["P1", "P2", "P3", "P1", "P2", "P3"],
            "product": ["p", "p", "p", "p", "p", "p"],
            "customer_id": ["C1", "C1", "C1", "C2", "C2", "C2"],
            "price": [10.0, 20.0, -5.0, 15.0, -10.0, 30.0],
            "quantity": [2, 1, 3, 1, 2, 1],
        }
    )
    df, warning, dropped, _ = load_transactions(io.BytesIO(raw.to_csv(index=False).encode()))

    # Return detection reported in warning
    assert "return row(s)" in warning.lower()
    assert "excluded" in warning.lower()
    assert dropped == 2  # rows with negative price: row 2 (price=-5) and row 4 (price=-10)

    # Returns are excluded from the kept data
    assert (df["price"] > 0).all()
    assert (df["quantity"] > 0).all()
    assert len(df) == 4  # 6 input - 2 returns = 4

    # RFM aggregates reflect only positive rows (no netting)
    rfm = compute_rfm_features(df)
    RFM_FEATURES.validate(rfm)  # no SchemaError
    assert (rfm["order_value_cv"] >= 0).all()
    assert rfm["order_value_cv"].notna().all()

    # C1 had: (10*2) + (20*1) = 40 kept; return was (-5*3) = -15 dropped
    c1 = rfm.set_index("customer_id").loc["C1"]
    assert c1["monetary"] == 40.0
    assert c1["frequency"] == 2

    # C2 had: (15*1) + (30*1) = 45 kept; return was (-10*2) = -20 dropped
    c2 = rfm.set_index("customer_id").loc["C2"]
    assert c2["monetary"] == 45.0
    assert c2["frequency"] == 2
