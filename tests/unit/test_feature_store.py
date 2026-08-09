"""Unit tests for the Feature Store (pure analytics, no Streamlit)."""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import sparse

from src.analytics.data import load_transactions
from src.analytics.feature_store import build_feature_store


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_store_contract(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    assert "customer_id" in fs.customer_features.columns
    assert "stockcode" in fs.product_features.columns
    assert "basket_revenue" in fs.basket_features.columns
    assert "iso_week" in fs.basket_features.columns
    assert fs.customer_product_binary.shape == (
        len(fs.customers),
        len(fs.products),
    )
    assert fs.customer_product_counts.shape == fs.customer_product_binary.shape


def test_store_product_lookup(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    assert set(fs.product_lookup["stockcode"]) == set(fs.products)
    assert "product" in fs.product_lookup.columns


def test_store_weekly_panel(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    expect = {"units", "revenue", "avg_price", "n_transactions", "n_customers", "active_days"}
    assert expect.issubset(set(fs.weekly_product_panel.columns))
    assert (fs.weekly_product_panel["units"] >= 0).all()


def test_store_sparse_counts_match_raw(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    c = sparse.csr_matrix
    dense = pd.DataFrame(
        fs.customer_product_counts.toarray(),
        index=fs.customers,
        columns=fs.products,
    )
    check = (
        sample_df.groupby(["customer_id", "stockcode"])["quantity"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=fs.customers, columns=fs.products)
        .fillna(0)
    )
    check.index.name = None
    check.columns.name = None
    pd.testing.assert_frame_equal(dense, check.astype("float32"))


def test_store_category_map(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    assert not fs.category_map().empty
    assert fs.category_map()["SKU000"] == "Coffee"


def test_store_weekly_revenue_matches_total(sample_df: pd.DataFrame) -> None:
    fs = build_feature_store(sample_df)
    total = float((sample_df["price"] * sample_df["quantity"]).sum())
    assert abs(float(fs.weekly_product_panel["revenue"].sum()) - total) < 0.01


def test_store_empty_input() -> None:
    fs = build_feature_store(pd.DataFrame(columns=["date", "transaction_id", "stockcode", "product", "customer_id", "price", "quantity"]))
    assert fs.customer_features.empty
    assert fs.weekly_product_panel.empty