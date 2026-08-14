"""Unit tests for the UI cache helpers in src/ui/features.py.

These exercise the underlying analytics builders (the st.cache_data wrapper
itself is a thin memoizer; contract validity flows from the analytics side).
The cached functions are imported but bypassed by calling the wrapped logic
through the same path used in the app.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.data import load_transactions
from src.ui.features import (
    get_basket_matrix,
    get_detected_promotions,
    get_product_lookup,
    get_product_metrics,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_lookup_rows_match_products(sample_df: pd.DataFrame) -> None:
    lookup = get_product_lookup(sample_df)
    assert set(lookup["stockcode"]) == set(sample_df["stockcode"].unique())
    assert "product" in lookup.columns


def test_product_metrics_contract_alignment(sample_df: pd.DataFrame) -> None:
    from src.analytics.schemas import PRODUCT_METRICS

    metrics = get_product_metrics(sample_df)
    PRODUCT_METRICS.validate(metrics)
    assert not metrics.empty


def test_basket_matrix_transaction_rows(sample_df: pd.DataFrame) -> None:
    basket = get_basket_matrix(sample_df)
    assert not basket.empty
    assert basket.index.nunique() <= sample_df["transaction_id"].nunique()


def test_detected_promotions_contract(sample_df: pd.DataFrame) -> None:
    from src.analytics.schemas import PROMO_PERIODS

    promos = get_detected_promotions(sample_df)
    if not promos.empty:
        PROMO_PERIODS.validate(promos)
