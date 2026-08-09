"""Streamlit cache layer for the Feature Store.

Keeps analytics Streamlit-free: only this module builds st.cache_data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.feature_store import FeatureStore, build_feature_store


@st.cache_data(show_spinner="Building feature store...")
def get_feature_store(df: pd.DataFrame) -> FeatureStore:
    """Return the cached Feature Store for the current line-item data.

    Builds once per distinct DataFrame (st.cache_data hashes the frame);
    subsequent reruns hit the cache.
    """
    return build_feature_store(df)


@st.cache_data(show_spinner=False)
def get_product_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Cached product lookup table (stockcode -> product/category/brand/...)."""
    from src.analytics.data import derive_product_lookup

    return derive_product_lookup(df)


@st.cache_data(show_spinner=False)
def get_segment_maps(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached per-customer segment columns and per-basket mission labels.

    Runs the expensive RFM + behavioral clustering + basket-mission pipeline
    once per dataset; returns compact (customer_id, *_segment) and
    (transaction_id, basket_mission) tables. Re-running this is exactly the
    per-rerun cost the cache removes.
    """
    from src.analytics.data import add_segment_columns, assign_basket_mission

    seg = add_segment_columns(df)
    seg_cols = [c for c in seg.columns if c.endswith("_segment")]
    if seg_cols:
        customer_segments = seg[["customer_id", *seg_cols]].drop_duplicates("customer_id")
    else:
        customer_segments = pd.DataFrame(columns=["customer_id"])

    mission = assign_basket_mission(df)
    baskets = mission[["transaction_id", "basket_mission"]].drop_duplicates("transaction_id")
    return customer_segments, baskets


@st.cache_data(show_spinner=False)
def get_basket_matrix(df: pd.DataFrame, sparse_output: bool = False) -> pd.DataFrame:
    """Cached transaction x product basket matrix (students of rules/affinity)."""
    from src.analytics.rules import create_basket_matrix

    return create_basket_matrix(df)


@st.cache_data(show_spinner=False)
def get_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Cached per-product scorecard: revenue, units, orders, customers, etc."""
    from src.analytics.performance import compute_product_metrics

    return compute_product_metrics(df)


@st.cache_data(show_spinner=False)
def get_detected_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """Cached promo-period table (transaction-level, PROMO_PERIODS contract)."""
    from src.analytics.promo import detect_promotions

    return detect_promotions(df)


__all__ = [
    "get_feature_store",
    "get_product_lookup",
    "get_segment_maps",
    "get_basket_matrix",
    "get_product_metrics",
    "get_detected_promotions",
    "FeatureStore",
    "build_feature_store",
]