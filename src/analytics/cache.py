"""Streamlit caching layer for expensive analytics computations.

Provides consistent cache keys and invalidation strategies for analytics pipelines.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st


def _df_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Fast hash of DataFrame content for cache keys."""
    if cols is None:
        cols = df.columns.tolist()
    # Use a subset of columns for hashing to avoid hashing huge DataFrames
    # Hash shape + first/last rows + column dtypes
    sample = df[cols].head(1000) if len(df) > 1000 else df[cols]
    content = f"{df.shape}{sample.dtypes.to_dict()}{sample.to_numpy().tobytes()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key from arguments."""
    parts = [prefix]
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            parts.append(_df_hash(arg))
        else:
            parts.append(str(arg))
    for k, v in sorted(kwargs.items()):
        if isinstance(v, pd.DataFrame):
            parts.append(f"{k}=<df:{_df_hash(v)}>")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_pricing_analysis(
    df_hash: str,
    min_periods: int = 5,
    min_price_variation: float = 0.05,
    kvi_method: str = "heuristic",
) -> Any:
    """Cached pricing analysis - wrapper that uses DataFrame hash as key."""
    # The actual computation is done in the caller which passes the real DataFrame
    # This is a placeholder - real implementation uses the hash for invalidation
    pass


def run_cached_pricing_analysis(
    transactions_df: pd.DataFrame,
    min_periods: int = 5,
    min_price_variation: float = 0.05,
    kvi_method: str = "heuristic",
) -> Any:
    """Run pricing analysis with automatic caching based on DataFrame content."""
    from src.analytics.pricing.pipeline import run_pricing_analysis

    df_hash = _df_hash(transactions_df, ["date", "transaction_id", "stockcode", "price", "quantity", "customer_id"])
    cache_key = _cache_key("pricing", df_hash, min_periods, min_price_variation, kvi_method)

    @st.cache_data(show_spinner="Computing pricing analysis...", max_entries=3)
    def _compute(key: str, df: pd.DataFrame) -> Any:
        return run_pricing_analysis(df, min_periods, min_price_variation, kvi_method)

    return _compute(cache_key, transactions_df)


@st.cache_data(show_spinner="Computing category enrichment...", max_entries=5)
def cached_enrich_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Cache category enrichment results."""
    from src.analytics.category import enrich_with_categories
    return enrich_with_categories(df)


@st.cache_data(show_spinner="Computing basket metrics...", max_entries=5)
def cached_basket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Cache basket metrics computation."""
    from src.analytics.basket_metrics import compute_basket_over_time, compute_basket_penetration
    penetration = compute_basket_penetration(df)
    over_time = compute_basket_over_time(df)
    return {"penetration": penetration, "over_time": over_time}


@st.cache_data(show_spinner="Computing cohort analysis...", max_entries=5)
def cached_cohort_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache cohort analysis results."""
    from src.analytics.cohort import (
        compute_cohort_ltv,
        compute_cohort_retention,
        compute_cohort_sizes,
    )
    return {
        "retention": compute_cohort_retention(df),
        "ltv": compute_cohort_ltv(df),
        "sizes": compute_cohort_sizes(df),
    }


@st.cache_data(show_spinner="Computing segmentation...", max_entries=3)
def cached_segmentation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache segmentation results."""
    from src.analytics.segmentation import (
        behavioral_segmentation,
        compute_rfm_features,
        rfm_segmentation,
        value_based_segmentation,
    )
    rfm_features = compute_rfm_features(df)
    rfm_seg = rfm_segmentation(rfm_features)
    beh_seg = behavioral_segmentation(df)
    val_seg = value_based_segmentation(df)
    return {
        "rfm_features": rfm_features,
        "rfm_segments": rfm_seg,
        "behavioral": beh_seg,
        "value_based": val_seg,
    }


@st.cache_data(show_spinner="Computing switching analysis...", max_entries=3)
def cached_switching(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache switching analysis results."""
    from src.analytics.switching import (
        compute_switching_matrix,
        compute_switching_status,
        compute_transition_matrix,
    )
    return {
        "matrix": compute_switching_matrix(df),
        "status": compute_switching_status(df),
        "transition": compute_transition_matrix(df),
    }


@st.cache_data(show_spinner="Computing demand transference...", max_entries=3)
def cached_transference(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache demand transference results."""
    from src.analytics.switching import compute_switching_matrix
    from src.analytics.transference import (
        compute_demand_transference_matrix,
        compute_recovery_hhi,
        compute_substitutable_demand_percentage,
    )
    matrix = compute_switching_matrix(df)
    transference = compute_demand_transference_matrix(df, matrix)
    sdp = compute_substitutable_demand_percentage(transference, df)
    hhi = compute_recovery_hhi(transference)
    return {
        "transference": transference,
        "sdp": sdp,
        "hhi": hhi,
    }


@st.cache_data(show_spinner="Computing copurchase rules...", max_entries=3)
def cached_copurchase(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache copurchase analysis results."""
    from src.analytics.addon import get_top_affinity_pairs
    from src.analytics.copurchase import compute_association_rules, compute_frequent_itemsets
    itemsets = compute_frequent_itemsets(df)
    rules = compute_association_rules(itemsets)
    affinity = get_top_affinity_pairs(df)
    return {"itemsets": itemsets, "rules": rules, "affinity": affinity}


@st.cache_data(show_spinner="Computing promotion analysis...", max_entries=3)
def cached_promotion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache promotion analysis results."""
    from src.analytics.promo import (
        compute_promo_baseline,
        detect_promotions,
        promo_roi_analysis,
    )
    promos = detect_promotions(df)
    baseline = compute_promo_baseline(df, promos)
    roi = promo_roi_analysis(df, promos)
    return {"promos": promos, "baseline": baseline, "roi": roi}


@st.cache_data(show_spinner="Computing CLV...", max_entries=2)
def cached_clv(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache CLV computation results."""
    from src.analytics.clv import fit_clv_model, predict_clv
    model = fit_clv_model(df)
    predictions = predict_clv(model, df)
    return {"model": model, "predictions": predictions}


@st.cache_data(show_spinner="Computing CDT...", max_entries=2)
def cached_cdt(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache CDT computation results."""
    from src.analytics.cdt import build_cdt, tree_to_dataframe
    tree = build_cdt(df)
    df_tree = tree_to_dataframe(tree)
    return {"tree": tree, "dataframe": df_tree}


@st.cache_data(show_spinner="Computing assortment optimization...", max_entries=2)
def cached_assortment(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cache assortment optimization results."""
    from src.analytics.assortment import optimize_assortment
    result = optimize_assortment(df)
    return {"solution": result}
