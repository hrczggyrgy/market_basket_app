"""Compatibility shim for Streamlit caching that redirects to ResultStore.

Provides the same function signatures as the original caching functions,
but uses the versioned ResultStore instead of Streamlit's caching mechanism.

All streamlit cache decorators have been removed. Functions now compute
results directly and cache them via the ResultStore with deterministic
parameter hashing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.orchestration.result_store import (
    DEFAULT_SCHEMA_VERSION,
    make_key,
    param_hash,
)
from src.orchestration.result_store import (
    set as set_result,
)
from src.performance.profiler import measure_analysis
from src.utils.hashing import df_hash as _df_hash

# Schema and feature versioning for cache invalidation
_schema_version: str = DEFAULT_SCHEMA_VERSION
_feature_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Compatibility shim functions (no streamlit cache decorators)
# Each function computes its result and caches via ResultStore
# ---------------------------------------------------------------------------


@measure_analysis
def cached_pricing_analysis(
    df: pd.DataFrame,
    min_periods: int = 5,
    min_price_variation: float = 0.05,
    kvi_method: str = "heuristic",
) -> Any:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.

    Note: This is a shim that delegates to the actual analytics pipeline.
    The actual computation is performed by the engine; this function
    serves as the entry point with ResultStore-based caching.
    """
    from src.analytics.pricing.pipeline import run_pricing_analysis

    result = run_pricing_analysis(df, min_periods=min_periods, min_price_variation=min_price_variation, kvi_method=kvi_method)
    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash, "min_periods": min_periods, "min_price_variation": min_price_variation, "kvi_method": kvi_method}, schema_version=_schema_version)
    key = make_key("default", "pricing", "1.0.0", ph)
    set_result("default", "pricing", "1.0.0", ph, result)
    return result


# Backward compatibility alias for tests
run_cached_pricing_analysis = cached_pricing_analysis


@measure_analysis
def cached_enrich_categories(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
    from src.analytics.category import enrich_with_categories

    result = enrich_with_categories(df)
    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "enrich_categories", "1.0.0", ph)
    set_result("default", "enrich_categories", "1.0.0", ph, result)
    return result


@measure_analysis
def cached_basket_metrics(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
    from src.analytics.basket_metrics import (
        basket_penetration_over_time,
        compute_basket_penetration,
    )

    penetration = compute_basket_penetration(df)
    over_time = basket_penetration_over_time(df)
    result = {"penetration": penetration, "over_time": over_time}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "basket_metrics", "1.0.0", ph)
    set_result("default", "basket_metrics", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_cohort_analysis(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
    from src.analytics.cohort import (
        compute_cohort_ltv_curve,
        compute_cohort_sizes,
        compute_cohorts,
    )

    retention = compute_cohorts(df)
    ltv = compute_cohort_ltv_curve(df)
    sizes = compute_cohort_sizes(df)
    result = {"retention": retention, "ltv": ltv, "sizes": sizes}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "cohort_analysis", "1.0.0", ph)
    set_result("default", "cohort_analysis", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_segmentation(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
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
    result = {
        "rfm_features": rfm_features,
        "rfm_segments": rfm_seg,
        "behavioral": beh_seg,
        "value_based": val_seg,
    }

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "segmentation", "1.0.0", ph)
    set_result("default", "segmentation", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_switching(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
    from src.analytics.switching import (
        compute_switching_matrix,
        compute_switching_status,
        compute_transition_matrix,
    )

    matrix = compute_switching_matrix(df)
    status = compute_switching_status(df)
    transition = compute_transition_matrix(df)
    result = {"matrix": matrix, "status": status, "transition": transition}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "switching", "1.0.0", ph)
    set_result("default", "switching", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_transference(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching."""
    from src.analytics.switching_engine import SwitchingEngine
    from src.analytics.transference import (
        compute_demand_transference_matrix,
        compute_recovery_hhi,
        compute_substitutable_demand_percentage,
    )

    # Use SwitchingEngine to get switching edges (single computation)
    engine = SwitchingEngine(df)
    switching_edges = engine.get_switching_edges()
    transference = compute_demand_transference_matrix(df, switching_edges=switching_edges)
    sdp = compute_substitutable_demand_percentage(transference, df)
    hhi = compute_recovery_hhi(transference)
    result = {
        "transference": transference,
        "sdp": sdp,
        "hhi": hhi,
    }

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "transference", "1.0.0", ph)
    set_result("default", "transference", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_copurchase(
    df: pd.DataFrame,
    top_n_products: int = 200,
    min_cooccurrence: int = 5,
    max_pairs: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    Fast co-purchase affinity analysis (Tier B). Does NOT run FP-Growth.
    For association rules, use cached_rules() which is Tier C (on demand).
    """
    from src.analytics.copurchase import get_top_affinity_pairs

    affinity = get_top_affinity_pairs(
        df,
        top_n_products=top_n_products,
        min_cooccurrence=min_cooccurrence,
        max_pairs=max_pairs,
    )
    result = {"affinity": affinity}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash, "top_n_products": top_n_products, "min_cooccurrence": min_cooccurrence, "max_pairs": max_pairs}, schema_version=_schema_version)
    key = make_key("default", "copurchase", "1.0.0", ph)
    set_result("default", "copurchase", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_rules(
    df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
    max_skus: int = 5000,
    min_confidence: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    FP-Growth + Association Rules (Tier C - on demand only).
    Run only when user explicitly requests rules analysis.
    """
    from src.analytics.rules import create_basket_matrix, generate_rules, run_fpgrowth

    basket = create_basket_matrix(df)
    itemsets = run_fpgrowth(basket, min_support=min_support, max_len=max_len, max_skus=max_skus)
    rules = generate_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    result = {"itemsets": itemsets, "rules": rules}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash, "min_support": min_support, "max_len": max_len, "max_skus": max_skus, "min_confidence": min_confidence}, schema_version=_schema_version)
    key = make_key("default", "rules", "1.0.0", ph)
    set_result("default", "rules", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_rules_bootstrap(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    Bootstrap lift CI for association rules (Tier C - on demand only).
    Run only when user explicitly requests robustness calculation.
    """
    from src.analytics.rules import bootstrap_lift_ci

    result = bootstrap_lift_ci(df, rules, n_resamples=n_resamples, seed=seed)

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash, "n_resamples": n_resamples, "seed": seed}, schema_version=_schema_version)
    key = make_key("default", "rules_bootstrap", "1.0.0", ph)
    set_result("default", "rules_bootstrap", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_promotion(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    Fast promotional analytics layer (Tier B): promo detection, baseline, ROI.
    For causal incrementality, use cached_promotion_advanced() which is Tier C (on demand).
    """
    from src.analytics.promo import (
        compute_promo_baseline,
        detect_promotions,
        promo_roi_analysis,
    )

    promos = detect_promotions(df)
    baseline = compute_promo_baseline(df, promos)
    roi = promo_roi_analysis(df, promos)
    result = {"promos": promos, "baseline": baseline, "roi": roi}

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "promotion", "1.0.0", ph)
    set_result("default", "promotion", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_promotion_advanced(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    Advanced promotional analytics layer (Tier C - on demand only): causal incrementality,
    cross-SKU effects, event study, bootstrap CI.
    Run only when user explicitly requests advanced promo analysis.
    """
    from src.analytics.promo import detect_promotions
    from src.analytics.promo.causal import (
        build_promo_causal_panel,
        compute_causal_waterfall,
        estimate_cross_sku_effects,
        estimate_event_study,
        estimate_twfe_promo_effect,
    )

    # Detect promotions first
    promos = detect_promotions(df)

    # Build causal panel
    panel = build_promo_causal_panel(df, promos)

    # Run advanced analyses
    twfe = estimate_twfe_promo_effect(panel)
    event_study = estimate_event_study(panel, promos)
    cross_effects = estimate_cross_sku_effects(panel, promos)
    waterfall = compute_causal_waterfall(df, promos)

    result = {
        "twfe": twfe,
        "event_study": event_study,
        "cross_effects": cross_effects,
        "waterfall": waterfall,
    }

    df_hash = _df_hash(df)
    ph = param_hash({"df_hash": df_hash}, schema_version=_schema_version)
    key = make_key("default", "promotion_advanced", "1.0.0", ph)
    set_result("default", "promotion_advanced", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_clv(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    min_repeat_customers: int = 10,
    discount_rate_pct: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    CLV prediction (Tier C - on demand only): BG/NBD + Gamma-Gamma.
    Run only when user explicitly requests CLV analysis.
    """
    from src.analytics.clv import CLVEngine

    engine = CLVEngine(df)
    predictions, diagnostics = engine.predict(
        prediction_horizon_days=prediction_horizon_days,
        freq=freq,
        min_repeat_customers=min_repeat_customers,
        discount_rate_pct=discount_rate_pct,
    )
    result = {"predictions": predictions, "diagnostics": diagnostics}

    df_hash = _df_hash(df)
    ph = param_hash({
        "df_hash": df_hash,
        "prediction_horizon_days": prediction_horizon_days,
        "freq": freq,
        "min_repeat_customers": min_repeat_customers,
        "discount_rate_pct": discount_rate_pct,
    }, schema_version=_schema_version)
    key = make_key("default", "clv", "1.0.0", ph)
    set_result("default", "clv", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_clv_customer(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    discount_rate_pct: float = 0.0,
) -> pd.DataFrame:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    CLV customer view with behavior metrics (Tier C - on demand only).
    Run only when user explicitly requests CLV customer analysis.
    """
    from src.analytics.clv import CLVEngine

    engine = CLVEngine(df)
    result = engine.compute_customer_view(
        prediction_horizon_days=prediction_horizon_days,
        freq=freq,
        discount_rate_pct=discount_rate_pct,
    )

    df_hash = _df_hash(df)
    ph = param_hash({
        "df_hash": df_hash,
        "prediction_horizon_days": prediction_horizon_days,
        "freq": freq,
        "discount_rate_pct": discount_rate_pct,
    }, schema_version=_schema_version)
    key = make_key("default", "clv_customer", "1.0.0", ph)
    set_result("default", "clv_customer", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_cdt(
    df: pd.DataFrame,
    min_cooccurrence: int = 5,
    similarity_method: str = "phi",
    linkage_method: str = "ward",
    min_clusters: int = 2,
    max_clusters: int = 15,
    min_cluster_size: int = 3,
    quality_threshold: float = 0.6,
    split_criterion: str = "mutual_info",
    resolution: float = 1.0,
    community_method: str = "none",
) -> dict[str, Any]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    CDT analysis (Tier C - on demand only): Customer Decision Tree construction.
    Run only when user explicitly requests CDT analysis.
    """
    from src.analytics.cdt.tree import CDTEngine

    engine = CDTEngine(df)
    result = engine.build_cdt(
        min_cooccurrence=min_cooccurrence,
        similarity_method=similarity_method,
        linkage_method=linkage_method,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        quality_threshold=quality_threshold,
        split_criterion=split_criterion,
        resolution=resolution,
        community_method=community_method,
    )

    df_hash = _df_hash(df)
    ph = param_hash({
        "df_hash": df_hash,
        "min_cooccurrence": min_cooccurrence,
        "similarity_method": similarity_method,
        "linkage_method": linkage_method,
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "min_cluster_size": min_cluster_size,
        "quality_threshold": quality_threshold,
        "split_criterion": split_criterion,
        "resolution": resolution,
        "community_method": community_method,
    }, schema_version=_schema_version)
    key = make_key("default", "cdt", "1.0.0", ph)
    set_result("default", "cdt", "1.0.0", ph, result)

    return result


@measure_analysis
def cached_assortment(
    df: pd.DataFrame,
    max_skus: int = 50,
    min_coverage: float = 0.8,
    min_category_coverage: float = 0.5,
    recovery_margin: float = 0.3,
) -> dict[str, Any]:
    """Compatibility shim - uses ResultStore instead of Streamlit caching.
    
    Assortment optimization (Tier C - on demand only): heuristic + MILP.
    Run only when user explicitly requests assortment analysis.
    """
    from src.analytics.assortment import AssortmentEngine

    engine = AssortmentEngine(df)
    result = engine.optimize_heuristic(
        max_skus=max_skus,
        min_coverage=min_coverage,
        min_category_coverage=min_category_coverage,
        recovery_margin=recovery_margin,
    )

    df_hash = _df_hash(df)
    ph = param_hash({
        "df_hash": df_hash,
        "max_skus": max_skus,
        "min_coverage": min_coverage,
        "min_category_coverage": min_category_coverage,
        "recovery_margin": recovery_margin,
    }, schema_version=_schema_version)
    key = make_key("default", "assortment", "1.0.0", ph)
    set_result("default", "assortment", "1.0.0", ph, {"solution": result})

    return {"solution": result}
