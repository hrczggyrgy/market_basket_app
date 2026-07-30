"""Configuration types for the Market Basket Analysis application."""

from typing import Any, Dict, List, Optional, TypedDict


class AnalysisParams(TypedDict, total=False):
    """Analysis-specific parameters for each mode."""

    # Association Rules / Co-purchase / Add-on / Switching
    top_n_products: int
    min_lift: float
    min_support: float
    top_n: int

    # Switching
    window_days: int
    min_transactions: int

    # Choice Prediction Model
    max_depth: int
    min_samples_leaf: int
    prediction_window: int

    # Decision Tree & Patterns (CDT)
    similarity_method: str
    similarity_methods: List[str]
    min_cooccurrence: int
    linkage_method: str
    min_k: int
    max_k: int
    min_cluster_size: int
    quality_threshold: int
    max_sub: float
    community_method: str
    community_resolution: float
    split_criterion: str
    split_alpha: float
    attribute_source: str

    # Customer Segmentation
    rfm_method: str
    n_segments: int
    behavioral_clusters: int
    value_horizon: int

    # Product Performance
    lifecycle_period: str
    elasticity_min_periods: int

    # Cohort Analysis
    cohort_period: str
    cohort_metric: str
    max_periods: int

    # Promotional Analytics
    price_change_threshold: int
    min_duration_days: int
    max_duration_days: int
    baseline_window: int
    promo_window: int

    # CDT & Assortment
    demand_transference: bool
    assortment_objective: str
    assortment_max_skus: int
    assortment_min_coverage: float
    assortment_solver: str
    assortment_time_limit: int

    # Pricing & Promotions
    elasticity_method: str
    kvi_method: str
    price_curve_method: str
    cost_col: str
    margin_pct: float

    # Promo Uplift
    uplift_method: str
    propensity_method: str


class Config(TypedDict):
    """Main application configuration."""

    uploaded_file: Any
    use_sample: bool
    column_mapping: Dict[str, str]
    min_support: float
    min_confidence: float
    max_itemset_len: int
    min_lift: float
    analysis_mode: str
    analysis_params: AnalysisParams
    run_analysis: bool
