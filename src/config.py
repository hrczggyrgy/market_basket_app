"""Configuration types for the Market Basket Analysis application."""

from typing import Any, Dict, TypedDict


class AnalysisParams(TypedDict, total=False):
    """Analysis-specific parameters for each mode."""

    # Co-purchase
    top_n_products: int
    min_lift: float

    # Add-on
    min_support: float
    top_n: int

    # Switching
    window_days: int
    min_transactions: int

    # Choice Prediction Model
    max_depth: int
    min_samples_leaf: int
    prediction_window: int

    # Decision Tree & Patterns
    similarity_method: str
    min_cooccurrence: int
    linkage_method: str
    min_k: int
    max_k: int
    min_cluster_size: int
    quality_threshold: int
    max_sub: float

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
