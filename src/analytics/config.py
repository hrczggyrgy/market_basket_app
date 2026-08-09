"""Configuration settings for the analytics package.

Centralizes all tunable parameters with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AnalyticsConfig:
    """Global configuration for the analytics pipeline."""

    # Data loading
    data_path: str = "sample_data/sample_transactions.csv"
    date_col: str = "date"
    transaction_id_col: str = "transaction_id"
    stockcode_col: str = "stockcode"
    product_col: str = "product"
    customer_id_col: str = "customer_id"
    price_col: str = "price"
    quantity_col: str = "quantity"
    category_col: str = "category"
    brand_col: str = "brand"
    size_col: str = "size"
    promo_flag_col: str = "promo_flag"
    cost_col: str = "cost"

    # Sampling / performance
    sample_top_n_products: int = 200
    min_cooccurrence: int = 5
    min_product_support: int = 2

    # FP-Growth / rules
    min_support: float = 0.01
    max_len_itemsets: int = 3
    min_threshold: float = 0.05

    # Switching
    switching_window_days: int = 90
    min_transactions_switching: int = 3

    # Co-purchase / affinity
    affinity_top_n: int = 20
    affinity_min_cooccurrence: int = 5

    # Add-on
    addon_top_n_anchors: int = 20
    addon_min_support: float = 0.01

    # Basket metrics
    basket_penetration_period: str = "W"
    ipt_cv_min_transactions: int = 3

    # Cohorts
    cohort_period: str = "W"
    ltv_horizon_periods: int = 12

    # Category / performance
    category_growth_threshold: float = 10.0  # % for scorecard RAG
    abc_thresholds: List[float] = field(default_factory=lambda: [0.7, 0.9])
    xyz_cv_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.25])
    lifecycle_growth_threshold: float = 25.0  # % for growth/decline

    # Promotional
    promo_price_drop_pct: float = 0.15
    promo_min_duration_days: int = 3
    promo_max_duration_days: int = 60
    promo_gap_days: int = 1
    baseline_stl_seasonal: int = 52
    promo_roi_n_resamples: int = 500

    # Uplift
    uplift_n_resamples: int = 100
    uplift_ci_level: float = 0.95
    uplift_max_pairs: int = 10

    # CLV
    clv_prediction_horizon_days: int = 90
    clv_observation_cutoff_days: int = 90
    clv_penalizer_escalation: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1.0, 2.0])

    # Transference
    transference_bootstrap_n_resamples: int = 100
    transference_max_pairs: int = 10

    # Assortment
    assortment_max_skus: int = 100
    assortment_min_coverage: float = 0.80
    assortment_time_limit_seconds: int = 60
    assortment_heuristic_iterations: int = 400

    # CDT
    cdt_n_clusters: int = 6
    cdt_min_cluster_size: int = 3
    cdt_max_depth: int = 4
    cdt_min_cooccurrence: int = 5

    # Segmentation
    segmentation_n_clusters: int = 6
    rfm_n_segments: int = 8
    clv_horizon_days: int = 90

    # Pricing
    elasticity_freq: str = "W"
    elasticity_min_periods: int = 10
    elasticity_min_price_variation: float = 0.05
    kvi_method: str = "heuristic"
    price_curve_n_tiers: int = 3
    price_curve_method: str = "kmeans"

    # Validation
    validation_min_obs: int = 10

    # Data Quality
    min_product_transactions: int = 50
    basket_outlier_percentile: float = 0.99
    min_viable_transactions: Dict[float, int] = field(default_factory=lambda: {
        200: 2000,
        1000: 5000,
        float('inf'): 10000,
    })

    # Output
    output_contract_check: bool = True
    output_detailed_diagnostics: bool = True

    # Quality gates: when True, data-quality issues raise DataQualityError
    # instead of being reported only (default False keeps current behavior)
    fail_on_quality_issues: bool = False


# Default singleton instance
DEFAULT_CONFIG = AnalyticsConfig()


def get_config() -> AnalyticsConfig:
    """Return the global default configuration."""
    return DEFAULT_CONFIG


from typing import Any
import copy


def update_config(**kwargs: Any) -> AnalyticsConfig:
    """Create a new config with overrides (immutable operation)."""
    # Create a deep copy to avoid mutating the global singleton
    new_config = copy.deepcopy(DEFAULT_CONFIG)
    
    for k, v in kwargs.items():
        if hasattr(new_config, k):
            setattr(new_config, k, v)
        else:
            raise KeyError(f"Unknown config key: {k}")
    
    return new_config


def set_global_config(config: AnalyticsConfig) -> None:
    """Set the global default configuration (use with caution)."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config