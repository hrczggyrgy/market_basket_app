"""Customer Segmentation package.

Public API
----------
- rfm: compute_rfm_features, rfm_segmentation
- behavioral: _create_behavioral_features_pandas, behavioral_segmentation
- survival: survival_analysis, kaplan_meier_estimates
- value_based: value_based_segmentation
- core: MIN_CLUSTER_SIZE, _label_rfm_clusters, _label_behavioral_clusters,
   compute_cluster_quality_metrics, compute_cluster_stability,
   format_quality_metrics, format_stability_metrics
- strategic: calculate_segment_value_metrics, calculate_segment_engagement_metrics,
   calculate_segment_retention_metrics, calculate_segment_basket_metrics,
   calculate_segment_price_behavior_metrics, calculate_segment_growth_metrics,
   calculate_segment_concentration_metrics, calculate_segment_stability_score,
   calculate_segment_distinctiveness
"""

from src.analytics.segmentation.behavioral import (
    _create_behavioral_features_pandas,
    behavioral_segmentation,
    compute_segment_migration,
    compute_segment_radar,
)
from src.analytics.segmentation.core import (
    MIN_CLUSTER_SIZE,
    _label_behavioral_clusters,
    _label_rfm_clusters,
    calculate_segment_basket_metrics,
    calculate_segment_concentration_metrics,
    calculate_segment_distinctiveness,
    calculate_segment_engagement_metrics,
    calculate_segment_growth_metrics,
    calculate_segment_price_behavior_metrics,
    calculate_segment_retention_metrics,
    calculate_segment_stability_score,
    calculate_segment_value_metrics,
    compute_cluster_quality_metrics,
    compute_cluster_stability,
    format_quality_metrics,
    format_stability_metrics,
)
from src.analytics.segmentation.rfm import compute_rfm_features, rfm_segmentation
from src.analytics.segmentation.survival import kaplan_meier_estimates, survival_analysis
from src.analytics.segmentation.value_based import value_based_segmentation

__all__ = [
    # rfm
    "compute_rfm_features",
    "rfm_segmentation",
    # behavioral
    "_create_behavioral_features_pandas",
    "behavioral_segmentation",
    "compute_segment_radar",
    "compute_segment_migration",
    # survival
    "kaplan_meier_estimates",
    "survival_analysis",
    # value_based
    "value_based_segmentation",
    # core
    "MIN_CLUSTER_SIZE",
    "_label_behavioral_clusters",
    "_label_rfm_clusters",
    "compute_cluster_quality_metrics",
    "compute_cluster_stability",
    "format_quality_metrics",
    "format_stability_metrics",
    # strategic
    "calculate_segment_value_metrics",
    "calculate_segment_engagement_metrics",
    "calculate_segment_retention_metrics",
    "calculate_segment_basket_metrics",
    "calculate_segment_price_behavior_metrics",
    "calculate_segment_growth_metrics",
    "calculate_segment_concentration_metrics",
    "calculate_segment_stability_score",
    "calculate_segment_distinctiveness",
]
