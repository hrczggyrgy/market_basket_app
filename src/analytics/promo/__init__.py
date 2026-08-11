"""Promotional Analytics: Causal Incrementality Engine.

Public API
----------
- build_promo_causal_panel: Build SKU-week panel for causal estimation
- estimate_twfe_promo_effect: Two-Way Fixed Effects promo effect
- estimate_event_study: Dynamic event study around promo start
- estimate_cross_sku_effects: Halo/cannibalization cross-SKU effects
- compute_causal_waterfall: Causal incrementality waterfall
- detect_promotions: Detect promo periods from price drops
- compute_promo_baseline: Compute promo baseline
- pre_post_promo_comparison: Pre/post promo comparison
- promo_roi_analysis: Promo ROI analysis
"""

from src.analytics import promo_core as _promo_module
from src.analytics.promo.causal import (
    build_promo_causal_panel,
    compute_causal_waterfall,
    estimate_cross_sku_effects,
    estimate_event_study,
    estimate_twfe_promo_effect,
)

# Re-export functions from the promo_core module
detect_promotions = _promo_module.detect_promotions
compute_promo_baseline = _promo_module.compute_promo_baseline
pre_post_promo_comparison = _promo_module.pre_post_promo_comparison
pre_post_promo_lift = _promo_module.pre_post_promo_comparison  # backwards-compatible alias
promo_roi_analysis = _promo_module.promo_roi_analysis
compute_incrementality_waterfall = _promo_module.compute_incrementality_waterfall
compute_cannibalization_analysis = _promo_module.compute_cannibalization_analysis
compute_category_cannibalization = _promo_module.compute_category_cannibalization
promotion_timing_analysis = _promo_module.promotion_timing_analysis
halo_effect_analysis = _promo_module.halo_effect_analysis
mark_promo_transactions = _promo_module.mark_promo_transactions
compute_category_promo_timeline = _promo_module.compute_category_promo_timeline
build_uplift_dataset = _promo_module.build_uplift_dataset
estimate_propensity_score = _promo_module.estimate_propensity_score
check_propensity_overlap = _promo_module.check_propensity_overlap
train_uplift_learner = _promo_module.train_uplift_learner
evaluate_uplift_model = _promo_module.evaluate_uplift_model
score_uplift_by_customer = _promo_module.score_uplift_by_customer

__all__ = [
    "build_promo_causal_panel",
    "estimate_twfe_promo_effect",
    "estimate_event_study",
    "estimate_cross_sku_effects",
    "compute_causal_waterfall",
    "detect_promotions",
    "compute_promo_baseline",
    "pre_post_promo_comparison",
    "pre_post_promo_lift",
    "promo_roi_analysis",
    "compute_incrementality_waterfall",
    "compute_cannibalization_analysis",
    "compute_category_cannibalization",
    "promotion_timing_analysis",
    "halo_effect_analysis",
    "mark_promo_transactions",
    "compute_category_promo_timeline",
    "build_uplift_dataset",
    "estimate_propensity_score",
    "check_propensity_overlap",
    "train_uplift_learner",
    "evaluate_uplift_model",
    "score_uplift_by_customer",
]
