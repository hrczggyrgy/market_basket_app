"""Pricing, Elasticity & KVI Analytics.

Public API
----------
- elasticity: estimate_loglog_elasticity, estimate_hierarchical_elasticity,
  estimate_cross_price_elasticity, compute_elasticity_status
- kvi: compute_kvi_score
- decision: compute_pricing_decision_matrix
- pipeline: run_pricing_analysis
- price_curves: diagnose_price_curves_1d, diagnose_price_curves_multivariate
- causal: local_price_response, iv_elasticity_manual_2sls,
  synthetic_control_estimate, causal_uplift_t_s
"""

from src.analytics.pricing.causal import (
    causal_uplift_t_s,
    iv_elasticity_manual_2sls,
    local_price_response,
    synthetic_control_estimate,
)
from src.analytics.pricing.decision import compute_pricing_decision_matrix
from src.analytics.pricing.elasticity import (
    classify_elasticity_confidence,
    compute_elasticity_status,
    estimate_cross_price_elasticity,
    estimate_hierarchical_elasticity,
    estimate_loglog_elasticity,
)
from src.analytics.pricing.kvi import compute_kvi_elasticity_quadrant, compute_kvi_score
from src.analytics.pricing.pipeline import PricingAnalysis, run_pricing_analysis
from src.analytics.pricing.price_curves import (
    diagnose_price_curves_1d,
    diagnose_price_curves_multivariate,
)

__all__ = [
    # elasticity
    "classify_elasticity_confidence",
    "compute_elasticity_status",
    "estimate_cross_price_elasticity",
    "estimate_hierarchical_elasticity",
    "estimate_loglog_elasticity",
    # kvi
    "compute_kvi_elasticity_quadrant",
    "compute_kvi_score",
    # decision
    "compute_pricing_decision_matrix",
    # pipeline
    "PricingAnalysis",
    "run_pricing_analysis",
    # price_curves
    "diagnose_price_curves_1d",
    "diagnose_price_curves_multivariate",
    # causal
    "causal_uplift_t_s",
    "iv_elasticity_manual_2sls",
    "local_price_response",
    "synthetic_control_estimate",
]
