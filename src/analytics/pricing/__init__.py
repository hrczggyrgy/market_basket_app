"""Pricing, Elasticity & KVI Analytics.

Public API
----------
- elasticity: estimate_loglog_elasticity, estimate_hierarchical_elasticity,
  estimate_cross_price_elasticity
- kvi: compute_kvi_score
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
from src.analytics.pricing.elasticity import (
    classify_elasticity_confidence,
    estimate_cross_price_elasticity,
    estimate_hierarchical_elasticity,
    estimate_loglog_elasticity,
)
from src.analytics.pricing.kvi import compute_kvi_elasticity_quadrant, compute_kvi_score
from src.analytics.pricing.price_curves import (
    diagnose_price_curves_1d,
    diagnose_price_curves_multivariate,
)

__all__ = [
    # elasticity
    "classify_elasticity_confidence",
    "estimate_cross_price_elasticity",
    "estimate_hierarchical_elasticity",
    "estimate_loglog_elasticity",
    # kvi
    "compute_kvi_elasticity_quadrant",
    "compute_kvi_score",
    # price_curves
    "diagnose_price_curves_1d",
    "diagnose_price_curves_multivariate",
    # causal
    "causal_uplift_t_s",
    "iv_elasticity_manual_2sls",
    "local_price_response",
    "synthetic_control_estimate",
]