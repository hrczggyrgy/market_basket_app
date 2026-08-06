"""Pricing, Elasticity & KVI Analytics.

Public API
----------
- elasticity: estimate_loglog_elasticity, estimate_hierarchical_elasticity,
  estimate_cross_price_elasticity
- kvi: compute_kvi_score
- price_curves: diagnose_price_curves_1d, diagnose_price_curves_multivariate
- causal: estimate_iv_elasticity, estimate_rdd_elasticity,
  estimate_synthetic_control_elasticity, causal_uplift_t_s
"""

from src.analytics.pricing.causal import (
    causal_uplift_t_s,
    estimate_iv_elasticity,
    estimate_rdd_elasticity,
    estimate_synthetic_control_elasticity,
)
from src.analytics.pricing.elasticity import (
    estimate_cross_price_elasticity,
    estimate_hierarchical_elasticity,
    estimate_loglog_elasticity,
)
from src.analytics.pricing.kvi import compute_kvi_score
from src.analytics.pricing.price_curves import (
    diagnose_price_curves_1d,
    diagnose_price_curves_multivariate,
)

__all__ = [
    # elasticity
    "estimate_cross_price_elasticity",
    "estimate_hierarchical_elasticity",
    "estimate_loglog_elasticity",
    # kvi
    "compute_kvi_score",
    # price_curves
    "diagnose_price_curves_1d",
    "diagnose_price_curves_multivariate",
    # causal
    "causal_uplift_t_s",
    "estimate_iv_elasticity",
    "estimate_rdd_elasticity",
    "estimate_synthetic_control_elasticity",
]