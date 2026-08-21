"""Pricing analysis pipeline.

Runs elasticity estimation, coverage status, confidence, KVI, the decision
matrix and the insight/opportunity engines in one pass so the Pricing tab and
the Decision Center share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analytics.insights import generate_pricing_insights
from src.analytics.opportunities import generate_pricing_opportunities
from src.analytics.pricing.decision import compute_pricing_decision_matrix
from src.analytics.pricing.elasticity import (
    classify_elasticity_confidence,
    compute_elasticity_status,
    estimate_loglog_elasticity,
    estimate_loglog_elasticity_from_weekly,
)
from src.analytics.pricing.kvi import compute_kvi_score


@dataclass
class PricingAnalysis:
    """Complete pricing analysis for a transaction frame."""

    elasticity: pd.DataFrame
    elasticity_status: pd.DataFrame
    confidence: pd.DataFrame
    kvi: pd.DataFrame
    decision_matrix: pd.DataFrame
    insights: pd.DataFrame
    opportunities: pd.DataFrame


def run_pricing_analysis(
    transactions_df: pd.DataFrame | None = None,
    weekly_panel: pd.DataFrame | None = None,
    min_periods: int = 5,
    min_price_variation: float = 0.05,
    kvi_method: str = "heuristic",
) -> PricingAnalysis:
    """Run the full pricing pipeline and return every stage's output.

    Can accept either:
    - transactions_df: Raw transaction data (will compute weekly panel internally)
    - weekly_panel: Pre-computed weekly product panel from FeatureStore (preferred for performance)
    """
    if weekly_panel is not None:
        # Use pre-computed weekly panel from FeatureStore
        elasticity = estimate_loglog_elasticity_from_weekly(
            weekly_panel,
            min_periods=min_periods,
            min_price_variation=min_price_variation,
        )
    elif transactions_df is not None:
        # Fall back to computing from raw transactions
        elasticity = estimate_loglog_elasticity(
            transactions_df,
            min_periods=min_periods,
            min_price_variation=min_price_variation,
        )
    else:
        raise ValueError("Either transactions_df or weekly_panel must be provided")

    elasticity_status = compute_elasticity_status(
        transactions_df if transactions_df is not None else pd.DataFrame(),
        elasticity_df=elasticity,
        min_periods=min_periods,
        min_price_variation=min_price_variation,
    )
    confidence = classify_elasticity_confidence(elasticity)

    # For KVI, we need either transactions or weekly panel
    if weekly_panel is not None:
        kvi = compute_kvi_score(
            weekly_panel=weekly_panel,
            elasticity_df=elasticity,
            method=kvi_method,
            elasticity_status_df=elasticity_status,
        )
    else:
        kvi = compute_kvi_score(
            transactions_df=transactions_df,
            elasticity_df=elasticity,
            method=kvi_method,
            elasticity_status_df=elasticity_status,
        )

    decision_matrix = compute_pricing_decision_matrix(kvi, elasticity)
    insights = generate_pricing_insights(elasticity, elasticity_status, kvi, decision_matrix)
    opportunities = generate_pricing_opportunities(
        elasticity, elasticity_status, kvi, decision_matrix
    )

    return PricingAnalysis(
        elasticity=elasticity,
        elasticity_status=elasticity_status,
        confidence=confidence,
        kvi=kvi,
        decision_matrix=decision_matrix,
        insights=insights,
        opportunities=opportunities,
    )
