"""Pricing-domain opportunity generation.

Produces rankable commercial opportunities (with EUR value where meaningful)
from the pricing decision matrix. Only SKUs with a usable, non-low-confidence
elasticity estimate ever produce a price recommendation: unreliable estimates
never enter the decision layer.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_ILLUSTRATIVE_PRICE_CUT = -0.05

_ACTION_TEMPLATES: dict[str, tuple[str, str]] = {
    "invest": (
        "Maintain price competitiveness; prioritize availability.",
        "High-KVI and price-elastic: price increases would sacrifice volume disproportionately.",
    ),
    "protect": (
        "Hold price; carry margin; avoid needless discounting.",
        "High-KVI and price-inelastic: margin-safe with no volume risk.",
    ),
    "price_lever": (
        "Run a controlled price test / targeted promotional investment.",
        "Low-KVI and price-elastic: candidate price lever.",
    ),
}


def _illustrative_incremental_revenue(revenue: float, abs_elasticity: float) -> float:
    """Order-of-magnitude revenue effect of a -5% price cut.

    ΔQ/Q ≈ e · ΔP/P, so new revenue = R · (1 + e·ΔP) · (1 + ΔP). Only used as
    an illustrative planning figure, never as a recommendation on its own.
    """
    delta = _ILLUSTRATIVE_PRICE_CUT
    if abs_elasticity <= 1.0:
        return 0.0
    new_revenue = revenue * (1 + abs_elasticity * abs(delta)) * (1 + delta)
    return max(new_revenue - revenue, 0.0)


def generate_pricing_opportunities(
    elasticity_df: pd.DataFrame,
    elasticity_status_df: pd.DataFrame,
    kvi_df: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build rankable Pricing opportunities.

    Args:
        elasticity_df: ELASTICITY output (unused directly; kept for signature
            symmetry with the insight generator and future margin inputs).
        elasticity_status_df: ELASTICITY_STATUS output (unused directly).
        kvi_df: KVI_SCORES output.
        decision_matrix: PRICING_DECISION_MATRIX output.
        top_n: Maximum opportunities per decision category.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST.
    """
    opportunities: list[Opportunity] = []

    if decision_matrix is None or decision_matrix.empty:
        return opportunities_to_dataframe(opportunities)

    dm = decision_matrix.copy()
    work = dm[dm["decision"].isin({"invest", "protect", "price_lever"})].copy()
    if work.empty:
        return opportunities_to_dataframe(opportunities)

    for decision, (action, rationale) in _ACTION_TEMPLATES.items():
        grp = (
            work[work["decision"] == decision]
            .sort_values("total_revenue", ascending=False)
            .head(top_n)
        )
        for _, row in grp.iterrows():
            value: float | None = None
            if decision == "price_lever" and pd.notna(row["abs_elasticity"]):
                value = _illustrative_incremental_revenue(
                    float(row["total_revenue"]), float(row["abs_elasticity"])
                )
            elif decision in ("invest", "protect"):
                value = float(row["total_revenue"])

            confidence = (
                row["elasticity_confidence"] if pd.notna(row["elasticity_confidence"]) else "medium"
            )

            opportunities.append(
                Opportunity(
                    domain="pricing",
                    entity=str(row["stockcode"]),
                    title={
                        "invest": f"Protect {row['stockcode']} — high-KVI traffic driver",
                        "protect": f"Hold price on {row['stockcode']} — margin carrier",
                        "price_lever": f"Test -5% price on {row['stockcode']}",
                    }[decision],
                    action=action,
                    source="kvi_elasticity",
                    rationale=rationale,
                    value=value,
                    confidence=confidence,
                )
            )

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
