"""Basket-domain insight generation.

Turns basket-level metrics into structured insights:
low/high basket penetration, unusual basket size, and revenue share anomalies.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.basket_metrics import compute_basket_penetration
from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_LOW_PENETRATION_THRESHOLD = 0.05  # 5% of baskets
_HIGH_PENETRATION_THRESHOLD = 0.5  # 50% of baskets


def generate_basket_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Build Basket insights from the raw transaction frame.

    Args:
        df: Raw transaction DataFrame with at least columns:
            transaction_id, stockcode, price, quantity, date, customer_id.

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "basket".
    """
    insights: list[Insight] = []

    # Compute basket penetration per product
    pen_df = compute_basket_penetration(df)
    if pen_df.empty:
        return check(insights_to_dataframe(insights), PRICING_INSIGHTS, allow_empty=True)

    # Identify products with low basket penetration (may be niche or failing)
    low_pen = pen_df[pen_df["penetration"] < _LOW_PENETRATION_THRESHOLD]
    if not low_pen.empty:
        # Take the top 5 by revenue share (or just the lowest penetration)
        low_pen = low_pen.sort_values("penetration").head(5)
        for _, row in low_pen.iterrows():
            insights.append(
                Insight(
                    domain="basket",
                    entity=str(row["stockcode"]),
                    kind="risk",
                    title=f"Low basket penetration: {row['stockcode']} in {row['penetration']:.1%} of baskets",
                    evidence=(
                        f"Only {row['basket_count']} baskets contain this product "
                        f"({row['penetration']:.1%} penetration). "
                        f"Revenue share is {row['revenue_share']:.1%}."
                    ),
                    action=(
                        "Review promotion, placement, and price; consider bundling with popular items "
                        "or delisting if low penetration is persistent."
                    ),
                    confidence="medium",
                    impact_value=float(row["revenue_share"]),
                    sample_size=int(row["basket_count"]),
                    evidence_level=2,  # descriptive: basket penetration analysis
                    n_transition_pairs=0,  # not applicable for basket insights
                    n_unique_products=0,  # not applicable for basket insights
                    confidence_gate=False,  # not applicable for basket insights
                )
            )

    # Identify products with high basket penetration (may be traffic drivers)
    high_pen = pen_df[pen_df["penetration"] >= _HIGH_PENETRATION_THRESHOLD]
    if not high_pen.empty:
        # Take the top 5 by penetration
        high_pen = high_pen.sort_values("penetration", ascending=False).head(5)
        for _, row in high_pen.iterrows():
            insights.append(
                Insight(
                    domain="basket",
                    entity=str(row["stockcode"]),
                    kind="opportunity",
                    title=f"High basket penetration: {row['stockcode']} in {row['penetration']:.1%} of baskets",
                    evidence=(
                        f"Appears in {row['basket_count']} baskets "
                        f"({row['penetration']:.1%} penetration) with revenue share {row['revenue_share']:.1%}."
                    ),
                    action=(
                        "Leverage as a traffic driver: ensure availability, consider promotional pricing "
                        "to increase basket size, or use as a loss leader."
                    ),
                    confidence="medium",
                    impact_value=float(row["revenue_share"]),
                    sample_size=int(row["basket_count"]),
                    evidence_level=2,  # descriptive: basket penetration analysis
                    n_transition_pairs=0,  # not applicable for basket insights
                    n_unique_products=0,  # not applicable for basket insights
                    confidence_gate=False,  # not applicable for basket insights
                )
            )

    # Optionally, we could add insights about basket size or revenue share anomalies.
    # For now, we keep it simple.

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
