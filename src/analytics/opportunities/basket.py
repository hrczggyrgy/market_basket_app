"""Basket opportunity generation.

Products with extreme basket penetration become opportunities for
assortment adjustments, bundling, or promotional strategies.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.basket_metrics import compute_basket_penetration
from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_HIGH_PENETRATION_THRESHOLD = 0.5   # 50% of baskets
_LOW_PENETRATION_THRESHOLD = 0.05   # 5% of baskets
_MIN_REVENUE_SHARE = 0.01           # 1% of revenue


def generate_basket_opportunities(
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build basket-based opportunities.

    Args:
        df: Raw transaction DataFrame.
        top_n: Maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "basket".
    """
    opportunities: list[Opportunity] = []

    pen_df = compute_basket_penetration(df)
    if pen_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # We'll consider two types of opportunities:
    # 1. High penetration, high revenue: leverage as traffic driver
    # 2. Low penetration, high revenue: increase penetration via promotion/placement
    # 3. Low penetration, low revenue: consider delisting or bundling

    # Filter to products with at least MIN_REVENUE_SHARE to avoid noise
    pen_df = pen_df[pen_df["revenue_share"] >= _MIN_REVENUE_SHARE]

    # Opportunity type 1: High penetration and high revenue -> traffic driver
    high_pen_high_rev = pen_df[
        (pen_df["penetration"] >= _HIGH_PENETRATION_THRESHOLD) &
        (pen_df["revenue_share"] >= pen_df["revenue_share"].quantile(0.5))
    ]
    if not high_pen_high_rev.empty:
        high_pen_high_rev = high_pen_high_rev.sort_values(
            ["penetration", "revenue_share"], ascending=[False, False]
        ).head(top_n // 2)
        for _, row in high_pen_high_rev.iterrows():
            opportunities.append(
                Opportunity(
                    domain="basket",
                    entity=str(row["stockcode"]),
                    title=f"Traffic driver: {row['stockcode']} (high penetration & revenue)",
                    action="Ensure availability, consider promotional pricing to increase basket size.",
                    source="basket_penetration",
                    rationale=(
                        f"Appears in {row['basket_count']} baskets "
                        f"({row['penetration']:.1%} penetration) with revenue share {row['revenue_share']:.1%}."
                    ),
                    value=round(float(row["revenue_share"] * 100), 1),  # as percentage
                    confidence="medium",
                )
            )

    # Opportunity type 2: Low penetration but high revenue -> increase penetration
    low_pen_high_rev = pen_df[
        (pen_df["penetration"] < _LOW_PENETRATION_THRESHOLD) &
        (pen_df["revenue_share"] >= pen_df["revenue_share"].quantile(0.5))
    ]
    if not low_pen_high_rev.empty:
        low_pen_high_rev = low_pen_high_rev.sort_values(
            ["penetration", "revenue_share"], ascending=[True, False]
        ).head(top_n // 2)
        for _, row in low_pen_high_rev.iterrows():
            opportunities.append(
                Opportunity(
                    domain="basket",
                    entity=str(row["stockcode"]),
                    title=f"Increase penetration: {row['stockcode']} (low penetration, high revenue)",
                    action="Improve placement, run promotions, or bundle with popular items.",
                    source="basket_penetration",
                    rationale=(
                        f"Only {row['basket_count']} baskets contain this product "
                        f"({row['penetration']:.1%} penetration) but revenue share is {row['revenue_share']:.1%}."
                    ),
                    value=round(float(row["revenue_share"] * 100), 1),
                    confidence="medium",
                )
            )

    # If we still need more opportunities, we can add products with extreme values
    if len(opportunities) < top_n:
        remaining = top_n - len(opportunities)
        # Take the top by penetration (high) and bottom by penetration (low) from what's left
        high_pen = pen_df[pen_df["penetration"] >= _HIGH_PENETRATION_THRESHOLD]
        low_pen = pen_df[pen_df["penetration"] < _LOW_PENETRATION_THRESHOLD]
        # Exclude those we already used
        used_entities = [opp.entity for opp in opportunities]
        high_pen = high_pen[~high_pen["stockcode"].isin(used_entities)]
        low_pen = low_pen[~low_pen["stockcode"].isin(used_entities)]

        # Take from high_pen first
        take_from_high = min(remaining // 2, len(high_pen))
        if take_from_high > 0:
            high_pen = high_pen.sort_values("penetration", ascending=False).head(take_from_high)
            for _, row in high_pen.iterrows():
                opportunities.append(
                    Opportunity(
                        domain="basket",
                        entity=str(row["stockcode"]),
                        title=f"High penetration product: {row['stockcode']}",
                        action="Leverage as a traffic driver or bundle to increase sales.",
                        source="basket_penetration",
                        rationale=(
                            f"Appears in {row['basket_count']} baskets "
                            f"({row['penetration']:.1%} penetration) with revenue share {row['revenue_share']:.1%}."
                        ),
                        value=round(float(row["revenue_share"] * 100), 1),
                        confidence="low",
                    )
                )
                remaining -= 1

        # Then take from low_pen
        if remaining > 0 and len(low_pen) > 0:
            take_from_low = min(remaining, len(low_pen))
            low_pen = low_pen.sort_values("penetration", ascending=True).head(take_from_low)
            for _, row in low_pen.iterrows():
                opportunities.append(
                    Opportunity(
                        domain="basket",
                        entity=str(row["stockcode"]),
                        title=f"Low penetration product: {row['stockcode']}",
                        action="Increase visibility or consider delisting if persistent.",
                        source="basket_penetration",
                        rationale=(
                            f"Only {row['basket_count']} baskets contain this product "
                            f"({row['penetration']:.1%} penetration) with revenue share {row['revenue_share']:.1%}."
                        ),
                        value=round(float(row["revenue_share"] * 100), 1),
                        confidence="low",
                    )
                )
                remaining -= 1

    # Sort by value (revenue share) descending and trim to top_n
    opportunities = sorted(opportunities, key=lambda x: x.value, reverse=True)[:top_n]

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
