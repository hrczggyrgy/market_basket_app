"""Product opportunity generation.

Turns product rationalization and performance data into structured opportunities:
pricing, promotion, assortment, and lifecycle actions.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check


def generate_product_opportunities(
    rational: pd.DataFrame,
    xyz: pd.DataFrame | None = None,
    lifecycle: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build product-based opportunities.

    Args:
        rational: SKU rationalization DataFrame (from compute_sku_rationalization_df).
        xyz: Optional XYZ analysis DataFrame.
        lifecycle: Optional lifecycle stage DataFrame.
        top_n: Maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "product".
    """
    opportunities: list[Opportunity] = []

    if rational is None or rational.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # Opportunity 1: Delist candidates (C + Z)
    delist_candidates = rational[rational["action"] == "delist_candidate"]
    if not delist_candidates.empty:
        # Take the top by revenue (if available) or just the first top_n
        if "revenue" in delist_candidates.columns:
            delist_candidates = delist_candidates.sort_values("revenue", ascending=False)
        delist_candidates = delist_candidates.head(top_n // 2)
        for _, row in delist_candidates.iterrows():
            opportunities.append(
                Opportunity(
                    domain="product",
                    entity=str(row["stockcode"]),
                    title=f"Delist candidate: {row['stockcode']}",
                    action="Delist and reallocate shelf space to higher-performing items.",
                    source="rationalization",
                    rationale=(
                        f"Classified as {row.get('abc_class', '?')}{row.get('xyz_class', '?')} "
                        f"with lifecycle stage {row.get('stage', '?')}."
                    ),
                    value=round(float(row.get("revenue", 0)), 0),
                    confidence="medium",
                )
            )

    # Opportunity 2: High velocity + high repeat (sticky fast-movers)
    velocity_col = "velocity" in rational.columns
    repeat_col = "repeat_rate" in rational.columns
    if velocity_col and repeat_col:
        sticky = rational[
            (rational["velocity"] > rational["velocity"].median())
            & (rational["repeat_rate"] > rational["repeat_rate"].median())
        ]
        if not sticky.empty:
            sticky = sticky.sort_values(["velocity", "repeat_rate"], ascending=[False, False]).head(
                top_n // 2
            )
            for _, row in sticky.iterrows():
                opportunities.append(
                    Opportunity(
                        domain="product",
                        entity=str(row["stockcode"]),
                        title=f"Sticky fast-mover: {row['stockcode']}",
                        action="Increase facings, consider promotional pricing to accelerate volume.",
                        source="velocity_repeat",
                        rationale=(
                            f"Velocity {row.get('velocity', 0):.1f} units/day, "
                            f"repeat rate {row.get('repeat_rate', 0):.1%}."
                        ),
                        value=round(float(row.get("revenue", 0)) * 0.1, 0),
                        confidence="medium",
                    )
                )

    # Opportunity 3: Low velocity + low repeat (slow movers)
    velocity_col = "velocity" in rational.columns
    repeat_col = "repeat_rate" in rational.columns
    if velocity_col and repeat_col:
        slow = rational[
            (rational["velocity"] <= rational["velocity"].median())
            & (rational["repeat_rate"] <= rational["repeat_rate"].median())
        ]
        if not slow.empty:
            slow = slow.sort_values(["velocity", "repeat_rate"], ascending=[True, True]).head(
                top_n // 2
            )
            for _, row in slow.iterrows():
                opportunities.append(
                    Opportunity(
                        domain="product",
                        entity=str(row["stockcode"]),
                        title=f"Slow mover: {row['stockcode']}",
                        action="Reduce inventory, consider promotion to clear, or delist if persistent.",
                        source="velocity_repeat",
                        rationale=(
                            f"Velocity {row.get('velocity', 0):.1f} units/day, "
                            f"repeat rate {row.get('repeat_rate', 0):.1%}."
                        ),
                        value=round(float(row.get("revenue", 0)) * 0.05, 0),
                        confidence="medium",
                    )
                )

    # If we need more opportunities, we can add from rationalization actions
    if len(opportunities) < top_n:
        remaining = top_n - len(opportunities)
        # Take from rationalization actions that we haven't used
        used_entities = [opp.entity for opp in opportunities]
        remaining_rational = rational[~rational["stockcode"].isin(used_entities)]
        if not remaining_rational.empty:
            # Prioritize by revenue if available
            if "revenue" in remaining_rational.columns:
                remaining_rational = remaining_rational.sort_values("revenue", ascending=False)
            remaining_rational = remaining_rational.head(remaining)
            for _, row in remaining_rational.iterrows():
                opportunities.append(
                    Opportunity(
                        domain="product",
                        entity=str(row["stockcode"]),
                        title=f"Action: {row['action']} for {row['stockcode']}",
                        action=row.get("action_recommendation", "Review"),
                        source="rationalization",
                        rationale=(
                            f"Classified as {row.get('abc_class', '?')}{row.get('xyz_class', '?')} "
                            f"with lifecycle stage {row.get('stage', '?')}."
                        ),
                        value=round(float(row.get("revenue", 0)) * 0.01, 0),
                        confidence="low",
                    )
                )

    # Sort by value descending and trim to top_n
    opportunities = sorted(opportunities, key=lambda x: x.value, reverse=True)[:top_n]

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
