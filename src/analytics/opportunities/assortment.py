"""Assortment opportunity generation.

Scenario-derived opportunities: the revenue upside of the best slimmer
assortment, and per-SKU keep/cut recommendations from the solution table.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check


def generate_assortment_opportunities(
    scenario_df: pd.DataFrame,
    solution_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build rankable assortment opportunities.

    Args:
        scenario_df: ASSORTMENT_SCENARIO output (method comparison).
        solution_df: optional ASSORTMENT_SOLUTION output (kept SKUs).
        top_n: maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "assortment".
    """
    opportunities: list[Opportunity] = []

    if scenario_df is None or scenario_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    best = scenario_df.sort_values("expected_revenue", ascending=False).iloc[0]
    coverage = float(best["coverage"])
    recovered = float(best["recovered_revenue"])
    lost = float(best["lost_revenue"])

    if coverage < 0.99:
        opportunities.append(
            Opportunity(
                domain="assortment",
                entity="assortment scenarios",
                title=f"Best scenario keeps {coverage:.0%} of revenue after slimming",
                action="Validate with a market test on the long tail before committing.",
                source="scenario_simulation",
                rationale=(
                    f"Recovered €{recovered:,.0f} of €{lost:,.0f} lost revenue via "
                    f"switching recovery (recovery rate {float(best['recovery_rate']):.0%})."
                ),
                value=round(recovered, 0),
                confidence="low",
            )
        )
    else:
        opportunities.append(
            Opportunity(
                domain="assortment",
                entity="assortment scenarios",
                title="Current assortment is close to optimal on coverage",
                action="No cut required; focus on the long tail one SKU at a time.",
                source="scenario_simulation",
                rationale=f"Best scenario achieves {coverage:.0%} revenue coverage.",
                value=0.0,
                confidence="medium",
            )
        )

    if solution_df is not None and not solution_df.empty:
        work = solution_df.copy()
        if "selected" in work.columns:
            work = work[work["selected"] == 1]
        work = work.sort_values("revenue", ascending=False).head(top_n)
        for _, row in work.iterrows():
            opportunities.append(
                Opportunity(
                    domain="assortment",
                    entity=str(row["stockcode"]),
                    title=f"Keep {row['stockcode']} in the optimized assortment",
                    action="Confirm availability; it is core to the coverage target.",
                    source="assortment_solution",
                    rationale=f"Rank #{int(row['rank'])}, revenue €{float(row['revenue']):,.0f}.",
                    value=round(float(row["revenue"]), 0),
                    confidence="medium",
                )
            )

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
