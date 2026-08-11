"""Cohort opportunity generation.

Cohorts with strong retention or high lifetime value become opportunities
for upselling, cross-selling, and loyalty programs.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.cohort import compute_cohorts, compute_cohort_sizes
from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_HIGH_RETENTION_THRESHOLD = 0.6   # 60% retention in period 1 is strong
_HIGH_LTV_THRESHOLD = 500.0       # Example threshold, adjust as needed


def generate_cohort_opportunities(
    df: pd.DataFrame,
    cohort_period: str = "M",
    top_n: int = 10,
) -> pd.DataFrame:
    """Build cohort-based opportunities.

    Args:
        df: Raw transaction DataFrame.
        cohort_period: Period for cohorting ('D', 'W', 'M', 'Q').
        top_n: Maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "cohort".
    """
    opportunities: list[Opportunity] = []

    # Compute cohort sizes and retention
    sizes_df = compute_cohort_sizes(df, cohort_period=cohort_period)
    retention_df = compute_cohorts(df, cohort_period=cohort_period)

    if sizes_df.empty or retention_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # Merge to get cohort size and retention in one df for the latest cohort
    latest_cohort = sizes_df["cohort"].max() if not sizes_df.empty else None
    if latest_cohort is None:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # Get the latest cohort's size and retention (period_index=1)
    cohort_size_row = sizes_df[sizes_df["cohort"] == latest_cohort]
    retention_row = retention_df[
        (retention_df["cohort"] == latest_cohort) & (retention_df["period_index"] == 1)
    ]

    if not cohort_size_row.empty and not retention_row.empty:
        cohort_size = cohort_size_row.iloc[0]["cohort_size"]
        retention_rate = retention_row.iloc[0]["retention_rate"]

        # Opportunity: High retention cohort -> upsell/cross-sell
        if retention_rate >= _HIGH_RETENTION_THRESHOLD:
            opportunities.append(
                Opportunity(
                    domain="cohort",
                    entity=str(latest_cohort),
                    title=f"High retention cohort: {latest_cohort} (retention {retention_rate:.1%})",
                    action="Upsell complementary products or launch a loyalty program.",
                    source="cohort_retention",
                    rationale=(
                        f"Cohort {latest_cohort} has {cohort_size:,.0f} customers with "
                        f"{retention_rate:.1%} retention in the following period."
                    ),
                    value=round(float(cohort_size * retention_rate), 0),
                    confidence="medium",
                )
            )

        # Opportunity: Large cohort size -> targeted acquisition
        if cohort_size >= sizes_df["cohort_size"].quantile(0.8):  # Top 20% by size
            opportunities.append(
                Opportunity(
                    domain="cohort",
                    entity=str(latest_cohort),
                    title=f"Large acquisition cohort: {latest_cohort} ({cohort_size:,.0f} new customers)",
                    action="Double down on acquisition channels that produced this cohort.",
                    source="cohort_size",
                    rationale=(
                        f"Cohort {latest_cohort} represents {cohort_size:,.0f} new customers, "
                        f"which is in the top 20% of cohort sizes."
                    ),
                    value=float(cohort_size),
                    confidence="medium",
                )
            )

    # If we need more opportunities, we can look at all cohorts for extreme values
    if len(opportunities) < top_n:
        remaining = top_n - len(opportunities)
        # Look for cohorts with high retention (any period)
        high_retention = retention_df[
            retention_df["retention_rate"] >= _HIGH_RETENTION_THRESHOLD
        ]
        # Exclude the latest cohort if we already used it
        used_entities = [opp.entity for opp in opportunities]
        high_retention = high_retention[~high_retention["cohort"].isin(used_entities)]
        if not high_retention.empty:
            high_retention = high_retention.sort_values("retention_rate", ascending=False).head(remaining)
            for _, row in high_retention.iterrows():
                cohort_size_row = sizes_df[sizes_df["cohort"] == row["cohort"]]
                if not cohort_size_row.empty:
                    cohort_size = cohort_size_row.iloc[0]["cohort_size"]
                    opportunities.append(
                        Opportunity(
                            domain="cohort",
                            entity=str(row["cohort"]),
                            title=f"High retention cohort: {row['cohort']} ({row['retention_rate']:.1%})",
                            action="Consider upsell or loyalty program to maximize value.",
                            source="cohort_retention",
                            rationale=(
                                f"Cohort {row['cohort']} has {cohort_size:,.0f} customers with "
                                f"{row['retention_rate']:.1%} retention."
                            ),
                            value=round(float(cohort_size * row["retention_rate"]), 0),
                            confidence="low",
                        )
                    )
                    remaining -= 1
                    if remaining <= 0:
                        break

    # Sort by value descending and trim to top_n
    opportunities = sorted(opportunities, key=lambda x: x.value, reverse=True)[:top_n]

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
