"""Cohort-domain insight generation.

Turns cohort analytics into structured insights:
cohort size trends, retention health, and value changes over time.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.cohort import compute_cohorts, compute_cohort_sizes
from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_DECLINING_COHORT_SIZE_THRESHOLD = -0.1  # 10% decline quarter over quarter
_LOW_RETENTION_THRESHOLD = 0.2          # 20% retention in period 1 is concerning


def generate_cohort_insights(df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Build Cohort insights from the raw transaction frame.

    Args:
        df: Raw transaction DataFrame with at least columns:
            customer_id, date, price, quantity.
        cohort_period: Period for cohorting ('D', 'W', 'M', 'Q').

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "cohort".
    """
    insights: list[Insight] = []

    # Compute cohort sizes over time
    sizes_df = compute_cohort_sizes(df, cohort_period=cohort_period)
    if sizes_df.empty:
        return check(insights_to_dataframe(insights), PRICING_INSIGHTS, allow_empty=True)

    # We need at least two cohorts to trend
    if len(sizes_df) >= 2:
        # Sort by cohort period
        sizes_df = sizes_df.sort_values("cohort")
        latest = sizes_df.iloc[-1]
        previous = sizes_df.iloc[-2]

        # Calculate quarter-over-quarter change in cohort size
        size_change = (
            (latest["cohort_size"] - previous["cohort_size"]) / previous["cohort_size"]
            if previous["cohort_size"] > 0 else 0.0
        )

        if size_change < _DECLINING_COHORT_SIZE_THRESHOLD:
            insights.append(
                Insight(
                    domain="cohort",
                    entity="new customers",
                    kind="risk",
                    title=f"New customer cohort size declining: {size_change:.1%} QoQ",
                    evidence=(
                        f"Cohort size dropped from {previous['cohort_size']:,.0f} to "
                        f"{latest['cohort_size']:,.0f} ({size_change:.1%} change)."
                    ),
                    action=(
                        "Investigate acquisition channels, promotions, and market conditions. "
                        "Consider win-back campaigns for previous cohorts."
                    ),
                    confidence="medium",
                    impact_value=float(size_change),
                    sample_size=int(latest["cohort_size"]),
                )
            )
        elif size_change > 0.1:  # Significant growth
            insights.append(
                Insight(
                    domain="cohort",
                    entity="new customers",
                    kind="growth",
                    title=f"New customer cohort size growing: {size_change:.1%} QoQ",
                    evidence=(
                        f"Cohort size increased from {previous['cohort_size']:,.0f} to "
                        f"{latest['cohort_size']:,.0f} ({size_change:.1%} change)."
                    ),
                    action=(
                        "Double down on successful acquisition channels. "
                        "Ensure onboarding and initial experience are optimized."
                    ),
                    confidence="medium",
                    impact_value=float(size_change),
                    sample_size=int(latest["cohort_size"]),
                )
            )

    # Compute cohort retention to check health
    retention_df = compute_cohorts(df, cohort_period=cohort_period)
    if not retention_df.empty:
        # Look at the first period retention (period_index=1) for the latest cohort
        latest_cohort = retention_df["cohort"].max()
        latest_retention = retention_df[
            (retention_df["cohort"] == latest_cohort) & (retention_df["period_index"] == 1)
        ]
        if not latest_retention.empty:
            retention_rate = latest_retention.iloc[0]["retention_rate"]
            if retention_rate < _LOW_RETENTION_THRESHOLD:
                insights.append(
                    Insight(
                        domain="cohort",
                        entity=str(latest_cohort),
                        kind="risk",
                        title=f"Low first-period retention: {retention_rate:.1%} for cohort {latest_cohort}",
                        evidence=(
                            f"Only {retention_rate:.1%} of customers from cohort {latest_cohort} "
                            f"made a repeat purchase in the following period."
                        ),
                        action=(
                            "Improve post-purchase experience, consider welcome offers, "
                            "or check for product/issues causing early churn."
                        ),
                        confidence="medium",
                        impact_value=float(retention_rate),
                        sample_size=int(latest_retention.iloc[0]["cohort_size"]),
                    )
                )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
