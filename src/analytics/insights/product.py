"""Product-performance-domain insight generation.

Converts ABC/XYZ classes, lifecycle stage and SKU rationalization actions into
structured insights: delist candidates, volatile high-revenue SKUs, and
declining revenue that needs intervention.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check


def generate_product_insights(
    sku_rationalization_df: pd.DataFrame,
    xyz_df: pd.DataFrame | None = None,
    lifecycle_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Product/Performance insights.

    Args:
        sku_rationalization_df: SKU_RATIONALIZATION output (per-SKU action).
        xyz_df: optional XYZ_CLASSES output (demand volatility).
        lifecycle_df: optional LIFECYCLE output (growth stage).

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "product".
    """
    insights: list[Insight] = []

    if sku_rationalization_df is not None and not sku_rationalization_df.empty:
        work = sku_rationalization_df.copy()
        n_delist = int((work["action"] == "delist").sum())
        n_review = int((work["action"] == "review").sum())
        if n_delist > 0:
            grp = work[work["action"] == "delist"].sort_values("revenue", ascending=False)
            delist_rev = float(grp["revenue"].sum())
            entities = ", ".join(grp["stockcode"].astype(str).head(5).tolist())
            if n_delist > 5:
                entities += f" +{n_delist - 5} more"
            insights.append(
                Insight(
                    domain="product",
                    entity=entities,
                    kind="efficiency",
                    title=f"{n_delist} SKUs are delist candidates (€{delist_rev:,.0f} revenue)",
                    evidence=(
                        "Delist candidates carry low revenue, volatile demand or poor "
                        "repeat rates relative to the shelf space they occupy."
                    ),
                    action="Verify demand transference and substitute depth before removing any.",
                    confidence="medium",
                    impact_value=delist_rev,
                    sample_size=n_delist,
                )
            )
        if n_review > 0:
            insights.append(
                Insight(
                    domain="product",
                    entity="SKUs flagged for review",
                    kind="watch",
                    title=f"{n_review} SKUs are flagged for review",
                    evidence="Review flags include insufficient demand history, uncertain profiles or borderline metrics.",
                    action="Give them more observation time or a causal test before deciding.",
                    confidence="medium",
                    sample_size=n_review,
                )
            )

        n_actionable = n_delist + n_review
        n_total = int(len(work))
        if n_total > 0 and n_actionable / n_total < 0.2:
            insights.append(
                Insight(
                    domain="product",
                    entity="all SKUs",
                    kind="efficiency",
                    title=f"Assortment quality is healthy ({n_actionable}/{n_total} SKUs need action)",
                    evidence="Fewer than 20% of SKUs are flagged for delist or review.",
                    action="Keep monitoring; focus energy on the flagged subset.",
                    confidence="high" if n_total >= 30 else "medium",
                    sample_size=n_total,
                    stability=round(1.0 - n_actionable / n_total, 3),
                )
            )

    if xyz_df is not None and not xyz_df.empty:
        volatile = xyz_df[xyz_df["demand_profile"].isin({"Lumpy", "Intermittent", "Seasonal"})]
        if not volatile.empty:
            top = volatile.sort_values("revenue", ascending=False).head(5)
            top_rev = float(top["revenue"].sum())
            entities = ", ".join(top["stockcode"].astype(str).tolist())
            insights.append(
                Insight(
                    domain="product",
                    entity=entities,
                    kind="risk",
                    title=f"{len(volatile)} SKUs have volatile demand (€{top_rev:,.0f} in the top 5)",
                    evidence=(
                        "Lumpy/intermittent/seasonal demand profiles make stockouts or "
                        "overstocking likely without demand-aware ordering."
                    ),
                    action="Use demand-profile-aware forecasting and safety stock for these SKUs.",
                    confidence="medium",
                    impact_value=top_rev,
                    sample_size=int(len(volatile)),
                )
            )

    if lifecycle_df is not None and not lifecycle_df.empty:
        declining = lifecycle_df[lifecycle_df["stage"] == "decline"]
        if not declining.empty:
            decline_rev = float(declining["recent_revenue"].sum())
            entities = ", ".join(
                declining.sort_values("recent_revenue", ascending=False)["stockcode"]
                .astype(str)
                .head(5)
                .tolist()
            )
            insights.append(
                Insight(
                    domain="product",
                    entity=entities,
                    kind="risk",
                    title=f"{len(declining)} SKUs are in lifecycle decline (€{decline_rev:,.0f})",
                    evidence="Recent-period revenue is contracting vs the prior period for these SKUs.",
                    action="Decide per SKU: revive (promo/test), defend, or plan an orderly delist.",
                    confidence="medium",
                    impact_value=decline_rev,
                    sample_size=int(len(declining)),
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
