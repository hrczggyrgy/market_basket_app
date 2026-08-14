"""Switching opportunity generation.

Products whose demand is recoverable within the assortment (high SDP) become
delist candidates; products with unique demand (low SDP) become availability
commitments with protected revenue.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_SDP_SUBSTITUTABLE = 0.8
_SDP_UNIQUE = 0.2
_MIN_REVENUE = 100.0


def generate_switching_opportunities(
    sdp_df: pd.DataFrame,
    delist_impact_df: pd.DataFrame | None = None,
    revenue_by_product: pd.Series | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build rankable switching / assortment opportunities.

    Args:
        sdp_df: SDP_SCORES output (substitutable demand %).
        delist_impact_df: optional DELIST_IMPACT output (net delist impact).
        revenue_by_product: optional per-product revenue Series.
        top_n: maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "switching".
    """
    opportunities: list[Opportunity] = []

    if sdp_df is None or sdp_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    work = sdp_df.copy()
    if revenue_by_product is not None and not revenue_by_product.empty:
        work = work.merge(
            revenue_by_product.rename("revenue"), left_on="stockcode", right_index=True, how="left"
        )
        work["revenue"] = work["revenue"].fillna(0.0)
    else:
        work["revenue"] = 0.0

    substitutable = (
        work[(work["sdp"] >= _SDP_SUBSTITUTABLE) & (work["revenue"] >= _MIN_REVENUE)]
        .sort_values(["sdp", "revenue"], ascending=[False, False])
        .head(top_n)
    )

    delist_net: dict[str, float] = {}
    if delist_impact_df is not None and not delist_impact_df.empty:
        delist_net = dict(
            zip(delist_impact_df["stockcode"], delist_impact_df["net_revenue_impact"], strict=True)
        )

    for _, row in substitutable.iterrows():
        sku = str(row["stockcode"])
        net = delist_net.get(sku)
        value = round(float(row["revenue"]) * float(row["sdp"]), 0)
        if net is not None and net > 0:
            rationale = (
                f"SDP {float(row['sdp']):.0%}; simulated delist is revenue-positive "
                f"(net €{net:,.0f})."
            )
        else:
            rationale = (
                f"SDP {float(row['sdp']):.0%} means most demand is recoverable in-assortment."
            )
        opportunities.append(
            Opportunity(
                domain="switching",
                entity=sku,
                title=f"Delist candidate: {sku} (demand recoverable)",
                action="Validate causally, then delist and reallocate shelf space.",
                source="substitutable_demand",
                rationale=rationale,
                value=value,
                confidence="low",
            )
        )

    if len(opportunities) < top_n:
        unique = work[work["sdp"] < _SDP_UNIQUE].sort_values("revenue", ascending=False)
        for _, row in unique.head(top_n - len(opportunities)).iterrows():
            sku = str(row["stockcode"])
            if row["revenue"] <= 0:
                continue
            opportunities.append(
                Opportunity(
                    domain="switching",
                    entity=sku,
                    title=f"Availability commitment: {sku} (non-substitutable)",
                    action="Never delist; protect stock and display.",
                    source="substitutable_demand",
                    rationale=f"SDP {float(row['sdp']):.0%} — its demand would leak out of the assortment.",
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
