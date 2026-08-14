"""Retention opportunity generation.

High-value customers at risk of churn become rankable win-back opportunities,
with the predicted CLV as the order-of-magnitude value at stake.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_P_ALIVE_RISK = 0.35


def generate_retention_opportunities(
    clv_customer_df: pd.DataFrame,
    top_n: int = 10,
    p_alive_risk: float = _P_ALIVE_RISK,
) -> pd.DataFrame:
    """Build rankable retention opportunities.

    Args:
        clv_customer_df: CLV_CUSTOMER output (predicted CLV, p_alive).
        top_n: maximum number of opportunities.
        p_alive_risk: p(alive) threshold below which a customer is at risk.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "retention".
    """
    opportunities: list[Opportunity] = []

    if clv_customer_df is None or clv_customer_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    work = clv_customer_df.copy()
    work["predicted_clv"] = pd.to_numeric(work["predicted_clv"], errors="coerce")
    work["p_alive"] = pd.to_numeric(work["p_alive"], errors="coerce")

    at_risk = work[
        work["p_alive"].notna() & (work["predicted_clv"].notna()) & (work["p_alive"] < p_alive_risk)
    ]
    at_risk = at_risk.sort_values("predicted_clv", ascending=False).head(top_n)

    for _, row in at_risk.iterrows():
        clv = float(row["predicted_clv"])
        p_alive = float(row["p_alive"])
        # Value at stake: CLV weighted by churn probability, conservative 50%.
        value = round(clv * (1.0 - p_alive) * 0.5, 0)
        opportunities.append(
            Opportunity(
                domain="retention",
                entity=str(row["customer_id"]),
                title=f"Win back {row['customer_id']} (p(alive) {p_alive:.0%})",
                action="Targeted win-back: personalized offer or contact within the week.",
                source="bg_nbd_clv",
                rationale=(
                    f"Predicted CLV €{clv:,.0f} with p(alive) {p_alive:.0%}; "
                    f"frequency {int(row.get('frequency', 0))}, "
                    f"recency {int(row.get('recency_days', 0))} days ago."
                ),
                value=value,
                confidence="medium",
            )
        )

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
