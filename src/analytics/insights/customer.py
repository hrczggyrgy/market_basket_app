"""Customer-domain (CLV) insight generation.

High-value-at-risk detection, CLV concentration and model-coverage signals
turned into structured insights.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_P_ALIVE_RISK = 0.35  # below: customer likely churned
_CLV_CONC_RISK = 0.5  # top-decile CLV share at/above this flags concentration


def generate_customer_insights(
    clv_customer_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build Customer (CLV) insights.

    Args:
        clv_customer_df: CLV_CUSTOMER output (per-customer CLV + p_alive).
        predictions_df: optional CLV_PREDICTIONS output (model diagnostics).
        top_n: how many at-risk customers to name.

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "customer".
    """
    insights: list[Insight] = []

    if clv_customer_df is None or clv_customer_df.empty:
        return check(insights_to_dataframe(insights), PRICING_INSIGHTS, allow_empty=True)

    work = clv_customer_df.copy()
    work["predicted_clv"] = pd.to_numeric(work["predicted_clv"], errors="coerce").fillna(0.0)
    work["p_alive"] = pd.to_numeric(work["p_alive"], errors="coerce")

    # --- High-value at-risk -------------------------------------------------
    at_risk = work[work["p_alive"].notna() & (work["p_alive"] < _P_ALIVE_RISK)]
    if not at_risk.empty:
        ranked = at_risk.sort_values("predicted_clv", ascending=False).head(top_n)
        clv_at_risk = float(ranked["predicted_clv"].sum())
        entities = ", ".join(ranked["customer_id"].astype(str).tolist())
        insights.append(
            Insight(
                domain="customer",
                entity=entities,
                kind="opportunity",
                title=f"{len(at_risk)} high-value customers are at risk of churn",
                evidence=(
                    f"Top {len(ranked)} by predicted CLV sum to €{clv_at_risk:,.0f} "
                    f"with p(alive) < {_P_ALIVE_RISK:.0%}. Predicted CLV is a model "
                    f"estimate, not a guarantee."
                ),
                action="Run a targeted win-back for these customers (offer, contact, or promo).",
                confidence="medium",
                impact_value=clv_at_risk,
                sample_size=int(len(at_risk)),
            )
        )

    # --- CLV concentration --------------------------------------------------
    total_clv = float(work["predicted_clv"].sum())
    if total_clv > 0:
        share_top = float(work["predicted_clv"].nlargest(max(1, len(work) // 10)).sum() / total_clv)
        if share_top >= _CLV_CONC_RISK:
            insights.append(
                Insight(
                    domain="customer",
                    entity="top-decile customers",
                    kind="risk",
                    title=f"Top 10% of customers hold {share_top:.0%} of predicted CLV",
                    evidence=(
                        "High CLV concentration means the base is fragile: losing a "
                        "handful of customers materially dents future revenue."
                    ),
                    action="Hard-wire retention for the top decile; diversify acquisition into the next tier.",
                    confidence="high",
                    stability=round(share_top, 3),
                )
            )
        else:
            insights.append(
                Insight(
                    domain="customer",
                    entity="all customers",
                    kind="efficiency",
                    title=f"CLV is spread across the base (top decile = {share_top:.0%})",
                    evidence="Lower concentration means the customer base is resilient to individual churn.",
                    action="Keep broad retention mechanics; no single-customer over-reliance.",
                    confidence="high",
                    stability=round(share_top, 3),
                )
            )

    # --- Coverage -----------------------------------------------------------
    n_modeled = int(work["customer_id"].nunique())
    if predictions_df is not None and not predictions_df.empty:
        n_pred = int(predictions_df["customer_id"].nunique())
        if n_pred > n_modeled:
            n_unmodeled = n_pred - n_modeled
            insights.append(
                Insight(
                    domain="customer",
                    entity="unmodeled customers",
                    kind="watch",
                    title=f"{n_unmodeled} customers have no CLV estimate",
                    evidence="They were dropped by the BG/NBD fit (insufficient history).",
                    action="Treat their value with rule-of-thumb; do not reprice or target on CLV.",
                    confidence="medium",
                    sample_size=n_unmodeled,
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
