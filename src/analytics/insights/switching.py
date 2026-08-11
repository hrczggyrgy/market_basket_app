"""Switching-domain insight generation.

Converts switching / demand-transference outputs into structured insights:
revenue at risk per product, net switching flows, and unique-demand (low SDP)
products whose absence would leak revenue out of the assortment.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_SDP_UNIQUE = 0.2  # below: non-substitutable demand
_SDP_SUBSTITUTABLE = 0.8  # above: highly substitutable


def _net_switching(demand_transference_df: pd.DataFrame) -> pd.DataFrame:
    """Net switching per product (in - out), from observed recovery proxies."""
    if demand_transference_df is None or demand_transference_df.empty:
        return pd.DataFrame()
    out = (
        demand_transference_df.groupby("from_product")["observed_switching_recovery_proxy"]
        .sum()
        .rename("outflow")
    )
    inflow = (
        demand_transference_df.groupby("to_product")["observed_switching_recovery_proxy"]
        .sum()
        .rename("inflow")
    )
    net = pd.concat([out, inflow], axis=1).fillna(0.0)
    net["net"] = net["inflow"] - net["outflow"]
    return net.sort_values("net")


def generate_switching_insights(
    demand_transference_df: pd.DataFrame,
    sdp_df: pd.DataFrame,
    delist_impact_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Switching insights.

    Args:
        demand_transference_df: DEMAND_TRANSFERENCE output (observed switching).
        sdp_df: SDP_SCORES output (substitutable demand % per product).
        delist_impact_df: optional DELIST_IMPACT output (net impact of simulated delists).

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "switching".
    """
    insights: list[Insight] = []

    net = _net_switching(demand_transference_df)
    if not net.empty:
        total_proxy = float(net["inflow"].sum())
        losers = net[net["net"] < 0].head(5)
        winners = net[net["net"] > 0].head(3)
        if not losers.empty and total_proxy > 0:
            top_loser = losers.iloc[0]
            top_loser_name = str(top_loser.name)
            loser_proxy = float(-top_loser["net"])
            insights.append(
                Insight(
                    domain="switching",
                    entity=top_loser_name,
                    kind="leakage",
                    title=f"{top_loser_name} leaks the most switching revenue",
                    evidence=(
                        f"Net switching outflow of €{loser_proxy:,.0f} (observed "
                        f"recovery-weighted). Switching is observed correlation, "
                        f"not a causal delist estimate."
                    ),
                    action=(
                        "Review price, availability and shelf position; verify whether "
                        "leakage is real before changing anything."
                    ),
                    confidence="medium",
                    impact_value=loser_proxy,
                    sample_size=int(len(net)),
                )
            )
        if not winners.empty:
            top_winner = winners.iloc[-1]
            winner_proxy = float(top_winner["net"])
            insights.append(
                Insight(
                    domain="switching",
                    entity=str(top_winner.name),
                    kind="growth",
                    title=f"{top_winner.name} is the top switching destination",
                    evidence=(
                        f"Net switching inflow of €{winner_proxy:,.0f}; customers "
                        f"substituting toward it should have availability protected."
                    ),
                    action="Protect stock and facings; consider promoting its substitutes to steer demand.",
                    confidence="medium",
                    impact_value=winner_proxy,
                )
            )

    if sdp_df is not None and not sdp_df.empty:
        unique = sdp_df[sdp_df["sdp"] < _SDP_UNIQUE]
        substitutable = sdp_df[sdp_df["sdp"] >= _SDP_SUBSTITUTABLE]
        if not unique.empty:
            unique_rev_pct = float(unique["sdp"].sum())
            entities = ", ".join(unique["stockcode"].astype(str).head(5).tolist())
            insights.append(
                Insight(
                    domain="switching",
                    entity=entities,
                    kind="risk",
                    title=f"{len(unique)} products carry non-substitutable demand",
                    evidence=(
                        f"Products with SDP < {_SDP_UNIQUE:.0%} retain "
                        f"~{unique_rev_pct:.0%} of switching-weighted revenue even when "
                        f"they go missing; their absence leaks demand out of the assortment."
                    ),
                    action="Never delist on volume grounds alone; they need substitute depth first.",
                    confidence="medium",
                    sample_size=int(len(unique)),
                )
            )
        if not substitutable.empty:
            sub_rev_pct = float(substitutable["sdp"].sum())
            insights.append(
                Insight(
                    domain="switching",
                    entity=", ".join(substitutable["stockcode"].astype(str).head(5).tolist()),
                    kind="efficiency",
                    title=f"{len(substitutable)} products have highly substitutable demand",
                    evidence=(
                        f"SDP >= {_SDP_SUBSTITUTABLE:.0%} (~{sub_rev_pct:.0%} of switching-weighted "
                        f"revenue): most of their demand is recoverable within the assortment."
                    ),
                    action="These are the least risky delist candidates — verify with a causal test.",
                    confidence="medium",
                    sample_size=int(len(substitutable)),
                )
            )

    if delist_impact_df is not None and not delist_impact_df.empty:
        positive = delist_impact_df[delist_impact_df["net_revenue_impact"] > 0]
        negative = delist_impact_df[delist_impact_df["net_revenue_impact"] < 0]
        if not negative.empty:
            worst = negative.iloc[0]
            insights.append(
                Insight(
                    domain="switching",
                    entity=str(worst["stockcode"]),
                    kind="risk",
                    title=f"Delisting {worst['stockcode']} would lose €{-worst['net_revenue_impact']:,.0f}",
                    evidence=(
                        f"Recovery of €{worst['estimated_revenue_recovered']:,.0f} against "
                        f"€{worst['product_revenue']:,.0f} of own revenue "
                        f"({worst['recovery_rate']:.0%} recovery)."
                    ),
                    action="Keep it in assortment; low internal recovery means revenue leaks.",
                    confidence="low",
                    impact_value=float(-worst["net_revenue_impact"]),
                    sample_size=int(len(negative)),
                )
            )
        if not positive.empty:
            best = positive.iloc[0]
            insights.append(
                Insight(
                    domain="switching",
                    entity=str(best["stockcode"]),
                    kind="opportunity",
                    title=f"Delisting {best['stockcode']} is revenue-neutral or better",
                    evidence=(
                        f"Recovery of €{best['estimated_revenue_recovered']:,.0f} vs "
                        f"€{best['product_revenue']:,.0f} own revenue; net "
                        f"€{best['net_revenue_impact']:,.0f}."
                    ),
                    action="Validate with a causal or market test before delisting.",
                    confidence="low",
                    impact_value=float(best["net_revenue_impact"]),
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
