"""Overview-domain insight generation.

Turns raw transaction aggregates into structured ``Insight`` objects:
revenue decomposition (customers x frequency x basket size x price-mix),
concentration (Pareto / HHI), and SPC anomaly flags on the revenue trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.basket_metrics import spc_revenue_trend
from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_TOP_SHARE_N = 10
_HHI_RISK = 0.25  # >= 0.25 ~ moderately concentrated
_TOP_SHARE_RISK = 0.5  # top-10 products >= 50% of revenue


def _revenue_components(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue = customers x frequency x basket_size x price_per_unit, weekly."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["revenue"] = work["price"] * work["quantity"]
    work["week"] = work["date"].dt.to_period("W").dt.start_time
    weekly = (
        work.groupby("week")
        .agg(
            customers=("customer_id", "nunique"),
            transactions=("transaction_id", "nunique"),
            units=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_index()
    )
    if weekly.empty:
        return weekly
    weekly["frequency"] = weekly["transactions"] / weekly["customers"].replace(0, np.nan)
    weekly["basket_size"] = weekly["units"] / weekly["transactions"].replace(0, np.nan)
    weekly["price_per_unit"] = weekly["revenue"] / weekly["units"].replace(0, np.nan)
    return weekly


def _growth_attribution(comps: pd.DataFrame) -> dict[str, float]:
    """Multiplicative growth attribution between the last two periods.

    g(R) ~= g(customers) + g(frequency) + g(basket_size) + g(price_per_unit),
    each g() a log-change. Returns shares summing to ~1.
    """
    if len(comps) < 2:
        return {}
    curr = comps.iloc[-1]
    prev = comps.iloc[-2]
    factors = ["customers", "frequency", "basket_size", "price_per_unit"]
    logs: dict[str, float] = {}
    total_log = 0.0
    for f in factors:
        if prev[f] and curr[f] and prev[f] > 0 and curr[f] > 0:
            val = float(np.log(curr[f] / prev[f]))
            logs[f] = val
            total_log += val
    if total_log == 0:
        return {f: 0.0 for f in logs}
    return {f: v / total_log for f, v in logs.items()}


def _concentration(df: pd.DataFrame) -> tuple[float, float]:
    """Top-N revenue share and HHI over the whole window."""
    work = df.copy()
    revenue = (work["price"] * work["quantity"]).groupby(work["stockcode"]).sum()
    total = float(revenue.sum())
    if total <= 0:
        return 0.0, 0.0
    shares = revenue / total
    top_share = float(shares.nlargest(_TOP_SHARE_N).sum())
    hhi = float((shares**2).sum())
    return top_share, hhi


def _anomaly_flag(df: pd.DataFrame) -> pd.DataFrame:
    """SPC anomalies on the weekly revenue series."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    revenue = (
        (work["price"] * work["quantity"])
        .groupby(work["date"].dt.to_period("W").dt.start_time)
        .sum()
        .sort_index()
    )
    if len(revenue) < 5:
        return pd.DataFrame()
    try:
        return spc_revenue_trend(revenue)
    except Exception:
        return pd.DataFrame()


def generate_overview_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Build Overview insights from the raw transaction frame.

    Returns a DataFrame validated against PRICING_INSIGHTS (the shared insight
    contract), with ``domain`` set to "overview".
    """
    insights: list[Insight] = []
    comps = _revenue_components(df)

    if len(comps) >= 2:
        curr = comps.iloc[-1]
        prev = comps.iloc[-2]
        growth = float(curr["revenue"] / prev["revenue"] - 1) if prev["revenue"] > 0 else 0.0
        attribution = _growth_attribution(comps)
        if attribution:
            driver = max(attribution, key=lambda k: attribution[k])
            share = attribution[driver]
            if abs(share) < 0.15:
                driver = "mix of customer, basket and price factors"
            driver_label = {
                "customers": "customer count",
                "frequency": "shopping frequency",
                "basket_size": "basket size",
                "price_per_unit": "price per unit",
            }.get(driver, driver)

            kind = "growth" if growth >= 0 else "risk"
            title = f"Weekly revenue {'up' if growth >= 0 else 'down'} {abs(growth):.1%}"
            if kind == "growth":
                title += f" — driven by {driver_label}"
            else:
                title += f" — {driver_label} the main drag"
            insights.append(
                Insight(
                    domain="overview",
                    entity="all customers",
                    kind=kind,
                    title=title,
                    evidence=(
                        f"Revenue moved from €{prev['revenue']:,.0f} to €{curr['revenue']:,.0f} "
                        f"({growth:+.1%}). Log-ratio attribution: customers "
                        f"{attribution.get('customers', 0.0):+.0%}, frequency "
                        f"{attribution.get('frequency', 0.0):+.0%}, basket size "
                        f"{attribution.get('basket_size', 0.0):+.0%}, price per unit "
                        f"{attribution.get('price_per_unit', 0.0):+.0%} of the change."
                    ),
                    action=(
                        "If customer count is the drag, prioritize acquisition/win-back; "
                        "if basket size, work cross-sell and add-ons; if price mix, "
                        "review promotions and assortment tiering."
                    ),
                    confidence="high" if len(comps) >= 8 else "medium",
                    sample_size=int(curr["transactions"]),
                )
            )

    top_share, hhi = _concentration(df)
    if top_share > 0:
        if top_share >= _TOP_SHARE_RISK or hhi >= _HHI_RISK:
            insights.append(
                Insight(
                    domain="overview",
                    entity="top products",
                    kind="risk",
                    title=f"Revenue is concentrated: top-{_TOP_SHARE_N} products = {top_share:.0%}",
                    evidence=(
                        f"Top-{_TOP_SHARE_N} products generate {top_share:.0%} of revenue "
                        f"(HHI {hhi:.2f}). Disruption to any of them — out-of-stock, "
                        f"delisting, or competitor price moves — hits revenue hard."
                    ),
                    action=(
                        "Protect availability and pricing on the top products; keep "
                        "substitute depth behind each of them."
                    ),
                    confidence="high" if top_share >= _TOP_SHARE_RISK else "medium",
                    impact_value=None,
                    stability=round(hhi, 3),
                )
            )
        else:
            insights.append(
                Insight(
                    domain="overview",
                    entity="all products",
                    kind="efficiency",
                    title=f"Revenue spread is healthy: top-{_TOP_SHARE_N} = {top_share:.0%}",
                    evidence=f"Top-{_TOP_SHARE_N} products account for {top_share:.0%} of revenue (HHI {hhi:.2f}).",
                    action="No concentration-driven action required; keep monitoring.",
                    confidence="high",
                    stability=round(hhi, 3),
                )
            )

    spc = _anomaly_flag(df)
    if not spc.empty and spc["anomaly"].any():
        n_anom = int(spc["anomaly"].sum())
        recent = spc.tail(3)
        recent_anom = recent[recent["anomaly"]]
        if not recent_anom.empty:
            periods = ", ".join(str(p) for p in recent_anom["period"].tolist())
            insights.append(
                Insight(
                    domain="overview",
                    entity="all products",
                    kind="anomaly",
                    title=f"{n_anom} SPC revenue anomalies, {len(recent_anom)} in the last 3 weeks",
                    evidence=(
                        f"Statistical process control flags unusual weekly revenue levels "
                        f"(recent: {periods}). Anomalies may be promos, stock-outs, "
                        f"price changes, or data issues."
                    ),
                    action="Investigate the anomalous weeks before extrapolating trends.",
                    confidence="medium",
                    sample_size=int(len(spc)),
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
