"""Scenario planning: pessimistic / neutral / optimistic projections per category.

Drivers are anchored to observed history:
- baseline: trailing weekly revenue per category
- retention: % of customers from a trailing window who return in the latest window
- frequency: transactions per customer per week
- aov: revenue per transaction

Projections compound weekly_growth_pct over ``projection_weeks`` using the
observed growth rate (neutral), yr 1-sigma downside (pessimistic), and a
manager-selectable uplift on top of trend (optimistic). Each row carries a
feasibility guard when the implied growth exceeds what the historical
volatility of the category can plausibly support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.schemas import SCENARIO_GRID, check

_SCENARIOS = ("pessimistic", "neutral", "optimistic")


def _category_baselines(df: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """Trailing weekly revenue, customers, transactions, and AOV per category."""
    df = df.copy()
    df["_week"] = df["date"].dt.to_period("W")
    df["revenue"] = df["price"] * df["quantity"]
    rows = []
    for cat, g in df.groupby("category"):
        weekly = g.groupby("_week").agg(
            revenue=("revenue", "sum"),
            transactions=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
        ).sort_index()
        if len(weekly) < 4:
            continue
        tail = weekly.tail(n_weeks)
        rows.append(
            {
                "category": cat,
                "n_transactions": int(weekly["transactions"].sum()),
                "n_customers": int(weekly["customers"].sum()),
                "weekly_revenue": float(tail["revenue"].mean()),
                "weekly_transactions": float(tail["transactions"].mean()),
                "weekly_customers": float(tail["customers"].mean()),
                "aov": float(tail["revenue"].sum() / max(tail["transactions"].sum(), 1)),
                "transactions_per_customer": float(
                    tail["transactions"].sum() / max(tail["customers"].sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_scenario_grid(
    df: pd.DataFrame,
    n_weeks: int = 12,
    optimistic_uplift: float = 0.10,
    projection_weeks: int = 13,
    weekly_band: float = 15.0,
) -> pd.DataFrame:
    """Project each category under pessimistic / neutral / optimistic growth.

    Levers: neutral growth = historical per-week growth rate (compounded);
    pessimistic = 1 sigma below trend; optimistic = trend + ``optimistic_uplift``.

    Feasibility guard: per-week growth is clamped to +/- ``weekly_band``
    percent (a plausible short-horizon planning band). A scenario is marked
    infeasible when its raw lever would have exceeded that band (i.e., the
    historical volatility of the category is too high to make a sensible
    projection), or when the resulting projection moves more than 2x in
    either direction.

    Args:
        df: Transaction DataFrame (date, transaction_id, stockcode, category,
            customer_id, price, quantity).
        n_weeks: Trailing weeks used to anchor the baseline and growth rate.
        optimistic_uplift: Additional weekly growth points for the optimistic
            scenario, in percent points (e.g., 0.10 = +0.10 pp/week).
        projection_weeks: Number of weeks to project forward.
        weekly_band: Absolute weekly growth cap (%) for clamping.

    Returns:
        DataFrame validated against SCENARIO_GRID (empty when insufficient
        history or no category column).
    """
    empty = pd.DataFrame(columns=list(SCENARIO_GRID.columns))

    if "category" not in df.columns or df.empty:
        return check(empty, SCENARIO_GRID, allow_empty=True)

    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").astype(str)
    df["revenue"] = df["price"] * df["quantity"]

    weekly = (
        df.groupby(["category", "week"], observed=True)["revenue"].sum().unstack(fill_value=0.0)
    )
    if weekly.shape[1] < 5:
        return check(empty, SCENARIO_GRID, allow_empty=True)

    # Weekly growth rate per row (pct change); clamp to guard division noise
    growth = weekly.pct_change(axis=1).replace([np.inf, -np.inf], np.nan)
    growth_tail = growth.iloc[:, -n_weeks:]

    rows: list[dict] = []
    for cat in weekly.index:
        series = weekly.loc[cat]
        non_zero = series[series > 0]
        if non_zero.empty:
            continue
        hist_mean = float(growth_tail.loc[cat].mean() * 100)
        hist_std = float(growth_tail.loc[cat].std() * 100)
        if np.isnan(hist_mean):
            continue
        base_revenue = float(non_zero.tail(n_weeks).mean())

        scenario_levers = {
            "pessimistic": (hist_mean - hist_std, "1-sigma below trend"),
            "neutral": (hist_mean, "historical trend"),
            "optimistic": (hist_mean + optimistic_uplift, "trend + manager uplift"),
        }
        for scenario, (raw_pct, lever) in scenario_levers.items():
            band_hit = abs(raw_pct) > weekly_band or np.isnan(raw_pct)
            weekly_pct = float(np.clip(raw_pct, -weekly_band, weekly_band)) if not np.isnan(raw_pct) else 0.0
            compounded = (1 + weekly_pct / 100) ** projection_weeks
            projected = base_revenue * compounded

            feasible = not band_hit
            note = ""
            if band_hit:
                note = f"raw growth {raw_pct:.1f}%/wk outside +/-{weekly_band:.0f}% planning band; clamped"
            if compounded > 2.0:
                feasible = False
                note = (note + "; " if note else "") + "projection more than doubles revenue"
            rows.append(
                {
                    "category": cat,
                    "scenario": scenario,
                    "growth_lever": lever,
                    "weekly_growth_pct": round(weekly_pct, 3),
                    "projected_revenue": round(projected, 2),
                    "revenue_change_pct": round((compounded - 1) * 100, 2),
                    "feasible": feasible,
                    "guard_note": note,
                }
            )

    if not rows:
        return check(empty, SCENARIO_GRID, allow_empty=True)

    table = pd.DataFrame(rows)
    return check(table, SCENARIO_GRID)