"""Cohort and period-over-period analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.analytics.schemas import (
    COHORT_DECAY,
    COHORT_LTV,
    COHORT_RETENTION,
    COHORT_SIZES,
    POP_COMPARISON,
    YOY_COMPARISON,
    check,
)

_PERIOD_FREQ = {"D": "D", "W": "W", "M": "M", "Q": "QE"}


def _cohort_period(df: pd.DataFrame, cohort_period: str) -> pd.Series:
    freq = _PERIOD_FREQ.get(cohort_period, "M")
    return df["date"].dt.to_period(freq)


def compute_cohorts(df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Retention table: cohort x period index -> retention rate."""
    df = df.copy()
    df["cohort"] = _cohort_period(df, cohort_period)
    df["period"] = _cohort_period(df, cohort_period)
    first = df.groupby("customer_id")["cohort"].min().rename("first_cohort")
    df = df.join(first, on="customer_id")
    df["first_cohort"] = df["first_cohort"].astype(df["cohort"].dtype)
    df["period_index"] = (df["period"] - df["first_cohort"]).apply(lambda p: p.n)
    df = df[df["period_index"].ge(0)]
    sizes = df.groupby("first_cohort")["customer_id"].nunique()
    retained = df.groupby(["first_cohort", "period_index"])["customer_id"].nunique().rename("retained")
    table = retained.reset_index().rename(columns={"first_cohort": "cohort"})
    table["cohort_size"] = table["cohort"].map(sizes)
    table["retention_rate"] = table["retained"] / table["cohort_size"]
    return check(table, COHORT_RETENTION)


def compute_cohort_sizes(df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Size and value of each acquisition cohort (customers grouped by first period)."""
    df = df.copy()
    df["cohort"] = _cohort_period(df, cohort_period)
    first = df.groupby("customer_id")["cohort"].min().rename("first_cohort")
    df = df.join(first, on="customer_id")
    df["first_cohort"] = df["first_cohort"].astype(df["cohort"].dtype)
    revenue = df["price"] * df["quantity"]
    table = pd.DataFrame(
        {
            "cohort": df.groupby("first_cohort")["customer_id"].nunique().index.astype(str),
            "n_customers": df.groupby("first_cohort")["customer_id"].nunique(),
            "n_transactions": df.groupby("first_cohort")["transaction_id"].nunique(),
            "revenue": revenue.groupby(df["first_cohort"]).sum(),
        }
    ).reset_index(drop=True)
    return check(table, COHORT_SIZES)


def period_over_period_comparison(df: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    """Period-over-period growth of revenue, transactions, and AOV."""
    df = df.copy()
    df["period"] = _cohort_period(df, period)
    revenue = df["price"] * df["quantity"]
    agg = pd.DataFrame(
        {
            "revenue": revenue.groupby(df["period"]).sum(),
            "transactions": df.groupby("period")["transaction_id"].nunique(),
            "customers": df.groupby("period")["customer_id"].nunique(),
        }
    )
    agg["aov"] = agg["revenue"] / agg["transactions"]
    agg["revenue_growth"] = agg["revenue"].pct_change() * 100
    agg["aov_growth"] = agg["aov"].pct_change() * 100
    table = agg.reset_index()
    table["period"] = table["period"].astype(str)
    return check(table, POP_COMPARISON)


def year_over_year_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year comparison by ISO week-of-year and year."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week
    df["revenue"] = df["price"] * df["quantity"]
    grouped = (
        df.groupby(["year", "week"])
        .agg(
            revenue=("revenue", "sum"),
            transactions=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    grouped["aov"] = grouped["revenue"] / grouped["transactions"]
    latest_year = grouped["year"].max()
    prior = grouped[grouped["year"] == latest_year - 1][["week", "revenue", "aov"]].rename(
        columns={"revenue": "prior_revenue", "aov": "prior_aov"}
    )
    current = grouped[grouped["year"] == latest_year]
    merged = current.merge(prior, on="week", how="left")
    merged["revenue_yoy_growth"] = (merged["revenue"] - merged["prior_revenue"]) / merged["prior_revenue"] * 100
    merged["aov_yoy_growth"] = (merged["aov"] - merged["prior_aov"]) / merged["prior_aov"] * 100
    return check(merged, YOY_COMPARISON)


def compute_cohort_ltv_curve(df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Cumulative revenue per customer by period index within cohort."""
    df = df.copy()
    df["cohort"] = _cohort_period(df, cohort_period)
    df["period"] = _cohort_period(df, cohort_period)
    first = df.groupby("customer_id")["cohort"].min().rename("first_cohort")
    df = df.join(first, on="customer_id")
    df["first_cohort"] = df["first_cohort"].astype(df["cohort"].dtype)
    df["period_index"] = (df["period"] - df["first_cohort"]).apply(lambda p: p.n)
    df = df[df["period_index"].ge(0)]
    revenue = df["price"] * df["quantity"]
    customers = df.groupby("first_cohort")["customer_id"].nunique()
    cumulative = (
        revenue.groupby([df["first_cohort"], df["period_index"]])
        .sum()
        .groupby(level=0)
        .cumsum()
        .rename("cumulative_revenue")
    )
    table = cumulative.reset_index().rename(columns={"first_cohort": "cohort"})
    table["ltv_per_customer"] = table["cumulative_revenue"] / table["cohort"].map(customers)
    return check(table, COHORT_LTV)


def compute_cohort_decay_rate(df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Exponential decay rate of each cohort's retention curve (log-linear slope)."""
    retention = compute_cohorts(df, cohort_period)
    if retention.empty:
        return check(pd.DataFrame(columns=list(COHORT_DECAY.columns)), COHORT_DECAY, allow_empty=True)
    rows = []
    for cohort, group in retention.groupby("cohort"):
        group = group[group["period_index"] > 0]
        if len(group) < 2:
            continue
        x = group["period_index"].to_numpy(dtype=float)
        y = np.log(group["retention_rate"].clip(lower=1e-6)).to_numpy()
        slope = linregress(x, y).slope
        rows.append({"cohort": cohort, "decay_rate": float(-slope) if not np.isnan(slope) else 0.0})
    if not rows:
        return check(pd.DataFrame(columns=list(COHORT_DECAY.columns)), COHORT_DECAY, allow_empty=True)
    table = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    return check(table, COHORT_DECAY)
