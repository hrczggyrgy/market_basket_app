"""Product performance: lifecycle, velocity, ABC/XYZ, repeat rate, SKU actions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import variation

from src.analytics.schemas import (
    ABC_CLASSES,
    LIFECYCLE,
    PRODUCT_METRICS,
    PRODUCT_VELOCITY,
    REPEAT_RATE,
    SECOND_PURCHASE,
    SKU_RATIONALIZATION,
    XYZ_CLASSES,
    check,
)


def compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Core per-product metrics."""
    revenue = df["price"] * df["quantity"]
    n_baskets = df["transaction_id"].nunique()
    table = pd.DataFrame(
        {
            "stockcode": df.groupby("stockcode")["product"].count().index,
            "revenue": revenue.groupby(df["stockcode"]).sum(),
            "units": df.groupby("stockcode")["quantity"].sum(),
            "transactions": df.groupby("stockcode")["transaction_id"].nunique(),
            "customers": df.groupby("stockcode")["customer_id"].nunique(),
        }
    ).reset_index(drop=True)
    table["avg_price"] = table["revenue"] / table["units"].replace(0, np.nan)
    table["penetration"] = table["transactions"] / n_baskets
    table = table.sort_values("revenue", ascending=False).reset_index(drop=True)
    return check(table, PRODUCT_METRICS)


def abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """ABC classification by cumulative revenue share (A <= 70%, B <= 90%, C rest)."""
    revenue = (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    cumulative = revenue.cumsum() / revenue.sum()
    # Clip to [0, 1] to handle floating-point precision issues on the last row
    cumulative = cumulative.clip(upper=1.0)
    table = pd.DataFrame({"stockcode": revenue.index, "revenue": revenue, "cumulative_share": cumulative})
    table["abc_class"] = np.select(
        [cumulative <= 0.7, cumulative <= 0.9],
        ["A", "B"],
        default="C",
    )
    return check(table.reset_index(drop=True), ABC_CLASSES)


def xyz_analysis(df: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    """XYZ classification by revenue volatility (X cv<=10%, Y<=25%, Z rest)."""
    df = df.copy()
    df["_period"] = df["date"].dt.to_period(period).astype(str)
    revenue = df["price"] * df["quantity"]
    pivot = revenue.groupby([df["stockcode"], df["_period"]]).sum().unstack(fill_value=0)
    cv = pivot.apply(lambda row: variation(row), axis=1)
    table = pd.DataFrame(
        {
            "stockcode": pivot.index,
            "revenue": pivot.sum(axis=1),
            "cv": cv,
        }
    ).reset_index(drop=True)
    table["xyz_class"] = np.select(
        [table["cv"] <= 0.10, table["cv"] <= 0.25],
        ["X", "Y"],
        default="Z",
    )
    return check(table, XYZ_CLASSES)


def product_lifecycle_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Lifecycle stage from recent vs prior period revenue growth."""
    df = df.copy()
    df["_period"] = df["date"].dt.to_period("W").astype(str)
    revenue = df["price"] * df["quantity"]
    recent_mask = df["_period"] == df["_period"].max()
    revenue_by = revenue.groupby(df["stockcode"])
    recent = revenue_by.sum()
    prior_mask = df["_period"] == sorted(df["_period"].unique())[-2] if len(df["_period"].unique()) > 1 else None
    prior = revenue[prior_mask].groupby(df.loc[prior_mask, "stockcode"]).sum() if prior_mask is not None else None
    if prior is None:
        return check(pd.DataFrame(columns=list(LIFECYCLE.columns)), LIFECYCLE, allow_empty=True)
    table = pd.DataFrame(
        {
            "stockcode": recent.index,
            "recent_revenue": recent,
            "prior_revenue": prior.reindex(recent.index).fillna(0.0),
        }
    ).reset_index(drop=True)
    prior_rev = prior.reindex(recent.index).fillna(0.0).mask(lambda s: s == 0)
    table["growth_pct"] = ((recent - prior_rev) / prior_rev * 100).fillna(0.0)
    table["stage"] = table["growth_pct"].apply(_stage_from_growth)
    return check(table, LIFECYCLE)


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Units sold per active day per product."""
    active = df.groupby("stockcode")["date"].nunique()
    units = df.groupby("stockcode")["quantity"].sum()
    table = pd.DataFrame(
        {
            "stockcode": active.index,
            "units": units,
            "active_days": active,
        }
    ).reset_index(drop=True)
    table["velocity"] = table["units"] / table["active_days"].replace(0, np.nan)
    return check(table, PRODUCT_VELOCITY)


def compute_repeat_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Share of a product's customers who bought it more than once."""
    counts = df.groupby(["stockcode", "customer_id"]).size().reset_index(name="purchases")
    grouped = counts.groupby("stockcode")
    table = pd.DataFrame(
        {
            "stockcode": grouped["customer_id"].nunique().index,
            "n_customers": grouped["customer_id"].nunique(),
            "repeat_customers": grouped["purchases"].apply(lambda s: (s > 1).sum()),
        }
    ).reset_index(drop=True)
    table["repeat_rate"] = table["repeat_customers"] / table["n_customers"].replace(0, np.nan)
    return check(table, REPEAT_RATE)


def compute_time_to_second_purchase(df: pd.DataFrame) -> pd.DataFrame:
    """Days between first and second purchase, per product."""
    df2 = df.sort_values(["stockcode", "customer_id", "date"])
    df2["_rank"] = df2.groupby(["stockcode", "customer_id"]).cumcount()
    first = df2[df2["_rank"] == 0].set_index(["stockcode", "customer_id"])["date"]
    second = df2[df2["_rank"] == 1].set_index(["stockcode", "customer_id"])["date"]
    delta = (second - first).dropna().dt.days
    if delta.empty:
        return check(pd.DataFrame(columns=list(SECOND_PURCHASE.columns)), SECOND_PURCHASE, allow_empty=True)
    grouped = delta.groupby(level=0)
    table = pd.DataFrame(
        {
            "stockcode": grouped.count().index,
            "n_second_purchasers": grouped.count(),
            "median_days_to_second": grouped.median(),
            "mean_days_to_second": grouped.mean(),
        }
    ).reset_index(drop=True)
    return check(table, SECOND_PURCHASE)


def compute_sku_rationalization_df(df: pd.DataFrame) -> pd.DataFrame:
    """Combined ABC/XYZ/velocity/repeat view with suggested action."""
    abc = abc_analysis(df).set_index("stockcode")
    xyz = xyz_analysis(df).set_index("stockcode")
    velocity = compute_velocity(df).set_index("stockcode")
    repeat = compute_repeat_rate(df).set_index("stockcode")
    table = abc.join(xyz[["xyz_class"]], how="outer").join(velocity[["velocity"]], how="outer").join(
        repeat[["repeat_rate"]], how="outer"
    )
    table = table.reset_index().rename(columns={"index": "stockcode"})
    table = table.fillna({"velocity": 0.0, "repeat_rate": 0.0, "xyz_class": "Z", "abc_class": "C"})
    table["action"] = table.apply(_classify_sku_action, axis=1)
    return check(table[["stockcode", "revenue", "abc_class", "xyz_class", "velocity", "repeat_rate", "action"]], SKU_RATIONALIZATION)


def _stage_from_growth(growth_pct: float) -> str:
    if growth_pct > 25:
        return "growth"
    if growth_pct < -25:
        return "decline"
    return "mature"


def _classify_sku_action(row: pd.Series) -> str:
    if row["abc_class"] == "A" and row["xyz_class"] in ("X", "Y"):
        return "keep"
    if row["abc_class"] == "C" and row["xyz_class"] == "Z":
        return "delist_candidate"
    if row["abc_class"] == "C" and row["xyz_class"] in ("X", "Y") and row["velocity"] > 0:
        return "review"
    return "review"
