"""Product performance: lifecycle, velocity, ABC/XYZ, repeat rate, SKU actions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import variation

from src.analytics.config import get_config
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
    """XYZ classification by UNIT-demand volatility, not revenue.

    Key differences from a naive revenue-CV approach:
    - CV is computed on weekly unit demand (quantity), so price-driven
      revenue swings do not masquerade as demand volatility.
    - Thresholds are config-driven (``xyz_cv_thresholds``) and can be
      switched to quantile-based assignment (``xyz_use_quantile_method``).
    - A minimum history guard (``xyz_min_periods``) prevents short-lived or
      recently-launched SKUs from being spuriously classified; those get
      ``demand_profile == "Insufficient History"`` and xyz_class "Z".
    - A demand profile distinguishes Regular / Seasonal / Intermittent /
      Lumpy demand (Syntetos-Boylan style) so volatility and intermittency
      are not conflated.

    Classes:
        X: stable (CV <= threshold[0]); Y: moderate; Z: erratic.
        SKUs with insufficient history default to Z (treated conservatively
        as high volatility) but are flagged via ``demand_profile``.
    """
    cfg = get_config()
    thresholds = sorted(cfg.xyz_cv_thresholds)
    min_periods = cfg.xyz_min_periods

    df = df.copy()
    df["_period"] = df["date"].dt.to_period(period).astype(str)
    units = df["quantity"]
    revenue = df["price"] * df["quantity"]
    pivot = (
        units.groupby([df["stockcode"], df["_period"]]).sum().unstack(fill_value=0)
    )
    # Per-SKU span: from first to last observed period (handles product
    # launches/end-of-life without counting pre-launch zeros).
    spans = []
    for _sku, row in pivot.iterrows():
        observed = row[row > 0]
        if observed.empty:
            spans.append(0)
            continue
        start = pivot.columns.get_loc(observed.index[0])
        end = pivot.columns.get_loc(observed.index[-1])
        spans.append(end - start + 1)
    n_periods = pd.Series(spans, index=pivot.index, name="n_periods")

    nonzero = (pivot > 0).sum(axis=1)
    cv = pivot.apply(lambda row: variation(row), axis=1).fillna(0.0)
    zero_demand_rate = (n_periods - nonzero).clip(lower=0) / n_periods.replace(0, np.nan)
    zero_demand_rate = zero_demand_rate.fillna(1.0).clip(0.0, 1.0)

    table = pd.DataFrame(
        {
            "stockcode": pivot.index,
            "revenue": revenue.groupby(df["stockcode"]).sum().reindex(pivot.index).fillna(0.0),
            "units": pivot.sum(axis=1),
            "cv": cv,
            "n_periods": n_periods,
            "nonzero_periods": nonzero,
            "zero_demand_rate": zero_demand_rate,
        }
    ).reset_index(drop=True)

    table["demand_profile"] = table.apply(
        lambda r: _demand_profile(r, min_periods=min_periods), axis=1
    )

    has_history = table["n_periods"] >= min_periods
    if cfg.xyz_use_quantile_method and has_history.sum() >= 3:
        q1, q2 = (cfg.xyz_quantiles + [0.5])[:2], cfg.xyz_quantiles
        cut_points = table.loc[has_history, "cv"].quantile([q1[0], q2[1]]).tolist()
        table["xyz_class"] = np.select(
            [
                table["cv"] <= cut_points[0],
                table["cv"] <= cut_points[1],
            ],
            ["X", "Y"],
            default="Z",
        )
    else:
        table["xyz_class"] = np.select(
            [table["cv"] <= thresholds[0], table["cv"] <= thresholds[1]],
            ["X", "Y"],
            default="Z",
        )
    # Insufficient history -> treated as unknown/high volatility, flagged.
    table.loc[~has_history, "xyz_class"] = "Z"
    return check(table, XYZ_CLASSES)


def _demand_profile(row: pd.Series, min_periods: int) -> str:
    """Syntetos-Boylan style profile: Regular/Seasonal/Intermittent/Lumpy.

    Insufficient History takes precedence: volatility cannot be assessed on
    fewer than ``min_periods`` of observed demand.
    """
    if row["n_periods"] < min_periods or row["nonzero_periods"] == 0:
        return "Insufficient History"
    adi = row["n_periods"] / max(row["nonzero_periods"], 1)  # avg inter-demand interval
    if adi >= 1.32:
        return "Lumpy" if row["cv"] >= 0.49 else "Intermittent"
    return "Seasonal" if row["cv"] >= 0.49 else "Regular"


def product_lifecycle_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Lifecycle stage from recent vs prior period revenue growth.

    Compares the latest weekly period against the second-to-latest: products
    whose most-recent week grew > ``+25%`` vs the prior week are "growth",
    <-25% are "decline", otherwise "mature" (thresholds from config).
    """
    df = df.copy()
    df["_period"] = df["date"].dt.to_period("W").astype(str)
    revenue = df["price"] * df["quantity"]

    periods = sorted(df["_period"].unique())
    if len(periods) < 2:
        return check(pd.DataFrame(columns=list(LIFECYCLE.columns)), LIFECYCLE, allow_empty=True)

    recent_period, prior_period = periods[-1], periods[-2]
    recent_mask = df["_period"] == recent_period
    prior_mask = df["_period"] == prior_period

    recent = revenue[recent_mask].groupby(df.loc[recent_mask, "stockcode"]).sum()
    prior = revenue[prior_mask].groupby(df.loc[prior_mask, "stockcode"]).sum()

    all_products = sorted(set(recent.index) | set(prior.index))
    recent_rev = recent.reindex(all_products).fillna(0.0).to_numpy()
    prior_rev = prior.reindex(all_products).fillna(0.0).to_numpy()

    table = pd.DataFrame(
        {
            "stockcode": all_products,
            "recent_revenue": recent_rev,
            "prior_revenue": prior_rev,
        }
    )
    table["growth_pct"] = np.divide(
        (recent_rev - prior_rev) * 100,
        prior_rev,
        out=np.zeros(len(table), dtype=float),
        where=prior_rev > 0,
    )
    # prior period had no sales but recent does -> new growth; both zero -> flat
    table.loc[(table["prior_revenue"] == 0) & (table["recent_revenue"] > 0), "growth_pct"] = 100.0
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
    table = abc.join(xyz[["xyz_class", "demand_profile"]], how="outer").join(velocity[["velocity"]], how="outer").join(
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
    if row.get("demand_profile") == "Insufficient History":
        return "review"
    if row["abc_class"] == "A" and row["xyz_class"] in ("X", "Y"):
        return "keep"
    if row["abc_class"] == "C" and row["xyz_class"] == "Z":
        return "delist_candidate"
    if row["abc_class"] == "C" and row["xyz_class"] in ("X", "Y") and row["velocity"] > 0:
        return "review"
    return "review"
