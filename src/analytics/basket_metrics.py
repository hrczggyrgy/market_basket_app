"""Basket-level metrics: penetration, composition, entropy, and variability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, variation

from src.analytics.schemas import (
    BASKET_COMPOSITION,
    BASKET_OVER_TIME,
    BASKET_PENETRATION,
    CUSTOMER_ENTROPY,
    IPT_CV,
    REVENUE_SPC,
    check,
)


def compute_basket_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product basket penetration and revenue share."""
    baskets = df["transaction_id"].nunique()
    grouped = df.groupby("stockcode")
    revenue = (df["price"] * df["quantity"]).sum()
    revenue_by = (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum()
    table = pd.DataFrame(
        {
            "stockcode": grouped.size().index,
            "basket_count": grouped["transaction_id"].nunique(),
            "penetration": grouped["transaction_id"].nunique() / baskets,
            "revenue_share": revenue_by.div(revenue),
        }
    ).reset_index(drop=True)
    table = table.sort_values("penetration", ascending=False).reset_index(drop=True)
    return check(table, BASKET_PENETRATION)


def basket_penetration_over_time(df: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    """Basket volume and value per time period."""
    df = df.copy()
    df["period"] = df["date"].dt.to_period(period)
    grouped = df.groupby("period")
    total_rev = (df["price"] * df["quantity"]).groupby(df["period"]).sum()
    table = pd.DataFrame(
        {
            "period": grouped["transaction_id"].nunique().index.astype(str),
            "n_baskets": grouped["transaction_id"].nunique(),
            "avg_basket_size": grouped["quantity"].sum() / grouped["transaction_id"].nunique(),
            "avg_basket_value": total_rev / grouped["transaction_id"].nunique(),
        }
    ).reset_index(drop=True)
    return check(table, BASKET_OVER_TIME)


def compute_basket_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of baskets by number of distinct items."""
    sizes = df.groupby("transaction_id")["stockcode"].nunique().rename("basket_size")
    table = (
        sizes.groupby(sizes).size().rename("n_baskets").reset_index()
    )
    table["pct"] = table["n_baskets"] / table["n_baskets"].sum()
    return check(table, BASKET_COMPOSITION)


def compute_customer_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer transaction-line mix entropy (variety of purchasing).

    Computes Shannon entropy over the distribution of SKUs in a customer's
    transaction lines. This is a measure of product-mix diversity within
    individual customer baskets, NOT a general purchase diversity metric.

    Renamed from "Product Mix Entropy" to "Transaction-line Mix Entropy"
    to avoid confusion with broader product diversity metrics.
    """
    counts = df.groupby(["customer_id", "stockcode"]).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    entropies = probs.apply(lambda row: entropy(row), axis=1)
    n_distinct = (counts > 0).sum(axis=1)
    max_entropy = np_log(n_distinct)
    
    # n_purchases should be transaction count (not line count)
    n_transactions = df.groupby("customer_id")["transaction_id"].nunique()
    
    table = pd.DataFrame(
        {
            "customer_id": counts.index,
            "n_distinct_products": n_distinct,
            "n_purchases": n_transactions,
            "entropy": entropies,
            "normalized_entropy": (entropies / max_entropy).fillna(0.0).clip(0, 1),
        }
    ).reset_index(drop=True)
    return check(table, CUSTOMER_ENTROPY)


def compute_ipt_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation of basket size for each product's baskets."""
    basket_sizes = df.groupby("transaction_id")["stockcode"].nunique()
    # Get unique (transaction_id, stockcode) pairs - each product in each basket
    prod_txn = df[["transaction_id", "stockcode"]].drop_duplicates()
    merged = prod_txn.merge(basket_sizes.rename("basket_size"), on="transaction_id", how="left")

    agg = merged.groupby("stockcode")["basket_size"].agg(
        mean_ipt="mean",
        std_ipt=lambda s: float(s.std(ddof=0)),
        cv_ipt=lambda s: float(variation(s)),
        n_transactions="size",
    ).reset_index()

    table = agg.sort_values("cv_ipt", ascending=False).reset_index(drop=True)
    table["mean_ipt"] = table["mean_ipt"].astype(float)
    table["std_ipt"] = table["std_ipt"].astype(float)
    table["cv_ipt"] = table["cv_ipt"].astype(float)
    table["n_transactions"] = table["n_transactions"].astype(int)
    return check(table, IPT_CV)


def spc_revenue_trend(
    values: pd.Series,
    index: pd.Index | None = None,
    term: str = "period",
    window: int = 8,
    k: float = 2.0,
    min_periods: int = 4,
    run_length: int = 7,
) -> pd.DataFrame:
    """SPC control-chart annotation for a revenue-like time series.

    Returns a REVENUE_SPC-validated table with one row per period:
    - center: trailing-window mean of values.
    - ucl / lcl: center +/- k * trailing-window std.
    - anomaly: True when a point is outside the control limits (Rule 1) OR
      part of a run of ``run_length`` consecutive points on one side of the
      center (Rule 3). Points within-bounds but slightly off-center do NOT
      count as anomalies.
    - rule: brief label of the rule that fired ("limit" / "run").

    ``values`` must be sorted ascending by time.
    """
    if len(values) == 0:
        return check(pd.DataFrame(columns=list(REVENUE_SPC.columns)), REVENUE_SPC, allow_empty=True)

    series = pd.Series(values.to_numpy(dtype=float), 
                        index=pd.RangeIndex(len(values)) if index is None else pd.Index(index))
    center = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=1)
    ucl = center + k * std
    lcl = center - k * std

    out_limits = (series > ucl) | (series < lcl)

    # runs: consecutive points on same side of center, min run_length
    side = pd.Series(np.sign(series - center), index=series.index)
    run_ids = (side != side.shift()).cumsum()
    run_sizes = side.groupby(run_ids).transform("size")
    in_run = (run_sizes >= run_length) & (side != 0)

    anomaly = out_limits | in_run
    rule_label = []
    for i, (is_lim, is_run) in enumerate(zip(out_limits.tolist(), in_run.tolist())):
        if is_lim:
            rule_label.append("limit")
        elif is_run:
            rule_label.append("run")
        else:
            rule_label.append("")

    table = pd.DataFrame(
        {
            term: [str(p) for p in series.index],
            "revenue": series.to_numpy(),
            "center": center.to_numpy(),
            "ucl": ucl.to_numpy(),
            "lcl": lcl.to_numpy(),
            "anomaly": anomaly.to_numpy(),
            "rule": rule_label,
        }
    )
    return check(table, REVENUE_SPC)


def np_log(series: pd.Series) -> pd.Series:
    """Natural log with zeros mapped to 0 (avoid -inf) with warning."""
    import numpy as np
    import warnings as _warnings

    zero_count = (series == 0).sum()
    if zero_count > 0:
        _warnings.warn(
            f"Entropy calculation: {zero_count} zero values replaced with 1 to avoid log(0). "
            "This may bias entropy estimates downward.",
            UserWarning,
            stacklevel=2
        )
    return series.replace(0, 1).apply(np.log)


# ============================================================
# Canonical Basket Metrics
# ============================================================

def compute_basket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute canonical basket-level metrics per transaction.

    Returns a DataFrame with one row per transaction_id and the following
    canonical columns:
    - basket_units: total quantity (sum of quantity)
    - basket_lines: number of distinct SKU lines in the basket
    - basket_distinct_skus: number of distinct SKUs (same as basket_lines for now)
    - basket_revenue: total revenue (sum of price * quantity)

    These canonical definitions ensure consistency across all analytics.
    """
    df = df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    
    agg = df.groupby("transaction_id").agg(
        basket_units=("quantity", "sum"),
        basket_lines=("stockcode", "nunique"),
        basket_distinct_skus=("stockcode", "nunique"),
        basket_revenue=("revenue", "sum"),
    ).reset_index()
    
    return agg


def compute_basket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics of canonical basket metrics across all transactions."""
    basket_metrics = compute_basket_metrics(df)
    
    summary = pd.DataFrame({
        "metric": [
            "basket_units",
            "basket_lines", 
            "basket_distinct_skus",
            "basket_revenue",
        ],
        "mean": [
            basket_metrics["basket_units"].mean(),
            basket_metrics["basket_lines"].mean(),
            basket_metrics["basket_distinct_skus"].mean(),
            basket_metrics["basket_revenue"].mean(),
        ],
        "median": [
            basket_metrics["basket_units"].median(),
            basket_metrics["basket_lines"].median(),
            basket_metrics["basket_distinct_skus"].median(),
            basket_metrics["basket_revenue"].median(),
        ],
        "std": [
            basket_metrics["basket_units"].std(),
            basket_metrics["basket_lines"].std(),
            basket_metrics["basket_distinct_skus"].std(),
            basket_metrics["basket_revenue"].std(),
        ],
        "min": [
            basket_metrics["basket_units"].min(),
            basket_metrics["basket_lines"].min(),
            basket_metrics["basket_distinct_skus"].min(),
            basket_metrics["basket_revenue"].min(),
        ],
        "max": [
            basket_metrics["basket_units"].max(),
            basket_metrics["basket_lines"].max(),
            basket_metrics["basket_distinct_skus"].max(),
            basket_metrics["basket_revenue"].max(),
        ],
    })
    return summary
