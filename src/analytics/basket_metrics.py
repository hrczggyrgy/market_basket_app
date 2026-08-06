"""Basket-level metrics: penetration, composition, entropy, and variability."""

from __future__ import annotations

import pandas as pd
from scipy.stats import entropy, variation

from src.analytics.schemas import (
    BASKET_COMPOSITION,
    BASKET_OVER_TIME,
    BASKET_PENETRATION,
    CUSTOMER_ENTROPY,
    IPT_CV,
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
    """Per-customer product-mix entropy (variety of purchasing)."""
    counts = df.groupby(["customer_id", "stockcode"]).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    entropies = probs.apply(lambda row: entropy(row), axis=1)
    n_distinct = (counts > 0).sum(axis=1)
    max_entropy = np_log(n_distinct)
    table = pd.DataFrame(
        {
            "customer_id": counts.index,
            "n_distinct_products": n_distinct,
            "n_purchases": counts.sum(axis=1),
            "entropy": entropies,
            "normalized_entropy": (entropies / max_entropy).fillna(0.0).clip(0, 1),
        }
    ).reset_index(drop=True)
    return check(table, CUSTOMER_ENTROPY)


def compute_ipt_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation of basket size for each product's baskets."""
    basket_sizes = df.groupby("transaction_id")["stockcode"].nunique()
    rows = []
    for stockcode, group in df.groupby("stockcode"):
        sizes = basket_sizes.loc[group["transaction_id"].unique()]
        rows.append(
            {
                "stockcode": stockcode,
                "mean_ipt": float(sizes.mean()),
                "std_ipt": float(sizes.std(ddof=0)),
                "cv_ipt": float(variation(sizes)),
                "n_transactions": int(len(sizes)),
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_ipt", ascending=False).reset_index(drop=True)
    return check(table, IPT_CV)


def np_log(series: pd.Series) -> pd.Series:
    """Natural log with zeros mapped to 0 (avoid -inf)."""
    import numpy as np

    return series.replace(0, 1).apply(np.log)
