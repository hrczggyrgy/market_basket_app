"""Data-sufficiency gate: assess whether a dataset is adequate for each analysis module.

Every module must work credibly from date, transaction_id, stockcode, product,
customer_id, price, quantity alone. This module flags when these minimal
columns are present and whether there is enough data for robust inference.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def assess_data_sufficiency(
    transactions_df: pd.DataFrame,
    *,
    min_transactions: int = 1000,
    min_customers: int = 50,
    min_products: int = 10,
    min_time_span_days: int = 90,
    min_price_variation_cv: float = 0.05,
    required_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Assess dataset adequacy per dimension.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transaction data with at minimum columns matching required_cols.
    min_transactions : int
        Minimum total transaction rows for robust inference.
    min_customers : int
        Minimum unique customers.
    min_products : int
        Minimum unique products.
    min_time_span_days : int
        Minimum calendar span (first to last transaction date).
    min_price_variation_cv : float
        Minimum median coefficient of variation of price per product.
    required_cols : list of str, optional
        Columns that must be present. Defaults to basic transaction columns.

    Returns
    -------
    dict
        Keys: ``'transactions'``, ``'customers'``, ``'products'``,
        ``'time_span'``, ``'price_variation'``, ``'columns'``,
        ``'overall'``.
        Values: ``'robust'``, ``'directional'``, or ``'insufficient'``.
    """
    if required_cols is None:
        required_cols = [
            "date", "transaction_id", "stockcode", "customer_id", "price", "quantity"
        ]

    result: Dict[str, str] = {}

    # --- Column check ---
    missing = [c for c in required_cols if c not in transactions_df.columns]
    if missing:
        result["columns"] = "insufficient"
    else:
        result["columns"] = "robust"

    n = len(transactions_df)
    if n < min_transactions:
        result["transactions"] = "insufficient"
    elif n < min_transactions * 2:
        result["transactions"] = "directional"
    else:
        result["transactions"] = "robust"

    n_customers = transactions_df["customer_id"].nunique()
    if n_customers < min_customers:
        result["customers"] = "insufficient"
    elif n_customers < min_customers * 2:
        result["customers"] = "directional"
    else:
        result["customers"] = "robust"

    n_products = transactions_df["stockcode"].nunique()
    if n_products < min_products:
        result["products"] = "insufficient"
    elif n_products < min_products * 2:
        result["products"] = "directional"
    else:
        result["products"] = "robust"

    if "date" in transactions_df.columns:
        date_min = pd.to_datetime(transactions_df["date"]).min()
        date_max = pd.to_datetime(transactions_df["date"]).max()
        span_days = (date_max - date_min).days
        if span_days < min_time_span_days:
            result["time_span"] = "insufficient"
        elif span_days < min_time_span_days * 2:
            result["time_span"] = "directional"
        else:
            result["time_span"] = "robust"
    else:
        result["time_span"] = "insufficient"

    # --- Price variation ---
    if "price" in transactions_df.columns and n_products > 0:
        cv_by_product = (
            transactions_df.groupby("stockcode")["price"]
            .apply(lambda x: x.std() / x.mean() if x.mean() > 0 else 0.0)
        )
        median_cv = cv_by_product.median()
        if median_cv < min_price_variation_cv:
            result["price_variation"] = "insufficient"
        elif median_cv < min_price_variation_cv * 2:
            result["price_variation"] = "directional"
        else:
            result["price_variation"] = "robust"
    else:
        result["price_variation"] = "insufficient"

    # --- Overall ---
    status_rank = {"robust": 3, "directional": 2, "insufficient": 1}
    overall = min(status_rank[result.get(k, "insufficient")] for k in result)
    overall_label = {3: "robust", 2: "directional", 1: "insufficient"}[overall]
    result["overall"] = overall_label

    return result


def sufficiency_badge(label: str) -> str:
    """Return an emoji badge for a sufficiency label."""
    return {"robust": "✅", "directional": "⚠️", "insufficient": "❌"}.get(label, "❓")


def format_sufficiency_summary(result: Dict[str, str]) -> str:
    """Format the sufficiency result as a compact multi-line string."""
    lines = [f"**Data Sufficiency:** {sufficiency_badge(result['overall'])} `{result['overall']}`"]
    for k in ("transactions", "customers", "products", "time_span", "price_variation", "columns"):
        if k in result:
            lines.append(f"  {sufficiency_badge(result[k])} {k}: {result[k]}")
    return "\n".join(lines)
