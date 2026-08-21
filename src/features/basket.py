"""Basket feature tables: fact tables for basket/transaction-level analytics."""

from __future__ import annotations

import pandas as pd


def build_fact_basket(df: pd.DataFrame) -> pd.DataFrame:
    """Build the basket-level fact table."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]
    fact = (
        d.groupby("transaction_id", as_index=False).agg(
            customer_id=("customer_id", "first"),
            date=("date", "first"),
            basket_size=("quantity", "sum"),
            n_products=("stockcode", "nunique"),
            basket_revenue=("line_revenue", "sum"),
            iso_week=("iso_week", "first"),
        )
    )
    fact["date"] = pd.to_datetime(fact["date"])
    if "iso_week" not in fact.columns or fact["iso_week"].isna().all():
        iso = fact["date"].dt.isocalendar()
        fact["iso_week"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    fact = fact.sort_values("transaction_id").reset_index(drop=True)
    return fact
