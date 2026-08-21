"""Customer feature tables: fact tables for customer-level analytics."""

from __future__ import annotations

import pandas as pd


def build_fact_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Build the customer fact table."""
    d = df.copy()
    if "customer_id" not in d.columns:
        raise ValueError("Missing required column 'customer_id' in DataFrame")
    d["line_revenue"] = d["price"] * d["quantity"]
    cust = (
        d.groupby("customer_id", as_index=False).agg(
            n_orders=("transaction_id", "nunique"),
            n_line_items=("transaction_id", "size"),
            n_products=("stockcode", "nunique"),
            total_revenue=("line_revenue", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
    )
    cust["active_days"] = (cust["last_date"] - cust["first_date"]).dt.days + 1
    cust["avg_order_value"] = cust["total_revenue"] / cust["n_orders"]
    col_order = [
        "customer_id", "n_orders", "n_line_items", "n_products",
        "total_revenue", "first_date", "last_date", "active_days", "avg_order_value",
    ]
    cust = cust[col_order]
    return cust

def build_fact_customer_product(df: pd.DataFrame) -> pd.DataFrame:
    """Build the customer-product interaction fact table."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]
    cp = (
        d.groupby(["customer_id", "stockcode"], as_index=False).agg(
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            n_transactions=("transaction_id", "nunique"),
        )
    )
    cp = cp.sort_values(["customer_id", "stockcode"]).reset_index(drop=True)
    return cp
