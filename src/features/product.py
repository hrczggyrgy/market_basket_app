"""Product feature tables: dimension and fact tables for product-level analytics."""

from __future__ import annotations

import pandas as pd


def build_dim_product(df: pd.DataFrame) -> pd.DataFrame:
    """Build the product dimension table."""
    d = df.copy()
    required = ["stockcode", "product", "price", "quantity"]
    for col in required:
        if col not in d.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame")
    d["line_revenue"] = d["price"] * d["quantity"]

    # Build agg spec dynamically based on available columns
    agg_dict = {
        "product": ("product", "first"),
        "total_revenue": ("line_revenue", "sum"),
        "n_orders": ("transaction_id", "nunique"),
        "n_line_items": ("transaction_id", "size"),
        "n_customers": ("customer_id", "nunique"),
        "units": ("quantity", "sum"),
        "avg_price": ("price", "mean"),
        "first_date": ("date", "min"),
        "last_date": ("date", "max"),
    }

    # Add optional columns if present
    optional_agg = {}
    if "category" in d.columns:
        agg_dict["category"] = ("category", "first")
    if "brand" in d.columns:
        agg_dict["brand"] = ("brand", "first")

    dim = d.groupby("stockcode", as_index=False).agg(agg_dict)

    # Fill NaN values for optional columns
    if "category" not in df.columns:
        dim["category"] = "Unknown"
    if "brand" not in df.columns:
        dim["brand"] = "Unknown"

    # Ensure product name falls back to stockcode if missing
    dim["product"] = dim["product"].fillna(dim["stockcode"])

    # Reorder columns
    col_order = [
        "stockcode", "product", "category", "brand",
        "total_revenue", "n_orders", "n_line_items", "n_customers",
        "units", "avg_price", "first_date", "last_date",
    ]
    dim = dim[col_order]

    return dim


def build_fact_product_day(df: pd.DataFrame) -> pd.DataFrame:
    """Build the daily product fact table."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]

    fact = (
        d.groupby(["stockcode", "date"], as_index=False).agg(
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            n_transactions=("transaction_id", "nunique"),
            n_customers=("customer_id", "nunique"),
        )
    )

    fact["date"] = pd.to_datetime(fact["date"])
    fact = fact.sort_values(["stockcode", "date"]).reset_index(drop=True)

    return fact


def build_fact_product_week(df: pd.DataFrame) -> pd.DataFrame:
    """Build the weekly product fact table."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]

    iso = d["date"].dt.isocalendar()
    d["iso_year"] = iso["year"].astype(int)
    d["iso_week_num"] = iso["week"].astype(int)
    d["iso_week"] = d["iso_year"] * 100 + d["iso_week_num"]

    fact = (
        d.groupby(["stockcode", "iso_week"], as_index=False).agg(
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            n_transactions=("transaction_id", "nunique"),
            n_customers=("customer_id", "nunique"),
        )
    )

    fact = fact.sort_values(["stockcode", "iso_week"]).reset_index(drop=True)
    fact = fact.drop(columns=["iso_year", "iso_week_num"])

    return fact
