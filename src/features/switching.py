"""Switching feature tables: customer sequence and switching analysis."""

from __future__ import annotations

import pandas as pd


def build_customer_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer sequence table for switching analysis."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]
    d = d.sort_values(["customer_id", "date"]).copy()
    d["prev_date"] = d.groupby("customer_id")["date"].shift(1)
    d["gap_days"] = (d["date"] - d["prev_date"]).dt.days
    d["is_new_sequence"] = (d["gap_days"] > 7) | (d["prev_date"].isna())
    d["sequence_id"] = d.groupby("customer_id")["is_new_sequence"].cumsum()
    seq = (
        d.groupby(["customer_id", "sequence_id"], as_index=False).agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_transactions=("transaction_id", "nunique"),
            n_products=("stockcode", "nunique"),
            revenue=("line_revenue", "sum"),
        )
    )
    seq["sequence_length_days"] = (seq["end_date"] - seq["start_date"]).dt.days + 1
    seq = seq.sort_values(["customer_id", "start_date"]).reset_index(drop=True)
    return seq
