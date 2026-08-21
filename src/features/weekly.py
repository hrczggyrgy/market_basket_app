"""Weekly aggregation utilities for the FeatureStore."""

from __future__ import annotations

import pandas as pd


def iso_week_label(date_series: pd.Series) -> pd.Series:
    """Compute ISO week label from a date series."""
    iso = date_series.dt.isocalendar()
    return iso["year"].astype(int) * 100 + iso["week"].astype(int)


def add_iso_week(df: pd.DataFrame, date_col: str = "date", week_col: str = "iso_week") -> pd.DataFrame:
    """Add an ISO week column to a DataFrame."""
    df = df.copy()
    df[week_col] = iso_week_label(df[date_col])
    return df


def weekly_agg(
    df: pd.DataFrame,
    groupby_cols: list[str],
    value_col: str,
    agg_func: str = "sum",
    additional_aggs: dict | None = None,
) -> pd.DataFrame:
    """Perform weekly aggregation over a DataFrame."""
    d = df.copy()
    if "iso_week" not in d.columns:
        d = add_iso_week(d)
    valid_funcs = {"sum", "mean", "count", "nunique"}
    if agg_func not in valid_funcs:
        raise ValueError(f"agg_func must be one of {valid_funcs}, got '{agg_func}'")
    if additional_aggs:
        for _new_col, (_src_col, func) in additional_aggs.items():
            if func not in valid_funcs:
                raise ValueError(f"agg_func for '{_new_col}' must be one of {valid_funcs}")
    agg_spec: dict[str, str] = {value_col: agg_func}
    if additional_aggs:
        for _new_col, (src_col, func) in additional_aggs.items():
            agg_spec[src_col] = func
    result = (
        d.groupby(groupby_cols + ["iso_week"], as_index=False)
            .agg(agg_spec)
    )
    result = result.sort_values(groupby_cols + ["iso_week"]).reset_index(drop=True)
    return result


def weekly_product_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the weekly product panel."""
    d = df.copy()
    d["line_revenue"] = d["price"] * d["quantity"]
    iso = d["date"].dt.isocalendar()
    d["iso_year"] = iso["year"].astype(int)
    d["iso_week_num"] = iso["week"].astype(int)
    d["iso_week"] = d["iso_year"] * 100 + d["iso_week_num"]
    panel = (
        d.groupby(["stockcode", "iso_week"], as_index=False).agg(
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            avg_price=("price", "mean"),
            n_transactions=("transaction_id", "nunique"),
            n_customers=("customer_id", "nunique"),
            active_days=("date", "nunique"),
        )
    )
    panel = panel.drop(columns=["iso_year", "iso_week_num"])
    panel = panel.sort_values(["stockcode", "iso_week"]).reset_index(drop=True)
    return panel
