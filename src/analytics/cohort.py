"""Cohort Analytics - Customer acquisition cohorts, retention, revenue."""

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_cohorts(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",  # Monthly cohorts (use 'M' for Period, 'ME' for Grouper)
    metric: str = "retention",  # 'retention', 'revenue', 'orders'
) -> pd.DataFrame:
    """
    Compute cohort analysis matrix.

    Args:
        transactions_df: Transaction data with date, customer_id, etc.
        cohort_period: Period for cohort definition ('W', 'M', 'Q')
        metric: What to measure ('retention', 'revenue', 'orders', 'avg_order_value')

    Returns:
        Cohort matrix with periods as columns, cohorts as rows
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Determine cohort for each customer (first purchase period)
    customer_cohorts = (
        df.groupby("customer_id")["date"].min().dt.to_period(cohort_period)
    )
    df["cohort"] = df["customer_id"].map(customer_cohorts)

    # Determine period number for each transaction
    df["period"] = df["date"].dt.to_period(cohort_period)
    df["period_number"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)

    # Filter out period -1 (shouldn't happen but just in case)
    df = df[df["period_number"] >= 0]

    if metric == "retention":
        # Count unique customers per cohort per period
        cohort_data = (
            df.groupby(["cohort", "period_number"])["customer_id"]
            .nunique()
            .reset_index()
        )
        cohort_data.columns = ["cohort", "period_number", "customers"]

        # Get cohort sizes (period 0)
        cohort_sizes = cohort_data[cohort_data["period_number"] == 0].set_index(
            "cohort"
        )["customers"]

        # Calculate retention rate
        cohort_data["retention_rate"] = cohort_data.apply(
            lambda row: (
                row["customers"] / cohort_sizes[row["cohort"]]
                if row["cohort"] in cohort_sizes.index
                else 0
            ),
            axis=1,
        )

        # Pivot to matrix
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="retention_rate"
        )

    elif metric == "revenue":
        cohort_data = (
            df.groupby(["cohort", "period_number"])["revenue"].sum().reset_index()
        )
        cohort_sizes = (
            df[df["period_number"] == 0].groupby("cohort")["customer_id"].nunique()
        )
        cohort_data["revenue_per_customer"] = cohort_data.apply(
            lambda row: (
                row["revenue"] / cohort_sizes[row["cohort"]]
                if row["cohort"] in cohort_sizes.index
                else 0
            ),
            axis=1,
        )
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="revenue_per_customer"
        )

    elif metric == "orders":
        cohort_data = (
            df.groupby(["cohort", "period_number"])["transaction_id"]
            .nunique()
            .reset_index()
        )
        cohort_data.columns = ["cohort", "period_number", "orders"]
        cohort_sizes = (
            df[df["period_number"] == 0].groupby("cohort")["customer_id"].nunique()
        )
        cohort_data["orders_per_customer"] = cohort_data.apply(
            lambda row: (
                row["orders"] / cohort_sizes[row["cohort"]]
                if row["cohort"] in cohort_sizes.index
                else 0
            ),
            axis=1,
        )
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="orders_per_customer"
        )

    elif metric == "avg_order_value":
        cohort_data = (
            df.groupby(["cohort", "period_number"])
            .agg(revenue=("revenue", "sum"), orders=("transaction_id", "nunique"))
            .reset_index()
        )
        cohort_data["avg_order_value"] = cohort_data["revenue"] / cohort_data[
            "orders"
        ].replace(0, np.nan)
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="avg_order_value"
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort columns
    matrix = matrix.sort_index(axis=1)
    matrix.index = matrix.index.astype(str)
    matrix.columns = [f"Period {int(c)}" for c in matrix.columns]

    return matrix


def compute_cohort_sizes(
    transactions_df: pd.DataFrame, cohort_period: str = "M"
) -> pd.DataFrame:
    """Get cohort sizes and basic stats."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    customer_cohorts = (
        df.groupby("customer_id")["date"].min().dt.to_period(cohort_period)
    )
    df["cohort"] = df["customer_id"].map(customer_cohorts)

    cohort_stats = (
        df[df["date"].dt.to_period(cohort_period) == df["cohort"]]
        .groupby("cohort")
        .agg(
            customers=("customer_id", "nunique"),
            first_orders=("transaction_id", "nunique"),
            first_revenue=("revenue", "sum"),
            avg_first_order=("revenue", "mean"),
        )
        .reset_index()
    )

    cohort_stats["cohort"] = cohort_stats["cohort"].astype(str)

    return cohort_stats


def period_over_period_comparison(
    transactions_df: pd.DataFrame, period: str = "M", metrics: List[str] = None
) -> pd.DataFrame:
    """Compare metrics period over period (PoP)."""
    if metrics is None:
        metrics = [
            "revenue",
            "orders",
            "customers",
            "avg_order_value",
            "items_per_order",
        ]

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period(period)

    period_stats = (
        df.groupby("period")
        .agg(
            revenue=("revenue", "sum"),
            orders=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
            total_items=("quantity", "sum"),
            avg_order_value=(
                "revenue",
                lambda x: x.sum() / df.loc[x.index, "transaction_id"].nunique(),
            ),
            items_per_order=(
                "quantity",
                lambda x: x.sum() / df.loc[x.index, "transaction_id"].nunique(),
            ),
        )
        .reset_index()
    )

    period_stats["period"] = period_stats["period"].astype(str)

    # Calculate PoP changes
    for metric in [
        "revenue",
        "orders",
        "customers",
        "avg_order_value",
        "items_per_order",
    ]:
        if metric in period_stats.columns:
            period_stats[f"{metric}_pop_pct"] = period_stats[metric].pct_change() * 100

    return period_stats


def year_over_year_comparison(
    transactions_df: pd.DataFrame, metrics: List[str] = None
) -> pd.DataFrame:
    """Compare same periods across years (YoY)."""
    if metrics is None:
        metrics = ["revenue", "orders", "customers", "avg_order_value"]

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    yoy_stats = (
        df.groupby(["year", "month"])
        .agg(
            revenue=("revenue", "sum"),
            orders=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
            avg_order_value=(
                "revenue",
                lambda x: x.sum() / x.index.map(df["transaction_id"].nunique),
            ),
        )
        .reset_index()
    )

    yoy_stats["period"] = (
        yoy_stats["year"].astype(str)
        + "-"
        + yoy_stats["month"].astype(str).str.zfill(2)
    )

    # Pivot for YoY comparison
    pivot_data = []
    for metric in ["revenue", "orders", "customers", "avg_order_value"]:
        if metric in yoy_stats.columns:
            pivot = yoy_stats.pivot(index="month", columns="year", values=metric)
            pivot.columns = [f"{metric}_{int(c)}" for c in pivot.columns]
            pivot_data.append(pivot)

    if pivot_data:
        result = pd.concat(pivot_data, axis=1).reset_index()
        result["month_name"] = result["month"].map(
            {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
        )

        # Calculate YoY changes
        years = sorted([c for c in result.columns if c.startswith("revenue_")])
        for i in range(1, len(years)):
            prev = years[i - 1]
            curr = years[i]
            for metric in ["revenue", "orders", "customers", "avg_order_value"]:
                if (
                    f"{metric}_{prev}" in result.columns
                    and f"{metric}_{curr}" in result.columns
                ):
                    result[f"{metric}_yoy_pct_{curr}_vs_{prev}"] = (
                        (result[f"{metric}_{curr}"] - result[f"{metric}_{prev}"])
                        / result[f"{metric}_{prev}"]
                        * 100
                    )
        return result

    return pd.DataFrame()


def cohort_comparison_summary(
    transactions_df: pd.DataFrame, cohort_period: str = "M", max_periods: int = 12
) -> Dict:
    """Generate summary statistics for cohort analysis."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Get retention matrix
    retention_matrix = compute_cohorts(
        df, cohort_period=cohort_period, metric="retention"
    )
    revenue_matrix = compute_cohorts(df, cohort_period=cohort_period, metric="revenue")

    # Limit to max_periods
    if len(retention_matrix.columns) > max_periods + 1:
        retention_matrix = retention_matrix.iloc[:, : max_periods + 1]
        revenue_matrix = revenue_matrix.iloc[:, : max_periods + 1]

    summary = {
        "n_cohorts": len(retention_matrix),
        "cohort_period": cohort_period,
        "avg_retention_period_1": (
            retention_matrix["Period 1"].mean()
            if "Period 1" in retention_matrix.columns
            else 0
        ),
        "avg_retention_period_3": (
            retention_matrix["Period 3"].mean()
            if "Period 3" in retention_matrix.columns
            else 0
        ),
        "avg_retention_period_6": (
            retention_matrix["Period 6"].mean()
            if "Period 6" in retention_matrix.columns
            else 0
        ),
        "avg_retention_period_12": (
            retention_matrix["Period 12"].mean()
            if "Period 12" in retention_matrix.columns
            else 0
        ),
        "avg_revenue_per_customer_period_1": (
            revenue_matrix["Period 1"].mean()
            if "Period 1" in revenue_matrix.columns
            else 0
        ),
        "avg_revenue_per_customer_period_3": (
            revenue_matrix["Period 3"].mean()
            if "Period 3" in revenue_matrix.columns
            else 0
        ),
        "best_cohort_retention": (
            retention_matrix.mean(axis=1).idxmax()
            if len(retention_matrix) > 0
            else None
        ),
        "best_cohort_revenue": (
            revenue_matrix.mean(axis=1).idxmax() if len(revenue_matrix) > 0 else None
        ),
    }

    return summary
