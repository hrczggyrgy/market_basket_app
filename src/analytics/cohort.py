"""Cohort Analytics - Customer acquisition cohorts, retention, revenue."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from scipy import optimize
from scipy.stats import norm


def _prepare_cohort_df(transactions_df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Prepare DataFrame with cohort and period columns for cohort analysis."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    customer_cohorts = df.groupby("customer_id")["date"].min().dt.to_period(cohort_period)
    df["cohort"] = df["customer_id"].map(customer_cohorts)
    df["period"] = df["date"].dt.to_period(cohort_period)
    df["period_number"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)
    df = df[df["period_number"] >= 0]
    return df


def compute_cohorts(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
    metric: str = "retention",
    _prepared: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute cohort analysis matrix.

    Args:
        transactions_df: Transaction data with date, customer_id, etc.
        cohort_period: Period for cohort definition ('W', 'M', 'Q')
        metric: What to measure ('retention', 'revenue', 'orders', 'avg_order_value')
        _prepared: Optional pre-prepared DataFrame (internal use, avoids re-computation)

    Returns:
        Cohort matrix with periods as columns, cohorts as rows
    """
    if _prepared is not None:
        df = _prepared
    else:
        df = _prepare_cohort_df(transactions_df, cohort_period)

    if metric == "retention":
        # Count unique customers per cohort per period
        cohort_data = df.groupby(["cohort", "period_number"])["customer_id"].nunique().reset_index()
        cohort_data.columns = ["cohort", "period_number", "customers"]

        # Get cohort sizes (period 0) — vectorized merge
        cohort_sizes = cohort_data[cohort_data["period_number"] == 0][
            ["cohort", "customers"]
        ].rename(columns={"customers": "cohort_size"})
        cohort_data = cohort_data.merge(cohort_sizes, on="cohort", how="left")
        cohort_data["retention_rate"] = (
            cohort_data["customers"] / cohort_data["cohort_size"]
        ).fillna(0)

        # Pivot to matrix
        matrix = cohort_data.pivot(index="cohort", columns="period_number", values="retention_rate")

    elif metric == "revenue":
        cohort_data = df.groupby(["cohort", "period_number"])["revenue"].sum().reset_index()
        cohort_sizes = (
            df[df["period_number"] == 0]
            .groupby("cohort")["customer_id"]
            .nunique()
            .rename("cohort_size")
        )
        cohort_data = cohort_data.merge(cohort_sizes, on="cohort", how="left")
        cohort_data["revenue_per_customer"] = (
            cohort_data["revenue"] / cohort_data["cohort_size"]
        ).fillna(0)
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="revenue_per_customer"
        )

    elif metric == "orders":
        cohort_data = (
            df.groupby(["cohort", "period_number"])["transaction_id"].nunique().reset_index()
        )
        cohort_data.columns = ["cohort", "period_number", "orders"]
        cohort_sizes = (
            df[df["period_number"] == 0]
            .groupby("cohort")["customer_id"]
            .nunique()
            .rename("cohort_size")
        )
        cohort_data = cohort_data.merge(cohort_sizes, on="cohort", how="left")
        cohort_data["orders_per_customer"] = (
            cohort_data["orders"] / cohort_data["cohort_size"]
        ).fillna(0)
        matrix = cohort_data.pivot(
            index="cohort", columns="period_number", values="orders_per_customer"
        )

    elif metric == "avg_order_value":
        cohort_data = (
            df.groupby(["cohort", "period_number"])
            .agg(revenue=("revenue", "sum"), orders=("transaction_id", "nunique"))
            .reset_index()
        )
        cohort_data["avg_order_value"] = cohort_data["revenue"] / cohort_data["orders"].replace(
            0, np.nan
        )
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


def compute_cohort_sizes(transactions_df: pd.DataFrame, cohort_period: str = "M") -> pd.DataFrame:
    """Get cohort sizes and basic stats."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    customer_cohorts = df.groupby("customer_id")["date"].min().dt.to_period(cohort_period)
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
        )
        .reset_index()
    )

    period_stats["avg_order_value"] = period_stats["revenue"] / period_stats["orders"].replace(
        0, np.nan
    )
    period_stats["items_per_order"] = period_stats["total_items"] / period_stats["orders"].replace(
        0, np.nan
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
        )
        .reset_index()
    )

    yoy_stats["avg_order_value"] = yoy_stats["revenue"] / yoy_stats["orders"].replace(0, np.nan)

    yoy_stats["period"] = (
        yoy_stats["year"].astype(str) + "-" + yoy_stats["month"].astype(str).str.zfill(2)
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
                if f"{metric}_{prev}" in result.columns and f"{metric}_{curr}" in result.columns:
                    result[f"{metric}_yoy_pct_{curr}_vs_{prev}"] = (
                        (result[f"{metric}_{curr}"] - result[f"{metric}_{prev}"])
                        / result[f"{metric}_{prev}"].replace(0, np.nan)
                        * 100
                    ).fillna(0)
        return result

    return pd.DataFrame()


def cohort_comparison_summary(
    transactions_df: pd.DataFrame, cohort_period: str = "M", max_periods: int = 12
) -> Dict:
    """Generate summary statistics for cohort analysis."""
    # Prepare cohort data once, reuse for both metrics
    prepared = _prepare_cohort_df(transactions_df, cohort_period)

    retention_matrix = compute_cohorts(
        transactions_df, cohort_period=cohort_period, metric="retention", _prepared=prepared
    )
    revenue_matrix = compute_cohorts(
        transactions_df, cohort_period=cohort_period, metric="revenue", _prepared=prepared
    )

    summary = {
        "n_cohorts": len(retention_matrix),
        "cohort_period": cohort_period,
        "avg_retention_period_1": (
            retention_matrix["Period 1"].mean() if "Period 1" in retention_matrix.columns else 0
        ),
        "avg_retention_period_3": (
            retention_matrix["Period 3"].mean() if "Period 3" in retention_matrix.columns else 0
        ),
        "avg_retention_period_6": (
            retention_matrix["Period 6"].mean() if "Period 6" in retention_matrix.columns else 0
        ),
        "avg_retention_period_12": (
            retention_matrix["Period 12"].mean() if "Period 12" in retention_matrix.columns else 0
        ),
        "avg_revenue_per_customer_period_1": (
            revenue_matrix["Period 1"].mean() if "Period 1" in revenue_matrix.columns else 0
        ),
        "avg_revenue_per_customer_period_3": (
            revenue_matrix["Period 3"].mean() if "Period 3" in revenue_matrix.columns else 0
        ),
        "best_cohort_retention": (
            retention_matrix.mean(axis=1).idxmax() if len(retention_matrix) > 0 else None
        ),
        "best_cohort_revenue": (
            revenue_matrix.mean(axis=1).idxmax() if len(revenue_matrix) > 0 else None
        ),
    }

    return summary


def compute_cohort_ltv_curve(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
    max_periods: int = 12,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute Cohort Lifetime Value (LTV) curve with power-law fit.

    Fits LTV(t) = a * t^b to cumulative revenue per cohort customer
    from period 0 to period N.

    Returns:
        DataFrame with fitted LTV curve per cohort
        Dict with fit parameters (a, b, R²) per cohort
    """
    prepared = _prepare_cohort_df(transactions_df, cohort_period)
    revenue_matrix = compute_cohorts(
        transactions_df, cohort_period=cohort_period, metric="revenue", _prepared=prepared
    )

    # Limit to max_periods
    period_cols = [c for c in revenue_matrix.columns if c.startswith("Period ")]
    if len(period_cols) > 12:
        period_cols = period_cols[:max_periods]
    revenue_matrix = revenue_matrix[period_cols]

    ltv_curves = []
    fit_params = {}

    for cohort_idx in revenue_matrix.index:
        cohort_revenue = revenue_matrix.loc[cohort_idx].values
        periods = np.arange(1, len(cohort_revenue) + 1)

        # Cumulative revenue per customer
        cum_revenue = np.cumsum(cohort_revenue)

        # Fit power law: LTV(t) = a * t^b
        # log(LTV) = log(a) + b * log(t)
        valid = cum_revenue > 0
        if valid.sum() < 3:
            fit_params = {"a": 0, "b": 0, "r2": 0}
            ltv_curve = np.zeros_like(cum_revenue)
        else:
            log_t = np.log(periods[valid])
            log_ltv = np.log(cum_revenue[valid])
            coeffs = np.polyfit(log_t, log_ltv, 1)
            b = coeffs[0]
            a = np.exp(coeffs[1])
            # R²
            y_pred = a * periods**b
            ss_res = np.sum((cum_revenue - y_pred) ** 2)
            ss_tot = np.sum((cum_revenue - np.mean(cum_revenue)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fit_params = {"a": a, "b": b, "r2": r2}
            ltv_curve = a * periods**b

        fit_params[cohort_idx] = fit_params
        ltv_curves.append(ltv_curve)

    ltv_df = pd.DataFrame(
        np.column_stack(ltv_curves).T if ltv_curves else np.array([]),
        index=revenue_matrix.index,
        columns=[f"Period {i + 1}" for i in range(max_periods)],
    )

    return ltv_df, fit_params


def compute_cohort_decay_rate(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
) -> pd.DataFrame:
    """
    Compute cohort decay rate λ by fitting R(t) = R₀ * e^(-λt) to retention series.

    Returns DataFrame with decay rate λ per cohort.
    """
    prepared = _prepare_cohort_df(transactions_df, cohort_period)
    retention_matrix = compute_cohorts(
        transactions_df, cohort_period=cohort_period, metric="retention", _prepared=prepared
    )

    decay_rates = {}
    for cohort_idx in retention_matrix.index:
        retention = retention_matrix.loc[cohort_idx].values
        periods = np.arange(len(retention))

        # Fit exponential decay: R(t) = R₀ * exp(-λt)
        # log(R) = log(R₀) - λt
        valid = retention > 0
        if valid.sum() < 3:
            decay_rates[cohort_idx] = {
                "lambda": 0,
                "r2": 0,
                "r0": retention[0] if len(retention) > 0 else 0,
            }
            continue

        log_r = np.log(retention[valid])
        t_valid = np.arange(len(retention))[valid]
        coeffs = np.polyfit(t_valid, log_r, 1)
        lambda_val = -coeffs[0]  # negative slope = decay rate
        r0 = np.exp(coeffs[1])

        # R²
        y_pred = np.exp(coeffs[1]) * np.exp(-lambda_val * np.arange(len(retention)))
        ss_res = np.sum((retention - y_pred) ** 2)
        ss_tot = np.sum((retention - np.mean(retention)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        decay_rates[cohort_idx] = {"lambda": max(0, lambda_val), "r2": r2, "r0": r0}

    return pd.DataFrame.from_dict(decay_rates, orient="index")


def compute_reactivation_rate(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
    inactivity_threshold: int = 2,
) -> pd.DataFrame:
    """
    Compute reactivation rate: among customers who churned (≥ inactivity_threshold periods inactive),
    what fraction returned in a subsequent period.

    Returns reactivation rate per cohort.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period(cohort_period)

    # Get all customer-period activity
    activity = df.groupby(["customer_id", "period"]).size().reset_index(name="active")

    # For each customer, find churn and reactivation
    reactivation_data = []
    for cust_id, cust_data in activity.groupby("customer_id"):
        cust_data = cust_data.sort_values("period")
        periods = cust_data["period"].values

        # Find gaps
        for i in range(1, len(periods)):
            gap = (periods[i] - periods[i - 1]).n
            if gap >= inactivity_threshold:
                # Check if customer returned after this gap
                returned = (periods[i + 1 :] > periods[i]).any()
                reactivation_data.append(
                    {
                        "customer_id": cust_id,
                        "churned_after_period": periods[i - 1],
                        "reactivated": returned,
                    }
                )

    if not reactivation_data:
        return pd.DataFrame()

    react_df = pd.DataFrame(reactivation_data)
    react_df["churned_period"] = react_df["churned_after_period"].astype(str)

    # Merge with cohort info
    from_date = df.groupby("customer_id")["date"].min().dt.to_period("M")
    react_df["cohort"] = react_df["customer_id"].map(from_date).astype(str)

    # Compute reactivation rate per cohort
    cohort_rates = (
        react_df.groupby("cohort")["reactivated"]
        .agg(
            reactivated="sum",
            churned="count",
        )
        .reset_index()
    )
    cohort_rates["reactivation_rate"] = cohort_rates["reactivated"] / cohort_rates["churned"]

    return cohort_rates


def compute_waterfall_decomposition(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
) -> pd.DataFrame:
    """
    Decompose period-over-period customer change into:
    - New customers (first purchase in period)
    - Retained customers (active in both periods)
    - Reactivated customers (inactive in t-1, active in t)
    - Churned customers (active in t-1, inactive in t)

    Returns period-over-period waterfall decomposition.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period(cohort_period)

    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return pd.DataFrame()

    waterfall_data = []

    for i in range(1, len(periods)):
        period_t = periods[i]
        period_t_1 = periods[i - 1]

        cust_t = set(df[df["period"] == period_t]["customer_id"])
        cust_t_1 = set(df[df["period"] == period_t_1]["customer_id"])

        new = cust_t - cust_t_1
        retained = cust_t & cust_t_1
        reactivated = (
            (cust_t - cust_t_1)
            & set(
                df[df["period"] < period_t_1]
                .groupby("customer_id")["period"]
                .max()[
                    df[df["period"] < period_t_1].groupby("customer_id")["period"].max()
                    == period_t_1
                ]
                .index
            )
            if i > 1
            else set()
        )
        churned = cust_t_1 - cust_t

        waterfall_data.append(
            {
                "period": str(period_t),
                "new_customers": len(new),
                "retained_customers": len(retained),
                "reactivated_customers": len(reactivated),
                "churned_customers": len(churned),
                "net_change": len(cust_t) - len(cust_t_1),
            }
        )

    return pd.DataFrame(waterfall_data)


def compute_cohort_p_alive(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
) -> pd.DataFrame:
    """
    Compute BG/NBD P(alive) for each customer in each cohort.

    Uses lifetimes library BG/NBD model to compute probability customer is still alive.
    """
    if not hasattr(compute_cohort_p_alive, "_lifetimes_available"):
        try:
            from lifetimes import BetaGeoFitter
            from lifetimes.utils import summary_data_from_transaction_data

            compute_cohort_p_alive._lifetimes_available = True
        except ImportError:
            compute_cohort_p_alive._lifetimes_available = False

    if not compute_cohort_p_alive._lifetimes_available:
        return pd.DataFrame()

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Prepare summary data for BG/NBD
    observation_period_end = df["date"].max()
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue" if "revenue" in df.columns else None,
        observation_period_end=observation_period_end,
        freq="D",
    )

    # Fit BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # Compute P(alive) for each customer
    p_alive = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    # Merge with cohort info
    summary["p_alive"] = p_alive
    summary = summary.reset_index()
    summary["cohort"] = (
        summary["customer_id"]
        .map(transactions_df.groupby("customer_id")["date"].min().dt.to_period("M"))
        .astype(str)
    )

    return summary[["customer_id", "cohort", "p_alive", "frequency", "recency", "T"]]
