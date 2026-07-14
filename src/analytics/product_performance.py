"""Product Performance Analytics - Product-level metrics, ABC analysis, lifecycle, seasonality."""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def compute_product_metrics(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive product performance metrics."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max()

    # Product-level aggregation
    product_metrics = (
        df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            total_transactions=("transaction_id", "nunique"),
            total_customers=("customer_id", "nunique"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
            avg_qty_per_txn=("quantity", "mean"),
            first_sale_date=("date", "min"),
            last_sale_date=("date", "max"),
        )
        .reset_index()
    )

    # Category and brand if available
    if "category" in df.columns:
        cat_info = df.groupby("stockcode")["category"].first().reset_index()
        product_metrics = product_metrics.merge(cat_info, on="stockcode", how="left")

    if "brand" in df.columns:
        brand_info = df.groupby("stockcode")["brand"].first().reset_index()
        product_metrics = product_metrics.merge(brand_info, on="stockcode", how="left")

    # Derived metrics
    product_metrics["days_on_market"] = (
        snapshot_date - product_metrics["first_sale_date"]
    ).dt.days
    product_metrics["days_since_last_sale"] = (
        snapshot_date - product_metrics["last_sale_date"]
    ).dt.days
    product_metrics["revenue_per_customer"] = product_metrics[
        "total_revenue"
    ] / product_metrics["total_customers"].replace(0, np.nan)
    product_metrics["revenue_per_txn"] = product_metrics[
        "total_revenue"
    ] / product_metrics["total_transactions"].replace(0, np.nan)
    product_metrics["txn_per_customer"] = product_metrics[
        "total_transactions"
    ] / product_metrics["total_customers"].replace(0, np.nan)
    product_metrics["sell_through_rate"] = (
        product_metrics["total_transactions"]
        / product_metrics["days_on_market"].replace(0, np.nan)
        * 30
    )  # monthly

    # Price volatility
    product_metrics["price_cv"] = product_metrics["price_std"] / product_metrics[
        "avg_price"
    ].replace(0, np.nan)

    return product_metrics


def abc_analysis(
    product_metrics: pd.DataFrame,
    revenue_col: str = "total_revenue",
    n_classes: int = 3,
) -> pd.DataFrame:
    """Perform ABC analysis on products based on revenue contribution."""
    df = product_metrics.sort_values(revenue_col, ascending=False).copy()
    df["cum_revenue"] = df[revenue_col].cumsum()
    df["cum_revenue_pct"] = df["cum_revenue"] / df[revenue_col].sum() * 100
    df["cum_quantity_pct"] = (
        (df["total_quantity"].cumsum() / df["total_quantity"].sum() * 100)
        if "total_quantity" in df.columns
        else np.nan
    )
    df["cum_customer_pct"] = (
        (df["total_customers"].cumsum() / df["total_customers"].sum() * 100)
        if "total_customers" in df.columns
        else np.nan
    )

    if n_classes == 3:
        # Classic ABC: A=80%, B=15%, C=5%
        conditions = [
            df["cum_revenue_pct"] <= 80,
            df["cum_revenue_pct"] <= 95,
        ]
        choices = ["A", "B"]
        df["abc_class"] = np.select(conditions, choices, default="C")
    elif n_classes == 4:
        # Extended: A=70%, B=20%, C=8%, D=2%
        conditions = [
            df["cum_revenue_pct"] <= 70,
            df["cum_revenue_pct"] <= 90,
            df["cum_revenue_pct"] <= 98,
        ]
        choices = ["A", "B", "C"]
        df["abc_class"] = np.select(conditions, choices, default="D")
    else:
        # Percentile-based
        df["abc_class"] = pd.qcut(
            df["cum_revenue_pct"],
            q=n_classes,
            labels=[chr(65 + i) for i in range(n_classes)],
            duplicates="drop",
        )

    return df


def xyz_analysis(
    transactions_df: pd.DataFrame,
    period: str = "ME",  # Monthly
) -> pd.DataFrame:
    """XYZ analysis for demand variability."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period(period)

    # Period-level demand per product
    demand = df.groupby(["stockcode", "period"])["quantity"].sum().unstack(fill_value=0)

    # Coefficient of variation
    demand_mean = demand.mean(axis=1)
    demand_std = demand.std(axis=1)
    cv = demand_std / demand_mean.replace(0, np.nan)

    xyz = pd.DataFrame(
        {
            "stockcode": demand.index,
            "avg_demand": demand_mean,
            "demand_std": demand_std,
            "cv": cv,
            "n_periods_active": (demand > 0).sum(axis=1),
        }
    ).reset_index(drop=True)

    # Classify
    conditions = [
        xyz["cv"] <= 0.5,
        xyz["cv"] <= 1.0,
    ]
    choices = ["X", "Y"]
    xyz["xyz_class"] = np.select(conditions, choices, default="Z")

    return xyz


def product_lifecycle_stage(
    product_metrics: pd.DataFrame, transactions_df: pd.DataFrame, period: str = "ME"
) -> pd.DataFrame:
    """Determine product lifecycle stage: Introduction, Growth, Maturity, Decline."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period(period)

    # Monthly revenue per product
    monthly_rev = (
        df.groupby(["stockcode", "period"])["revenue"].sum().unstack(fill_value=0)
    )

    lifecycle = []
    for stockcode in product_metrics["stockcode"]:
        if stockcode not in monthly_rev.index:
            lifecycle.append(
                {
                    "stockcode": stockcode,
                    "stage": "Unknown",
                    "trend": "Unknown",
                    "growth_rate": 0,
                }
            )
            continue

        rev_series = monthly_rev.loc[stockcode]
        rev_series = rev_series[rev_series > 0]

        if len(rev_series) < 2:
            lifecycle.append(
                {
                    "stockcode": stockcode,
                    "stage": "Introduction",
                    "trend": "Insufficient data",
                    "growth_rate": 0,
                }
            )
            continue

        # Calculate growth trend
        x = np.arange(len(rev_series))
        y = rev_series.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        growth_rate = slope / np.mean(y) if np.mean(y) > 0 else 0

        # Peak detection
        peak_idx = np.argmax(y)
        peak_period = peak_idx / len(y)

        # Stage classification
        if peak_period < 0.3 and growth_rate > 0.05:
            stage = "Introduction"
        elif growth_rate > 0.05:
            stage = "Growth"
        elif abs(growth_rate) <= 0.05:
            stage = "Maturity"
        else:
            stage = "Decline"

        lifecycle.append(
            {
                "stockcode": stockcode,
                "stage": stage,
                "trend": "Growing" if growth_rate > 0 else "Declining",
                "growth_rate": growth_rate,
                "r_squared": r_value**2,
                "peak_period": peak_period,
                "months_active": len(rev_series),
            }
        )

    lifecycle_df = pd.DataFrame(lifecycle)
    return product_metrics.merge(lifecycle_df, on="stockcode", how="left")


def product_seasonality(transactions_df: pd.DataFrame, product_id: str) -> Dict:
    """Analyze seasonality for a specific product."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    prod_df = df[df["stockcode"] == product_id].copy()

    if len(prod_df) < 10:
        return {"has_seasonality": False, "message": "Insufficient data"}

    # Monthly aggregation
    monthly = prod_df.set_index("date").groupby(pd.Grouper(freq="M"))["quantity"].sum()

    if len(monthly) < 12:
        return {"has_seasonality": False, "message": "Less than 12 months of data"}

    # Simple seasonality detection: month-over-month pattern
    monthly.index = monthly.index.month
    month_avg = monthly.groupby(monthly.index).mean()

    # Coefficient of variation across months
    cv = month_avg.std() / month_avg.mean() if month_avg.mean() > 0 else 0

    # Peak month
    peak_month = month_avg.idxmax()
    trough_month = month_avg.idxmin()

    # Seasonal strength (simplified without statsmodels)
    # Use CV as a proxy for seasonal strength
    seasonal_strength = cv

    # Peak to trough ratio
    peak_to_trough_ratio = (
        month_avg.max() / month_avg.min() if month_avg.min() > 0 else np.inf
    )

    has_seasonality = seasonal_strength > 0.3

    return {
        "has_seasonality": has_seasonality,
        "seasonal_strength": seasonal_strength,
        "cv_across_months": cv,
        "peak_month": int(peak_month),
        "trough_month": int(trough_month),
        "monthly_pattern": month_avg.to_dict(),
        "peak_to_trough_ratio": peak_to_trough_ratio,
    }


def product_affinity_score(
    transactions_df: pd.DataFrame, target_product: str, min_support: float = 0.001
) -> pd.DataFrame:
    """Calculate affinity scores between target product and all others."""
    from src.algorithms.fpgrowth import create_basket_matrix

    basket = create_basket_matrix(transactions_df)

    if target_product not in basket.columns:
        return pd.DataFrame()

    # Get transactions with target product
    target_transactions = basket[basket[target_product] == 1]

    if len(target_transactions) == 0:
        return pd.DataFrame()

    # Calculate co-occurrence
    other_products = target_transactions.drop(columns=[target_product])

    results = []
    p_target = basket[target_product].mean()

    for product in other_products.columns:
        p_product = basket[product].mean()
        p_both = (target_transactions[product] == 1).mean()

        if p_target > 0 and p_product > 0:
            lift = p_both / (p_target * p_product)
            confidence = p_both / p_target
            leverage = p_both - (p_target * p_product)

            if p_both >= min_support:
                results.append(
                    {
                        "product": product,
                        "support": p_both,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": leverage,
                        "p_target": p_target,
                        "p_product": p_product,
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("lift", ascending=False)
    return df


def cross_sell_opportunity_matrix(
    transactions_df: pd.DataFrame, top_n: int = 50
) -> pd.DataFrame:
    """Generate cross-sell opportunity matrix for top N products."""
    from src.algorithms.fpgrowth import create_basket_matrix

    basket = create_basket_matrix(transactions_df)

    # Get top N products by frequency
    product_freq = basket.mean().sort_values(ascending=False)
    top_products = product_freq.head(top_n).index.tolist()

    # Create matrix
    matrix = pd.DataFrame(index=top_products, columns=top_products, dtype=float)

    for i, prod_a in enumerate(top_products):
        for prod_b in top_products:
            if prod_a == prod_b:
                matrix.loc[prod_a, prod_b] = 1.0
            else:
                p_a = basket[prod_a].mean()
                p_b = basket[prod_b].mean()
                p_both = ((basket[prod_a] == 1) & (basket[prod_b] == 1)).mean()

                if p_a > 0 and p_b > 0:
                    lift = p_both / (p_a * p_b)
                    matrix.loc[prod_a, prod_b] = lift
                else:
                    matrix.loc[prod_a, prod_b] = 0

    return matrix.astype(float)


def price_elasticity_analysis(
    transactions_df: pd.DataFrame, product_id: str, min_periods: int = 10
) -> Dict:
    """Estimate price elasticity for a product."""
    prod_df = transactions_df[transactions_df["stockcode"] == product_id].copy()
    prod_df["date"] = pd.to_datetime(prod_df["date"])
    prod_df["revenue"] = prod_df["price"] * prod_df["quantity"]

    if len(prod_df) < min_periods:
        return {"elasticity": None, "message": "Insufficient data"}

    # Aggregate to weekly
    weekly = (
        prod_df.set_index("date")
        .groupby(pd.Grouper(freq="W"))
        .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
        .dropna()
    )

    if len(weekly) < min_periods:
        return {"elasticity": None, "message": "Insufficient weekly data"}

    # Log-log regression
    log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
    log_qty = np.log(
        weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna()
    )

    if len(log_price) < min_periods:
        return {"elasticity": None, "message": "Insufficient valid data after cleaning"}

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)

    elasticity = slope

    # Interpretation
    if elasticity < -1:
        interpretation = "Elastic (demand sensitive to price)"
    elif elasticity < 0:
        interpretation = "Inelastic (demand not very sensitive)"
    elif elasticity == 0:
        interpretation = "Unit elastic"
    else:
        interpretation = "Positive elasticity (unusual for normal goods)"

    return {
        "elasticity": elasticity,
        "r_squared": r_value**2,
        "p_value": p_value,
        "interpretation": interpretation,
        "n_observations": len(weekly),
        "avg_price": weekly["avg_price"].mean(),
        "avg_weekly_qty": weekly["total_qty"].mean(),
    }
