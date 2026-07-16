"""Product Performance Analytics - Product-level metrics, ABC analysis, lifecycle, seasonality."""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_product_metrics(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive product performance metrics.

    Note: avg_price is pulled down by promotions; use median_price for
    baseline pricing decisions.
    """
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
    product_metrics["days_on_market"] = (snapshot_date - product_metrics["first_sale_date"]).dt.days
    product_metrics["days_since_last_sale"] = (
        snapshot_date - product_metrics["last_sale_date"]
    ).dt.days
    product_metrics["revenue_per_customer"] = product_metrics["total_revenue"] / product_metrics[
        "total_customers"
    ].replace(0, np.nan)
    product_metrics["revenue_per_txn"] = product_metrics["total_revenue"] / product_metrics[
        "total_transactions"
    ].replace(0, np.nan)
    product_metrics["txn_per_customer"] = product_metrics["total_transactions"] / product_metrics[
        "total_customers"
    ].replace(0, np.nan)
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
    period: str = "M",  # Monthly
) -> pd.DataFrame:
    """XYZ analysis for demand variability."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period(period)

    # Period-level demand per product
    demand = df.groupby(["stockcode", "period"])["quantity"].sum().unstack(fill_value=0)

    # CV over active periods only (exclude zero-filled periods for intermittent products)
    def _active_cv(row):
        active = row[row > 0]
        if len(active) < 2:
            return np.nan
        return active.std() / active.mean() if active.mean() > 0 else np.nan

    cv = demand.apply(_active_cv, axis=1)

    xyz = pd.DataFrame(
        {
            "stockcode": demand.index,
            "avg_demand": demand.mean(axis=1),
            "demand_std": demand.std(axis=1),
            "cv": cv,
            "n_periods_active": (demand > 0).sum(axis=1),
        }
    ).reset_index(drop=True)

    # Classify: intermittent products (few active periods) get separate label
    total_periods = demand.shape[1]
    intermittent = xyz["n_periods_active"] < total_periods * 0.5

    conditions = [
        ~intermittent & (xyz["cv"].fillna(1) <= 0.5),
        ~intermittent & (xyz["cv"].fillna(1) <= 1.0),
    ]
    choices = ["X", "Y"]
    xyz["xyz_class"] = np.select(conditions, choices, default="Z")
    xyz.loc[intermittent & (xyz["xyz_class"] == "Z"), "xyz_class"] = "I"

    return xyz


def product_lifecycle_stage(
    product_metrics: pd.DataFrame, transactions_df: pd.DataFrame, period: str = "M"
) -> pd.DataFrame:
    """Determine product lifecycle stage: Introduction, Growth, Maturity, Decline."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period(period)

    # Monthly revenue per product
    monthly_rev = df.groupby(["stockcode", "period"])["revenue"].sum().unstack(fill_value=0)

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

        # Calculate growth trend from recent months (last 3 or 25%)
        n_recent = max(3, int(len(rev_series) * 0.25))
        recent = rev_series.tail(n_recent)
        x = np.arange(len(recent))
        y = recent.values
        slope, intercept = np.polyfit(x, y, 1)
        if len(x) > 1:
            corr_matrix = np.corrcoef(x, y)
            r_value = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
        else:
            r_value = 0

        growth_rate = slope / max(np.mean(y), 1e-10) if np.mean(y) > 0 else 0

        # Stage classification based on recent trend
        if growth_rate > 0.05:
            stage = "Growth"
        elif growth_rate < -0.05:
            stage = "Decline"
        else:
            stage = "Maturity"

        lifecycle.append(
            {
                "stockcode": stockcode,
                "stage": stage,
                "trend": "Growing" if growth_rate > 0 else "Declining",
                "growth_rate": growth_rate,
                "r_squared": r_value**2,
                "peak_period": None,
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
    monthly = prod_df.set_index("date").groupby(pd.Grouper(freq="ME"))["quantity"].sum()

    if len(monthly) < 12:
        return {"has_seasonality": False, "message": "Less than 12 months of data"}

    # Detrend: divide each month by the annual mean to remove trend before averaging
    monthly_df = monthly.to_frame("qty")
    monthly_df["year"] = monthly_df.index.year
    annual_mean = monthly_df.groupby("year")["qty"].transform("mean").replace(0, np.nan)
    monthly_df["detrended"] = monthly_df["qty"] / annual_mean
    monthly_df["month"] = monthly_df.index.month
    month_avg = monthly_df.groupby("month")["detrended"].mean()

    # Coefficient of variation across months
    cv = month_avg.std() / month_avg.mean() if month_avg.mean() > 0 else 0

    # Peak month
    peak_month = month_avg.idxmax()
    trough_month = month_avg.idxmin()

    # Seasonal strength (simplified without statsmodels)
    # Use CV as a proxy for seasonal strength
    seasonal_strength = cv

    # Peak to trough ratio
    peak_to_trough_ratio = month_avg.max() / month_avg.min() if month_avg.min() > 0 else np.inf

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


def cross_sell_opportunity_matrix(transactions_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Generate cross-sell opportunity matrix for top N products."""
    from src.algorithms.fpgrowth import create_basket_matrix

    basket = create_basket_matrix(transactions_df)

    # Get top N products by frequency
    product_freq = basket.mean().sort_values(ascending=False)
    top_products = product_freq.head(top_n).index.tolist()

    # Precompute marginals once
    marginals = basket.mean()

    # Create matrix
    matrix = pd.DataFrame(index=top_products, columns=top_products, dtype=float)

    for prod_a in top_products:
        p_a = marginals[prod_a]
        for prod_b in top_products:
            if prod_a == prod_b:
                matrix.loc[prod_a, prod_b] = 1.0
            else:
                p_b = marginals[prod_b]
                p_both = ((basket[prod_a] == 1) & (basket[prod_b] == 1)).mean()
                lift = p_both / (p_a * p_b) if p_a > 0 and p_b > 0 else 0
                matrix.loc[prod_a, prod_b] = lift

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
    log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

    if len(log_price) < min_periods:
        return {"elasticity": None, "message": "Insufficient valid data after cleaning"}

    # Linear regression via numpy (avoids statsmodels dependency)
    slope, intercept = np.polyfit(log_price, log_qty, 1)
    corr_matrix = np.corrcoef(log_price, log_qty)
    r_squared = corr_matrix[0, 1] ** 2

    elasticity = slope

    # Interpretation
    if elasticity < -1:
        interpretation = "Elastic (demand sensitive to price)"
    elif elasticity < -0.05:
        interpretation = "Inelastic (demand not very sensitive)"
    elif abs(elasticity) <= 0.05:
        interpretation = "Unit elastic (quantity changes proportionally to price)"
    else:
        interpretation = "Positive elasticity — likely omitted variable bias (promotions raise both price and recorded quantity)"

    return {
        "elasticity": elasticity,
        "r_squared": r_squared,
        "interpretation": interpretation,
        "n_observations": len(weekly),
        "avg_price": weekly["avg_price"].mean(),
        "avg_weekly_qty": weekly["total_qty"].mean(),
    }


# =====================================================================
# Transaction-Only Product Dashboard Metrics
# =====================================================================

def compute_velocity(
    transactions_df: pd.DataFrame,
    period: str = "M",  # "W" for weekly, "M" for monthly
) -> pd.DataFrame:
    """Compute product velocity: units sold per active period.

    Velocity = total_quantity / number_of_active_periods
    Active period = period where product had at least 1 sale.

    Returns DataFrame: stockcode, velocity, active_periods, total_quantity
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period(period)

    # Active periods per product
    active = (
        df.groupby(["stockcode", "period"])["quantity"]
        .sum()
        .reset_index()
    )
    active_periods = active.groupby("stockcode")["period"].nunique().rename("active_periods")

    # Total quantity per product
    total_qty = df.groupby("stockcode")["quantity"].sum().rename("total_quantity")

    result = pd.concat([total_qty, active_periods], axis=1).reset_index()
    result["velocity"] = result["total_quantity"] / result["active_periods"].replace(0, np.nan)

    return result[["stockcode", "velocity", "active_periods", "total_quantity"]]


def compute_repeat_rate(
    transactions_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute repeat purchase rate per product.

    Repeat rate = % of buyers who purchased the product 2+ times.

    Returns DataFrame: stockcode, total_buyers, repeat_buyers, repeat_rate
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if snapshot_date is None:
        snapshot_date = df["date"].max()

    rows = []
    for stockcode in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == stockcode]
        buyers = prod_df["customer_id"].unique()
        n_buyers = len(buyers)
        if n_buyers == 0:
            continue

        # Count unique purchase dates per customer
        purchases_per_customer = (
            prod_df.groupby("customer_id")["date"].nunique()
        )
        repeat_buyers = (purchases_per_customer >= 2).sum()
        repeat_rate = repeat_buyers / n_buyers

        rows.append({
            "stockcode": stockcode,
            "total_buyers": n_buyers,
            "repeat_buyers": int(repeat_buyers),
            "repeat_rate": repeat_rate,
        })

    return pd.DataFrame(rows)


def compute_time_to_second_purchase(
    transactions_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute median time to second purchase per product.

    Returns DataFrame: stockcode, total_buyers, median_days_to_second, p25, p75
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if snapshot_date is None:
        snapshot_date = df["date"].max()

    rows = []
    for stockcode in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == stockcode]
        buyers = prod_df["customer_id"].unique()
        n_buyers = len(buyers)
        if n_buyers == 0:
            continue

        tt2p_list = []
        for cid, grp in prod_df.groupby("customer_id"):
            dates = grp["date"].drop_duplicates().sort_values().tolist()
            if len(dates) >= 2:
                tt2p_list.append((dates[1] - dates[0]).days)

        if tt2p_list:
            rows.append({
                "stockcode": stockcode,
                "total_buyers": n_buyers,
                "median_days_to_second": np.median(tt2p_list),
                "p25_days_to_second": np.percentile(tt2p_list, 25),
                "p75_days_to_second": np.percentile(tt2p_list, 75),
            })

    return pd.DataFrame(rows)


def compute_price_positioning_index(
    transactions_df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    """Compute price positioning index: product price vs category median.

    Index = product_median_price / category_median_price
    - Index > 1.0: premium priced vs category
    - Index < 1.0: value priced vs category
    - Index ~ 1.0: at category average

    Returns DataFrame: stockcode, category, product_price, category_price, price_index
    """
    df = transactions_df.copy()
    if "category" not in df.columns:
        return pd.DataFrame()

    # Get median price per product
    product_prices = (
        df.groupby("stockcode")[price_col]
        .median()
        .rename("product_price")
        .reset_index()
    )

    # Get product categories
    cat_map = df.groupby("stockcode")["category"].first().reset_index()
    product_prices = product_prices.merge(cat_map, on="stockcode", how="left")

    # Category median price
    cat_prices = (
        product_prices.groupby("category")["product_price"]
        .median()
        .rename("category_median_price")
        .reset_index()
    )

    product_prices = product_prices.merge(cat_prices, on="category", how="left")
    product_prices["price_index"] = (
        product_prices["product_price"] / product_prices["category_median_price"]
    ).replace([np.inf, -np.inf], np.nan)

    return product_prices[["stockcode", "category", "product_price", "category_median_price", "price_index"]]


def compute_switching_gain_loss(
    transactions_df: pd.DataFrame,
    window_days: int = 90,
) -> pd.DataFrame:
    """Compute net switching gain/loss per product.

    Gain = customers switching TO this product
    Loss = customers switching FROM this product
    Net = Gain - Loss (positive = net winner)

    Returns DataFrame: stockcode, gain_customers, loss_customers, net_gain, gain_rate, loss_rate
    """
    from src.analytics.switching import compute_switching_matrix

    switch_matrix = compute_switching_matrix(transactions_df, window_days=window_days)

    if switch_matrix.empty:
        return pd.DataFrame()

    # Gain: switches TO this product
    gain = (
        switch_matrix.groupby("to_product")["unique_customers"]
        .sum()
        .rename("gain_customers")
        .reset_index()
        .rename(columns={"to_product": "stockcode"})
    )

    # Loss: switches FROM this product
    loss = (
        switch_matrix.groupby("from_product")["unique_customers"]
        .sum()
        .rename("loss_customers")
        .reset_index()
        .rename(columns={"from_product": "stockcode"})
    )

    result = gain.merge(loss, on="stockcode", how="outer").fillna(0)
    result["net_gain"] = result["gain_customers"] - result["loss_customers"]

    # Rates (per total customers who ever bought)
    from src.analytics.product_performance import compute_product_metrics
    product_metrics = compute_product_metrics(transactions_df)
    total_customers = product_metrics.set_index("stockcode")["total_customers"]

    result["total_customers"] = result["stockcode"].map(total_customers).fillna(0)
    result["gain_rate"] = result["gain_customers"] / result["total_customers"].replace(0, np.nan)
    result["loss_rate"] = result["loss_customers"] / result["total_customers"].replace(0, np.nan)

    return result


def compute_basket_uplift(
    transactions_df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Compute basket value uplift per product.

    Wrapper around basket_metrics.compute_basket_value_uplift.
    """
    from src.analytics.basket_metrics import compute_basket_value_uplift
    return compute_basket_value_uplift(transactions_df, top_n=top_n)


def compute_product_dashboard_metrics(
    transactions_df: pd.DataFrame,
    product_lookup: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute comprehensive transaction-only product dashboard metrics.

    Combines all available metrics into a single DataFrame for the product dashboard.

    Returns DataFrame with columns:
    - Core: stockcode, product_name, category, brand
    - Revenue: total_revenue, total_quantity, avg_price, median_price
    - Penetration: basket_penetration, unique_shopper_penetration
    - Loyalty: repeat_rate, median_days_to_second, time_to_second_p75
    - Velocity: velocity, active_periods
    - Switching: gain_customers, loss_customers, net_gain, gain_rate, loss_rate
    - Basket uplift: avg_basket_with, avg_basket_without, basket_value_uplift_pct
    - Price positioning: product_price, category_median_price, price_index
    """
    from src.analytics.basket_metrics import compute_basket_penetration
    from src.analytics.product_performance import (
        compute_product_metrics,
        compute_repeat_rate,
        compute_time_to_second_purchase,
    )

    # Base product metrics
    metrics = compute_product_metrics(transactions_df)

    # Basket penetration
    penetration = compute_basket_penetration(transactions_df)
    metrics = metrics.merge(penetration, on="stockcode", how="left")

    # Repeat rate
    repeat = compute_repeat_rate(transactions_df)
    metrics = metrics.merge(repeat, on="stockcode", how="left")

    # Time to second purchase
    tt2p = compute_time_to_second_purchase(transactions_df)
    metrics = metrics.merge(tt2p, on="stockcode", how="left")

    # Velocity
    velocity = compute_velocity(transactions_df)
    metrics = metrics.merge(velocity, on="stockcode", how="left")

    # Price positioning
    price_idx = compute_price_positioning_index(transactions_df)
    metrics = metrics.merge(price_idx, on="stockcode", how="left")

    # Switching gain/loss
    switching = compute_switching_gain_loss(transactions_df)
    metrics = metrics.merge(switching, on="stockcode", how="left")

    # Basket uplift
    uplift = compute_basket_uplift(transactions_df, top_n=100)
    uplift_small = uplift[["stockcode", "avg_basket_value_with", "avg_basket_value_without", "basket_value_uplift_pct"]].rename(
        columns={"basket_value_uplift_pct": "basket_uplift_pct"}
    )
    metrics = metrics.merge(uplift_small, on="stockcode", how="left")

    # Add product names if lookup provided
    if product_lookup:
        metrics["product_name"] = metrics["stockcode"].map(product_lookup)

    return metrics
