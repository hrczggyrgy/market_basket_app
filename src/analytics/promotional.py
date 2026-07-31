"""Promotional Lift Analysis - Measure impact of promotions on sales and customer behavior."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def detect_promotions(
    transactions_df: pd.DataFrame,
    price_change_threshold: float = 0.15,
    min_duration_days: int = 3,
    max_duration_days: int = 30,
    gap_threshold_days: int = 1,
) -> pd.DataFrame:
    """
    Detect promotional periods based on price drops.

    Args:
        transactions_df: Transaction data
        price_change_threshold: Minimum price drop to consider as promotion (e.g., 0.15 = 15%)
        min_duration_days: Minimum days for a promotion
        max_duration_days: Maximum days for a promotion
        gap_threshold_days: Max gap between transactions to merge into same promo period (default 1)

    Returns:
        DataFrame with detected promotions
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Calculate baseline price per product (90th percentile — robust regular price)
    baseline_prices = df.groupby("stockcode")["price"].quantile(0.9).to_dict()
    df["baseline_price"] = df["stockcode"].map(baseline_prices)
    df["price_drop_pct"] = (df["baseline_price"] - df["price"]) / df["baseline_price"]

    # Flag potential promotional transactions
    df["is_promo"] = df["price_drop_pct"] >= price_change_threshold

    # Group by product and identify contiguous promo periods
    promotions = []

    for product in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == product].sort_values("date")
        prod_df = prod_df[prod_df["is_promo"]]

        if len(prod_df) == 0:
            continue

        # Find contiguous promo periods
        prod_df = prod_df.copy()
        prod_df["date_diff"] = prod_df["date"].diff().dt.days
        prod_df["new_period"] = (prod_df["date_diff"] > gap_threshold_days) | (
            prod_df["date_diff"].isna()
        )
        prod_df["promo_group"] = prod_df["new_period"].cumsum()

        for group_id, group in prod_df.groupby("promo_group"):
            start_date = group["date"].min()
            end_date = group["date"].max()
            duration = (end_date - start_date).days + 1

            if min_duration_days <= duration <= max_duration_days:
                # Calculate metrics during promo vs baseline
                promo_sales = group
                baseline_sales = df[(df["stockcode"] == product) & (~df["is_promo"])]

                promo_revenue = promo_sales["revenue"].sum()
                promo_qty = promo_sales["quantity"].sum()
                promo_orders = promo_sales["transaction_id"].nunique()
                promo_customers = promo_sales["customer_id"].nunique()
                avg_promo_price = promo_sales["price"].mean()
                avg_promo_discount = promo_sales["price_drop_pct"].mean()

                baseline_revenue = baseline_sales["revenue"].sum()
                baseline_qty = baseline_sales["quantity"].sum()
                baseline_orders = baseline_sales["transaction_id"].nunique()
                baseline_customers = baseline_sales["customer_id"].nunique()
                avg_baseline_price = (
                    baseline_sales["price"].mean() if len(baseline_sales) > 0 else 0
                )

                # Calculate lift against non-promo daily rate (excludes promo days)
                non_promo_days = df[~df["is_promo"]]["date"].nunique()
                if baseline_qty > 0 and non_promo_days > 0:
                    qty_lift = (promo_qty / duration) / (baseline_qty / non_promo_days) - 1
                else:
                    qty_lift = 0

                if baseline_revenue > 0 and non_promo_days > 0:
                    revenue_lift = (promo_revenue / duration) / (
                        baseline_revenue / non_promo_days
                    ) - 1
                else:
                    revenue_lift = 0

                promotions.append(
                    {
                        "stockcode": product,
                        "product_name": (
                            promo_sales["product"].iloc[0]
                            if "product" in promo_sales.columns
                            else product
                        ),
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration_days": duration,
                        "avg_discount_pct": avg_promo_discount * 100,
                        "promo_revenue": promo_revenue,
                        "baseline_revenue": baseline_revenue,
                        "promo_qty": promo_qty,
                        "baseline_qty": baseline_qty,
                        "promo_orders": promo_orders,
                        "baseline_orders": baseline_orders,
                        "promo_customers": promo_customers,
                        "baseline_customers": baseline_customers,
                        "qty_lift": qty_lift,
                        "revenue_lift": revenue_lift,
                        "avg_promo_price": avg_promo_price,
                        "avg_baseline_price": avg_baseline_price,
                    }
                )

    return pd.DataFrame(promotions)


def calculate_promotional_lift(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame = None,
    control_products: List[str] = None,
) -> Dict:
    """
    Calculate promotional lift using difference-in-differences or simple comparison.

    Args:
        transactions_df: Transaction data
        promo_periods: DataFrame with promotional periods (from detect_promotions)
        control_products: List of product IDs to use as control group

    Returns:
        Dictionary with lift metrics
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if promo_periods is None or len(promo_periods) == 0:
        # Auto-detect promotions
        from .promotional import detect_promotions

        promo_periods = detect_promotions(df)

    if len(promo_periods) == 0:
        return {"error": "No promotions detected"}

    # Create promo flag for each transaction
    df["is_promo"] = False
    for _, promo in promo_periods.iterrows():
        mask = (
            (df["stockcode"] == promo["stockcode"])
            & (df["date"] >= promo["start_date"])
            & (df["date"] <= promo["end_date"])
        )
        df.loc[mask, "is_promo"] = True

    # Separate promo and non-promo transactions
    promo_trans = df[df["is_promo"]]
    non_promo_trans = df[~df["is_promo"]]

    # Overall lift metrics
    promo_revenue = promo_trans["revenue"].sum()
    promo_qty = promo_trans["quantity"].sum()
    promo_orders = promo_trans["transaction_id"].nunique()
    promo_customers = promo_trans["customer_id"].nunique()

    non_promo_revenue = non_promo_trans["revenue"].sum()
    non_promo_qty = non_promo_trans["quantity"].sum()
    non_promo_orders = non_promo_trans["transaction_id"].nunique()
    non_promo_customers = non_promo_trans["customer_id"].nunique()

    # Calculate daily rates for fair comparison
    promo_days = promo_trans["date"].nunique()
    non_promo_days = non_promo_trans["date"].nunique()

    results = {
        "promo_transactions": len(promo_trans),
        "non_promo_transactions": len(non_promo_trans),
        "promo_revenue": promo_revenue,
        "non_promo_revenue": non_promo_revenue,
        "promo_qty": promo_qty,
        "non_promo_qty": non_promo_qty,
        "promo_orders": promo_orders,
        "non_promo_orders": non_promo_orders,
        "promo_customers": promo_customers,
        "non_promo_customers": non_promo_customers,
        "revenue_per_day_promo": promo_revenue / max(promo_days, 1),
        "revenue_per_day_non_promo": non_promo_revenue / max(non_promo_days, 1),
        "qty_per_day_promo": promo_qty / max(promo_days, 1),
        "qty_per_day_non_promo": non_promo_qty / max(non_promo_days, 1),
        "orders_per_day_promo": promo_orders / max(promo_days, 1),
        "orders_per_day_non_promo": non_promo_orders / max(non_promo_days, 1),
    }

    # Calculate lifts
    if results["revenue_per_day_non_promo"] > 0:
        results["revenue_lift_pct"] = (
            results["revenue_per_day_promo"] / results["revenue_per_day_non_promo"] - 1
        ) * 100
    else:
        results["revenue_lift_pct"] = 0

    if results["qty_per_day_non_promo"] > 0:
        results["qty_lift_pct"] = (
            results["qty_per_day_promo"] / results["qty_per_day_non_promo"] - 1
        ) * 100
    else:
        results["qty_lift_pct"] = 0

    # Statistical significance test (customer-level, Mann-Whitney U — no normality/independence assumption)
    if len(promo_trans) > 10 and len(non_promo_trans) > 10:
        promo_per_customer = promo_trans.groupby("customer_id")["revenue"].sum()
        non_promo_per_customer = non_promo_trans.groupby("customer_id")["revenue"].sum()

        if len(promo_per_customer) > 1 and len(non_promo_per_customer) > 1:
            u_stat, p_val = mannwhitneyu(
                promo_per_customer, non_promo_per_customer, alternative="two-sided"
            )
            results["u_statistic"] = u_stat
            results["p_value"] = p_val
            results["significant"] = p_val < 0.05

    # Product-level lift
    product_lifts = []
    for product in promo_trans["stockcode"].unique():
        promo_prod = promo_trans[promo_trans["stockcode"] == product]
        non_promo_prod = non_promo_trans[non_promo_trans["stockcode"] == product]

        if len(promo_prod) > 0 and len(non_promo_prod) > 0:
            promo_rev = promo_prod["revenue"].sum() / max(promo_prod["date"].nunique(), 1)
            non_promo_rev = non_promo_prod["revenue"].sum() / max(
                non_promo_prod["date"].nunique(), 1
            )

            if non_promo_rev > 0:
                lift = (promo_rev / non_promo_rev - 1) * 100
            else:
                lift = 0

            product_lifts.append(
                {
                    "stockcode": product,
                    "promo_revenue_per_day": promo_rev,
                    "non_promo_revenue_per_day": non_promo_rev,
                    "lift_pct": lift,
                    "promo_transactions": len(promo_prod),
                    "baseline_transactions": len(non_promo_prod),
                }
            )

    results["product_lifts"] = pd.DataFrame(product_lifts).sort_values("lift_pct", ascending=False)

    return results


def calculate_incremental_revenue(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    baseline_period_days: int = 30,
) -> pd.DataFrame:
    """
    Calculate incremental revenue from promotions by comparing to baseline period.

    Args:
        transactions_df: Transaction data
        promo_periods: Detected promotional periods
        baseline_period_days: Days before promotion to use as baseline

    Returns:
        DataFrame with incremental revenue per promotion
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    incremental = []

    for _, promo in promo_periods.iterrows():
        product = promo["stockcode"]
        start = promo["start_date"]
        end = promo["end_date"]
        duration = (end - start).days + 1

        # Baseline period (same duration before promotion)
        baseline_start = start - pd.Timedelta(duration, unit="D")
        baseline_end = start - pd.Timedelta(1, unit="D")

        # Promo sales
        promo_sales = df[(df["stockcode"] == product) & (df["date"] >= start) & (df["date"] <= end)]

        # Baseline sales
        baseline_sales = df[
            (df["stockcode"] == product)
            & (df["date"] >= baseline_start)
            & (df["date"] <= baseline_end)
        ]

        promo_revenue = promo_sales["revenue"].sum()
        promo_qty = promo_sales["quantity"].sum()
        promo_orders = promo_sales["transaction_id"].nunique()

        baseline_revenue = baseline_sales["revenue"].sum()
        baseline_qty = baseline_sales["quantity"].sum()
        baseline_orders = baseline_sales["transaction_id"].nunique()

        # Incremental
        inc_revenue = promo_revenue - baseline_revenue
        inc_qty = promo_qty - baseline_qty
        inc_orders = promo_orders - baseline_orders

        # Lift
        revenue_lift = (promo_revenue / baseline_revenue - 1) * 100 if baseline_revenue > 0 else 0
        qty_lift = (promo_qty / baseline_qty - 1) * 100 if baseline_qty > 0 else 0

        incremental.append(
            {
                "stockcode": product,
                "promo_start": start,
                "promo_end": end,
                "duration_days": duration,
                "promo_revenue": promo_revenue,
                "baseline_revenue": baseline_revenue,
                "incremental_revenue": inc_revenue,
                "revenue_lift_pct": revenue_lift,
                "incremental_qty": inc_qty,
                "qty_lift_pct": qty_lift,
                "incremental_orders": inc_orders,
            }
        )

    return pd.DataFrame(incremental)


def promotion_roi_analysis(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    promo_costs: Dict[str, float] = None,
    margin_pct: float = 0.3,
    promo_cost_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Calculate ROI of promotions.

    Args:
        transactions_df: Transaction data
        promo_periods: Detected promotional periods
        promo_costs: Dict mapping product to promotional cost (e.g., marketing spend, margin loss)
        margin_pct: Assumed profit margin (e.g., 0.3 = 30%)
        promo_cost_pct: Promotional cost as fraction of revenue (e.g., 0.15 = 15%)

    Returns:
        DataFrame with ROI metrics per promotion
    """
    # First get incremental revenue
    inc_revenue = calculate_incremental_revenue(transactions_df, promo_periods)

    if len(inc_revenue) == 0:
        return pd.DataFrame()

    # Calculate ROI
    results = []
    for _, row in inc_revenue.iterrows():
        product = row["stockcode"]

        # Estimate promo cost if not provided
        if promo_costs and product in promo_costs:
            promo_cost = promo_costs[product]
        else:
            promo_cost = row["incremental_revenue"] * promo_cost_pct

        incremental_profit = row["incremental_revenue"] * margin_pct - promo_cost
        roi_pct = (incremental_profit / promo_cost * 100) if promo_cost > 0 else 0.0

        results.append(
            {
                "stockcode": row["stockcode"],
                "incremental_revenue": row["incremental_revenue"],
                "incremental_profit": incremental_profit,
                "promo_cost": promo_cost,
                "roi_pct": roi_pct,
                "revenue_lift_pct": row["revenue_lift_pct"],
            }
        )

    return pd.DataFrame(results)


def halo_effect_analysis(
    transactions_df: pd.DataFrame, promo_periods: pd.DataFrame, window_days: int = 7
) -> pd.DataFrame:
    """
    Analyze halo effect - impact of promotions on related/non-promoted products.

    Args:
        transactions_df: Transaction data
        promo_periods: Promotional periods
        window_days: Days around promotion to analyze

    Returns:
        DataFrame with halo effect metrics per promo product
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Mark promo transactions
    df["is_promo"] = False
    for _, promo in promo_periods.iterrows():
        mask = (
            (df["stockcode"] == promo["stockcode"])
            & (df["date"] >= promo["start_date"])
            & (df["date"] <= promo["end_date"])
        )
        df.loc[mask, "is_promo"] = True

    # For each promo, look at other products in same transactions
    halo_results = []

    for _, promo in promo_periods.iterrows():
        promo_product = promo["stockcode"]
        start = promo["start_date"]
        end = promo["end_date"]

        # Get transactions with promo product
        promo_trans = df[
            (df["stockcode"] == promo_product) & (df["date"] >= start) & (df["date"] <= end)
        ]

        promo_txn_ids = promo_trans["transaction_id"].unique()

        # Get other products in those transactions
        basket_trans = df[
            (df["transaction_id"].isin(promo_txn_ids)) & (df["stockcode"] != promo_product)
        ]

        # Baseline: same basket co-occurrence filter in the 30 days before promo
        baseline_start = start - pd.Timedelta(window_days * 4, unit="D")
        baseline_end = start - pd.Timedelta(1, unit="D")
        pre_promo_products = df[
            (df["stockcode"] == promo_product)
            & (df["date"] >= baseline_start)
            & (df["date"] <= baseline_end)
        ]
        pre_promo_txn_ids = pre_promo_products["transaction_id"].unique()

        baseline_basket_trans = df[
            (df["transaction_id"].isin(pre_promo_txn_ids)) & (df["stockcode"] != promo_product)
        ]

        halo_products = (
            basket_trans.groupby("stockcode")
            .agg(
                halo_revenue=("revenue", "sum"),
                halo_qty=("quantity", "sum"),
                halo_orders=("transaction_id", "nunique"),
            )
            .reset_index()
        )

        baseline_products = (
            baseline_basket_trans.groupby("stockcode")
            .agg(
                base_revenue=("revenue", "sum"),
                base_qty=("quantity", "sum"),
                base_orders=("transaction_id", "nunique"),
            )
            .reset_index()
        )

        merged = halo_products.merge(baseline_products, on="stockcode", how="outer").fillna(0)
        merged["revenue_lift"] = (
            merged["halo_revenue"] / merged["base_revenue"].replace(0, np.nan) - 1
        ) * 100

        for _, row in merged.iterrows():
            halo_results.append(
                {
                    "promo_product": promo_product,
                    "halo_product": row["stockcode"],
                    "halo_revenue": row["halo_revenue"],
                    "base_revenue": row["base_revenue"],
                    "revenue_lift_pct": row["revenue_lift"],
                    "halo_orders": row["halo_orders"],
                    "base_orders": row["base_orders"],
                }
            )

    return pd.DataFrame(halo_results)


def promotion_timing_analysis(transactions_df: pd.DataFrame, promo_periods: pd.DataFrame) -> Dict:
    """
    Analyze optimal timing for promotions (day of week, month, etc.)

    Returns:
        Dict with keys 'by_day_of_week' and 'by_month', each a DataFrame
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["dow"] = df["date"].dt.dayofweek  # 0=Monday
    df["week_of_month"] = (df["date"].dt.day - 1) // 7 + 1
    df["month"] = df["date"].dt.month

    # Mark promo transactions
    df["is_promo"] = False
    for _, promo in promo_periods.iterrows():
        mask = (
            (df["stockcode"] == promo["stockcode"])
            & (df["date"] >= promo["start_date"])
            & (df["date"] <= promo["end_date"])
        )
        df.loc[mask, "is_promo"] = True

    # Analyze by day of week
    dow_analysis = (
        df[df["is_promo"]]
        .groupby("dow")
        .agg(
            promo_revenue=("revenue", "sum"),
            promo_orders=("transaction_id", "nunique"),
            promo_qty=("quantity", "sum"),
        )
        .reset_index()
    )

    dow_base = (
        df[~df["is_promo"]]
        .groupby("dow")
        .agg(
            base_revenue=("revenue", "sum"),
            base_orders=("transaction_id", "nunique"),
            base_qty=("quantity", "sum"),
        )
        .reset_index()
    )

    dow_merged = dow_analysis.merge(dow_base, on="dow", how="outer").fillna(0)
    dow_merged["revenue_lift"] = (
        dow_merged["promo_revenue"] / dow_merged["base_revenue"].replace(0, np.nan) - 1
    ) * 100

    dow_merged["day_name"] = dow_merged["dow"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )

    # Monthly analysis
    month_analysis = (
        df[df["is_promo"]]
        .groupby("month")
        .agg(promo_revenue=("revenue", "sum"), promo_orders=("transaction_id", "nunique"))
        .reset_index()
    )

    month_base = (
        df[~df["is_promo"]]
        .groupby("month")
        .agg(base_revenue=("revenue", "sum"), base_orders=("transaction_id", "nunique"))
        .reset_index()
    )

    month_merged = month_analysis.merge(month_base, on="month", how="outer").fillna(0)
    month_merged["revenue_lift"] = (
        month_merged["promo_revenue"] / month_merged["base_revenue"].replace(0, np.nan) - 1
    ) * 100

    return {"by_day_of_week": dow_merged, "by_month": month_merged}


def detect_promotions_adaptive(
    transactions_df: pd.DataFrame,
    baseline_window: int = 28,
    z_score_threshold: float = -2.0,
    min_duration_days: int = 3,
    max_duration_days: int = 30,
    gap_threshold_days: int = 1,
) -> pd.DataFrame:
    """Detect promotional periods using per-SKU rolling z-score.

    For each product, computes a rolling mean and std of price over the
    baseline window. Flags weeks where price z-score < z_score_threshold
    (i.e., price is unusually low relative to the product's own history).

    Args:
        transactions_df: Transaction data
        baseline_window: Rolling window size in days for mean/std estimation
        z_score_threshold: Z-score threshold (default -2.0)
        min_duration_days: Minimum days for a promotion
        max_duration_days: Maximum days for a promotion
        gap_threshold_days: Max gap between transactions to merge same promo

    Returns:
        DataFrame with detected promotions (same schema as detect_promotions)
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Daily price per product
    daily = df.groupby(["stockcode", "date"])["price"].mean().reset_index()
    daily = daily.sort_values(["stockcode", "date"])

    # Rolling statistics per product
    daily["rolling_mean"] = daily.groupby("stockcode")["price"].transform(
        lambda x: x.rolling(window=baseline_window, min_periods=5).mean()
    )
    daily["rolling_std"] = daily.groupby("stockcode")["price"].transform(
        lambda x: x.rolling(window=baseline_window, min_periods=5).std()
    )
    daily["z_score"] = (daily["price"] - daily["rolling_mean"]) / daily["rolling_std"].replace(0, 1)

    # Flag promo days
    daily["is_promo"] = daily["z_score"] < z_score_threshold

    # Merge back
    df = df.merge(
        daily[["stockcode", "date", "is_promo", "z_score"]], on=["stockcode", "date"], how="left"
    )
    df["is_promo"] = df["is_promo"].fillna(False)

    # Baseline price (90th percentile for reporting)
    baseline_prices = df.groupby("stockcode")["price"].quantile(0.9).to_dict()
    df["baseline_price"] = df["stockcode"].map(baseline_prices)
    df["price_drop_pct"] = (df["baseline_price"] - df["price"]) / df["baseline_price"]

    # Group by product and identify contiguous promo periods (same logic as detect_promotions)
    promotions = []

    for product in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == product].sort_values("date")
        prod_df = prod_df[prod_df["is_promo"]]

        if len(prod_df) == 0:
            continue

        prod_df = prod_df.copy()
        prod_df["date_diff"] = prod_df["date"].diff().dt.days
        prod_df["new_period"] = (prod_df["date_diff"] > gap_threshold_days) | (
            prod_df["date_diff"].isna()
        )
        prod_df["promo_group"] = prod_df["new_period"].cumsum()

        for group_id, group in prod_df.groupby("promo_group"):
            start_date = group["date"].min()
            end_date = group["date"].max()
            duration = (end_date - start_date).days + 1

            if min_duration_days <= duration <= max_duration_days:
                promo_sales = group
                baseline_sales = df[(df["stockcode"] == product) & (~df["is_promo"])]

                promo_revenue = promo_sales["revenue"].sum()
                promo_qty = promo_sales["quantity"].sum()
                promo_orders = promo_sales["transaction_id"].nunique()
                promo_customers = promo_sales["customer_id"].nunique()
                avg_promo_price = promo_sales["price"].mean()
                avg_promo_discount = promo_sales["price_drop_pct"].mean()

                baseline_revenue = baseline_sales["revenue"].sum()
                baseline_qty = baseline_sales["quantity"].sum()
                baseline_orders = baseline_sales["transaction_id"].nunique()
                baseline_customers = baseline_sales["customer_id"].nunique()
                avg_baseline_price = (
                    baseline_sales["price"].mean() if len(baseline_sales) > 0 else 0
                )

                non_promo_days = df[~df["is_promo"]]["date"].nunique()
                if baseline_qty > 0 and non_promo_days > 0:
                    qty_lift = (promo_qty / duration) / (baseline_qty / non_promo_days) - 1
                else:
                    qty_lift = 0

                if baseline_revenue > 0 and non_promo_days > 0:
                    revenue_lift = (promo_revenue / duration) / (
                        baseline_revenue / non_promo_days
                    ) - 1
                else:
                    revenue_lift = 0

                promotions.append(
                    {
                        "stockcode": product,
                        "product_name": (
                            promo_sales["product"].iloc[0]
                            if "product" in promo_sales.columns
                            else product
                        ),
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration_days": duration,
                        "avg_discount_pct": avg_promo_discount * 100,
                        "promo_revenue": promo_revenue,
                        "baseline_revenue": baseline_revenue,
                        "promo_qty": promo_qty,
                        "baseline_qty": baseline_qty,
                        "promo_orders": promo_orders,
                        "baseline_orders": baseline_orders,
                        "promo_customers": promo_customers,
                        "baseline_customers": baseline_customers,
                        "qty_lift": qty_lift,
                        "revenue_lift": revenue_lift,
                        "avg_promo_price": avg_promo_price,
                        "avg_baseline_price": avg_baseline_price,
"detection_method": "adaptive_zscore",
                }
            )

    return pd.DataFrame(promotions)


# ============================================================================
# PROMO BASELINE & INCREMENTALITY (Phase 2)
# ============================================================================


def compute_promo_baseline_stl(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    seasonal_period: int = 52,  # weekly seasonality
) -> pd.DataFrame:
    """Compute baseline sales using STL decomposition (trend + seasonal).

    Replaces simple price-comparison baseline with proper time-series decomposition.
    Returns DataFrame with baseline, actual, incremental per product-week.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        # Fallback to simple moving average baseline
        return _compute_promo_baseline_fallback(transactions_df, promo_periods)

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")

    # Weekly aggregation per product
    weekly = df.groupby(["stockcode", "week"]).agg(
        actual_units=("quantity", "sum"),
        actual_revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
    ).reset_index()

    # Promo flag per product-week
    promo_weeks = promo_periods.copy()
    promo_weeks["date"] = pd.to_datetime(promo_weeks["date"])
    promo_weeks["week"] = promo_weeks["date"].dt.to_period("W")
    promo_weekly = promo_weeks.groupby(["stockcode", "week"])["is_promo"].any().reset_index()

    weekly = weekly.merge(promo_weekly, on=["stockcode", "week"], how="left")
    weekly["is_promo"] = weekly["is_promo"].fillna(False)

    results = []
    for sku, sku_data in weekly.groupby("stockcode"):
        sku_data = sku_data.sort_values("week").reset_index(drop=True)

        if len(sku_data) < seasonal_period * 2:
            # Not enough data for STL, use simple baseline
            baseline_units = sku_data["actual_units"].rolling(4, min_periods=1).mean().shift(1)
            baseline_revenue = sku_data["actual_revenue"].rolling(4, min_periods=1).mean().shift(1)
        else:
            # STL decomposition on units
            units_series = sku_data["actual_units"].values
            try:
                stl = STL(units_series, seasonal=seasonal_period, period=seasonal_period, robust=True)
                res = stl.fit()
                # Baseline = trend + seasonal (no residual/remainder)
                baseline_units = res.trend + res.seasonal
                baseline_units = np.maximum(baseline_units, 0)  # No negative baseline

                # For revenue, use same approach or ratio
                revenue_series = sku_data["actual_revenue"].values
                stl_rev = STL(revenue_series, seasonal=seasonal_period, period=seasonal_period, robust=True)
                res_rev = stl_rev.fit()
                baseline_revenue = res_rev.trend + res_rev.seasonal
                baseline_revenue = np.maximum(baseline_revenue, 0)
            except Exception:
                baseline_units = sku_data["actual_units"].rolling(4, min_periods=1).mean().shift(1).fillna(0)
                baseline_revenue = sku_data["actual_revenue"].rolling(4, min_periods=1).mean().shift(1).fillna(0)

        sku_data["baseline_units"] = baseline_units
        sku_data["baseline_revenue"] = baseline_revenue
        sku_data["incremental_units"] = sku_data["actual_units"] - sku_data["baseline_units"]
        sku_data["incremental_revenue"] = sku_data["actual_revenue"] - sku_data["baseline_revenue"]
        sku_data["incrementality_pct"] = np.where(
            sku_data["actual_revenue"] > 0,
            sku_data["incremental_revenue"] / sku_data["actual_revenue"] * 100,
            0,
        )

        results.append(sku_data)

    return pd.concat(results, ignore_index=True)


def _compute_promo_baseline_fallback(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame,
) -> pd.DataFrame:
    """Fallback baseline: 4-week moving average excluding promo weeks."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")

    weekly = df.groupby(["stockcode", "week"]).agg(
        actual_units=("quantity", "sum"),
        actual_revenue=("revenue", "sum"),
    ).reset_index()

    promo_weeks = promo_periods.copy()
    promo_weeks["date"] = pd.to_datetime(promo_weeks["date"])
    promo_weeks["week"] = promo_weeks["date"].dt.to_period("W")
    promo_weekly = promo_weeks.groupby(["stockcode", "week"])["is_promo"].any().reset_index()

    weekly = weekly.merge(promo_weekly, on=["stockcode", "week"], how="left")
    weekly["is_promo"] = weekly["is_promo"].fillna(False)

    # For non-promo weeks, use as baseline; for promo weeks, use rolling avg of prior non-promo
    results = []
    for sku, sku_data in weekly.groupby("stockcode"):
        sku_data = sku_data.sort_values("week").reset_index(drop=True)
        baseline_units = []
        baseline_revenue = []

        for i, row in sku_data.iterrows():
            if row["is_promo"]:
                # Look back at prior non-promo weeks
                prior_non_promo = sku_data[(sku_data.index < i) & (~sku_data["is_promo"])]
                if len(prior_non_promo) >= 2:
                    baseline_units.append(prior_non_promo["actual_units"].tail(4).mean())
                    baseline_revenue.append(prior_non_promo["actual_revenue"].tail(4).mean())
                else:
                    baseline_units.append(sku_data["actual_units"].mean())
                    baseline_revenue.append(sku_data["actual_revenue"].mean())
            else:
                baseline_units.append(row["actual_units"])
                baseline_revenue.append(row["actual_revenue"])

        sku_data["baseline_units"] = baseline_units
        sku_data["baseline_revenue"] = baseline_revenue
        sku_data["incremental_units"] = sku_data["actual_units"] - sku_data["baseline_units"]
        sku_data["incremental_revenue"] = sku_data["actual_revenue"] - sku_data["baseline_revenue"]
        sku_data["incrementality_pct"] = np.where(
            sku_data["actual_revenue"] > 0,
            sku_data["incremental_revenue"] / sku_data["actual_revenue"] * 100,
            0,
        )

        results.append(sku_data)

    return pd.concat(results, ignore_index=True)


def compute_incrementality_waterfall(
    baseline_df: pd.DataFrame,
    halo_revenue: Optional[pd.DataFrame] = None,
    cannibalization_revenue: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute True Incrementality Waterfall components.

    Waterfall: Baseline → +Incremental → +Halo → -Cannibalization = Net Incremental
    """
    # Aggregate by product
    agg = baseline_df.groupby("stockcode").agg(
        baseline_revenue=("baseline_revenue", "sum"),
        actual_revenue=("actual_revenue", "sum"),
        incremental_revenue=("incremental_revenue", "sum"),
        incremental_units=("incremental_units", "sum"),
        avg_incrementality_pct=("incrementality_pct", "mean"),
    ).reset_index()

    # Add halo revenue if provided
    if halo_revenue is not None and not halo_revenue.empty:
        agg = agg.merge(
            halo_revenue.groupby("promo_product")["halo_revenue"].sum().reset_index()
            .rename(columns={"promo_product": "stockcode", "halo_revenue": "halo_revenue"}),
            on="stockcode", how="left"
        )
    else:
        agg["halo_revenue"] = 0

    # Add cannibalization if provided
    if cannibalization_revenue is not None and not cannibalization_revenue.empty:
        agg = agg.merge(
            cannibalization_revenue.groupby("from_product")["cannibalization_revenue"].sum().reset_index()
            .rename(columns={"from_product": "stockcode"}),
            on="stockcode", how="left"
        )
    else:
        agg["cannibalization_revenue"] = 0

    agg["halo_revenue"] = agg["halo_revenue"].fillna(0)
    agg["cannibalization_revenue"] = agg["cannibalization_revenue"].fillna(0)

    agg["net_incremental_revenue"] = (
        agg["incremental_revenue"] + agg["halo_revenue"] - agg["cannibalization_revenue"]
    )
    agg["roi"] = np.where(
        agg["baseline_revenue"] > 0,
        agg["net_incremental_revenue"] / agg["baseline_revenue"],
        0,
    )

    return agg


def compute_promo_calendar_heatmap(
    transactions_df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    top_n_products: int = 30,
) -> pd.DataFrame:
    """Generate promo calendar heatmap matrix: weeks x products."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W")

    # Get top products by revenue
    top_products = (
        df.groupby("stockcode")["price"].count()
        .sort_values(ascending=False)
        .head(top_n_products)
        .index.tolist()
    )

    # All weeks in data
    all_weeks = sorted(df["week"].unique())

    # Promo weeks per product
    promo_weeks = promo_periods.copy()
    promo_weeks["date"] = pd.to_datetime(promo_weeks["date"])
    promo_weeks["week"] = promo_weeks["date"].dt.to_period("W")
    promo_matrix = promo_weeks[promo_weeks["stockcode"].isin(top_products)]
    promo_matrix = promo_matrix.groupby(["stockcode", "week"])["is_promo"].any().unstack(fill_value=False)

    # Reindex to all weeks
    promo_matrix = promo_matrix.reindex(columns=all_weeks, fill_value=False)

    return promo_matrix
