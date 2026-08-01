"""CLV Customer Analytics - BG/NBD model output joined with basket metrics.

Computes customer-level CLV predictions and joins with IPT-CV, entropy,
and other behavioral metrics for a complete customer health view.
"""

import numpy as np
import pandas as pd

from src.analytics.basket_metrics import compute_customer_entropy, compute_ipt_cv
from src.analytics.segmentation import predict_clv_bg_nbd


def compute_clv_customer_df(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
) -> pd.DataFrame:
    """Compute CLV customer dataframe with BG/NBD + basket metrics.

    Returns one row per customer with:
    - BG/NBD outputs: p_alive, expected_orders_90d, expected_avg_value, predicted_clv, clv_segment
    - Basket metrics: ipt_cv, entropy_score
    - Derived: avg_order_value, clv_12m
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Filter out customers with zero/negative total revenue for BG/NBD
    customer_revenue = df.groupby("customer_id")["revenue"].sum()
    valid_customers = customer_revenue[customer_revenue > 0].index
    df_valid = df[df["customer_id"].isin(valid_customers)].copy()

    # Also ensure positive average revenue per transaction for Gamma-Gamma
    customer_avg_revenue = df_valid.groupby("customer_id")["revenue"].mean()
    valid_customers_gg = customer_avg_revenue[customer_avg_revenue > 0].index
    df_valid = df_valid[df_valid["customer_id"].isin(valid_customers_gg)].copy()

    if len(df_valid) == 0:
        return pd.DataFrame()

    # 1. BG/NBD CLV prediction
    try:
        clv_result, diagnostics = predict_clv_bg_nbd(
            df_valid,
            prediction_horizon_days=prediction_horizon_days,
            freq=freq,
        )
    except Exception as e:
        raise ValueError(f"BG/NBD model failed: {e}")

    # 2. Customer entropy
    entropy_df = compute_customer_entropy(df_valid)

    # 3. IPT-CV
    ipt_df = compute_ipt_cv(df_valid)

    # 4. Base customer metrics
    customer_metrics = (
        df_valid.groupby("customer_id")
        .agg(
            frequency=("transaction_id", "nunique"),
            total_revenue=("revenue", "sum"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
        )
        .reset_index()
    )
    customer_metrics["avg_order_value"] = (
        customer_metrics["total_revenue"] / customer_metrics["frequency"]
    ).replace([np.inf, -np.inf], np.nan)
    customer_metrics["customer_lifetime_days"] = (
        customer_metrics["last_purchase"] - customer_metrics["first_purchase"]
    ).dt.days
    customer_metrics["recency_days"] = (
        df["date"].max() - customer_metrics["last_purchase"]
    ).dt.days

    # 5. Merge all together
    result = clv_result.merge(customer_metrics, on="customer_id", how="left")
    result = result.merge(entropy_df, on="customer_id", how="left")
    result = result.merge(ipt_df, on="customer_id", how="left")

    # 6. Compute 12-month CLV projection
    # expected_orders_90d * 4 * avg_order_value
    result["clv_12m"] = (
        result["expected_avg_value"] *
        result["predicted_purchases"] *
        4  # 4 quarters per year
    )

    # 7. CLV segment labels (2x2: p_alive x clv_12m)
    result["clv_segment"] = result.apply(_label_clv_segment, axis=1)

    # 8. Reorder columns
    cols_order = [
        "customer_id",
        "frequency",
        "recency_days",
        "customer_lifetime_days",
        "total_revenue",
        "avg_order_value",
        "p_alive",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "clv_12m",
        "clv_segment",
        "ipt_mean",
        "ipt_std",
        "ipt_cv",
        "entropy",
        "normalized_entropy",
    ]
    # Only keep columns that exist
    cols_order = [c for c in cols_order if c in result.columns]

    return result[cols_order].sort_values("clv_12m", ascending=False).reset_index(drop=True)


def _label_clv_segment(row) -> str:
    """Label customer segment based on p_alive and predicted CLV."""
    p_alive = row.get("p_alive", 0)
    clv_12m = row.get("clv_12m", 0)

    # Compute median CLV for threshold
    # This will be set properly in the UI after computing all rows
    # Use a default threshold here, UI will recompute
    clv_median = getattr(_label_clv_segment, "clv_median", 0)

    if p_alive >= 0.5 and clv_12m >= clv_median:
        return "Champions"
    elif p_alive >= 0.5 and clv_12m < clv_median:
        return "Promising"
    elif p_alive < 0.5 and clv_12m >= clv_median:
        return "At Risk"
    else:
        return "Lost"


def compute_clv_segment_profiles(clv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate profiles per CLV segment."""
    # Recompute median for labeling
    clv_median = clv_df["clv_12m"].median()
    _label_clv_segment.clv_median = clv_median

    # Relabel with correct median
    clv_df = clv_df.copy()
    clv_df["clv_segment"] = clv_df.apply(_label_clv_segment, axis=1)

    profiles = (
        clv_df.groupby("clv_segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_p_alive=("p_alive", "mean"),
            avg_predicted_purchases=("predicted_purchases", "mean"),
            avg_expected_avg_value=("expected_avg_value", "mean"),
            avg_predicted_clv=("predicted_clv", "mean"),
            avg_clv_12m=("clv_12m", "mean"),
            total_clv_12m=("clv_12m", "sum"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency_days", "mean"),
            avg_ipt_cv=("ipt_cv", "mean"),
            avg_entropy=("normalized_entropy", "mean"),
        )
        .reset_index()
    )

    profiles["customer_share"] = profiles["n_customers"] / profiles["n_customers"].sum()
    profiles["revenue_share"] = profiles["total_clv_12m"] / profiles["total_clv_12m"].sum()

    return profiles


def get_rfm_heatmap_data(clv_df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """Prepare RFM heatmap data: recency decile x frequency decile -> mean p_alive."""
    df = clv_df.copy()

    # Create deciles
    df["recency_decile"] = pd.qcut(
        df["recency_days"].rank(method="first"),
        q=n_deciles,
        labels=[f"D{i+1}" for i in range(n_deciles)],
        duplicates="drop",
    )
    df["frequency_decile"] = pd.qcut(
        df["frequency"].rank(method="first"),
        q=n_deciles,
        labels=[f"D{i+1}" for i in range(n_deciles)],
        duplicates="drop",
    )

    heatmap = (
        df.groupby(["recency_decile", "frequency_decile"])
        .agg(
            mean_p_alive=("p_alive", "mean"),
            n_customers=("customer_id", "count"),
        )
        .reset_index()
    )

    # Pivot for heatmap
    heatmap_pivot = heatmap.pivot(
        index="recency_decile", columns="frequency_decile", values="mean_p_alive"
    )

    return heatmap_pivot