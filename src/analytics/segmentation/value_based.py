"""Value-Based Segmentation with CLV prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.schemas import VALUE_BASED_SEGMENTS, check


def value_based_segmentation(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Value-based segmentation with predicted CLV.

    Args:
        transactions_df: Transaction data
        prediction_horizon_days: Horizon for prediction; used to define historical vs future
        as_of_date: If provided, compute features ONLY on data up to this date
            (temporal holdout to prevent leakage). Use for training on historical data.

    Returns:
        DataFrame with features, predicted CLV, and value_segment per customer.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Apply temporal holdout if as_of_date provided
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(prediction_horizon_days, unit="D")

    # Historical (before cutoff) and future (after cutoff)
    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

    if hist.empty:
        raise ValueError(
            f"No historical data before cutoff ({cutoff_date.date()}). "
            f"Reduce prediction_horizon_days (currently {prediction_horizon_days})."
        )

    # Historical features
    features = (
        hist.groupby("customer_id")
        .agg(
            recency=("date", lambda x: int((cutoff_date - x.max()).days)),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            lifetime_days=("date", lambda x: int((x.max() - x.min()).days + 1)),
        )
        .reset_index()
    )

    # Ensure lifetime_days is numeric
    features["lifetime_days"] = pd.to_numeric(features["lifetime_days"], errors="coerce").fillna(1)

    # Ensure recency is numeric (days as int)
    features["recency"] = pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)

    # Future actuals (for validation)
    future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
    future_rev.columns = ["customer_id", "future_revenue"]
    features = features.merge(future_rev, on="customer_id", how="left").fillna(
        {"future_revenue": 0}
    )

    # CLV prediction with churn adjustment
    # Annualized historical spend per customer
    annual_value = features["monetary"] / (features["lifetime_days"].clip(lower=1) / 365)
    # Survival probability: customers with long recency relative to lifetime likely churned
    survival_prob = np.clip(
        1 - features["recency"] / (features["lifetime_days"].clip(lower=1) + features["recency"]),
        0,
        1,
    )
    features["predicted_clv"] = annual_value * survival_prob * 2  # 2-year horizon

    # Segments — first matching condition wins (priority: high CLV + recent > loyal > new > churned)
    conditions = [
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.8))
        & (features["recency"] < 30)
        & (features["frequency"] > 1),
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.6))
        & (features["recency"] < 60)
        & (features["frequency"] > 1),
        (features["frequency"] > 5) & (features["recency"] < 90),
        (features["frequency"] == 1) & (features["recency"] < 30),
        (features["recency"] > 180),
    ]
    choices = ["VIP", "High Potential", "Loyal", "New", "Churned"]
    features["value_segment"] = np.select(conditions, choices, default="Regular")

    return check(features, VALUE_BASED_SEGMENTS)
